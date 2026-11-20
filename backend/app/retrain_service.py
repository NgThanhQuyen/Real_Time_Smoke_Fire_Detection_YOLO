from __future__ import annotations

import base64
import io
import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import ROOT, RUNTIME, runtime_file, settings
from .models import (
    AuditLog,
    Feedback,
    FeedbackReview,
    ModelVersion,
    RetrainRun,
    new_id,
    utcnow,
)
from .serializers import iso

logger = logging.getLogger(__name__)

KAGGLE_KERNEL_SLUG = "flare-retrain-yolo11n"
BASE_DATASET_SOURCE = "quynnguynthanh/finalsfd-yolo-dataset"
BASELINE_MLFLOW_RUN_ID = "fafc4786b6f24e2f851b1164ac3574f2"


def get_kaggle_client():
    """Khởi tạo và xác thực client Kaggle API từ biến môi trường."""
    username = os.getenv("KAGGLE_USERNAME") or ""
    key = os.getenv("KAGGLE_KEY") or ""
    token = os.getenv("KAGGLE_API_TOKEN") or ""

    if not username or (not key and not token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chưa cấu hình KAGGLE_USERNAME và KAGGLE_KEY / KAGGLE_API_TOKEN trong file .env",
        )

    # Đảm bảo biến môi trường được truyền đúng cho SDK
    os.environ["KAGGLE_USERNAME"] = username
    if key:
        os.environ["KAGGLE_KEY"] = key
    if token:
        os.environ["KAGGLE_API_TOKEN"] = token

    import kaggle

    api = kaggle.KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Xác thực Kaggle API thất bại: {exc}",
        )
    return api


def get_retrain_statistics(db: Session) -> dict[str, Any]:
    """Tổng hợp số liệu thống kê dữ liệu feedback sẵn sàng cho huấn luyện lại."""
    rows = db.execute(
        select(Feedback, FeedbackReview)
        .join(FeedbackReview, Feedback.id == FeedbackReview.feedback_id)
        .where(FeedbackReview.decision == "accepted")
    ).all()

    hard_negatives = 0
    hard_positives = 0
    total_boxes = 0

    for item, review in rows:
        if review.annotation_status == "background":
            hard_negatives += 1
        elif review.annotation_status == "objects":
            hard_positives += 1
            if review.annotations:
                total_boxes += len(review.annotations)

    pending_count = db.scalar(
        select(Feedback)
        .where(Feedback.status == "pending")
    )

    active_model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active").limit(1))

    kaggle_configured = bool(
        os.getenv("KAGGLE_USERNAME") and (os.getenv("KAGGLE_KEY") or os.getenv("KAGGLE_API_TOKEN"))
    )

    latest_run = db.scalar(
        select(RetrainRun).order_by(RetrainRun.created_at.desc()).limit(1)
    )

    return {
        "accepted_total": len(rows),
        "hard_negatives": hard_negatives,
        "hard_positives": hard_positives,
        "total_boxes": total_boxes,
        "pending_reviews": db.query(Feedback).filter(Feedback.status == "pending").count(),
        "active_model_name": active_model.name if active_model else "Chưa có",
        "active_model_id": active_model.id if active_model else None,
        "kaggle_configured": kaggle_configured,
        "kaggle_username": os.getenv("KAGGLE_USERNAME", ""),
        "latest_run": retrain_run_data(latest_run) if latest_run else None,
    }


def build_feedback_bundle(db: Session) -> dict[str, Any]:
    """Gom toàn bộ ảnh + nhãn feedback đã duyệt thành file zip và chuỗi base64."""
    rows = db.execute(
        select(Feedback, FeedbackReview)
        .join(FeedbackReview, Feedback.id == FeedbackReview.feedback_id)
        .where(
            FeedbackReview.decision == "accepted",
            FeedbackReview.annotation_status.in_(["objects", "background"]),
        )
        .order_by(Feedback.created_at)
    ).all()

    if not rows:
        raise ValueError("Chưa có phản hồi nào được duyệt để làm dữ liệu huấn luyện lại")

    zip_buffer = io.BytesIO()
    manifest = []

    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item, review in rows:
            img_rel = f"images/{item.id}.jpg"
            lbl_rel = f"labels/{item.id}.txt"

            archive.write(runtime_file(item.image_path), img_rel)
            lines = []
            if review.annotation_status == "objects" and review.annotations:
                for box in review.annotations:
                    width = box["x2"] - box["x1"]
                    height = box["y2"] - box["y1"]
                    cx = box["x1"] + width / 2
                    cy = box["y1"] + height / 2
                    lines.append(f"{box['class_id']} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")

            archive.writestr(lbl_rel, "\n".join(lines) + ("\n" if lines else ""))
            manifest.append(
                {
                    "feedback_id": item.id,
                    "type": item.type,
                    "annotation_status": review.annotation_status,
                    "boxes_count": len(lines),
                    "image": img_rel,
                    "label": lbl_rel,
                }
            )

        archive.writestr("classes.json", json.dumps({"0": "smoke", "1": "fire"}, indent=2))
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr(
            "dataset.yaml",
            "path: .\ntrain: images\nval: images\nnames:\n  0: smoke\n  1: fire\n",
        )

    zip_bytes = zip_buffer.getvalue()
    b64_str = base64.b64encode(zip_bytes).decode("ascii")

    # Lưu bản copy local tại runtime/retrain_feedback.zip
    local_zip = RUNTIME / "retrain_feedback.zip"
    local_zip.write_bytes(zip_bytes)

    return {
        "count": len(manifest),
        "manifest": manifest,
        "zip_b64": b64_str,
        "zip_size_kb": round(len(zip_bytes) / 1024, 1),
    }


def generate_retrain_notebook_content(
    feedback_b64: str,
    epochs: int = 25,
    run_tag: str = "",
) -> str:
    """Tạo cấu trúc JSON của Jupyter Notebook hoàn chỉnh để chạy trên Kaggle."""
    mlflow_user = settings.mlflow_tracking_username or os.getenv("MLFLOW_TRACKING_USERNAME", "")
    mlflow_pass = settings.mlflow_tracking_password or os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    mlflow_uri = (
        settings.mlflow_tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI", "")
        or "https://dagshub.com/nguyenthanhquyen145/Real_Time_Smoke_Fire_Detection_YOLO.mlflow"
    )

    if not mlflow_user or not mlflow_pass:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Chưa cấu hình MLFLOW_TRACKING_USERNAME hoặc MLFLOW_TRACKING_PASSWORD trong file .env để đồng bộ kết quả lên DagsHub.",
        )

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 🚀 FLARE — Huấn luyện lại YOLO11n trên dữ liệu phản hồi (Continual Learning)\n",
                "> Notebook này được hệ thống FLARE Web tự động sinh và kích hoạt trên Kaggle GPU.\n",
                f"> **Mục tiêu:** Fine-tune mô hình phát hiện Khói/Lửa bằng cách trộn dữ liệu phản hồi đã duyệt (Hard Negatives + Hard Positives) với tập huấn luyện gốc.",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Cài đặt các thư viện cần thiết\n",
                "!pip install -q ultralytics mlflow\n",
                "import os, sys, shutil, zipfile, io, base64, time, glob\n",
                "from pathlib import Path\n",
                "import torch\n",
                "from ultralytics import YOLO\n",
                "import mlflow\n",
                "\n",
                "print('GPU sẵn sàng:', torch.cuda.is_available())\n",
                "if torch.cuda.is_available():\n",
                "    print('Thiết bị GPU:', torch.cuda.get_device_name(0))\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Giải nén tập dữ liệu Feedback được Web đóng gói tự động\n",
                f'FEEDBACK_B64 = "{feedback_b64}"\n',
                "WORK_DIR = Path('/kaggle/working')\n",
                "FEEDBACK_DIR = WORK_DIR / 'feedback_dataset'\n",
                "FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "zip_data = base64.b64decode(FEEDBACK_B64)\n",
                "with zipfile.ZipFile(io.BytesIO(zip_data)) as z:\n",
                "    z.extractall(FEEDBACK_DIR)\n",
                "\n",
                "fb_images = list((FEEDBACK_DIR / 'images').glob('*.*'))\n",
                "fb_labels = list((FEEDBACK_DIR / 'labels').glob('*.txt'))\n",
                "print(f'✅ Đã giải nén thành công {len(fb_images)} ảnh và {len(fb_labels)} file nhãn feedback!')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 3. Tải checkpoint Baseline Model từ DagsHub MLflow\n",
                f'os.environ["MLFLOW_TRACKING_USERNAME"] = "{mlflow_user}"\n',
                f'os.environ["MLFLOW_TRACKING_PASSWORD"] = "{mlflow_pass}"\n',
                f'mlflow.set_tracking_uri("{mlflow_uri}")\n',
                "\n",
                "print('Đang tải checkpoint Baseline Model từ DagsHub MLflow...')\n",
                "baseline_pt = WORK_DIR / 'baseline_best.pt'\n",
                "try:\n",
                "    # Tải artifact từ run gốc\n",
                f'    downloaded = mlflow.artifacts.download_artifacts(run_id="{BASELINE_MLFLOW_RUN_ID}", artifact_path="model/best.pt", dst_path=str(WORK_DIR))\n',
                "    if Path(downloaded).is_file():\n",
                "        shutil.copy2(downloaded, baseline_pt)\n",
                "        print('✅ Tải baseline weights thành công từ DagsHub!')\n",
                "except Exception as e:\n",
                "    print('⚠️ Tải từ MLflow gặp lỗi, chuyển sang nạp weights mặc định:', e)\n",
                "    # Fallback dự phòng nếu có checkpoint cục bộ hoặc tải yolo11n gốc\n",
                "    baseline_pt = 'yolo11n.pt'\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 4. Xác định tập dữ liệu huấn luyện gốc (Base Dataset)\n",
                "BASE_CANDIDATES = [\n",
                "    Path('/kaggle/input/finalsfd-yolo-dataset/FinalSFD_YOLO_Dataset'),\n",
                "    Path('/kaggle/input/datasets/quynnguynthanh/finalsfd-yolo-dataset/FinalSFD_YOLO_Dataset'),\n",
                "    Path('/kaggle/input/finalsfd-yolo-dataset'),\n",
                "]\n",
                "BASE_DIR = None\n",
                "for cand in BASE_CANDIDATES:\n",
                "    if cand.is_dir() and (cand / 'images' / 'train').is_dir():\n",
                "        BASE_DIR = cand\n",
                "        break\n",
                "\n",
                "if not BASE_DIR:\n",
                "    print('⚠️ Không tìm thấy FinalSFD_YOLO_Dataset, tìm kiếm các thư mục trong /kaggle/input:')\n",
                "    print(list(Path('/kaggle/input').glob('*')))\n",
                "    # Fallback: sử dụng trực tiếp thư mục feedback nếu dataset gốc chưa mount\n",
                "    BASE_DIR = FEEDBACK_DIR\n",
                "else:\n",
                "    print(f'✅ Đã phát hiện Base Dataset tại: {BASE_DIR}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 5. Trộn dữ liệu (Data Mixing / Experience Replay)\n",
                "MERGED_DIR = WORK_DIR / 'dataset_retrain'\n",
                "MERGED_TRAIN_IMG = MERGED_DIR / 'images' / 'train'\n",
                "MERGED_TRAIN_LBL = MERGED_DIR / 'labels' / 'train'\n",
                "MERGED_VAL_IMG = MERGED_DIR / 'images' / 'val'\n",
                "MERGED_VAL_LBL = MERGED_DIR / 'labels' / 'val'\n",
                "\n",
                "MERGED_TRAIN_IMG.mkdir(parents=True, exist_ok=True)\n",
                "MERGED_TRAIN_LBL.mkdir(parents=True, exist_ok=True)\n",
                "MERGED_VAL_IMG.mkdir(parents=True, exist_ok=True)\n",
                "MERGED_VAL_LBL.mkdir(parents=True, exist_ok=True)\n",
                "\n",
                "# Tạo symlink hoặc copy từ Base Dataset để tiết kiệm dung lượng đĩa\n",
                "if BASE_DIR != FEEDBACK_DIR:\n",
                "    print('Đang liên kết dữ liệu huấn luyện gốc...')\n",
                "    for p in (BASE_DIR / 'images' / 'train').glob('*.*'):\n",
                "        dst = MERGED_TRAIN_IMG / p.name\n",
                "        if not dst.exists(): os.symlink(p, dst)\n",
                "    for p in (BASE_DIR / 'labels' / 'train').glob('*.txt'):\n",
                "        dst = MERGED_TRAIN_LBL / p.name\n",
                "        if not dst.exists(): os.symlink(p, dst)\n",
                "    for p in (BASE_DIR / 'images' / 'val').glob('*.*'):\n",
                "        dst = MERGED_VAL_IMG / p.name\n",
                "        if not dst.exists(): os.symlink(p, dst)\n",
                "    for p in (BASE_DIR / 'labels' / 'val').glob('*.txt'):\n",
                "        dst = MERGED_VAL_LBL / p.name\n",
                "        if not dst.exists(): os.symlink(p, dst)\n",
                "\n",
                "# Đưa toàn bộ các mẫu feedback vào tập Train\n",
                "print('Đang hòa trộn các mẫu feedback đã kiểm duyệt vào tập Train...')\n",
                "for p in (FEEDBACK_DIR / 'images').glob('*.*'):\n",
                "    shutil.copy2(p, MERGED_TRAIN_IMG / f'fb_{p.name}')\n",
                "for p in (FEEDBACK_DIR / 'labels').glob('*.txt'):\n",
                "    shutil.copy2(p, MERGED_TRAIN_LBL / f'fb_{p.name}')\n",
                "\n",
                "# Tạo file retrain_data.yaml chuẩn YOLO\n",
                "yaml_content = f'''# Cấu hình tự động sinh cho Fine-tuning FLARE\n",
                "path: {MERGED_DIR.as_posix()}\n",
                "train: images/train\n",
                "val: images/val\n",
                "test: images/val\n",
                "names:\n",
                "  0: smoke\n",
                "  1: fire\n",
                "'''\n",
                "DATA_YAML = WORK_DIR / 'retrain_data.yaml'\n",
                "DATA_YAML.write_text(yaml_content, encoding='utf-8')\n",
                "print(f'✅ Hoàn tất hòa trộn dữ liệu! Tổng ảnh train: {len(list(MERGED_TRAIN_IMG.glob(\"*.*\")))}')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 6. Khởi chạy Fine-tuning với YOLO11 (Continual Learning)\n",
                "print('🚀 Bắt đầu quá trình Fine-tuning...')\n",
                "model = YOLO(str(baseline_pt))\n",
                "\n",
                "# Siêu tham số tối ưu cho fine-tuning: lr0 thấp, freeze 10 layers đầu, 25 epochs\n",
                f"EPOCHS = {epochs}\n",
                "results = model.train(\n",
                "    data=str(DATA_YAML),\n",
                "    epochs=EPOCHS,\n",
                "    imgsz=640,\n",
                "    batch=32,\n",
                "    lr0=0.002,\n",
                "    lrf=0.01,\n",
                "    freeze=10,\n",
                "    device=0 if torch.cuda.is_available() else 'cpu',\n",
                "    project='/kaggle/working/retrain_run',\n",
                "    name='exp',\n",
                "    exist_ok=True,\n",
                "    save=True,\n",
                "    plots=True,\n",
                ")\n",
                "print('✅ Huấn luyện hoàn tất!')\n",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 7. Đánh giá và Tự động đẩy kết quả lên DagsHub MLflow\n",
                "best_weights = Path('/kaggle/working/retrain_run/exp/weights/best.pt')\n",
                "if not best_weights.is_file():\n",
                "    best_weights = Path('/kaggle/working/retrain_run/exp/weights/last.pt')\n",
                "\n",
                "print('Đang đánh giá mô hình mới trên tập val...')\n",
                "eval_model = YOLO(str(best_weights))\n",
                "val_metrics = eval_model.val(data=str(DATA_YAML), split='val', imgsz=640)\n",
                "\n",
                f"timestamp = time.strftime('%Y%m%d_%H%M%S')\n",
                f"run_name = f'retrain_yolo11n_{{timestamp}}'\n",
                "print(f'Đang đẩy mô hình {{run_name}} lên DagsHub MLflow...')\n",
                "\n",
                "with mlflow.start_run(run_name=run_name):\n",
                "    mlflow.log_params({\n",
                "        'epochs': EPOCHS,\n",
                "        'imgsz': 640,\n",
                "        'batch': 32,\n",
                "        'lr0': 0.002,\n",
                "        'freeze': 10,\n",
                "        'base_model': 'final_sfd_yolo11n',\n",
                "        'feedback_count': len(fb_images),\n",
                "    })\n",
                "    mlflow.log_metrics({\n",
                "        'mAP50': float(val_metrics.box.map50),\n",
                "        'mAP50-95': float(val_metrics.box.map),\n",
                "        'precision': float(val_metrics.box.mp),\n",
                "        'recall': float(val_metrics.box.mr),\n",
                "    })\n",
                "    if best_weights.is_file():\n",
                "        mlflow.log_artifact(str(best_weights), artifact_path='model')\n",
                "        print('✅ Đã đẩy file trọng số weights/best.pt lên MLflow!')\n",
                "\n",
                "print('🎉 TOÀN BỘ CHU TRÌNH HUẤN LUYỆN LẠI ĐÃ HOÀN TẤT THÀNH CÔNG!')\n",
            ],
        },
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    return json.dumps(notebook_dict, ensure_ascii=False, indent=2)


def trigger_kaggle_retrain(
    db: Session,
    actor_id: str | None = None,
    epochs: int = 25,
) -> RetrainRun:
    """Đóng gói dữ liệu feedback, sinh notebook và kích hoạt chạy trên Kaggle GPU."""
    # 1. Kiểm tra xem có đợt retrain nào đang chạy không
    existing_running = db.scalar(
        select(RetrainRun).where(RetrainRun.status.in_(["queued", "running"])).limit(1)
    )
    if existing_running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Đang có đợt huấn luyện lại đang chạy (ID: {existing_running.id[:8]}, Kaggle: {existing_running.kaggle_status})",
        )

    # 2. Đóng gói dữ liệu feedback
    bundle = build_feedback_bundle(db)

    # 3. Sinh nội dung notebook với dữ liệu feedback đã nhúng
    notebook_json = generate_retrain_notebook_content(
        feedback_b64=bundle["zip_b64"],
        epochs=epochs,
        run_tag=f"retrain_{int(time.time())}",
    )

    # 4. Lưu notebook cục bộ vào notebooks/model_training/
    nb_dir = ROOT / "notebooks" / "model_training"
    nb_dir.mkdir(parents=True, exist_ok=True)
    nb_path = nb_dir / "smoke-fire-retrain-yolo11n.ipynb"
    nb_path.write_text(notebook_json, encoding="utf-8")

    # 5. Chuẩn bị thư mục tạm để gọi Kaggle API kernels_push
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kernel_slug = f"{kaggle_user}/{KAGGLE_KERNEL_SLUG}"

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        code_file = temp_path / "smoke-fire-retrain-yolo11n.ipynb"
        code_file.write_text(notebook_json, encoding="utf-8")

        meta = {
            "id": kernel_slug,
            "title": "flare-retrain-yolo11n",
            "code_file": "smoke-fire-retrain-yolo11n.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [BASE_DATASET_SOURCE],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": [],
        }
        (temp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        api = get_kaggle_client()
        try:
            logger.info("Đang đẩy notebook lên Kaggle Kernel: %s", kernel_slug)
            api.kernels_push(str(temp_path))
        except Exception as exc:
            logger.exception("Kaggle kernels_push thất bại: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Không thể khởi chạy kernel trên Kaggle: {exc}",
            )

    # 6. Tạo bản ghi RetrainRun trong CSDL
    active_model = db.scalar(select(ModelVersion).where(ModelVersion.status == "active").limit(1))
    run_id = new_id()
    run = RetrainRun(
        id=run_id,
        kernel_slug=kernel_slug,
        status="running",
        feedback_count=bundle["count"],
        base_model_name=active_model.name if active_model else "final_sfd_yolo11n",
        kaggle_status="queued",
        created_by=actor_id,
    )
    db.add(run)
    db.add(
        AuditLog(
            actor_id=actor_id,
            action="retrain_triggered",
            target_type="retrain_run",
            target_id=run_id,
            metadata_json={
                "kernel_slug": kernel_slug,
                "feedback_count": bundle["count"],
                "epochs": epochs,
            },
        )
    )
    db.commit()
    db.refresh(run)
    return run


def check_and_update_retrain_status(run_id: str, db: Session) -> dict[str, Any]:
    """Kiểm tra tiến độ thực thi của Kernel trên Kaggle và cập nhật DB."""
    run = db.get(RetrainRun, run_id)
    if not run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Không tìm thấy đợt huấn luyện")

    if run.status in ["completed", "failed"]:
        return retrain_run_data(run)

    api = get_kaggle_client()
    try:
        status_res = api.kernels_status(run.kernel_slug)
        k_status = ""
        if hasattr(status_res, "status"):
            k_status = str(status_res.status).replace("KernelWorkerStatus.", "").lower()
        elif isinstance(status_res, dict):
            k_status = str(status_res.get("status", "")).lower()

        fail_msg = getattr(status_res, "failureMessage", None) or (
            status_res.get("failureMessage") if isinstance(status_res, dict) else None
        )

        run.kaggle_status = k_status

        if k_status in ["complete", "completed"]:
            run.status = "completed"
            run.completed_at = utcnow()
        elif k_status in ["error", "failed", "cancel_acknowledged"]:
            run.status = "failed"
            run.error_message = fail_msg or "Lỗi trong quá trình chạy kernel trên Kaggle"
            run.completed_at = utcnow()
        elif k_status in ["running", "queued"]:
            run.status = "running"

        db.commit()
    except Exception as exc:
        logger.warning("Không thể lấy trạng thái từ Kaggle: %s", exc)

    return retrain_run_data(run)


def retrain_run_data(run: RetrainRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "kernel_slug": run.kernel_slug,
        "status": run.status,
        "feedback_count": run.feedback_count,
        "base_model_name": run.base_model_name,
        "kaggle_status": run.kaggle_status,
        "mlflow_run_id": run.mlflow_run_id,
        "error_message": run.error_message,
        "created_by": run.created_by,
        "created_at": iso(run.created_at),
        "completed_at": iso(run.completed_at),
        "kaggle_url": f"https://www.kaggle.com/code/{run.kernel_slug}",
    }
