from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from .config import ROOT, settings
from .db import SessionLocal
from .inference.coordination import inference_slot
from .models import AuditLog, BenchmarkRun, ModelVersion, utcnow

logger = logging.getLogger(__name__)

BENCHMARK_DIR = ROOT / "data" / "test_benchmark"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_BENCHMARK_LOCK = threading.Lock()
_ACTIVE_BENCHMARKS: set[str] = set()


def get_dataset_paths() -> dict[str, Path]:
    """Tìm thư mục ảnh và nhãn của tập test (hỗ trợ cả images/ và test/images/)."""
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = BENCHMARK_DIR / "images"
    labels_dir = BENCHMARK_DIR / "labels"

    # Kiểm tra cấu trúc lồng test/images nếu người dùng giải nén dạng Roboflow/Kaggle
    if not any(images_dir.glob("*.*")):
        nested_images = BENCHMARK_DIR / "test" / "images"
        nested_labels = BENCHMARK_DIR / "test" / "labels"
        if nested_images.is_dir() and any(nested_images.glob("*.*")):
            images_dir = nested_images
            labels_dir = nested_labels

    yaml_path = BENCHMARK_DIR / "dataset.yaml"
    return {
        "root": BENCHMARK_DIR,
        "images": images_dir,
        "labels": labels_dir,
        "yaml": yaml_path,
    }


def ensure_dataset_yaml(images_dir: Path, yaml_path: Path, class_names: dict[int, str] | None = None) -> Path:
    """Tự động đồng bộ đường dẫn dataset.yaml với hệ điều hành cục bộ."""
    if not class_names:
        class_names = {0: "khoi", 1: "lua"}
    names_block = "\n".join(f"  {k}: {v}" for k, v in sorted(class_names.items()))
    content = f"""# Cấu hình tự động sinh cho bộ kiểm định Golden Test
path: "{images_dir.parent.as_posix()}"
train: "{images_dir.name}"
val: "{images_dir.name}"
test: "{images_dir.name}"
names:
{names_block}
"""
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def get_dataset_status() -> dict[str, Any]:
    """Kiểm tra số lượng ảnh, nhãn và trạng thái sẵn sàng của bộ dữ liệu test."""
    paths = get_dataset_paths()
    images_dir = paths["images"]
    labels_dir = paths["labels"]

    images_count = 0
    labels_count = 0
    total_bytes = 0

    if images_dir.is_dir():
        for file in images_dir.iterdir():
            if file.is_file() and file.suffix.lower() in IMAGE_EXTS:
                images_count += 1
                try:
                    total_bytes += file.stat().st_size
                except OSError:
                    pass

    if labels_dir.is_dir():
        for file in labels_dir.iterdir():
            if file.is_file() and file.suffix.lower() == ".txt":
                labels_count += 1

    total_mb = round(total_bytes / (1024 * 1024), 2)
    is_ready = images_count > 0 and labels_count > 0

    if is_ready:
        ensure_dataset_yaml(images_dir, paths["yaml"])

    return {
        "is_ready": is_ready,
        "images_count": images_count,
        "labels_count": labels_count,
        "total_mb": total_mb,
        "folder_path": str(BENCHMARK_DIR.resolve()),
        "images_dir": str(images_dir.resolve()) if images_dir.is_dir() else None,
        "labels_dir": str(labels_dir.resolve()) if labels_dir.is_dir() else None,
        "yaml_path": str(paths["yaml"].resolve()) if paths["yaml"].is_file() else None,
        "recommended_count": 3857,
    }


def execute_benchmark_task(benchmark_id: str, actor_id: str | None = None) -> None:
    """Tiến trình chạy đánh giá model với Ultralytics YOLO trong luồng nền."""
    with SessionLocal() as db:
        run = db.get(BenchmarkRun, benchmark_id)
        if not run:
            logger.error("Không tìm thấy lượt benchmark %s", benchmark_id)
            return
        model_version = db.get(ModelVersion, run.model_version_id)
        if not model_version:
            run.status = "failed"
            run.error_message = "Phiên bản mô hình không tồn tại"
            db.commit()
            return

        run.status = "running"
        db.commit()

    try:
        paths = get_dataset_paths()
        status_info = get_dataset_status()
        if not status_info["is_ready"]:
            raise ValueError(
                f"Tập dữ liệu kiểm định chưa sẵn sàng (có {status_info['images_count']} ảnh, "
                f"{status_info['labels_count']} file nhãn). Hãy copy tập test vào data/test_benchmark/images và labels"
            )

        model_artifact = (ROOT / model_version.artifact_path).resolve()
        if not model_artifact.is_file():
            if Path(model_version.artifact_path).is_file():
                model_artifact = Path(model_version.artifact_path).resolve()
            else:
                raise FileNotFoundError(f"Không tìm thấy file checkpoint: {model_artifact}")

        cls_map = model_version.class_map if isinstance(model_version.class_map, dict) else None
        if cls_map:
            cls_map_int = {int(k): str(v) for k, v in cls_map.items()}
        else:
            cls_map_int = {0: "khoi", 1: "lua"}
        yaml_path = ensure_dataset_yaml(paths["images"], paths["yaml"], class_names=cls_map_int)

        logger.info(
            "Bắt đầu đánh giá benchmark %s cho model '%s' trên %d ảnh",
            benchmark_id,
            model_version.name,
            status_info["images_count"],
        )

        from ultralytics import YOLO

        # Dùng inference_slot để điều phối không đụng độ với worker video/webcam
        with inference_slot(blocking=True):
            model = YOLO(str(model_artifact))
            started_time = time.perf_counter()

            # Gọi hàm val chuẩn của Ultralytics
            val_results = model.val(
                data=str(yaml_path),
                split="test",
                imgsz=settings.model_image_size,
                conf=settings.detection_confidence,
                iou=settings.detection_iou,
                device="cpu",
                verbose=False,
                save=False,
                plots=False,
            )
            total_duration_s = time.perf_counter() - started_time

        # Trích xuất số liệu
        box = val_results.box
        p_overall = float(box.mp) if hasattr(box, "mp") else 0.0
        r_overall = float(box.mr) if hasattr(box, "mr") else 0.0
        map50_overall = float(box.map50) if hasattr(box, "map50") else 0.0
        map50_95_overall = float(box.map) if hasattr(box, "map") else 0.0

        def extract_cls(idx: int) -> dict[str, float]:
            try:
                p, r, ap50, ap95 = box.class_result(idx)
                return {
                    "precision": round(float(p), 4),
                    "recall": round(float(r), 4),
                    "map50": round(float(ap50), 4),
                    "map50_95": round(float(ap95), 4),
                }
            except Exception:
                return {"precision": 0.0, "recall": 0.0, "map50": 0.0, "map50_95": 0.0}

        # Map theo lớp: 0 = khoi (smoke), 1 = lua (fire)
        khoi_metrics = extract_cls(0)
        lua_metrics = extract_cls(1)

        speed_dict = dict(val_results.speed) if hasattr(val_results, "speed") else {}
        inference_ms = float(speed_dict.get("inference", 0.0))
        fps = round(1000.0 / inference_ms, 1) if inference_ms > 0 else 0.0

        metrics_summary = {
            "overall": {
                "precision": round(p_overall, 4),
                "recall": round(r_overall, 4),
                "map50": round(map50_overall, 4),
                "map50_95": round(map50_95_overall, 4),
                "f1": round(2 * (p_overall * r_overall) / max(p_overall + r_overall, 1e-6), 4),
            },
            "classes": {
                "khoi": khoi_metrics,
                "lua": lua_metrics,
            },
            "speed": {
                "preprocess_ms": round(float(speed_dict.get("preprocess", 0.0)), 2),
                "inference_ms": round(inference_ms, 2),
                "loss_ms": round(float(speed_dict.get("loss", 0.0)), 2),
                "postprocess_ms": round(float(speed_dict.get("postprocess", 0.0)), 2),
                "total_duration_s": round(total_duration_s, 2),
                "fps": fps,
            },
            "dataset_info": {
                "image_count": status_info["images_count"],
                "labels_count": status_info["labels_count"],
            },
        }

        with SessionLocal() as db:
            run = db.get(BenchmarkRun, benchmark_id)
            if run:
                run.status = "completed"
                run.image_count = status_info["images_count"]
                run.metrics_summary = metrics_summary
                run.completed_at = utcnow()
                # Cập nhật số liệu chuẩn hóa vào ModelVersion nếu model chưa có metrics
                mv = db.get(ModelVersion, run.model_version_id)
                if mv and not mv.metrics:
                    mv.metrics = {
                        "mAP50": map50_overall,
                        "mAP50-95": map50_95_overall,
                        "precision": p_overall,
                        "recall": r_overall,
                        "benchmark_id": benchmark_id,
                    }
                db.add(
                    AuditLog(
                        actor_id=actor_id,
                        action="benchmark_completed",
                        target_type="benchmark_run",
                        target_id=benchmark_id,
                        metadata_json={
                            "model_id": run.model_version_id,
                            "map50": map50_overall,
                            "map50_95": map50_95_overall,
                        },
                    )
                )
                db.commit()
                logger.info("Hoàn tất benchmark %s: mAP50=%.4f", benchmark_id, map50_overall)

    except Exception as exc:
        logger.exception("Benchmark %s thất bại: %s", benchmark_id, exc)
        with SessionLocal() as db:
            run = db.get(BenchmarkRun, benchmark_id)
            if run:
                run.status = "failed"
                run.error_message = str(exc)
                run.completed_at = utcnow()
                db.commit()
    finally:
        with _BENCHMARK_LOCK:
            _ACTIVE_BENCHMARKS.discard(benchmark_id)


def start_benchmark(
    model_version_id: str,
    actor_id: str | None = None,
    db: Session | None = None,
) -> BenchmarkRun:
    """Tạo bản ghi BenchmarkRun và khởi chạy luồng đánh giá trong background."""
    should_close = False
    if db is None:
        db = SessionLocal()
        should_close = True

    try:
        mv = db.get(ModelVersion, model_version_id)
        if not mv:
            raise ValueError("Không tìm thấy phiên bản mô hình cần benchmark")

        # Kiểm tra xem có đang chạy benchmark cho chính model này không
        existing_running = db.scalar(
            select(BenchmarkRun)
            .options(joinedload(BenchmarkRun.model_version))
            .where(
                BenchmarkRun.model_version_id == model_version_id,
                BenchmarkRun.status.in_(["queued", "running"]),
            )
            .limit(1)
        )
        if existing_running:
            return existing_running

        run = BenchmarkRun(
            model_version_id=model_version_id,
            dataset_name="golden_test",
            status="queued",
            created_by=actor_id,
        )
        db.add(run)
        db.commit()
        db.refresh(run)
        run.model_version = mv
        benchmark_id = run.id
    finally:
        if should_close:
            db.close()

    with _BENCHMARK_LOCK:
        _ACTIVE_BENCHMARKS.add(benchmark_id)

    # Chạy worker trong thread riêng để không block HTTP event loop
    thread = threading.Thread(
        target=execute_benchmark_task,
        args=(benchmark_id, actor_id),
        name=f"benchmark-{benchmark_id[:8]}",
        daemon=True,
    )
    thread.start()
    return run


def compare_benchmarks_metrics(candidate_summary: dict, base_summary: dict) -> dict[str, Any]:
    """So sánh chi tiết 2 bộ metrics giữa Candidate và Active Model."""
    c_over = candidate_summary.get("overall", {})
    b_over = base_summary.get("overall", {})

    c_map50 = float(c_over.get("map50", 0.0))
    b_map50 = float(b_over.get("map50", 0.0))
    delta_map50 = round(c_map50 - b_map50, 4)
    pct_map50 = round((delta_map50 / max(b_map50, 1e-6)) * 100, 2)

    c_map50_95 = float(c_over.get("map50_95", 0.0))
    b_map50_95 = float(b_over.get("map50_95", 0.0))
    delta_map50_95 = round(c_map50_95 - b_map50_95, 4)

    c_prec = float(c_over.get("precision", 0.0))
    b_prec = float(b_over.get("precision", 0.0))
    delta_prec = round(c_prec - b_prec, 4)

    c_rec = float(c_over.get("recall", 0.0))
    b_rec = float(b_over.get("recall", 0.0))
    delta_rec = round(c_rec - b_rec, 4)

    c_speed = candidate_summary.get("speed", {}).get("inference_ms", 0.0)
    b_speed = base_summary.get("speed", {}).get("inference_ms", 0.0)
    delta_speed = round(c_speed - b_speed, 2)

    # Đánh giá lớp khói
    c_khoi = candidate_summary.get("classes", {}).get("khoi", {})
    b_khoi = base_summary.get("classes", {}).get("khoi", {})
    khoi_delta_map50 = round(float(c_khoi.get("map50", 0.0)) - float(b_khoi.get("map50", 0.0)), 4)

    # Đánh giá lớp lửa
    c_lua = candidate_summary.get("classes", {}).get("lua", {})
    b_lua = base_summary.get("classes", {}).get("lua", {})
    lua_delta_map50 = round(float(c_lua.get("map50", 0.0)) - float(b_lua.get("map50", 0.0)), 4)

    if delta_map50 > 0.005:
        verdict = "improved"
        verdict_text = f"Mô hình ứng viên vượt trội (+{pct_map50}% mAP50). Rất khuyến nghị kích hoạt!"
    elif delta_map50 < -0.01:
        verdict = "regressed"
        verdict_text = f"Mô hình ứng viên sụt giảm ({pct_map50}% mAP50). Cần kiểm tra lại trước khi kích hoạt!"
    else:
        verdict = "comparable"
        verdict_text = "Hiệu năng tương đương với mô hình đang chạy hiện tại."

    return {
        "verdict": verdict,
        "verdict_text": verdict_text,
        "overall": {
            "map50": {"candidate": c_map50, "base": b_map50, "delta": delta_map50, "pct": pct_map50},
            "map50_95": {"candidate": c_map50_95, "base": b_map50_95, "delta": delta_map50_95},
            "precision": {"candidate": c_prec, "base": b_prec, "delta": delta_prec},
            "recall": {"candidate": c_rec, "base": b_rec, "delta": delta_rec},
            "speed_ms": {"candidate": c_speed, "base": b_speed, "delta": delta_speed},
        },
        "classes": {
            "khoi": {
                "candidate": c_khoi,
                "base": b_khoi,
                "delta_map50": khoi_delta_map50,
            },
            "lua": {
                "candidate": c_lua,
                "base": b_lua,
                "delta_map50": lua_delta_map50,
            },
        },
    }

