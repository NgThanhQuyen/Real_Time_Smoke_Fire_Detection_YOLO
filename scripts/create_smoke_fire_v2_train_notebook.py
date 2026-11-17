"""Tạo notebook Kaggle huấn luyện YOLO11n trên data 2 đã xử lý."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "model_training" / "train_yolo11n_smoke_fire_detection.ipynb"


def main() -> None:
    cells = [
        nbf.v4.new_markdown_cell(
            """# Huấn luyện YOLO11n phát hiện lửa và khói trên data 2 đã xử lý

Notebook này dùng Kaggle Input **Smoke-Fire-Detection-Dataset** đã giải nén như ảnh chụp người dùng cung cấp. Bộ dữ liệu có 13.769 ảnh train, 1.684 ảnh valid và 1.684 ảnh test; nhãn `0 = Fire`, `1 = Smoke`. Không tải lại Roboflow và không sửa Kaggle Input.

Recipe cơ bản lấy từ notebook tham khảo `smoke-fire-detection-yolo11.ipynb`: YOLO11n pretrained, tối đa 250 epoch, `imgsz=640`, `batch=32`, `patience=20`. Bản này thêm kiểm tra dữ liệu trước train, đánh giá `best.pt` trên valid và test, báo cáo ảnh nền, ảnh dự đoán mẫu và metadata dùng cho web.

**Trước khi chạy:** chọn GPU trong Kaggle Settings và bật Internet nếu Kaggle cần cài Ultralytics hoặc tải `yolo11n.pt`. Notebook sao chép dữ liệu từ Kaggle Input sang thư mục run trên `/kaggle/working` một lần để tránh đọc ảnh qua mount chậm ở mọi epoch. Cần đủ dung lượng trống cho bản sao khoảng 3,5 GB và file kết quả. Cuối notebook, kết quả được nén vào một ZIP; bản sao dữ liệu không được đưa vào ZIP và chỉ bị xóa cùng thư mục run sau khi ZIP qua CRC. Nếu train hoặc đóng gói lỗi, thư mục run được giữ để kiểm tra.

Không dùng test để chọn epoch hay chỉnh ngưỡng. Chỉ số từ bộ test này cần được xác nhận thêm trên video/camera độc lập vì việc chia theo tên nguồn chưa chứng minh hết rò rỉ cùng cảnh."""
        ),
        nbf.v4.new_markdown_cell("## 1. Cài thư viện và ghi nhận môi trường"),
        nbf.v4.new_code_cell(
            """from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ULTRALYTICS_VERSION = "8.4.203"
try:
    installed_version = importlib.metadata.version("ultralytics")
except importlib.metadata.PackageNotFoundError:
    installed_version = None

if installed_version != ULTRALYTICS_VERSION:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet",
        f"ultralytics=={ULTRALYTICS_VERSION}",
    ])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
import yaml
from IPython.display import FileLink, Image as NotebookImage, display
from ultralytics import YOLO

if not torch.cuda.is_available():
    raise RuntimeError("Chưa có GPU. Hãy chọn GPU trong Kaggle Settings trước khi train.")

print("Python:", platform.python_version())
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0))"""
        ),
        nbf.v4.new_markdown_cell("## 2. Cấu hình và tìm Kaggle Input"),
        nbf.v4.new_code_cell(
            """INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
DATASET_OVERRIDE = None
DATASET_HINT = "smoke-fire-detection-dataset"
RUN_ID = "smoke_fire_v2_yolo11n_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_DIR = WORK_ROOT / RUN_ID
RESULT_ZIP = WORK_ROOT / f"{RUN_ID}_results.zip"

IMAGE_SIZE = 640
BATCH_SIZE = 32
EPOCHS = 250
PATIENCE = 20
WORKERS = 4
SEED = 0
EVAL_CONF = 0.001
VIS_CONF = 0.25
NMS_IOU = 0.7
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_COUNTS = {"train": 13769, "valid": 1684, "test": 1684}
EXPECTED_CATEGORIES = {
    "train": {"fire_only": 2819, "smoke_only": 3110, "both": 6340, "background": 1500},
    "valid": {"fire_only": 352, "smoke_only": 389, "both": 791, "background": 152},
    "test": {"fire_only": 352, "smoke_only": 389, "both": 793, "background": 150},
}


def has_dataset_layout(folder: Path) -> bool:
    return (folder / "data.yaml").is_file() and all(
        (folder / split / "images").is_dir() and (folder / split / "labels").is_dir()
        for split in ("train", "valid", "test")
    )


def find_dataset_root() -> Path:
    if DATASET_OVERRIDE:
        candidate = Path(DATASET_OVERRIDE).expanduser().resolve()
        if not has_dataset_layout(candidate):
            raise RuntimeError(f"DATASET_OVERRIDE không có đủ data.yaml, ảnh và nhãn: {candidate}")
        return candidate

    # Ưu tiên đúng đường dẫn Kaggle Input người dùng đã cung cấp.
    candidates = [
        INPUT_ROOT / "datasets" / "quynnguyenthanh" / DATASET_HINT,
        INPUT_ROOT / DATASET_HINT,
    ]
    for yaml_path in sorted(INPUT_ROOT.rglob("data.yaml")):
        if DATASET_HINT in str(yaml_path.parent).lower():
            candidates.append(yaml_path.parent)
    for candidate in candidates:
        if has_dataset_layout(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        f"Không thấy dataset '{DATASET_HINT}' đã giải nén trong {INPUT_ROOT}. "
        "Hãy kiểm tra Kaggle Input hoặc đặt DATASET_OVERRIDE."
    )


DATASET_ROOT = find_dataset_root()
SOURCE_YAML = DATASET_ROOT / "data.yaml"
source_config = yaml.safe_load(SOURCE_YAML.read_text(encoding="utf-8"))
names = source_config.get("names")
if isinstance(names, dict):
    names = [names[key] for key in sorted(names, key=lambda value: int(value))]
if [str(name).lower() for name in names or []] != ["fire", "smoke"]:
    raise RuntimeError(f"Class map phải là 0=Fire, 1=Smoke; hiện tại: {names}")

if RUN_DIR.exists() or RESULT_ZIP.exists():
    raise FileExistsError(f"Run đã tồn tại: {RUN_DIR} hoặc {RESULT_ZIP}. Hãy chạy lại ô cấu hình để tạo RUN_ID mới.")
RUN_DIR.mkdir(parents=True)
(RUN_DIR / "reports").mkdir()

# Sao chép một lần để các epoch không đọc hàng chục nghìn ảnh từ Kaggle Input.
LOCAL_DATASET_ROOT = RUN_DIR / "dataset_local"
source_files = [path for path in DATASET_ROOT.rglob("*") if path.is_file()]
source_bytes = sum(path.stat().st_size for path in source_files)
free_bytes = shutil.disk_usage(WORK_ROOT).free
reserve_bytes = 2 * 1024**3
if free_bytes < source_bytes + reserve_bytes:
    raise RuntimeError(
        f"Không đủ dung lượng để sao chép dataset: cần ít nhất "
        f"{(source_bytes + reserve_bytes) / 1024**3:.2f} GB, "
        f"còn {free_bytes / 1024**3:.2f} GB."
    )
copy_started = time.perf_counter()
shutil.copytree(DATASET_ROOT, LOCAL_DATASET_ROOT)
copy_seconds = time.perf_counter() - copy_started
if not has_dataset_layout(LOCAL_DATASET_ROOT):
    raise RuntimeError(f"Bản sao dataset thiếu ảnh hoặc nhãn: {LOCAL_DATASET_ROOT}")

TRAIN_YAML = RUN_DIR / "data_for_training.yaml"
TRAIN_YAML.write_text(yaml.safe_dump({
    "path": str(LOCAL_DATASET_ROOT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": {0: "Fire", 1: "Smoke"},
}, sort_keys=False, allow_unicode=True), encoding="utf-8")

print("Dataset:", DATASET_ROOT)
print("Bản sao cục bộ:", LOCAL_DATASET_ROOT)
print("Dung lượng dữ liệu (GB):", round(source_bytes / 1024**3, 2))
print("Thời gian sao chép (giây):", round(copy_seconds, 1))
print("YAML train:", TRAIN_YAML)
print("Thư mục run:", RUN_DIR)
print("ZIP kết quả sẽ tạo:", RESULT_ZIP)"""
        ),
        nbf.v4.new_markdown_cell("## 3. Kiểm tra ảnh, nhãn và thống kê trước train"),
        nbf.v4.new_code_cell(
            """def inspect_split(split: str) -> dict:
    image_dir = LOCAL_DATASET_ROOT / split / "images"
    label_dir = LOCAL_DATASET_ROOT / split / "labels"
    images = sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    orphan_labels = [path.name for path in labels if path.stem not in image_stems]
    if orphan_labels:
        raise RuntimeError(f"{split} có nhãn không khớp ảnh: {orphan_labels[:5]}")

    categories = Counter()
    boxes = Counter()
    records = []
    # Ảnh nền có nhãn rỗng; nền không phải lớp thứ ba.
    for image_path in images:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.is_file():
            raise RuntimeError(f"Thiếu nhãn: {label_path}")
        present = set()
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5 or fields[0] not in {"0", "1"}:
                raise RuntimeError(f"Nhãn không đúng YOLO box: {label_path}:{line_number}")
            values = [float(value) for value in fields[1:]]
            if not all(np.isfinite(value) for value in values):
                raise RuntimeError(f"Tọa độ không hữu hạn: {label_path}:{line_number}")
            x, y, width, height = values
            if width <= 0 or height <= 0 or any(value < 0 or value > 1 for value in values):
                raise RuntimeError(f"Tọa độ không hợp lệ: {label_path}:{line_number}")
            if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
                raise RuntimeError(f"Box vượt ảnh: {label_path}:{line_number}")
            if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
                raise RuntimeError(f"Box vượt ảnh: {label_path}:{line_number}")
            class_id = int(fields[0])
            present.add(class_id)
            boxes[class_id] += 1

        category = (
            "fire_only" if present == {0} else
            "smoke_only" if present == {1} else
            "both" if present == {0, 1} else "background"
        )
        categories[category] += 1
        records.append({"split": split, "image": str(image_path), "category": category})

    if len(images) != EXPECTED_COUNTS[split] or dict(categories) != EXPECTED_CATEGORIES[split]:
        raise RuntimeError(
            f"Sai thống kê {split}: {len(images)} ảnh, {dict(categories)}. "
            f"Mong đợi {EXPECTED_COUNTS[split]} ảnh, {EXPECTED_CATEGORIES[split]}"
        )
    return {"images": len(images), "categories": categories, "boxes": boxes, "records": records}


audit = {split: inspect_split(split) for split in ("train", "valid", "test")}
image_records = pd.DataFrame(record for item in audit.values() for record in item["records"])
dataset_rows = []
for split, item in audit.items():
    counts = item["categories"]
    dataset_rows.append({
        "split": split, "images": item["images"],
        "fire_only": counts["fire_only"], "smoke_only": counts["smoke_only"],
        "both": counts["both"], "background": counts["background"],
        "fire_present": counts["fire_only"] + counts["both"],
        "smoke_present": counts["smoke_only"] + counts["both"],
        "fire_boxes": item["boxes"][0], "smoke_boxes": item["boxes"][1],
    })
dataset_table = pd.DataFrame(dataset_rows).set_index("split")
display(dataset_table)
dataset_table.to_csv(RUN_DIR / "reports" / "dataset_summary.csv")
(RUN_DIR / "reports" / "dataset_summary.json").write_text(
    json.dumps(dataset_rows, ensure_ascii=False, indent=2), encoding="utf-8"
)

fig, axis = plt.subplots(figsize=(9, 4))
dataset_table[["fire_only", "smoke_only", "both", "background"]].plot.bar(
    stacked=True, ax=axis, rot=0,
    color=["#d95f02", "#4c78a8", "#7a5195", "#7f8c8d"],
)
axis.set_title("Image categories by split")
axis.set_ylabel("Number of images")
fig.tight_layout()
fig.savefig(RUN_DIR / "reports" / "dataset_distribution.png", dpi=160)
display(fig)
plt.close(fig)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Huấn luyện YOLO11n

Chọn `best.pt` theo validation trong quá trình train. Tập test chưa được dùng ở bước này. Nếu GPU báo hết bộ nhớ, đổi `BATCH_SIZE` từ 32 xuống 16 ở ô cấu hình rồi bắt đầu một run mới; ghi lại sự thay đổi trong báo cáo."""
        ),
        nbf.v4.new_code_cell(
            """# Tải checkpoint pretrained vào đúng thư mục run để có thể dọn sau khi đóng gói.
os.chdir(RUN_DIR)
train_started = time.perf_counter()
train_model = YOLO("yolo11n.pt")
train_model.train(
    data=str(TRAIN_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    seed=SEED,
    deterministic=True,
    workers=WORKERS,
    device=0,
    project=str(RUN_DIR),
    name="train",
    exist_ok=True,
    plots=True,
    save=True,
    cache=False,
)
training_seconds = time.perf_counter() - train_started
TRAIN_DIR = Path(train_model.trainer.save_dir).resolve()
BEST_PT = TRAIN_DIR / "weights" / "best.pt"
if not BEST_PT.is_file():
    raise FileNotFoundError(f"Không có checkpoint tốt nhất: {BEST_PT}")
print("Checkpoint tốt nhất:", BEST_PT)
print("Thời gian train (giây):", round(training_seconds, 1))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Đánh giá `best.pt` trên valid và test

Hai lần đánh giá dùng cùng `imgsz`, `batch`, confidence thấp để tính mAP và NMS IoU. Test chỉ được đọc sau khi `best.pt` đã được chọn bằng validation."""
        ),
        nbf.v4.new_code_cell(
            """# Chỉ đánh giá test sau khi checkpoint đã được chọn bằng validation.
best_model = YOLO(str(BEST_PT))


def summarize_metrics(metrics, split: str) -> dict:
    per_class = []
    for metric_index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, map50, map50_95 = metrics.box.class_result(metric_index)
        per_class.append({
            "class_id": int(class_id),
            "class_name": str(metrics.names[int(class_id)]),
            "precision": float(precision),
            "recall": float(recall),
            "mAP50": float(map50),
            "mAP50_95": float(map50_95),
        })
    mean_precision, mean_recall, mean_map50, mean_map50_95 = metrics.box.mean_results()
    return {
        "split": split,
        "overall": {
            "precision": float(mean_precision), "recall": float(mean_recall),
            "mAP50": float(mean_map50), "mAP50_95": float(mean_map50_95),
        },
        "per_class": per_class,
        "validation_conf": EVAL_CONF,
        "nms_iou": NMS_IOU,
    }


evaluation = {}
for split_name, ultralytics_split in (("valid", "val"), ("test", "test")):
    metrics = best_model.val(
        data=str(TRAIN_YAML),
        split=ultralytics_split,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=EVAL_CONF,
        iou=NMS_IOU,
        workers=WORKERS,
        device=0,
        plots=True,
        project=str(RUN_DIR),
        name=f"evaluation_{split_name}",
        exist_ok=True,
        verbose=True,
    )
    evaluation[split_name] = summarize_metrics(metrics, split_name)
    (RUN_DIR / "reports" / f"metrics_{split_name}.json").write_text(
        json.dumps(evaluation[split_name], ensure_ascii=False, indent=2), encoding="utf-8"
    )

metric_rows = []
for split_name, report in evaluation.items():
    metric_rows.append({"split": split_name, "class": "all", **report["overall"]})
    metric_rows.extend({"split": split_name, "class": row["class_name"],
                        **{key: row[key] for key in ("precision", "recall", "mAP50", "mAP50_95")}}
                       for row in report["per_class"])
metrics_table = pd.DataFrame(metric_rows)
display(metrics_table)
metrics_table.to_csv(RUN_DIR / "reports" / "metrics_comparison.csv", index=False)"""
        ),
        nbf.v4.new_markdown_cell("## 6. Ảnh nền test và dự đoán minh họa"),
        nbf.v4.new_code_cell(
            """test_records = image_records[image_records["split"] == "test"]
background_paths = test_records.loc[test_records["category"] == "background", "image"].tolist()
background_predictions = []
for result in best_model.predict(
    source=background_paths, stream=True, batch=BATCH_SIZE,
    conf=VIS_CONF, iou=NMS_IOU, imgsz=IMAGE_SIZE, device=0, verbose=False,
):
    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = result.boxes.conf.cpu().numpy().tolist()
    background_predictions.append({
        "image": Path(result.path).name,
        "detections": len(classes),
        "fire_detections": classes.count(0),
        "smoke_detections": classes.count(1),
        "max_confidence": max(confidences, default=0.0),
    })
if len(background_predictions) != len(background_paths):
    raise RuntimeError("Số kết quả dự đoán ảnh nền không khớp số ảnh nguồn")
background_table = pd.DataFrame(background_predictions)
background_table.to_csv(RUN_DIR / "reports" / "background_test_predictions.csv", index=False)
background_false_positive_images = int((background_table["detections"] > 0).sum())
background_report = {
    "split": "test", "confidence": VIS_CONF,
    "background_images": len(background_table),
    "images_with_detection": background_false_positive_images,
    "image_false_positive_rate": background_false_positive_images / len(background_table),
}
(RUN_DIR / "reports" / "background_test_summary.json").write_text(
    json.dumps(background_report, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(pd.DataFrame([background_report]))

sample_rng = random.Random(SEED)
sample_paths = []
for category in ("fire_only", "smoke_only", "both", "background"):
    candidates = sorted(test_records.loc[test_records["category"] == category, "image"].tolist())
    sample_paths.extend(sample_rng.sample(candidates, min(3, len(candidates))))

sample_dir = RUN_DIR / "reports" / "sample_predictions"
sample_dir.mkdir(parents=True, exist_ok=True)
category_by_path = dict(zip(test_records["image"], test_records["category"]))
sample_rows = []
for result in best_model.predict(
    source=sample_paths, batch=BATCH_SIZE, conf=VIS_CONF,
    iou=NMS_IOU, imgsz=IMAGE_SIZE, device=0, verbose=False,
):
    source_path = str(Path(result.path))
    output_path = sample_dir / Path(source_path).name
    result.save(filename=str(output_path))
    sample_rows.append({
        "image": Path(source_path).name,
        "ground_truth_category": category_by_path[source_path],
        "detections": len(result.boxes),
        "annotated_file": output_path.name,
    })
sample_table = pd.DataFrame(sample_rows)
sample_table.to_csv(RUN_DIR / "reports" / "sample_predictions.csv", index=False)
display(sample_table)
for output_path in sorted(sample_dir.iterdir())[:4]:
    display(NotebookImage(filename=str(output_path), width=500))"""
        ),
        nbf.v4.new_markdown_cell("## 7. Lưu biểu đồ, checkpoint và metadata cho báo cáo/web"),
        nbf.v4.new_code_cell(
            """for relative_path in (
    "train/results.png", "train/confusion_matrix.png",
    "evaluation_valid/PR_curve.png", "evaluation_test/confusion_matrix.png",
):
    chart_path = RUN_DIR / relative_path
    if chart_path.is_file():
        print(relative_path)
        display(NotebookImage(filename=str(chart_path), width=700))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


run_summary = {
    "run_id": RUN_ID,
    "dataset": str(DATASET_ROOT),
    "dataset_yaml_sha256": file_sha256(SOURCE_YAML),
    "dataset_images": dataset_rows,
    "class_map": {"0": "Fire", "1": "Smoke"},
    "model_source": "yolo11n.pt (COCO pretrained)",
    "checkpoint_in_zip": "best.pt",
    "best_checkpoint_sha256": file_sha256(BEST_PT),
    "training": {
        "epochs_max": EPOCHS, "patience": PATIENCE, "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE, "seed": SEED, "workers": WORKERS,
        "dataset_copy_seconds": round(copy_seconds, 2),
        "duration_seconds": round(training_seconds, 2),
    },
    "evaluation": evaluation,
    "background_test": background_report,
    "software": {
        "python": platform.python_version(),
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
    },
    "limits": [
        "Ngưỡng 0.25 chỉ dùng để minh họa và đếm ảnh nền, chưa tối ưu cho triển khai.",
        "Tập test chưa được chứng minh độc lập hoàn toàn theo cảnh/video.",
        "Metrics ảnh có nhãn không phải metrics cảnh báo theo sự kiện video.",
    ],
}
(RUN_DIR / "reports" / "run_summary.json").write_text(
    json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_DIR / "reports" / "labelmap.json").write_text(
    json.dumps({"0": "Fire", "1": "Smoke"}, ensure_ascii=False, indent=2), encoding="utf-8"
)
(RUN_DIR / "reports" / "README.md").write_text(
    "# Kết quả YOLO11n Fire/Smoke\\n\\n"
    "`best.pt` là checkpoint được chọn bằng validation. `reports/run_summary.json` ghi nguồn dữ liệu, "
    "thông số train, metrics valid/test và SHA-256 checkpoint. `reports/labelmap.json` ghi thứ tự lớp "
    "0=Fire, 1=Smoke. Các file `evaluation_*` là biểu đồ đánh giá. "
    "Ảnh mẫu và bảng dự đoán ở `reports/sample_predictions`. "
    "Không dùng metrics ảnh làm kết luận độ chính xác cảnh báo video.\\n",
    encoding="utf-8",
)
print("Đã lưu run_summary.json, labelmap.json và biểu đồ.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 8. Đóng gói và dọn bộ nhớ

Chỉ ZIP cuối được giữ trong `/kaggle/working`. Bước dọn dẹp chỉ xóa đúng `RUN_DIR` do notebook này tạo, sau khi kiểm tra ZIP qua CRC và xác nhận có `best.pt` cùng báo cáo. Không xóa Kaggle Input hoặc file kết quả khác."""
        ),
        nbf.v4.new_code_cell(
            """def package_and_cleanup(run_dir: Path, final_zip: Path, working_root: Path) -> dict:
    base = working_root.resolve()
    target = run_dir.resolve()
    if target.parent != base or run_dir.is_symlink() or not target.name.startswith("smoke_fire_v2_yolo11n_"):
        raise RuntimeError(f"Từ chối xóa thư mục ngoài run được tạo: {target}")
    if final_zip.resolve().parent != base or final_zip.exists():
        raise RuntimeError(f"Đường dẫn ZIP không hợp lệ hoặc đã tồn tại: {final_zip}")

    temporary_zip = final_zip.with_suffix(".zip.part")
    if temporary_zip.exists():
        raise FileExistsError(f"File đóng gói tạm đã tồn tại: {temporary_zip}")
    excluded_names = {"last.pt", "yolo11n.pt"}
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(target.rglob("*")):
            if (not path.is_file() or path.is_symlink() or path.name in excluded_names
                    or path.suffix == ".cache" or path.relative_to(target).parts[0] == "dataset_local"):
                continue
            relative_path = path.relative_to(target)
            archive_path = "best.pt" if path.resolve() == BEST_PT.resolve() else relative_path.as_posix()
            archive.write(path, archive_path)

    with zipfile.ZipFile(temporary_zip) as archive:
        bad_member = archive.testzip()
        required = {"best.pt", "reports/run_summary.json", "reports/labelmap.json", "reports/metrics_comparison.csv"}
        missing = required - set(archive.namelist())
        if bad_member or missing:
            raise RuntimeError(f"ZIP không đạt kiểm tra: bad_member={bad_member}, thiếu={sorted(missing)}")
        packaged_files = len(archive.namelist())
    # Chỉ thay file tạm bằng ZIP cuối khi CRC và các file bắt buộc đều đạt.
    os.replace(temporary_zip, final_zip)
    if not final_zip.is_file():
        raise RuntimeError("Chưa thấy ZIP cuối; giữ thư mục run để kiểm tra")

    # Chỉ xóa thư mục run do notebook tạo; Kaggle Input và ZIP cuối được giữ.
    os.chdir(base)
    shutil.rmtree(target)
    return {"zip": str(final_zip), "bytes": final_zip.stat().st_size, "files": packaged_files,
            "run_directory_removed": not target.exists()}


package_result = package_and_cleanup(RUN_DIR, RESULT_ZIP, WORK_ROOT)
display(pd.DataFrame([package_result]))
display(FileLink(str(RESULT_ZIP)))"""
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
