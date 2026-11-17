"""Tạo notebook so sánh độ phân giải suy luận trên FinalSFD validation."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/model_analysis/compare_final_sfd_resolution_640_800.ipynb"


def code(source: str):
    return nbformat.v4.new_code_cell(source.strip())


def markdown(source: str):
    return nbformat.v4.new_markdown_cell(source.strip())


cells = [
    markdown("""
# So sánh `imgsz=640` và `800` trên FinalSFD validation

Notebook **chỉ đánh giá**, không gọi `train()`. Dùng cùng `best.pt` pretrained trong ZIP kết quả run IoU 0,5. Đầu vào cần gắn vào Kaggle: dataset FinalSFD đã dùng để train và ZIP `final_sfd_yolo11n_20261112_170520_results.zip`. Chỉ sao chép ảnh/nhãn **val**; nhãn trong Kaggle Input và tập test không thay đổi.

Hai độ phân giải dùng cùng NMS IoU 0,5 và confidence 0,001 cho P/R/mAP. Để so recall theo kích thước box, notebook dùng confidence triển khai 0,29 đã chọn ở run cũ, ghép box cùng lớp với ground truth tại IoU ≥0,5. Các nhóm `<1%`, `1–10%`, `≥10%` là diện tích box chia cho diện tích ảnh gốc, không phải COCO AP small/medium/large. Tốc độ đo trên Kaggle GPU không đại diện cho CPU laptop.
"""),
    code("""
from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import platform
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
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet",
                           f"ultralytics=={ULTRALYTICS_VERSION}"])

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
import yaml
from IPython.display import FileLink, display
from ultralytics import YOLO

print("Python:", platform.python_version())
print("Ultralytics:", ultralytics.__version__)
print("PyTorch:", torch.__version__)
print("Thiết bị:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
"""),
    markdown("""
## 1. Xác nhận đầu vào và chuẩn bị riêng tập val

Nếu tên ZIP trên Kaggle khác, sửa `RESULTS_ZIP_NAME`. Notebook tự tìm đúng một ZIP có tên đó trong `/kaggle/input`. Dataset được kiểm tra bằng SHA của `data.yaml` và số ảnh/box val được lưu trong ZIP kết quả. Các box lỗi hình học được sửa **trên bản sao val** theo đúng quy tắc notebook train.
"""),
    code("""
INPUT_ROOT = Path("/kaggle/input")
DATASET_ROOT = Path("/kaggle/input/datasets/quynnguynthanh/finalsfd-yolo-dataset/FinalSFD_YOLO_Dataset")
RESULTS_ZIP_NAME = "final_sfd_yolo11n_20261112_170520_results.zip"
RESULTS_ZIP_OVERRIDE = None  # Nếu tự tìm ra nhiều ZIP cùng tên, đặt đường dẫn Kaggle Input ở đây.
SIZES = (640, 800)
NMS_IOU = 0.5
EVAL_CONF = 0.001
DEPLOY_CONF = 0.29
MATCH_IOU = 0.5
VAL_BATCH = 8
PREDICT_BATCH = 8
WORKERS = 4
DEVICE = 0 if torch.cuda.is_available() else "cpu"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if RESULTS_ZIP_OVERRIDE is None:
    candidates = sorted(INPUT_ROOT.rglob(RESULTS_ZIP_NAME))
    if len(candidates) != 1:
        raise RuntimeError(f"Cần đúng một ZIP {RESULTS_ZIP_NAME} trong Kaggle Input; "
                           f"tìm thấy {len(candidates)}: {candidates[:5]}")
    results_zip = candidates[0]
else:
    results_zip = Path(RESULTS_ZIP_OVERRIDE)
if not results_zip.is_file():
    raise FileNotFoundError(f"Không thấy ZIP kết quả: {results_zip}")
if not (DATASET_ROOT / "data.yaml").is_file():
    raise FileNotFoundError(f"Không thấy FinalSFD tại {DATASET_ROOT}")

run_dir = Path("/kaggle/working") / ("final_sfd_resolution_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"))
if run_dir.exists():
    raise FileExistsError(run_dir)
run_dir.mkdir(parents=True)
report_dir = run_dir / "reports"
report_dir.mkdir()
local_dataset = run_dir / "dataset_val"
for subdir in ("images/val", "labels/val"):
    (local_dataset / subdir).mkdir(parents=True)

with zipfile.ZipFile(results_zip) as archive:
    required = {"best.pt", "reports/run_summary.json"}
    if not required.issubset(archive.namelist()):
        raise RuntimeError(f"ZIP thiếu: {sorted(required - set(archive.namelist()))}")
    previous = json.loads(archive.read("reports/run_summary.json"))
    checkpoint_bytes = archive.read("best.pt")
if previous.get("nms_iou_main") != NMS_IOU or previous.get("class_map") != {"0": "smoke", "1": "fire"}:
    raise RuntimeError("ZIP không phải run FinalSFD với 0=smoke, 1=fire và NMS IoU 0,5")
source_yaml_hash = hashlib.sha256((DATASET_ROOT / "data.yaml").read_bytes()).hexdigest()
if source_yaml_hash != previous.get("dataset_yaml_sha256"):
    raise RuntimeError("data.yaml khác run đã train; hãy gắn đúng phiên bản FinalSFD")
checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()
if checkpoint_hash != previous.get("best_checkpoint_sha256"):
    raise RuntimeError("Hash best.pt khác metadata trong ZIP")
checkpoint = run_dir / "best.pt"
checkpoint.write_bytes(checkpoint_bytes)

expected_val = next(row for row in previous["dataset_images"] if row["split"] == "val")
source_images = DATASET_ROOT / "images/val"
source_labels = DATASET_ROOT / "labels/val"
images = sorted(path for path in source_images.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
if len(images) != expected_val["images"] or len({p.stem for p in images}) != len(images):
    raise RuntimeError(f"Số ảnh val không khớp ZIP: {len(images)} / {expected_val['images']}")
for image_path in images:
    label_path = source_labels / f"{image_path.stem}.txt"
    if not label_path.is_file():
        raise FileNotFoundError(f"Thiếu nhãn: {label_path}")
    shutil.copy2(image_path, local_dataset / "images/val" / image_path.name)
    shutil.copy2(label_path, local_dataset / "labels/val" / label_path.name)

fixes = Counter()
for label_path in sorted((local_dataset / "labels/val").glob("*.txt")):
    corrected = []
    changed = False
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] not in {"0", "1"}:
            raise RuntimeError(f"Nhãn không đúng YOLO box: {label_path}:{line_number}")
        x, y, width, height = map(float, fields[1:])
        if not np.isfinite([x, y, width, height]).all():
            raise RuntimeError(f"Tọa độ không hữu hạn: {label_path}:{line_number}")
        left, right = x - width / 2, x + width / 2
        top, bottom = y - height / 2, y + height / 2
        cl, cr = max(0.0, left), min(1.0, right)
        ct, cb = max(0.0, top), min(1.0, bottom)
        if width <= 0 or height <= 0 or cr <= cl or cb <= ct:
            fixes["drop_zero_area"] += 1
            changed = True
            continue
        if (any(value < 0 or value > 1 for value in (x, y, width, height))
                or max(abs(cl-left), abs(cr-right), abs(ct-top), abs(cb-bottom)) > 1e-6):
            line = (f"{fields[0]} {(cl+cr)/2:.12g} {(ct+cb)/2:.12g} "
                    f"{cr-cl:.12g} {cb-ct:.12g}")
            fixes["clip_to_image"] += 1
            changed = True
        corrected.append(line)
    if changed:
        label_path.write_text("\\n".join(corrected) + ("\\n" if corrected else ""), encoding="utf-8")

label_counts = Counter()
for label_path in (local_dataset / "labels/val").glob("*.txt"):
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            label_counts[int(line.split()[0])] += 1
if label_counts[0] != expected_val["smoke_boxes"] or label_counts[1] != expected_val["fire_boxes"]:
    raise RuntimeError(f"Số box val khác run gốc: {label_counts} / {expected_val}")

eval_yaml = run_dir / "data_val.yaml"
eval_yaml.write_text(yaml.safe_dump({
    "path": str(local_dataset), "train": "images/val", "val": "images/val",
    "test": "images/val", "nc": 2, "names": ["smoke", "fire"]
}, sort_keys=False, allow_unicode=True), encoding="utf-8")
print("ZIP:", results_zip)
print("Checkpoint SHA-256:", checkpoint_hash)
print("Val:", len(images), "ảnh; smoke:", label_counts[0], "box; fire:", label_counts[1], "box")
print("Sửa nhãn cục bộ:", dict(fixes))
print("Thư mục báo cáo:", report_dir)
"""),
    markdown("""
## 2. Đánh giá P/R/mAP với cùng checkpoint

P/R của Ultralytics là chỉ số tóm tắt từ các đường precision–recall, không phải recall ở confidence triển khai 0,29. Hai lượt này chỉ khác `imgsz`.
"""),
    code("""
model = YOLO(str(checkpoint))
metric_rows = []
for size in SIZES:
    started = time.perf_counter()
    metrics = model.val(
        data=str(eval_yaml), split="val", imgsz=size, batch=VAL_BATCH,
        conf=EVAL_CONF, iou=NMS_IOU, workers=WORKERS, device=DEVICE,
        plots=False, save=False, verbose=False, project=str(run_dir),
        name=f"val_{size}", exist_ok=True,
    )
    elapsed = time.perf_counter() - started
    precision, recall, map50, map50_95 = metrics.box.mean_results()
    metric_rows.append({"imgsz": size, "class": "all", "precision": float(precision),
                        "recall": float(recall), "mAP50": float(map50),
                        "mAP50_95": float(map50_95), "validation_seconds": elapsed})
    for index, class_id in enumerate(metrics.box.ap_class_index):
        precision, recall, map50, map50_95 = metrics.box.class_result(index)
        metric_rows.append({"imgsz": size, "class": metrics.names[int(class_id)],
                            "precision": float(precision), "recall": float(recall),
                            "mAP50": float(map50), "mAP50_95": float(map50_95),
                            "validation_seconds": elapsed})
    print(f"Đánh giá {size}: {elapsed:.1f} giây")
    del metrics
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

metrics_table = pd.DataFrame(metric_rows).sort_values(["class", "imgsz"])
metrics_table.to_csv(report_dir / "metrics_val_640_800.csv", index=False)
display(metrics_table)
"""),
    markdown("""
## 3. Recall lửa theo kích thước và tốc độ dự đoán

Cùng 2.342 ảnh val, confidence 0,29 và NMS IoU 0,5. Ghép dự đoán lửa với box lửa thật theo IoU ≥0,5, mỗi box thật tối đa một lần. Bảng `missed_fire_val.csv` ghi ảnh và nhóm box còn bỏ sót để xem sau. Thời gian toàn lượt có cả đọc ảnh và Python; `result.speed` là thời gian pipeline Ultralytics báo cáo, không phải FPS webcam trên laptop.
"""),
    code("""
def read_fire_boxes(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    label_path = local_dataset / "labels/val" / f"{image_path.stem}.txt"
    boxes, areas = [], []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x, y, w, h = map(float, line.split())
        if int(class_id) != 1:
            continue
        boxes.append([x-w/2, y-h/2, x+w/2, y+h/2])
        areas.append(w*h)
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(areas, dtype=float)


def box_iou(one: np.ndarray, many: np.ndarray) -> np.ndarray:
    top_left = np.maximum(one[:2], many[:, :2])
    bottom_right = np.minimum(one[2:], many[:, 2:])
    intersection = np.prod(np.maximum(bottom_right - top_left, 0), axis=1)
    one_area = np.prod(np.maximum(one[2:] - one[:2], 0))
    many_area = np.prod(np.maximum(many[:, 2:] - many[:, :2], 0), axis=1)
    return intersection / np.maximum(one_area + many_area - intersection, 1e-12)


def size_group(area: float) -> str:
    return "<1%" if area < 0.01 else "1-10%" if area < 0.10 else ">=10%"


size_rows, speed_rows, missed_rows = [], [], []
for size in SIZES:
    counts = Counter()
    speed_ms = Counter()
    fire_images, detected_fire_images = 0, 0
    background_images, background_alarms = 0, 0
    started = time.perf_counter()
    predictions = model.predict(
        source=[str(path) for path in images], stream=True, batch=PREDICT_BATCH,
        imgsz=size, conf=DEPLOY_CONF, iou=NMS_IOU, device=DEVICE,
        verbose=False, save=False,
    )
    processed = 0
    for result in predictions:
        processed += 1
        image_path = Path(result.path)
        true_boxes, true_areas = read_fire_boxes(image_path)
        height, width = result.orig_shape
        if len(true_boxes):
            fire_images += 1
        elif not (local_dataset / "labels/val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").strip():
            background_images += 1
            background_alarms += int(len(result.boxes) > 0)

        predicted_classes = result.boxes.cls.cpu().numpy().astype(int)
        fire_indices = np.flatnonzero(predicted_classes == 1)
        fire_predicted = len(fire_indices) > 0
        if len(true_boxes) and fire_predicted:
            detected_fire_images += 1
        predicted_boxes = result.boxes.xyxyn.cpu().numpy()[fire_indices]
        predicted_conf = result.boxes.conf.cpu().numpy()[fire_indices]
        matched = np.zeros(len(true_boxes), dtype=bool)
        for pred_index in np.argsort(-predicted_conf):
            available = np.flatnonzero(~matched)
            if not len(available):
                break
            ious = box_iou(predicted_boxes[pred_index], true_boxes[available])
            best = int(np.argmax(ious))
            if ious[best] >= MATCH_IOU:
                matched[available[best]] = True
        for area, found in zip(true_areas, matched):
            group = size_group(float(area))
            counts[(group, "ground_truth")] += 1
            counts[(group, "matched")] += int(found)
            if not found:
                missed_rows.append({"imgsz": size, "image": image_path.name,
                                    "size_group": group, "box_area": float(area)})
        for stage in ("preprocess", "inference", "postprocess"):
            speed_ms[stage] += float(result.speed.get(stage, 0.0))
    elapsed = time.perf_counter() - started
    if processed != len(images):
        raise RuntimeError(f"Predict {size} xử lý {processed}/{len(images)} ảnh")
    for group in ("<1%", "1-10%", ">=10%"):
        total = counts[(group, "ground_truth")]
        matched_count = counts[(group, "matched")]
        size_rows.append({"imgsz": size, "size_group": group,
                          "ground_truth_boxes": total, "matched_boxes": matched_count,
                          "recall_at_conf_029": matched_count / total if total else float("nan")})
    speed_rows.append({
        "imgsz": size, "images": processed, "wall_seconds": elapsed,
        "images_per_second": processed / elapsed,
        "preprocess_ms_per_image": speed_ms["preprocess"] / processed,
        "inference_ms_per_image": speed_ms["inference"] / processed,
        "postprocess_ms_per_image": speed_ms["postprocess"] / processed,
        "fire_images": fire_images, "fire_images_with_detection": detected_fire_images,
        "fire_image_detection_rate": detected_fire_images / fire_images,
        "background_images": background_images,
        "background_images_with_detection": background_alarms,
    })
    print(f"Predict {size}: {elapsed:.1f} giây, {processed / elapsed:.2f} ảnh/giây")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

size_table = pd.DataFrame(size_rows)
speed_table = pd.DataFrame(speed_rows)
missed_table = pd.DataFrame(missed_rows)
size_table.to_csv(report_dir / "fire_recall_by_size_val.csv", index=False)
speed_table.to_csv(report_dir / "predict_speed_and_image_flags_val.csv", index=False)
missed_table.to_csv(report_dir / "missed_fire_val.csv", index=False)
display(size_table)
display(speed_table)
"""),
    markdown("""
## 4. So sánh, biểu đồ và xuất báo cáo

Chỉ chọn `imgsz` từ **val**. Tăng recall nhưng giảm mAP50–95 hoặc tăng báo nhầm là một đánh đổi cần ghi rõ; không khẳng định 800 tốt hơn chỉ vì tìm được nhiều box hơn.
"""),
    code("""
fire_metrics = metrics_table[metrics_table["class"] == "fire"].set_index("imgsz")
small_recall = size_table[size_table["size_group"] == "<1%"].set_index("imgsz")
comparison = pd.DataFrame({
    "fire_mAP50": fire_metrics["mAP50"],
    "fire_mAP50_95": fire_metrics["mAP50_95"],
    "fire_reported_recall": fire_metrics["recall"],
    "fire_small_recall_at_conf_029": small_recall["recall_at_conf_029"],
    "images_per_second_kaggle": speed_table.set_index("imgsz")["images_per_second"],
    "val_background_alarms_at_conf_029": speed_table.set_index("imgsz")["background_images_with_detection"],
})
comparison.loc["delta_800_minus_640"] = comparison.loc[800] - comparison.loc[640]
comparison.to_csv(report_dir / "resolution_comparison.csv")
display(comparison)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fire_metrics[["mAP50", "mAP50_95"]].plot.bar(ax=axes[0], rot=0)
axes[0].set(title="Fire AP trên val", xlabel="imgsz", ylabel="Điểm")
small_recall[["recall_at_conf_029"]].plot.bar(ax=axes[1], rot=0, legend=False, color="#d95f02")
axes[1].set(title="Recall box lửa <1% tại conf 0,29", xlabel="imgsz", ylabel="Recall")
fig.tight_layout()
fig.savefig(report_dir / "resolution_comparison.png", dpi=160)
display(fig)
plt.close(fig)

metadata = {
    "dataset": str(DATASET_ROOT), "source_zip": str(results_zip),
    "checkpoint_sha256": checkpoint_hash, "dataset_yaml_sha256": source_yaml_hash,
    "val_images": len(images), "sizes": list(SIZES), "nms_iou": NMS_IOU,
    "map_conf": EVAL_CONF, "deploy_conf": DEPLOY_CONF, "match_iou": MATCH_IOU,
    "label_fixes_in_val_copy": dict(fixes), "device": str(DEVICE),
    "ultralytics": ultralytics.__version__,
}
(report_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
output_zip = Path("/kaggle/working") / f"{run_dir.name}_results.zip"
with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(report_dir.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(run_dir).as_posix())
print("Báo cáo:", output_zip)
display(FileLink(str(output_zip)))
"""),
]

notebook = nbformat.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbformat.validate(notebook)
nbformat.write(notebook, OUTPUT)
print(OUTPUT)
