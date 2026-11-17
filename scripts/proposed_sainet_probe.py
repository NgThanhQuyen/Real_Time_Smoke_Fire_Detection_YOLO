"""Đo ảnh âm tính SAINetset nào làm baseline D-Fire báo nhầm."""

import csv
import json
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime/ultralytics"))
from ultralytics import YOLO  # noqa: E402

SOURCE = ROOT / "runtime/proposed_sources/sainet_sample"
rows = list(csv.DictReader((SOURCE / "sample_index.csv").open(encoding="utf-8")))
negative = [r for r in rows if int(r["boxes"]) == 0]
paths = [str(SOURCE / r["image"]) for r in negative]
model = YOLO("C:/Users/User/Downloads/dfire_clean_v2_yolo11n_best.pt")
results = model.predict(paths, imgsz=640, conf=0.25, iou=0.7, batch=12, device="cpu", verbose=False)
records = []
counts = Counter()
for row, result in zip(negative, results):
    detections = len(result.boxes)
    classes = sorted({int(c) for c in result.boxes.cls.tolist()})
    if detections:
        counts["images_with_detection"] += 1
    counts["boxes"] += detections
    records.append({"image": row["image"], "detections": detections,
                    "classes": ";".join(map(str, classes)),
                    "max_conf": round(float(result.boxes.conf.max()), 5) if detections else 0})
with (SOURCE / "baseline_probe.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, list(records[0]))
    writer.writeheader()
    writer.writerows(records)
counts["negative_sample"] = len(negative)
(SOURCE / "baseline_probe_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
print(json.dumps(counts, indent=2))
