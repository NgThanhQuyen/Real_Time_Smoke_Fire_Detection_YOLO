"""Khảo sát lỗi model cũ trên ảnh Catargiu đã duyệt và video kiểm thử."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "runtime/ultralytics"))
from ultralytics import YOLO  # noqa: E402


MODEL = Path("C:/Users/User/Downloads/dfire_clean_v2_yolo11n_best.pt")
DATASET = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1"
MANIFEST = DATASET / "metadata/catargiu_selected_manifest.csv"
VIDEOS = Path("F:/6/KLTN/Smoke-Fire-Dataset/Video_test")
OUTPUT = ROOT / "reports/catargiu_fire_smoke_audit/model_probe"
CONFIDENCE = 0.25
IMAGE_SIZE = 640
VIDEO_SAMPLE_FPS = 5


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def boxes_from_result(result) -> list[dict]:
    return [
        {"class_id": int(cls), "confidence": round(float(conf), 5),
         "xyxy": [round(float(value), 2) for value in coords]}
        for cls, conf, coords in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                     result.boxes.xyxy.tolist())
    ]


def iou(first: list[float], second: list[float]) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    b = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = a + b - overlap
    return overlap / union if union else 0.0


def fire_truth(path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        cls, x, y, w, h = map(float, line.split())
        if int(cls) != 1:
            raise RuntimeError(f"Ảnh tuyển chọn có nhãn ngoài lớp lửa: {path}")
        boxes.append([(x - w / 2) * width, (y - h / 2) * height,
                      (x + w / 2) * width, (y + h / 2) * height])
    return boxes


def infer_images(model: YOLO, selected: list[dict[str, str]]) -> list[dict]:
    rows = []
    for start in range(0, len(selected), 16):
        batch = selected[start:start + 16]
        paths = [str(DATASET / item["output_image"]) for item in batch]
        predictions = model.predict(paths, imgsz=IMAGE_SIZE, conf=CONFIDENCE, iou=0.7,
                                    batch=16, device="cpu", verbose=False)
        if len(predictions) != len(batch):
            raise RuntimeError("Số kết quả ảnh không khớp số đầu vào")
        for item, result in zip(batch, predictions):
            height, width = result.orig_shape
            gt = fire_truth(DATASET / item["output_label"], width, height)
            detected = boxes_from_result(result)
            fire = [box for box in detected if box["class_id"] == 1]
            smoke = [box for box in detected if box["class_id"] == 0]
            rows.append({"review_id": item["review_id"], "output_image": item["output_image"],
                         "category": item["output_class"], "gt_fire_boxes": len(gt),
                         "matched_fire_boxes_iou50": sum(any(iou(truth, box["xyxy"]) >= 0.5
                                                              for box in fire) for truth in gt),
                         "fire_predictions": len(fire), "smoke_predictions": len(smoke),
                         "max_fire_conf": max((box["confidence"] for box in fire), default=0.0),
                         "max_smoke_conf": max((box["confidence"] for box in smoke), default=0.0),
                         "ground_truth": gt, "predictions": detected})
        print(f"images {len(rows)}/{len(selected)}", flush=True)
    return rows


def sample_video(path: Path) -> tuple[list[tuple[float, object]], float]:
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or fps <= 0:
        raise RuntimeError(f"Không đọc được video: {path}")
    interval = max(1, round(fps / VIDEO_SAMPLE_FPS))
    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % interval == 0:
            frames.append((round(index / fps, 3), frame))
        index += 1
    capture.release()
    return frames, fps


def infer_videos(model: YOLO) -> list[dict]:
    rows = []
    for path in sorted(VIDEOS.glob("test_*.mp4")):
        frames, fps = sample_video(path)
        for start in range(0, len(frames), 12):
            batch = frames[start:start + 12]
            results = model.predict([frame for _, frame in batch], imgsz=IMAGE_SIZE,
                                    conf=CONFIDENCE, iou=0.7, batch=12, device="cpu", verbose=False)
            if len(results) != len(batch):
                raise RuntimeError(f"Số kết quả video không khớp: {path}")
            for (time_s, _), result in zip(batch, results):
                boxes = boxes_from_result(result)
                rows.append({"video": path.name, "time_s": time_s, "boxes": boxes,
                             "fire": sum(box["class_id"] == 1 for box in boxes),
                             "smoke": sum(box["class_id"] == 0 for box in boxes),
                             "max_fire_conf": max((box["confidence"] for box in boxes
                                                   if box["class_id"] == 1), default=0.0),
                             "max_smoke_conf": max((box["confidence"] for box in boxes
                                                    if box["class_id"] == 0), default=0.0)})
        print(f"video {path.name} {len(frames)} frames at approx {round(fps / round(fps / VIDEO_SAMPLE_FPS), 2)} fps", flush=True)
    return rows


def make_sheet(rows: list[dict], destination: Path, title: str) -> None:
    subset = rows[:12]
    canvas = Image.new("RGB", (1200, ((len(subset) + 3) // 4) * 260), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(subset):
        with Image.open(DATASET / row["output_image"]) as source:
            source = source.convert("RGB")
            scale = min(290 / source.width, 225 / source.height)
            image = source.resize((round(source.width * scale), round(source.height * scale)))
        x, y = (index % 4) * 300, (index // 4) * 260
        canvas.paste(image, (x, y + 25))
        draw.text((x + 3, y + 3), row["review_id"] + " F" + str(row["fire_predictions"]) +
                  " S" + str(row["smoke_predictions"]), fill="black")
        for box in row["ground_truth"]:
            draw.rectangle((x + box[0] * scale, y + 25 + box[1] * scale,
                            x + box[2] * scale, y + 25 + box[3] * scale), outline="lime", width=2)
        for box in row["predictions"]:
            coords = box["xyxy"]
            color = "red" if box["class_id"] == 0 else "magenta"
            draw.rectangle((x + coords[0] * scale, y + 25 + coords[1] * scale,
                            x + coords[2] * scale, y + 25 + coords[3] * scale), outline=color, width=2)
    if subset:
        canvas.save(destination, quality=90)
        print(title, destination, flush=True)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    model = YOLO(str(MODEL))
    if model.names != {0: "smoke", 1: "fire"}:
        raise RuntimeError(f"Sai thứ tự lớp của checkpoint: {model.names}")
    selected = read_manifest()
    images = infer_images(model, selected)
    videos = infer_videos(model)
    (OUTPUT / "images.json").write_text(json.dumps(images, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "video_frames.json").write_text(json.dumps(videos, ensure_ascii=False, indent=2), encoding="utf-8")
    positives = [row for row in images if row["category"] == "fire"]
    negatives = [row for row in images if row["category"] == "background"]
    video_stats = {}
    for path in sorted(VIDEOS.glob("test_*.mp4")):
        group = [row for row in videos if row["video"] == path.name]
        video_stats[path.name] = {"sampled_frames": len(group),
                                  "frames_with_any_prediction": sum(bool(row["boxes"]) for row in group),
                                  "frames_with_fire": sum(row["fire"] > 0 for row in group),
                                  "frames_with_smoke": sum(row["smoke"] > 0 for row in group),
                                  "max_fire_conf": max((row["max_fire_conf"] for row in group), default=0.0),
                                  "max_smoke_conf": max((row["max_smoke_conf"] for row in group), default=0.0)}
    summary = {"checkpoint": str(MODEL), "confidence": CONFIDENCE,
               "video_sampling_fps": VIDEO_SAMPLE_FPS, "image_fire_count": len(positives),
               "image_negative_count": len(negatives),
               "fire_images_with_fire_prediction": sum(row["fire_predictions"] > 0 for row in positives),
               "fire_gt_boxes": sum(row["gt_fire_boxes"] for row in positives),
               "fire_gt_boxes_matched_iou50": sum(row["matched_fire_boxes_iou50"] for row in positives),
               "fire_images_with_smoke_prediction": sum(row["smoke_predictions"] > 0 for row in positives),
               "negative_images_with_any_prediction": sum(bool(row["predictions"]) for row in negatives),
               "negative_images_with_fire_prediction": sum(row["fire_predictions"] > 0 for row in negatives),
               "negative_images_with_smoke_prediction": sum(row["smoke_predictions"] > 0 for row in negatives),
               "videos": video_stats}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    make_sheet(sorted((row for row in positives if row["matched_fire_boxes_iou50"] == 0),
                      key=lambda row: row["max_fire_conf"]), OUTPUT / "missed_fire.jpg", "missed_fire")
    make_sheet(sorted((row for row in negatives if row["predictions"]),
                      key=lambda row: max(row["max_fire_conf"], row["max_smoke_conf"]), reverse=True),
               OUTPUT / "negative_false_alarms.jpg", "negative_false_alarms")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
