"""Kiểm tra dự đoán YOLO trên test AoF và tạo ảnh ghép các lỗi đại diện."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def intersection_over_union(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    b = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / (a + b - intersection) if a + b > intersection else 0.0


def read_ground_truth(path: Path, image_width: int, image_height: int) -> list[dict]:
    boxes = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        cls, x, y, width, height = map(float, line.split())
        boxes.append({
            "class_id": int(cls),
            "xyxy": ((x - width / 2) * image_width, (y - height / 2) * image_height,
                     (x + width / 2) * image_width, (y + height / 2) * image_height),
        })
    return boxes


def hamming_hex(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def diverse_rows(rows: list[dict], hashes: dict[str, str], limit: int = 12) -> list[dict]:
    selected = []
    for threshold in (14, 10, 6, 0):
        for row in rows:
            if row in selected:
                continue
            key = f"test/images/{Path(row['output_image']).stem}.jpg"
            value = hashes.get(key)
            if value and all(hamming_hex(value, hashes.get(
                    f"test/images/{Path(other['output_image']).stem}.jpg", value)) >= threshold
                    for other in selected):
                selected.append(row)
            if len(selected) >= limit:
                return selected
    return selected


def contact_sheet(rows: list[dict], dataset: Path, destination: Path, title: str) -> None:
    width, height = 350, 260
    columns = 4
    sheet = Image.new("RGB", (columns * width, ((len(rows) + columns - 1) // columns) * height), "white")
    drawer = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        with Image.open(dataset / row["output_image"]) as original:
            image = original.convert("RGB")
        scale = min((width - 12) / image.width, (height - 48) / image.height)
        resized = image.resize((round(image.width * scale), round(image.height * scale)))
        offset_x = (index % columns) * width + 6
        offset_y = (index // columns) * height + 32
        sheet.paste(resized, (offset_x, offset_y))
        drawer.text((offset_x, offset_y - 29), Path(row["output_image"]).name, fill="black", font=font)
        drawer.text((offset_x, offset_y - 16),
                    f"GT:{row['category']} pred S:{row['max_smoke_conf']:.2f} F:{row['max_fire_conf']:.2f}",
                    fill="black", font=font)
        for box in row["ground_truth"]:
            color = "lime" if box["class_id"] == 0 else "yellow"
            coords = [offset_x + box["xyxy"][0] * scale, offset_y + box["xyxy"][1] * scale,
                      offset_x + box["xyxy"][2] * scale, offset_y + box["xyxy"][3] * scale]
            drawer.rectangle(coords, outline=color, width=3)
        for box in row["predictions"]:
            color = "red" if box["class_id"] == 0 else "magenta"
            coords = [offset_x + box["xyxy"][0] * scale, offset_y + box["xyxy"][1] * scale,
                      offset_x + box["xyxy"][2] * scale, offset_y + box["xyxy"][3] * scale]
            drawer.rectangle(coords, outline=color, width=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, quality=92)
    print(title, destination, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError(arguments.output)
    arguments.output.mkdir(parents=True)

    with (arguments.dataset / "metadata/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        manifest = [row for row in csv.DictReader(stream)
                    if row["output_split"] == "test" and row["source_prefix"] == "AoF"]
    with arguments.index.open(encoding="utf-8-sig", newline="") as stream:
        hashes = {row["image"]: row["phash"] for row in csv.DictReader(stream)}

    model = YOLO(str(arguments.model))
    predictions = []
    for start in range(0, len(manifest), 32):
        batch = manifest[start:start + 32]
        result_batch = model.predict(
            source=[str(arguments.dataset / row["output_image"]) for row in batch],
            imgsz=640, conf=0.25, iou=0.7, batch=16, device="cpu", stream=True, verbose=False,
        )
        results = list(result_batch)
        if len(results) != len(batch):
            raise RuntimeError("Số kết quả dự đoán không khớp số ảnh.")
        for row, result in zip(batch, results):
            image_height, image_width = result.orig_shape
            ground_truth = read_ground_truth(arguments.dataset / row["output_label"],
                                             image_width, image_height)
            boxes = [
                {"class_id": int(cls), "confidence": float(conf), "xyxy": tuple(map(float, xyxy))}
                for cls, conf, xyxy in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                           result.boxes.xyxy.tolist())
            ]
            smoke_gt = [box for box in ground_truth if box["class_id"] == 0]
            smoke_pred = [box for box in boxes if box["class_id"] == 0]
            matched_smoke_gt = sum(any(intersection_over_union(gt["xyxy"], pred["xyxy"]) >= 0.5
                                       for pred in smoke_pred) for gt in smoke_gt)
            predictions.append({
                "output_image": row["output_image"], "category": row["output_category"],
                "ground_truth": ground_truth, "predictions": boxes,
                "gt_smoke_count": len(smoke_gt), "pred_smoke_count": len(smoke_pred),
                "matched_smoke_gt": matched_smoke_gt,
                "max_smoke_conf": max([box["confidence"] for box in smoke_pred], default=0.0),
                "max_fire_conf": max([box["confidence"] for box in boxes if box["class_id"] == 1], default=0.0),
            })
        print(f"reviewed={len(predictions)}/{len(manifest)}", flush=True)

    with (arguments.output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    background = [row for row in predictions if row["category"] == "background"]
    false_smoke = sorted([row for row in background if row["pred_smoke_count"]],
                         key=lambda row: row["max_smoke_conf"], reverse=True)
    smoke_positive = [row for row in predictions if row["gt_smoke_count"]]
    missed_smoke = sorted([row for row in smoke_positive if row["matched_smoke_gt"] == 0],
                          key=lambda row: row["max_smoke_conf"])
    summary = {
        "aof_test_images": len(predictions),
        "background_images": len(background),
        "background_with_any_prediction": sum(bool(row["predictions"]) for row in background),
        "background_with_smoke_prediction": len(false_smoke),
        "smoke_positive_images": len(smoke_positive),
        "smoke_positive_images_without_iou50_match": len(missed_smoke),
        "categories": dict(Counter(row["category"] for row in predictions)),
    }
    (arguments.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    contact_sheet(diverse_rows(false_smoke, hashes), arguments.dataset,
                  arguments.output / "aof_background_false_smoke.jpg", "false_smoke")
    contact_sheet(diverse_rows(missed_smoke, hashes), arguments.dataset,
                  arguments.output / "aof_smoke_missed.jpg", "missed_smoke")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
