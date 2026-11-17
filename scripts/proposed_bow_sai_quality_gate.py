"""Kiểm tra nhãn, cặp ảnh và truy nguyên của bộ D-Fire bổ sung BoWFire/SAINetset."""

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_bow_sai_v1"
BASE = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    counts = {}
    for split, expected in (("train", 8011), ("val", 1371), ("test", 4249)):
        image_dir, label_dir = DATA / split / "images", DATA / split / "labels"
        images = {path.stem: path for path in image_dir.iterdir() if path.is_file()}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        if len(images) != expected or set(images) != set(labels):
            raise ValueError(f"Số lượng hoặc cặp ảnh/nhãn sai ở {split}")
        category = Counter()
        boxes = Counter()
        for stem, label_path in labels.items():
            classes = set()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5 or parts[0] not in ("0", "1"):
                    raise ValueError(f"Nhãn sai cú pháp: {label_path}")
                x, y, w, h = map(float, parts[1:])
                if not all(map(math.isfinite, (x, y, w, h))) or w <= 0 or h <= 0:
                    raise ValueError(f"Box vô hiệu: {label_path}")
                if min(x-w/2, y-h/2) < -1e-5 or max(x+w/2, y+h/2) > 1+1e-5:
                    raise ValueError(f"Box vượt biên: {label_path}")
                classes.add(parts[0])
                boxes[parts[0]] += 1
            category["both" if len(classes) == 2 else "smoke" if classes == {"0"}
                     else "fire" if classes == {"1"} else "empty"] += 1
        counts[split] = {"images": len(images), "categories": dict(category), "boxes": dict(boxes)}

    with (DATA / "metadata/bow_sai_selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
        selected = list(csv.DictReader(handle))
    if len(selected) != 24 or len({row["sha256"] for row in selected}) != 24:
        raise ValueError("Manifest bổ sung thiếu hoặc có ảnh trùng byte")
    for row in selected:
        image = DATA / row["output_image"]
        label = DATA / row["output_label"]
        if digest(image) != row["sha256"]:
            raise ValueError(f"SHA-256 không khớp: {image}")
        with Image.open(image) as decoded:
            decoded.load()
            if decoded.size != (int(row["width"]), int(row["height"])):
                raise ValueError(f"Kích thước không khớp: {image}")
        if row["category"] == "negative" and label.read_text(encoding="utf-8").strip():
            raise ValueError(f"Ảnh âm tính có box: {image}")
        if row["category"] == "fire" and not label.read_text(encoding="utf-8").startswith("1 "):
            raise ValueError(f"Ảnh lửa thiếu box lớp 1: {image}")

    for split in ("val", "test"):
        for kind in ("images", "labels"):
            current = DATA / split / kind
            original = BASE / split / kind
            for path in current.iterdir():
                if digest(path) != digest(original / path.name):
                    raise ValueError(f"Split cố định thay đổi: {path}")
    result = {"splits": counts, "selected": len(selected), "selected_sha256_valid": True,
              "val_test_byte_identical": True}
    (DATA / "metadata/quality_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
