"""Kiểm kê hai ZIP FireAndSmoke, giữ nguyên dữ liệu nguồn và split D-Fire."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


SOURCES = (
    "FireSmokeDataset.v2i.yolov8.zip",
    "FireSmokeNEWdataset.v1i.yolov8.zip",
)
FIELDS = (
    "source", "image", "split", "scene_key", "category", "fire_boxes",
    "smoke_boxes", "other_boxes", "width", "height", "sha256", "dhash",
    "phash", "issue",
)


def scene_key(name: str) -> str:
    stem = Path(name).stem.split("_jpg.rf.")[0]
    stem = re.sub(r"_f\d+$", "", stem, flags=re.IGNORECASE)
    if re.match(r"(?i)^fire1-\d+-?$", stem):
        return "fire1_sequence"
    if re.match(r"(?i)^Img_\d+$", stem):
        return "Img_sequence"
    pexels_video = re.match(r"(?i)^(pexels-.+?-\d+)-\d+x\d+-\d+fps-\d+$", stem)
    if pexels_video:
        return pexels_video.group(1)
    return stem


def hashes(image: Image.Image) -> tuple[str, str]:
    gray = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
    dhash = int("".join(map(str, (pixels[:, 1:] > pixels[:, :-1]).astype(int).ravel())), 2)
    low = dctn(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32), norm="ortho")[:8, :8]
    bits = low.ravel() > np.median(low.ravel()[1:])
    phash = int("".join(map(str, bits.astype(int))), 2)
    return f"{dhash:016x}", f"{phash:016x}"


def label_counts(text: str) -> tuple[Counter[int], str]:
    counts: Counter[int] = Counter()
    issues = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("column_count")
            values = list(map(float, parts))
            cls, x, y, width, height = values
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite")
            if cls not in (0, 1, 2):
                raise ValueError("class")
            if width <= 0 or height <= 0:
                raise ValueError("nonpositive")
            if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
                raise ValueError("outside")
            counts[int(cls)] += 1
        except ValueError as error:
            issues.append(f"{number}:{error}")
    return counts, ";".join(issues)


def audit(source_root: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "image_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for source_name in SOURCES:
            source = source_root / source_name
            with zipfile.ZipFile(source) as archive:
                names = set(archive.namelist())
                images = sorted(name for name in names if "/images/" in name and name.lower().endswith(".jpg"))
                for number, name in enumerate(images, 1):
                    label = name.replace("/images/", "/labels/")[:-4] + ".txt"
                    counts, issue = label_counts(archive.read(label).decode("utf-8-sig")) if label in names else (Counter(), "missing_label")
                    image_bytes = archive.read(name)
                    digest = hashlib.sha256(image_bytes).hexdigest()
                    width = height = 0
                    dhash = phash = ""
                    try:
                        with Image.open(BytesIO(image_bytes)) as image:
                            width, height = image.size
                            dhash, phash = hashes(image)
                    except Exception as error:
                        issue = f"{issue};image:{type(error).__name__}" if issue else f"image:{type(error).__name__}"
                    category = "".join(word for cls, word in ((0, "fire"), (2, "smoke"), (1, "other")) if counts[cls]) or "empty"
                    writer.writerow({
                        "source": source_name, "image": name, "split": name.split("/", 1)[0],
                        "scene_key": scene_key(name), "category": category,
                        "fire_boxes": counts[0], "smoke_boxes": counts[2], "other_boxes": counts[1],
                        "width": width, "height": height, "sha256": digest,
                        "dhash": dhash, "phash": phash, "issue": issue,
                    })
                    if number % 1000 == 0:
                        handle.flush()
                        print(f"{source_name}: {number}/{len(images)}", flush=True)


def rekey(output: Path) -> None:
    path = output / "image_index.csv"
    temporary = output / "image_index.rekey.csv"
    with path.open(newline="", encoding="utf-8") as source, temporary.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["scene_key"] = scene_key(row["image"])
            writer.writerow(row)
    temporary.replace(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path(r"F:\6\KLTN\Smoke-Fire-Dataset"))
    parser.add_argument("--output", type=Path, default=Path("runtime/catargiu_audit"))
    parser.add_argument("--rekey", action="store_true")
    arguments = parser.parse_args()
    if arguments.rekey:
        rekey(arguments.output)
    else:
        audit(arguments.source_root, arguments.output)
