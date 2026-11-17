"""Tạo ảnh ghép kiểm duyệt mẫu SAINetset đã tải."""

import csv
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources/sainet_sample"
rows = list(csv.DictReader((SOURCE / "sample_index.csv").open(encoding="utf-8")))
negatives = [r for r in rows if int(r["boxes"]) == 0]
positives = [r for r in rows if int(r["boxes"]) > 0]
chosen = random.Random(9).sample(negatives, min(42, len(negatives))) + positives
width, height, columns = 280, 205, 6
canvas = Image.new("RGB", (width * columns, height * ((len(chosen) + columns - 1) // columns)), "white")
draw = ImageDraw.Draw(canvas)
dimensions = Counter()
for index, row in enumerate(chosen):
    with Image.open(SOURCE / row["image"]) as image:
        dimensions[image.size] += 1
        thumbnail = ImageOps.contain(image.convert("RGB"), (width, height - 25))
    x, y = index % columns * width, index // columns * height
    canvas.paste(thumbnail, (x + (width - thumbnail.width) // 2, y + 25))
    draw.text((x + 3, y + 3), f"{row['boxes']} {Path(row['image']).name[:24]}", fill="black")
target = ROOT / "reports/proposed_sources/sainet_sample.jpg"
target.parent.mkdir(parents=True, exist_ok=True)
canvas.save(target, quality=88)
print(f"sample={len(rows)} negatives={len(negatives)} positives={len(positives)} dimensions={dimensions}")
print(target)
