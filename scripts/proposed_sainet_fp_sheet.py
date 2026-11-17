"""Tạo ảnh ghép các mẫu âm tính SAINetset làm model D-Fire báo nhầm."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources/sainet_sample"
rows = list(csv.DictReader((SOURCE / "baseline_probe.csv").open(encoding="utf-8")))
selected = sorted((r for r in rows if int(r["detections"]) > 0), key=lambda r: -float(r["max_conf"]))
width, height, columns = 300, 220, 5
canvas = Image.new("RGB", (width * columns, height * ((len(selected) + columns - 1) // columns)), "white")
draw = ImageDraw.Draw(canvas)
for index, row in enumerate(selected):
    with Image.open(SOURCE / row["image"]) as image:
        thumb = ImageOps.contain(image.convert("RGB"), (width, height - 25))
    x, y = index % columns * width, index // columns * height
    canvas.paste(thumb, (x + (width - thumb.width)//2, y + 25))
    draw.text((x + 3, y + 3), f"{Path(row['image']).name} class={row['classes']} conf={row['max_conf']}", fill="black")
target = ROOT / "reports/proposed_sources/sainet_false_positive_sample.jpg"
canvas.save(target, quality=88)
print(len(selected), target)
