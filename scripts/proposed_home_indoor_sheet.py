"""Hiển thị các cặp hash Home Fire và Indoor để kiểm tra bằng mắt."""

import csv
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
rows = list(csv.DictReader((ROOT / "runtime/proposed_sources/home_indoor_candidates.csv").open(encoding="utf-8")))
canvas = Image.new("RGB", (700, 250 * len(rows)), "white")
draw = ImageDraw.Draw(canvas)
with zipfile.ZipFile("F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip") as indoor:
    home = {}
    for index, row in enumerate(rows):
        split = row["home_split"]
        if split not in home:
            home[split] = zipfile.ZipFile(ROOT / f"runtime/proposed_sources/HomeFire_{split}.zip")
        for offset, data in ((0, home[split].read(row["home_image"])), (350, indoor.read(row["indoor_image"]))):
            with Image.open(BytesIO(data)) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (350, 225))
            canvas.paste(thumb, (offset + (350 - thumb.width) // 2, index * 250 + 25))
        draw.text((3, index * 250 + 3), row["home_image"], fill="black")
        draw.text((353, index * 250 + 3), Path(row["indoor_image"]).name[:35], fill="black")
    for archive in home.values():
        archive.close()
target = ROOT / "reports/proposed_sources/home_indoor_candidates.jpg"
canvas.save(target, quality=88)
print(target)
