"""Tạo ảnh ghép các cặp nghi trùng với D-Fire để xác nhận bằng mắt."""

import csv
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
DFIRE = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1"
rows = list(csv.DictReader((SOURCE / "overlap_candidates.csv").open(encoding="utf-8")))
selected = [r for r in rows if r["source"] in ("home", "bowfire")]
width, height = 350, 240
canvas = Image.new("RGB", (width * 2, height * max(1, len(selected))), "white")
draw = ImageDraw.Draw(canvas)
archives = {}
for index, row in enumerate(selected):
    if row["source"] == "home":
        name = f"HomeFire_{row['split']}.zip"
    elif row["source"] == "bowfire":
        name = "BoWFireDataset.zip"
    else:
        name = None
    if name and name not in archives:
        archives[name] = zipfile.ZipFile(SOURCE / name)
    if name:
        left_bytes = archives[name].read(row["image"])
    else:
        left_bytes = (SOURCE / "sainet_sample" / row["image"]).read_bytes()
    with Image.open(BytesIO(left_bytes)) as image:
        left = ImageOps.contain(image.convert("RGB"), (width, height - 26))
    with Image.open(DFIRE / row["nearest_dfire"]) as image:
        right = ImageOps.contain(image.convert("RGB"), (width, height - 26))
    y = index * height
    canvas.paste(left, ((width - left.width) // 2, y + 26))
    canvas.paste(right, (width + (width - right.width) // 2, y + 26))
    draw.text((3, y + 3), f"{row['source']} {Path(row['image']).name}", fill="black")
    draw.text((width + 3, y + 3), f"{row['nearest_split']} {Path(row['nearest_dfire']).name} d={row['dhash_distance']} p={row['phash_distance']}", fill="black")
for archive in archives.values():
    archive.close()
target = ROOT / "reports/proposed_sources/dfire_overlap_candidates.jpg"
canvas.save(target, quality=88)
print(len(selected), target)
