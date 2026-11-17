"""Visual review sheet for new-source hash candidates near D-Fire val/test."""

import csv
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("F:/6/KLTN/Smoke-Fire-Dataset")
BASE = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1"
ROWS = list(csv.DictReader((ROOT / "runtime/new_sources_audit/dfire_overlap_candidates.csv").open(encoding="utf-8")))
MANIFEST = {r["source_image"]: r["output_image"] for r in csv.DictReader((BASE / "metadata/dfire_manifest.csv").open(encoding="utf-8-sig")) if r["output_split"] in ("train", "val", "test")}
SELECT = [r for r in ROWS if r["nearest_split"] in ("val", "test")]
WIDTH, HEIGHT = 340, 230
canvas = Image.new("RGB", (WIDTH * 2, HEIGHT * max(1, len(SELECT))), "white")
draw = ImageDraw.Draw(canvas)
archives = {}
for n, row in enumerate(SELECT):
    name = row["source"]
    if name not in archives:
        archives[name] = zipfile.ZipFile(SOURCE / name)
    with Image.open(BytesIO(archives[name].read(row["image"]))) as img:
        left = ImageOps.contain(img.convert("RGB"), (WIDTH, HEIGHT - 28))
    with Image.open(BASE / MANIFEST[row["nearest_dfire"]]) as img:
        right = ImageOps.contain(img.convert("RGB"), (WIDTH, HEIGHT - 28))
    y = n * HEIGHT
    canvas.paste(left, ((WIDTH - left.width)//2, y + 25))
    canvas.paste(right, (WIDTH + (WIDTH - right.width)//2, y + 25))
    draw.text((3, y + 3), f"{name[:8]} {Path(row['image']).stem[:18]}", fill="black")
    draw.text((WIDTH + 3, y + 3), f"{row['nearest_split']} {Path(row['nearest_dfire']).name} d={row['dhash_distance']} p={row['phash_distance']}", fill="black")
for archive in archives.values():
    archive.close()
target = ROOT / "reports/new_sources_audit/dfire_val_test_hash_candidates.jpg"
canvas.save(target, quality=88)
print(len(SELECT), target)
