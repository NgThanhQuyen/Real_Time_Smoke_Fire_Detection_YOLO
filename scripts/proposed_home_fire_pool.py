"""Lấy mẫu ảnh chỉ có lửa của Home Fire để kiểm duyệt mức đa dạng và nhãn."""

import csv
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
OUTPUT = ROOT / "reports/proposed_sources/home_fire_pool"
OUTPUT.mkdir(parents=True, exist_ok=True)

rows = []
for split in ("train", "val", "test"):
    with (SOURCE / f"home_index_{split}.csv").open(newline="", encoding="utf-8") as handle:
        rows.extend(row for row in csv.DictReader(handle)
                    if int(row["fire_boxes"]) > 0 and int(row["smoke_boxes"]) == 0 and not row["issue"])
blocked = {(row["home_split"], row["home_image"])
           for row in csv.DictReader((SOURCE / "home_indoor_candidates.csv").open(newline="", encoding="utf-8"))}
blocked |= {(row["split"], row["image"])
            for row in csv.DictReader((SOURCE / "overlap_candidates.csv").open(newline="", encoding="utf-8"))
            if row["source"] == "home"}
rows = [row for row in rows if (row["split"], row["image"]) not in blocked]

by_hash = defaultdict(list)
for row in rows:
    by_hash[(row["dhash"], row["phash"])].append(row)
unique = [group[0] for group in by_hash.values()]
chosen = random.Random(73).sample(unique, min(180, len(unique)))
chosen.sort(key=lambda row: (row["split"], row["image"]))

with (OUTPUT / "pool.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(chosen[0]))
    writer.writeheader()
    writer.writerows(chosen)

archives = {split: zipfile.ZipFile(SOURCE / f"HomeFire_{split}.zip") for split in ("train", "val", "test")}
try:
    for start in range(0, len(chosen), 30):
        width, height, columns = 350, 250, 5
        canvas = Image.new("RGB", (width * columns, height * 6), "white")
        draw = ImageDraw.Draw(canvas)
        for index, row in enumerate(chosen[start:start+30]):
            archive = archives[row["split"]]
            label_path = row["image"].replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
            label = archive.read(label_path).decode("utf-8-sig")
            with Image.open(BytesIO(archive.read(row["image"]))) as image:
                image = image.convert("RGB")
            boxes = ImageDraw.Draw(image)
            for line in label.splitlines():
                cls, x, y, w, h = line.split()
                x, y, w, h = map(float, (x, y, w, h))
                boxes.rectangle(((x-w/2)*image.width, (y-h/2)*image.height,
                                 (x+w/2)*image.width, (y+h/2)*image.height), outline="lime", width=5)
            thumb = ImageOps.contain(image, (width-8, height-30))
            left = index % columns * width
            top = index // columns * height
            canvas.paste(thumb, (left+(width-thumb.width)//2, top+25))
            draw.text((left+4, top+5), f"{row['split']}:{Path(row['image']).stem}", fill="black")
        target = OUTPUT / f"pool_{start//30+1:02d}.jpg"
        canvas.save(target, quality=90)
        print(target)
finally:
    for archive in archives.values():
        archive.close()
