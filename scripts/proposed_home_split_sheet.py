"""Minh họa các cảnh Home Fire có hash giống nhau ở train và val."""

import csv
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
rows = []
for split in ("train", "val"):
    with (SOURCE / f"home_index_{split}.csv").open(encoding="utf-8") as handle:
        rows += list(csv.DictReader(handle))
groups = defaultdict(list)
for row in rows:
    groups[(row["dhash"], row["phash"])].append(row)
cross = [group for group in groups.values() if {r["split"] for r in group} == {"train", "val"}]
cross.sort(key=lambda group: (-len(group), group[0]["image"]))
selected = cross[:12]
canvas = Image.new("RGB", (700, 250 * len(selected)), "white")
draw = ImageDraw.Draw(canvas)
with zipfile.ZipFile(SOURCE / "HomeFire_train.zip") as train, zipfile.ZipFile(SOURCE / "HomeFire_val.zip") as val:
    archives = {"train": train, "val": val}
    for index, group in enumerate(selected):
        for offset, split in ((0, "train"), (350, "val")):
            row = next(r for r in group if r["split"] == split)
            with Image.open(BytesIO(archives[split].read(row["image"]))) as image:
                thumb = ImageOps.contain(image.convert("RGB"), (350, 225))
            canvas.paste(thumb, (offset + (350 - thumb.width) // 2, index * 250 + 25))
            draw.text((offset + 3, index * 250 + 3), f"{split} {Path(row['image']).name}", fill="black")
target = ROOT / "reports/proposed_sources/home_train_val_same_hash.jpg"
canvas.save(target, quality=88)
print(len(cross), sum(len(group) for group in cross), target)
