"""Tạo các bảng ảnh BoWFire để duyệt toàn bộ ảnh và vùng lửa được gắn nhãn."""

import csv
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
OUTPUT = ROOT / "reports/proposed_sources/bowfire_full"
OUTPUT.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader((SOURCE / "bowfire_index.csv").open(encoding="utf-8")))

with zipfile.ZipFile(SOURCE / "BoWFireDataset.zip") as archive:
    for category in ("fire", "negative"):
        selected = [row for row in rows if row["category"] == category]
        for start in range(0, len(selected), 30):
            batch = selected[start : start + 30]
            tile_width, tile_height, columns = 340, 260, 5
            canvas = Image.new("RGB", (tile_width * columns, tile_height * 6), "white")
            draw = ImageDraw.Draw(canvas)
            for index, row in enumerate(batch):
                with Image.open(BytesIO(archive.read(row["image"]))) as image:
                    image = image.convert("RGB")
                    thumb = ImageOps.contain(image, (tile_width - 8, tile_height - 30))
                x = index % columns * tile_width
                y = index // columns * tile_height
                canvas.paste(thumb, (x + (tile_width - thumb.width) // 2, y + 25))
                draw.text((x + 4, y + 5), Path(row["image"]).stem, fill="black")
            target = OUTPUT / f"{category}_{start//30 + 1:02d}.jpg"
            canvas.save(target, quality=90)
            print(target)
