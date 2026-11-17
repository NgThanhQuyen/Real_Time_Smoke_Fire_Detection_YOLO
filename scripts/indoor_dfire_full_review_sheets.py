"""Tạo bảng xem toàn bộ ảnh chỉ lửa Indoor mà không sửa dữ liệu nguồn."""

import csv
import io
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip")
INDEX = ROOT / "reports/indoor_fire_audit/fire_only_cross_locked.csv"
OUT = ROOT / "reports/indoor_dfire_original_review/full_review"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with INDEX.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for number, row in enumerate(rows, 1):
        row["review_id"] = f"I{number:04d}"

    with (OUT / "index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", *[key for key in rows[0] if key != "review_id"]])
        writer.writeheader()
        writer.writerows(rows)

    per_page = 80
    columns = 8
    cell_w, cell_h = 215, 175
    with zipfile.ZipFile(SOURCE) as archive:
        for page_no in range(math.ceil(len(rows) / per_page)):
            page_rows = rows[page_no * per_page : (page_no + 1) * per_page]
            canvas = Image.new("RGB", (columns * cell_w, math.ceil(len(page_rows) / columns) * cell_h), "white")
            draw = ImageDraw.Draw(canvas)
            for position, row in enumerate(page_rows):
                image = Image.open(io.BytesIO(archive.read(row["image"]))).convert("RGB")
                label = archive.read(row["label"]).decode("utf-8-sig")
                boxes = ImageDraw.Draw(image)
                for line in label.splitlines():
                    parts = line.split()
                    if len(parts) != 5 or parts[0] != "0":
                        continue
                    _, x, y, width, height = map(float, parts)
                    iw, ih = image.size
                    boxes.rectangle(
                        ((x - width / 2) * iw, (y - height / 2) * ih,
                         (x + width / 2) * iw, (y + height / 2) * ih),
                        outline="lime", width=max(2, iw // 250),
                    )
                thumb = ImageOps.contain(image, (cell_w - 6, cell_h - 31))
                px, py = (position % columns) * cell_w, (position // columns) * cell_h
                canvas.paste(thumb, (px + (cell_w - thumb.width) // 2, py + 27))
                name = Path(row["image"]).stem.split(".rf.")[0]
                draw.text((px + 3, py + 2), f'{row["review_id"]} {name[:24]}', fill="black")
            canvas.save(OUT / f"page_{page_no + 1:02d}.jpg", quality=82)
            print(f"page {page_no + 1:02d}: {len(page_rows)}")


if __name__ == "__main__":
    main()
