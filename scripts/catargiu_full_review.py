"""Tạo trang ảnh đủ 640 px cho mẫu được chọn sơ bộ."""

from __future__ import annotations

import csv
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw

from catargiu_contact_sheets import ROOT, SOURCE_ROOT, COLORS


def tile(archive: zipfile.ZipFile, row: dict[str, str]) -> Image.Image:
    name = row["image"]
    with Image.open(BytesIO(archive.read(name))) as image:
        image = image.convert("RGB")
        canvas = Image.new("RGB", (640, 668), "#141414")
        canvas.paste(image, (0, 0))
    label = name.replace("/images/", "/labels/")[:-4] + ".txt"
    draw = ImageDraw.Draw(canvas)
    for line in archive.read(label).decode("utf-8-sig").splitlines():
        values = line.split()
        if len(values) != 5:
            continue
        cls = int(float(values[0]))
        x, y, width, height = map(float, values[1:])
        draw.rectangle(((x - width / 2) * 640, (y - height / 2) * 640,
                        (x + width / 2) * 640, (y + height / 2) * 640),
                       outline=COLORS[cls], width=3)
    draw.text((5, 645), row["review_id"] + " " + row["scene_key"][:60], fill="white")
    return canvas


def main() -> None:
    pages = ROOT / "review_pages"
    with (pages / "visual_review.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["decision"] == "provisional"]
    with zipfile.ZipFile(SOURCE_ROOT / "FireSmokeDataset.v2i.yolov8.zip") as first, zipfile.ZipFile(SOURCE_ROOT / "FireSmokeNEWdataset.v1i.yolov8.zip") as second:
        archives = {"FireSmokeDataset.v2i.yolov8.zip": first, "FireSmokeNEWdataset.v1i.yolov8.zip": second}
        for kind in ("fire_only", "hard_negative"):
            candidates = [row for row in rows if row["review_id"].startswith(kind)]
            for start in range(0, len(candidates), 6):
                canvas = Image.new("RGB", (1920, 1336), "white")
                for offset, row in enumerate(candidates[start:start + 6]):
                    canvas.paste(tile(archives[row["source"]], row), (640 * (offset % 3), 668 * (offset // 3)))
                canvas.save(pages / f"full_{kind}_{start // 6 + 1:02d}.jpg", quality=90)
    print("full review images", len(rows))


if __name__ == "__main__":
    main()
