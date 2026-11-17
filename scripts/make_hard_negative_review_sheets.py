"""Tạo bảng ảnh để duyệt các ứng viên âm tính khó từ ZIP nguồn."""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "fire_smoke_v2_hardneg500" / "review"
SOURCES = {
    "rain": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Rain.zip"),
    "fog": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Fog.zip"),
}
TILE_W, TILE_H, LABEL_H, COLS, PAGE_SIZE = 230, 150, 22, 5, 50


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for source, path in SOURCES.items():
        entries: list[tuple[int, str, str, bytes]] = []
        seen: set[str] = set()
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name for name in archive.namelist()
                if name.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            for name in members:
                data = archive.read(name)
                sha256 = hashlib.sha256(data).hexdigest()
                if sha256 in seen:
                    continue
                seen.add(sha256)
                with Image.open(io.BytesIO(data)) as image:
                    image.load()
                entries.append((len(entries), name, sha256, data))

        with (OUT / f"{source}_candidates.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["index", "source_member", "sha256", "review_status"])
            writer.writerows((index, name, sha256, "pending") for index, name, sha256, _ in entries)

        for start in range(0, len(entries), PAGE_SIZE):
            batch = entries[start:start + PAGE_SIZE]
            rows = (len(batch) + COLS - 1) // COLS
            sheet = Image.new("RGB", (COLS * TILE_W, rows * (TILE_H + LABEL_H)), "white")
            draw = ImageDraw.Draw(sheet)
            for position, (index, name, _, data) in enumerate(batch):
                x = position % COLS * TILE_W
                y = position // COLS * (TILE_H + LABEL_H)
                with Image.open(io.BytesIO(data)) as image:
                    thumb = ImageOps.contain(image.convert("RGB"), (TILE_W, TILE_H))
                sheet.paste(thumb, (x + (TILE_W - thumb.width) // 2, y))
                draw.text((x + 3, y + TILE_H + 3), f"{index:03d} {Path(name).name}", fill="black")
            sheet.save(OUT / f"{source}_{start:03d}.jpg", quality=87)
        print(f"{source}: {len(entries)} unique images")


if __name__ == "__main__":
    main()
