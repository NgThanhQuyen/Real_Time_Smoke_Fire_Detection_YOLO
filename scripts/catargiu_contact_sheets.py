"""Tạo ảnh ghép có box để kiểm tra nhãn và trùng cảnh bằng mắt."""

from __future__ import annotations

import csv
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path("runtime/catargiu_audit")
SOURCE_ROOT = Path(r"F:\6\KLTN\Smoke-Fire-Dataset")
COLORS = {0: "#ff351c", 1: "#ffe43b", 2: "#43ddff"}


def read_csv(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render(archive: zipfile.ZipFile, name: str, caption: str, boxes: bool = True) -> Image.Image:
    with Image.open(BytesIO(archive.read(name))) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((310, 216))
        tile = Image.new("RGB", (320, 252), "#1a1a1a")
        tile.paste(image, ((320 - image.width) // 2, 4))
        if boxes:
            label = name.replace("/images/", "/labels/")[:-4] + ".txt"
            original = Image.open(BytesIO(archive.read(name)))
            width, height = original.size
            original.close()
            draw = ImageDraw.Draw(tile)
            left, top = (320 - image.width) // 2, 4
            for line in archive.read(label).decode("utf-8-sig").splitlines():
                values = line.split()
                if len(values) != 5:
                    continue
                cls = int(float(values[0]))
                x, y, w, h = map(float, values[1:])
                draw.rectangle((left + (x - w / 2) * image.width, top + (y - h / 2) * image.height,
                                left + (x + w / 2) * image.width, top + (y + h / 2) * image.height),
                               outline=COLORS[cls], width=2)
        ImageDraw.Draw(tile).text((5, 224), caption[:43], fill="white")
        return tile


def sheet(name: str, rows: list[dict[str, str]], archives: dict[str, zipfile.ZipFile], limit: int = 20) -> None:
    tiles = [render(archives[row["source"]], row["image"], f'{row["source"][:16]} {row["scene_key"]}') for row in rows[:limit]]
    output = Image.new("RGB", (320 * 5, 252 * ((len(tiles) + 4) // 5)), "white")
    for index, tile in enumerate(tiles):
        output.paste(tile, (320 * (index % 5), 252 * (index // 5)))
    output.save(ROOT / name, quality=90)


def distinct_groups(rows: list[dict[str, str]], n: int, seed: int) -> list[dict[str, str]]:
    random.Random(seed).shuffle(rows)
    chosen = []
    used = set()
    for row in rows:
        key = (row["source"], row["scene_key"])
        if key not in used:
            chosen.append(row)
            used.add(key)
        if len(chosen) >= n:
            break
    return chosen


def main() -> None:
    rows = read_csv("image_index.csv")
    with zipfile.ZipFile(SOURCE_ROOT / "FireSmokeDataset.v2i.yolov8.zip") as first, zipfile.ZipFile(SOURCE_ROOT / "FireSmokeNEWdataset.v1i.yolov8.zip") as second:
        archives = {"FireSmokeDataset.v2i.yolov8.zip": first, "FireSmokeNEWdataset.v1i.yolov8.zip": second}
        fire = [row for row in rows if int(row["fire_boxes"]) and not int(row["smoke_boxes"])]
        other = [row for row in rows if row["category"] == "other"]
        mixed = [row for row in rows if int(row["fire_boxes"]) and int(row["smoke_boxes"])]
        sheet("fire_only_review.jpg", distinct_groups(fire, 20, 42), archives)
        sheet("other_only_review.jpg", distinct_groups(other, 20, 43), archives)
        sheet("fire_smoke_review.jpg", distinct_groups(mixed, 20, 44), archives)
        near = read_csv("dfire_near_matches.csv")
        near = [row for row in near if row["dfire_split"] == "test" and int(row["dhash_distance"]) <= 3 and int(row["phash_distance"]) <= 12]
        selected = distinct_groups(near, 12, 45)
        with zipfile.ZipFile(SOURCE_ROOT / "D-Fire-YOLO.zip") as dfire:
            tiles = []
            for row in selected:
                tiles.extend((render(archives[row["source"]], row["image"], f'New {row["scene_key"]} d{row["dhash_distance"]} p{row["phash_distance"]}'),
                              render(dfire, row["dfire_image"], f'D-Fire test {Path(row["dfire_image"]).stem}', boxes=False)))
            output = Image.new("RGB", (640 * 2, 252 * ((len(tiles) + 3) // 4)), "white")
            for index, tile in enumerate(tiles):
                output.paste(tile, (320 * (index % 4), 252 * (index // 4)))
            output.save(ROOT / "dfire_test_near_review.jpg", quality=90)


if __name__ == "__main__":
    main()
