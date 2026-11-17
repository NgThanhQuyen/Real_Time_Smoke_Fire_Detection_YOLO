"""Tạo trang ảnh ghép có số thứ tự để duyệt thủ công từng mẫu."""

from __future__ import annotations

import csv
import random
import zipfile
from pathlib import Path

from PIL import Image

from catargiu_contact_sheets import ROOT, SOURCE_ROOT, render


PAGE_SIZE = 20


def main() -> None:
    with (ROOT / "review_queue.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pages_root = ROOT / "review_pages"
    pages_root.mkdir(parents=True, exist_ok=True)
    selected = []
    with zipfile.ZipFile(SOURCE_ROOT / "FireSmokeDataset.v2i.yolov8.zip") as first, zipfile.ZipFile(SOURCE_ROOT / "FireSmokeNEWdataset.v1i.yolov8.zip") as second:
        archives = {"FireSmokeDataset.v2i.yolov8.zip": first, "FireSmokeNEWdataset.v1i.yolov8.zip": second}
        for kind, limit, seed in (("fire_only", 240, 20261030), ("hard_negative", 139, 20261031)):
            candidates = [row for row in rows if row["priority"] == kind]
            random.Random(seed).shuffle(candidates)
            candidates = candidates[:limit]
            for start in range(0, len(candidates), PAGE_SIZE):
                page_number = start // PAGE_SIZE + 1
                page = candidates[start:start + PAGE_SIZE]
                canvas = Image.new("RGB", (1600, 1008), "white")
                for offset, row in enumerate(page):
                    number = start + offset + 1
                    tile = render(archives[row["source"]], row["image"], f"{number:03d} {row['scene_key']} {row['scene_frames']}f")
                    canvas.paste(tile, (320 * (offset % 5), 252 * (offset // 5)))
                    selected.append({"review_id": f"{kind}:{number:03d}", "page": f"{kind}_{page_number:02d}.jpg",
                                     "source": row["source"], "image": row["image"], "scene_key": row["scene_key"],
                                     "scene_frames": row["scene_frames"], "sha256": row["sha256"],
                                     "decision": "pending", "reason": ""})
                canvas.save(pages_root / f"{kind}_{page_number:02d}.jpg", quality=92)
    with (pages_root / "review_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("review_id", "page", "source", "image", "scene_key", "scene_frames", "sha256", "decision", "reason"))
        writer.writeheader()
        writer.writerows(selected)
    print("review pages", len(list(pages_root.glob("*.jpg"))), "images", len(selected))


if __name__ == "__main__":
    main()
