"""Tạo báo cáo ứng viên cần kiểm duyệt từ ZIP D-Fire gốc, không đổi dữ liệu nguồn."""

from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.fft import dctn


def hashes_and_hist(image: Image.Image) -> tuple[int, int, np.ndarray]:
    gray = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
    dhash = int("".join(map(str, (pixels[:, 1:] > pixels[:, :-1]).astype(int).ravel())), 2)
    ph = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    low = dctn(ph, norm="ortho")[:8, :8]
    bits = low.ravel() > np.median(low.ravel()[1:])
    phash = int("".join(map(str, bits.astype(int))), 2)
    rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB").resize((64, 64)))
    histogram = np.concatenate([np.histogram(rgb[:, :, c], bins=8, range=(0, 256))[0] for c in range(3)])
    histogram = histogram.astype(np.float32) / (64 * 64 * 3)
    return dhash, phash, histogram


def label_info(text: str, image_key: str) -> tuple[str, int, int, list[dict]]:
    classes = set()
    counts = Counter()
    problems = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("column_count")
            values = [float(part) for part in parts]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite")
            cls, x, y, w, h = values
            if not cls.is_integer() or cls not in (0, 1):
                raise ValueError("class")
            classes.add(int(cls))
            counts[int(cls)] += 1
            if w <= 0 or h <= 0:
                problems.append(dict(image=image_key, line=line_number, reason="nonpositive", raw=line))
            if x - w / 2 < 0 or y - h / 2 < 0 or x + w / 2 > 1 or y + h / 2 > 1:
                problems.append(dict(image=image_key, line=line_number, reason="outside", raw=line))
        except ValueError as exc:
            problems.append(dict(image=image_key, line=line_number, reason=str(exc), raw=line))
    category = "both" if len(classes) == 2 else "smoke_only" if 0 in classes else "fire_only" if 1 in classes else "background"
    return category, counts[0], counts[1], problems


def prefix_for(key: str) -> str:
    stem = Path(key).stem
    return "".join(char for char in stem if not char.isdigit())


def inspect_zip(source: Path) -> tuple[pd.DataFrame, list[np.ndarray], pd.DataFrame]:
    rows = []
    histograms = []
    issues = []
    with zipfile.ZipFile(source) as archive:
        images = sorted(name for name in archive.namelist() if name.endswith(".jpg"))
        for number, name in enumerate(images, 1):
            label = name.replace("/images/", "/labels/")[:-4] + ".txt"
            category, smoke_boxes, fire_boxes, label_issues = label_info(
                archive.read(label).decode("utf-8-sig"), name
            )
            issues.extend(label_issues)
            data = archive.read(name)
            with Image.open(BytesIO(data)) as image:
                dhash, phash, histogram = hashes_and_hist(image)
                width, height = image.size
                actual_format = image.format
            rows.append(dict(image=name, split=name.split("/")[0], prefix=prefix_for(name), category=category,
                             smoke_boxes=smoke_boxes, fire_boxes=fire_boxes, width=width, height=height,
                             format=actual_format, sha256=hashlib.sha256(data).hexdigest(),
                             dhash=f"{dhash:016x}", phash=f"{phash:016x}"))
            histograms.append(histogram)
            if number % 5000 == 0:
                print(f"read_images={number}", flush=True)
    return pd.DataFrame(rows), histograms, pd.DataFrame(issues)


def candidate_pairs(index: pd.DataFrame, histograms: list[np.ndarray]) -> pd.DataFrame:
    dhashes = [int(value, 16) for value in index.dhash]
    phashes = [int(value, 16) for value in index.phash]
    dimensions = list(zip(index.width, index.height))
    buckets = defaultdict(list)
    for number, (value, dimension) in enumerate(zip(dhashes, dimensions)):
        for band in range(4):
            buckets[(dimension, band, (value >> (band * 16)) & 65535)].append(number)
    seen = set()
    pairs = []
    for members in buckets.values():
        for offset, left in enumerate(members):
            for right in members[offset + 1:]:
                key = (left, right)
                if key in seen:
                    continue
                seen.add(key)
                dh = (dhashes[left] ^ dhashes[right]).bit_count()
                if dh > 3:
                    continue
                ph = (phashes[left] ^ phashes[right]).bit_count()
                color = float(np.minimum(histograms[left], histograms[right]).sum())
                if ph > 12 or color < 0.72:
                    continue
                a, b = index.iloc[left], index.iloc[right]
                kind = "cross_split" if a.split != b.split else f"within_{a.split}"
                pairs.append(dict(image_a=a.image, image_b=b.image, split_a=a.split, split_b=b.split,
                                  prefix_a=a.prefix, prefix_b=b.prefix, category_a=a.category,
                                  category_b=b.category, dhash_distance=dh, phash_distance=ph,
                                  color_overlap=round(color, 4), pair_type=kind))
    return pd.DataFrame(pairs)


def selected_pairs(candidates: pd.DataFrame) -> pd.DataFrame:
    cross = candidates[candidates.pair_type == "cross_split"].copy()
    cross["source"] = np.where(cross.prefix_a == cross.prefix_b, cross.prefix_a, "mixed")
    cross["test_image"] = np.where(cross.split_a == "test", cross.image_a, cross.image_b)
    cross["priority"] = cross.dhash_distance + cross.phash_distance / 4 + (1 - cross.color_overlap) * 10
    cross = cross.sort_values(["priority", "test_image"]).drop_duplicates("test_image")
    chosen = []
    for (_, _), group in cross.groupby(["source", "category_a"], sort=True):
        used_blocks = set()
        for row in group.itertuples():
            digits = "".join(char for char in Path(row.test_image).stem if char.isdigit())
            block = int(digits) // 200 if digits else row.test_image
            if block in used_blocks:
                continue
            chosen.append(row.Index)
            used_blocks.add(block)
            if len(used_blocks) == 4:
                break
    return cross.loc[chosen].sort_values(["source", "category_a", "priority"])


def draw_sheets(source: Path, selected: pd.DataFrame, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for page, start in enumerate(range(0, len(selected), 6), 1):
            sheet = Image.new("RGB", (1000, 6 * 245), "white")
            draw = ImageDraw.Draw(sheet)
            for row_number, row in enumerate(selected.iloc[start:start + 6].itertuples()):
                for column, key in enumerate((row.image_a, row.image_b)):
                    with Image.open(BytesIO(archive.read(key))) as raw:
                        image = ImageOps.exif_transpose(raw).convert("RGB")
                        image.thumbnail((490, 210))
                        sheet.paste(image, (column * 500 + (500 - image.width) // 2, row_number * 245))
                    caption = f"{key} | {row.category_a if column == 0 else row.category_b}"
                    draw.text((column * 500 + 5, row_number * 245 + 214), caption[:69], fill="black")
                draw.text((5, row_number * 245 + 229),
                          f"dHash={row.dhash_distance} pHash={row.phash_distance} color={row.color_overlap:.2f}", fill="black")
            sheet.save(folder / f"cross_split_{page:02d}.jpg", quality=85)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    index, histograms, issues = inspect_zip(args.source)
    index.to_csv(args.output / "image_index.csv", index=False, encoding="utf-8-sig")
    issues.to_csv(args.output / "label_issues.csv", index=False, encoding="utf-8-sig")
    candidates = candidate_pairs(index, histograms)
    candidates.to_csv(args.output / "candidate_pairs.csv", index=False, encoding="utf-8-sig")
    selected = selected_pairs(candidates)
    selected.to_csv(args.output / "visual_review_queue.csv", index=False, encoding="utf-8-sig")
    draw_sheets(args.source, selected, args.output / "sheets")
    summary = dict(source=str(args.source), images=len(index), label_issue_rows=len(issues),
                   candidate_pairs=len(candidates), cross_split_pairs=int((candidates.pair_type == "cross_split").sum()),
                   test_repetition_pairs=int((candidates.pair_type == "within_test").sum()),
                   selected_visual_pairs=len(selected),
                   note="Ứng viên hash không phải xác nhận cùng cảnh; không dùng để tự xóa hoặc chia dữ liệu.")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
