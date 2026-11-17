"""Kiểm tra nhãn, ảnh và nguồn Pyro-SDIS trước khi tuyển chọn vào D-Fire."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError

from dfire_raw_review import hashes_and_hist


EXPECTED_FILES = (
    "train-00000-of-00003.parquet",
    "train-00001-of-00003.parquet",
    "train-00002-of-00003.parquet",
    "val-00000-of-00001.parquet",
)
FIELDS = (
    "parquet_file",
    "source_row",
    "source_split",
    "image_name",
    "partner",
    "camera",
    "camera_key",
    "date",
    "day",
    "positive",
    "box_count",
    "min_box_area",
    "max_box_area",
    "width",
    "height",
    "format",
    "sha256",
    "dhash",
    "phash",
    "dfire_exact_match",
    "issue",
)


def read_dfire_hashes(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sha256"] for row in csv.DictReader(handle) if row["sha256"]}


def parse_boxes(annotation: str | None) -> tuple[list[float], list[str]]:
    areas: list[float] = []
    issues: list[str] = []
    for line_number, line in enumerate((annotation or "").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"nhan_{line_number}_sai_so_cot")
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(f"nhan_{line_number}_khong_phai_so")
            continue
        if class_id != 0:
            issues.append(f"nhan_{line_number}_sai_lop")
        if width <= 0 or height <= 0:
            issues.append(f"nhan_{line_number}_box_rong")
        if (
            x_center - width / 2 < 0
            or x_center + width / 2 > 1
            or y_center - height / 2 < 0
            or y_center + height / 2 > 1
        ):
            issues.append(f"nhan_{line_number}_vuot_bien")
        areas.append(width * height)
    return areas, issues


def audit(source: Path, output: Path, dfire_index: Path) -> None:
    files = [source / "data" / name for name in EXPECTED_FILES]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Thiếu Parquet Pyro-SDIS: " + ", ".join(missing))
    output.mkdir(parents=True, exist_ok=True)
    dfire_hashes = read_dfire_hashes(dfire_index)
    totals: Counter[str] = Counter()
    cameras_by_split: dict[str, set[str]] = defaultdict(set)
    names: set[str] = set()
    hashes: set[str] = set()
    per_camera: Counter[str] = Counter()
    small_box_counts: Counter[str] = Counter()
    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for path in files:
            split = "train" if path.name.startswith("train") else "val"
            parquet = pq.ParquetFile(path)
            row_number = 0
            for batch in parquet.iter_batches(batch_size=32):
                columns = batch.to_pydict()
                for index in range(batch.num_rows):
                    name = str(columns["image_name"][index])
                    partner = str(columns["partner"][index])
                    camera = str(columns["camera"][index])
                    date = str(columns["date"][index])
                    camera_key = f"{partner}/{camera}"
                    areas, label_issues = parse_boxes(columns["annotations"][index])
                    issues = label_issues.copy()
                    image_bytes = columns["image"][index]["bytes"]
                    sha256 = hashlib.sha256(image_bytes).hexdigest() if image_bytes else ""
                    width = height = 0
                    image_format = ""
                    dhash = phash = ""
                    if not image_bytes:
                        issues.append("anh_rong")
                    else:
                        try:
                            with Image.open(BytesIO(image_bytes)) as image:
                                image.load()
                                width, height = image.size
                                image_format = image.format or ""
                                dvalue, pvalue, _ = hashes_and_hist(image)
                                dhash, phash = f"{dvalue:016x}", f"{pvalue:016x}"
                        except (OSError, ValueError, UnidentifiedImageError):
                            issues.append("anh_khong_doc_duoc")
                    if name in names:
                        issues.append("ten_anh_trung")
                    names.add(name)
                    if sha256 and sha256 in hashes:
                        issues.append("anh_trung_tuyet_doi_trong_nguon")
                    hashes.add(sha256)
                    if not date or len(date) < 10:
                        issues.append("thieu_ngay")
                    if not partner or not camera:
                        issues.append("thieu_nguon_camera")
                    exact_match = bool(sha256 and sha256 in dfire_hashes)
                    row = {
                        "parquet_file": path.name,
                        "source_row": row_number,
                        "source_split": split,
                        "image_name": name,
                        "partner": partner,
                        "camera": camera,
                        "camera_key": camera_key,
                        "date": date,
                        "day": date[:10],
                        "positive": int(bool(areas)),
                        "box_count": len(areas),
                        "min_box_area": f"{min(areas):.8f}" if areas else "",
                        "max_box_area": f"{max(areas):.8f}" if areas else "",
                        "width": width,
                        "height": height,
                        "format": image_format,
                        "sha256": sha256,
                        "dhash": dhash,
                        "phash": phash,
                        "dfire_exact_match": int(exact_match),
                        "issue": ";".join(sorted(set(issues))),
                    }
                    writer.writerow(row)
                    totals["images"] += 1
                    totals[f"split_{split}"] += 1
                    totals["positive" if areas else "negative"] += 1
                    totals["boxes"] += len(areas)
                    totals["with_issue"] += bool(issues)
                    totals["dfire_exact_matches"] += exact_match
                    for area in areas:
                        if area < 0.005:
                            small_box_counts["lt_0_5pct"] += 1
                        if area < 0.02:
                            small_box_counts["lt_2pct"] += 1
                    cameras_by_split[split].add(camera_key)
                    per_camera[camera_key] += 1
                    row_number += 1
            print(f"Audited {path.name}: {row_number} images", flush=True)
    summary = {
        "source_revision": "ebfc02d0cba7f7dfe0623d13bcf0f0aa8a38e3ee",
        "source": "pyronear/pyro-sdis v2.0",
        "counts": dict(totals),
        "small_box_counts": dict(small_box_counts),
        "camera_count": len(per_camera),
        "cameras_in_both_original_splits": len(cameras_by_split["train"] & cameras_by_split["val"]),
        "largest_camera_groups": per_camera.most_common(15),
        "dfire_hash_index": str(dfire_index),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("runtime/pyro_sdis_v2_source"))
    parser.add_argument("--output", type=Path, default=Path("runtime/pyro_sdis_v2_audit"))
    parser.add_argument("--dfire-index", type=Path, default=Path("runtime/dfire_raw_review/image_index.csv"))
    args = parser.parse_args()
    audit(args.source, args.output, args.dfire_index)


if __name__ == "__main__":
    main()
