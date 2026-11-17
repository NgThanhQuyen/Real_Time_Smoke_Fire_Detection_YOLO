"""Tạo validation từ train D-Fire sạch, giữ test nguyên byte."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


SEED = 20261110
TARGET_VAL = 4244
CATEGORIES = ("background", "smoke_only", "fire_only", "both")
SOURCES = ("AoF", "WEB", "PublicDataset")


def image_hashes(raw: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(raw)) as image:
        gray = ImageOps.exif_transpose(image).convert("L")
        dh_pixels = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
        dhash = int("".join(map(str, (dh_pixels[:, 1:] > dh_pixels[:, :-1]).astype(int).ravel())), 2)
        ph_pixels = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
        low = dctn(ph_pixels, norm="ortho")[:8, :8].ravel()
        phash = int("".join(map(str, (low > np.median(low[1:])).astype(int))), 2)
    return dhash, phash


def source_and_number(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(AoF|WEB|PublicDataset)(\d+)\.jpg", name)
    if match is None:
        raise ValueError(f"Tên ảnh D-Fire không đúng mẫu: {name}")
    return match.group(1), int(match.group(2))


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def join(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


def row_features(row: dict[str, str]) -> Counter[str]:
    return Counter({f"category:{row['category']}": 1, f"source:{row['source']}": 1})


def score(counts: Counter[str], targets: dict[str, float]) -> float:
    return sum((counts[key] - target) ** 2 / max(target, 1) for key, target in targets.items())


def choose_groups(groups: list[list[int]], rows: list[dict[str, str]], target_size: int) -> set[int]:
    eligible = [group for group in groups if all(rows[index]["original_split"] == "train" for index in group)]
    if sum(map(len, eligible)) < target_size:
        raise ValueError("Không đủ nhóm train độc lập với test để tạo validation.")
    total_train = sum(row["original_split"] == "train" for row in rows)
    targets = {
        key: sum(row_features(row)[key] for row in rows if row["original_split"] == "train") * target_size / total_train
        for key in [*(f"category:{item}" for item in CATEGORIES), *(f"source:{item}" for item in SOURCES)]
    }
    group_features = [sum((row_features(rows[index]) for index in group), Counter()) for group in eligible]
    best: tuple[float, set[int]] | None = None
    for attempt in range(80):
        rng = random.Random(SEED + attempt)
        order = list(range(len(eligible)))
        rng.shuffle(order)
        selected: set[int] = set()
        counts: Counter[str] = Counter()
        size = 0
        for group_index in order:
            group = eligible[group_index]
            if size + len(group) > target_size:
                continue
            new_counts = counts + group_features[group_index]
            current_gap = score(counts, targets)
            new_gap = score(new_counts, targets)
            if new_gap <= current_gap or size < target_size * 0.75:
                selected.add(group_index)
                counts = new_counts
                size += len(group)
            if size == target_size:
                break
        if size < target_size:
            remaining = [index for index in range(len(eligible)) if index not in selected]
            remaining.sort(key=lambda index: (len(eligible[index]), score(counts + group_features[index], targets), index))
            for group_index in remaining:
                if size + len(eligible[group_index]) <= target_size:
                    selected.add(group_index)
                    counts += group_features[group_index]
                    size += len(eligible[group_index])
                if size == target_size:
                    break
        if size != target_size:
            continue
        value = score(counts, targets)
        if best is None or value < best[0]:
            best = (value, {index for group_index in selected for index in eligible[group_index]})
    if best is None:
        raise ValueError("Không tìm được validation đúng số ảnh mà giữ nguyên nhóm ứng viên.")
    return best[1]


def summarize(rows: list[dict[str, str]]) -> dict[str, dict[str, int]]:
    output: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        counts = output[row["split"]]
        counts["images"] += 1
        counts[row["category"]] += 1
        counts["smoke_images"] += int(int(row["smoke_boxes"]) > 0)
        counts["fire_images"] += int(int(row["fire_boxes"]) > 0)
        counts["smoke_boxes"] += int(row["smoke_boxes"])
        counts["fire_boxes"] += int(row["fire_boxes"])
        counts[f"source_{row['source']}"] += 1
    return {split: dict(output[split]) for split in ("train", "val", "test")}


def build(input_zip: Path, input_manifest: Path, output_zip: Path, report_dir: Path) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with input_manifest.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "kept"]
    rows.sort(key=lambda row: (row["split"], row["image"]))
    for row in rows:
        row["original_split"] = row["split"]
        row["source"], number = source_and_number(row["image"])
        row["number"] = str(number)

    with zipfile.ZipFile(input_zip) as source:
        archive_images = {name for name in source.namelist() if name.endswith(".jpg")}
        expected_images = {f"{row['original_split']}/images/{row['image']}" for row in rows}
        if archive_images != expected_images:
            raise ValueError("Manifest không khớp ảnh trong ZIP sạch.")
        features = []
        for index, row in enumerate(rows, 1):
            raw = source.read(f"{row['original_split']}/images/{row['image']}")
            if hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError(f"SHA-256 sai: {row['image']}")
            features.append(image_hashes(raw))
            if index % 3000 == 0:
                print(f"hashes: {index}/{len(rows)}", flush=True)

        union = UnionFind(len(rows))
        by_source: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            by_source[row["source"]].append(index)
        candidate_edges = 0
        for indices in by_source.values():
            indices.sort(key=lambda index: int(rows[index]["number"]))
            for first, second in zip(indices, indices[1:]):
                if int(rows[second]["number"]) - int(rows[first]["number"]) > 2:
                    continue
                dh_distance = (features[first][0] ^ features[second][0]).bit_count()
                ph_distance = (features[first][1] ^ features[second][1]).bit_count()
                if dh_distance <= 24 and ph_distance <= 20:
                    union.join(first, second)
                    candidate_edges += 1

        components: dict[int, list[int]] = defaultdict(list)
        for index in range(len(rows)):
            components[union.find(index)].append(index)
        groups = list(components.values())
        selected = choose_groups(groups, rows, TARGET_VAL)
        for index, row in enumerate(rows):
            row["split"] = "test" if row["original_split"] == "test" else "val" if index in selected else "train"
            row["candidate_group"] = str(union.find(index))
            row["dhash"] = f"{features[index][0]:016x}"
            row["phash"] = f"{features[index][1]:016x}"

        counts = summarize(rows)
        if [counts[split]["images"] for split in ("train", "val", "test")] != [12699, 4244, 4244]:
            raise ValueError("Số ảnh sau chia tập không đúng mục tiêu.")
        test_groups = {row["candidate_group"] for row in rows if row["split"] == "test"}
        val_groups = {row["candidate_group"] for row in rows if row["split"] == "val"}
        train_groups = {row["candidate_group"] for row in rows if row["split"] == "train"}
        if val_groups & (train_groups | test_groups):
            raise ValueError("Nhóm ứng viên đi qua validation và tập khác.")

        temporary = output_zip.with_suffix(output_zip.suffix + ".partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
            for index, row in enumerate(rows, 1):
                old_split, new_split, name = row["original_split"], row["split"], row["image"]
                for folder, suffix in (("images", ".jpg"), ("labels", ".txt")):
                    old_path = f"{old_split}/{folder}/{name[:-4]}{suffix}"
                    new_path = f"{new_split}/{folder}/{name[:-4]}{suffix}"
                    target.writestr(new_path, source.read(old_path))
                if index % 4000 == 0:
                    print(f"written: {index}/{len(rows)}", flush=True)
            for split in ("train", "val", "test"):
                names = sorted(row["image"] for row in rows if row["split"] == split)
                target.writestr(f"splits/{split}.txt", "\n".join(names) + "\n")
            target.writestr("data.yaml", "train: train/images\nval: val/images\ntest: test/images\nnc: 2\nnames:\n  0: smoke\n  1: fire\n")
            target.writestr("DATASET_INFO.md", "# D-Fire sạch với validation cố định\n\nNguồn: dfire_original_clean_v1.zip. Test giữ nguyên byte.\nValidation gồm 4.244 ảnh từ train sạch. Nhãn gốc: 0 = smoke, 1 = fire.\nNhóm cảnh ứng viên dựa trên ID liền kề và dHash/pHash, không phải xác nhận thủ công cùng cảnh.\n")
        temporary.replace(output_zip)

    with (report_dir / "split_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        fields = ("image", "original_split", "split", "category", "smoke_boxes", "fire_boxes", "source", "number", "sha256", "candidate_group", "dhash", "phash")
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, object] = {
        "input_zip": str(input_zip),
        "output_zip": str(output_zip),
        "seed": SEED,
        "target_validation_images": TARGET_VAL,
        "class_map": {"0": "smoke", "1": "fire"},
        "counts": counts,
        "candidate_edges": candidate_edges,
        "candidate_groups": len(groups),
        "candidate_groups_touching_test_and_train": sum(
            {rows[index]["split"] for index in group} == {"train", "test"} for group in groups
        ),
        "largest_candidate_group": max(map(len, groups)),
        "validation_group_overlap": 0,
        "method_limit": "ID liền kề và hash cảm nhận chỉ là tín hiệu; chưa kiểm duyệt thủ công toàn bộ cặp cùng cảnh.",
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-zip", type=Path, default=root / "runtime/dfire_original_clean_v1.zip")
    parser.add_argument("--input-manifest", type=Path, default=root / "reports/dfire_original_clean_v1/image_manifest.csv")
    parser.add_argument("--output-zip", type=Path, default=root / "runtime/dfire_original_clean_split_v1.zip")
    parser.add_argument("--report-dir", type=Path, default=root / "reports/dfire_original_split_v1")
    args = parser.parse_args()
    result = build(args.input_zip, args.input_manifest, args.output_zip, args.report_dir)
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
