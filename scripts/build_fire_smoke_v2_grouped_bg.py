"""Tạo bản thử nghiệm data 2 với split theo nguồn và ảnh nền D-Fire."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

import numpy as np
from PIL import Image
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"F:\6\KLTN\Smoke-Fire-Dataset\fire and smoke detection.v2i.yolov11.zip")
BACKGROUND_SOURCE = ROOT / "runtime" / "fire_smoke_dfire_final.zip"
OUTPUT = ROOT / "runtime" / "Smoke-Fire-Detection-Dataset.zip"
REPORT = ROOT / "reports" / "fire_smoke_v2_grouped_bg_v1"
SEED = 20261109
BACKGROUND_QUOTA = {"train": 1500, "valid": 150, "test": 150}
SPLITS = ("train", "valid", "test")
CATEGORIES = ("fire_only", "smoke_only", "both", "background")
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]+$")
VIDEO_ID = re.compile(r"video_(\d+)", re.IGNORECASE)


def image_is_valid(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as image:
            image.load()
            return image.width > 0 and image.height > 0
    except Exception:
        return False


def source_group(image_path: str) -> tuple[str, str]:
    stem = RF_SUFFIX.sub("", Path(image_path).stem)
    video = VIDEO_ID.search(stem)
    if video:
        return f"video_{video.group(1)}", "video"
    return stem, "youtube"


def label_category(label: bytes) -> tuple[str, Counter]:
    classes = Counter()
    for line in label.decode("utf-8-sig").splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5 or fields[0] not in {"0", "1"}:
            raise ValueError(f"Nhãn không hợp lệ: {line[:100]}")
        values = [float(value) for value in fields[1:]]
        x, y, width, height = values
        if width <= 0 or height <= 0 or any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"Tọa độ không hợp lệ: {line[:100]}")
        if x - width / 2 < -1e-6 or x + width / 2 > 1 + 1e-6:
            raise ValueError(f"Box vượt mép ngang: {line[:100]}")
        if y - height / 2 < -1e-6 or y + height / 2 > 1 + 1e-6:
            raise ValueError(f"Box vượt mép dọc: {line[:100]}")
        classes[int(fields[0])] += 1
    if not classes:
        return "background", classes
    if set(classes) == {0}:
        return "fire_only", classes
    if set(classes) == {1}:
        return "smoke_only", classes
    return "both", classes


def read_positive_inventory() -> tuple[list[dict], list[str]]:
    records = []
    corrupt = []
    with ZipFile(SOURCE) as archive:
        names = set(archive.namelist())
        image_paths = sorted(
            name for name in names if "/images/" in name and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        for image_path in image_paths:
            label_path = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            if label_path not in names:
                raise ValueError(f"Thiếu nhãn: {image_path}")
            image_data = archive.read(image_path)
            if not image_is_valid(image_data):
                corrupt.append(image_path)
                continue
            label_data = archive.read(label_path)
            category, boxes = label_category(label_data)
            group, family = source_group(image_path)
            records.append(
                {
                    "source_image": image_path,
                    "source_label": label_path,
                    "old_split": image_path.split("/", 1)[0],
                    "group": group,
                    "family": family,
                    "category": category,
                    "fire_boxes": boxes[0],
                    "smoke_boxes": boxes[1],
                    "sha256": hashlib.sha256(image_data).hexdigest(),
                }
            )
    return records, corrupt


def assign_groups(records: list[dict]) -> dict[str, str]:
    groups: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        group = groups[record["group"]]
        group["images"] += 1
        group[record["category"]] += 1
        group[f"{record['family']}_images"] += 1
    group_names = sorted(groups)
    features = ("images", "fire_only", "smoke_only", "both", "video_images", "youtube_images")
    totals = Counter()
    for group in groups.values():
        totals.update(group)
    target_splits = ("valid", "test")
    group_vars = len(group_names) * 3
    deviations = len(target_splits) * len(features) * 2
    variable_count = group_vars + deviations
    rows = len(group_names) + len(target_splits) * len(features) + 4
    matrix = lil_matrix((rows, variable_count), dtype=float)
    lower = np.full(rows, -np.inf)
    upper = np.full(rows, np.inf)
    objective = np.zeros(variable_count)
    integrality = np.zeros(variable_count)
    integrality[:group_vars] = 1
    row = 0
    for index, _ in enumerate(group_names):
        for split_index in range(3):
            matrix[row, index * 3 + split_index] = 1
        lower[row] = upper[row] = 1
        row += 1
    weights = {"images": 12, "fire_only": 4, "smoke_only": 4, "both": 4, "video_images": 3, "youtube_images": 3}
    for split_index, _ in enumerate(target_splits, start=1):
        for feature_index, feature in enumerate(features):
            for group_index, name in enumerate(group_names):
                matrix[row, group_index * 3 + split_index] = groups[name][feature]
            deviation_index = group_vars + ((split_index - 1) * len(features) + feature_index) * 2
            matrix[row, deviation_index] = -1
            matrix[row, deviation_index + 1] = 1
            target = totals[feature] * 0.1
            lower[row] = upper[row] = target
            objective[deviation_index : deviation_index + 2] = weights[feature] / max(target, 1)
            row += 1
    for split_index in (1, 2):
        for group_index, name in enumerate(group_names):
            if name.startswith("video_"):
                matrix[row, group_index * 3 + split_index] = 1
        lower[row] = 2
        row += 1
        for group_index, name in enumerate(group_names):
            matrix[row, group_index * 3 + split_index] = groups[name]["youtube_images"]
        lower[row] = 300
        row += 1
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(variable_count), np.concatenate((np.ones(group_vars), np.full(deviations, np.inf)))),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 90, "mip_rel_gap": 0.02},
    )
    if result.x is None:
        raise RuntimeError(f"Không tìm được cách chia nhóm: {result.message}")
    assignment = {}
    for index, name in enumerate(group_names):
        split_index = int(np.argmax(result.x[index * 3 : index * 3 + 3]))
        assignment[name] = SPLITS[split_index]
    return assignment


def select_backgrounds(positive_hashes: set[str]) -> list[dict]:
    rng = random.Random(SEED)
    with ZipFile(BACKGROUND_SOURCE) as archive:
        provenance = {}
        for line in archive.read("metadata/selection_manifest.jsonl").decode("utf-8").splitlines():
            item = json.loads(line)
            if item["category"] == "background":
                provenance[item["output_image"]] = item
        selected = []
        used_hashes = set(positive_hashes)
        for split in SPLITS:
            candidates = [item for item in provenance.values() if item["split"] == split]
            rng.shuffle(candidates)
            count = 0
            for item in candidates:
                image_path = item["output_image"]
                label_path = item["output_label"]
                if archive.getinfo(label_path).file_size != 0:
                    raise ValueError(f"Nhãn nền không rỗng: {label_path}")
                image_data = archive.read(image_path)
                digest = hashlib.sha256(image_data).hexdigest()
                if digest in used_hashes or not image_is_valid(image_data):
                    continue
                used_hashes.add(digest)
                selected.append(
                    {
                        "source_image": image_path,
                        "source_label": label_path,
                        "old_split": split,
                        "new_split": split,
                        "group": f"dfire_{item['source_image']}",
                        "family": "dfire_background",
                        "category": "background",
                        "fire_boxes": 0,
                        "smoke_boxes": 0,
                        "sha256": digest,
                        "background_original": item["source_image"],
                    }
                )
                count += 1
                if count == BACKGROUND_QUOTA[split]:
                    break
            if count != BACKGROUND_QUOTA[split]:
                raise ValueError(f"Không đủ ảnh nền hợp lệ cho {split}: {count}")
    return selected


def write_dataset(positives: list[dict], backgrounds: list[dict], summary: dict) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists():
        raise FileExistsError(f"Đầu ra đã tồn tại: {OUTPUT}")
    temp = OUTPUT.with_suffix(".zip.tmp")
    if temp.exists():
        raise FileExistsError(f"File tạm đã tồn tại: {temp}")
    manifest_path = REPORT / "manifest.csv"
    fields = (
        "source_dataset", "source_image", "source_label", "old_split", "new_split", "output_image", "output_label",
        "group", "family", "category", "fire_boxes", "smoke_boxes", "sha256", "background_original",
    )
    with ZipFile(SOURCE) as positive_archive, ZipFile(BACKGROUND_SOURCE) as background_archive, ZipFile(temp, "w", compression=ZIP_STORED, allowZip64=True) as output, manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        output.writestr("data.yaml", "train: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames:\n  0: Fire\n  1: Smoke\n")
        for source_name, archive, records in (
            ("fire_and_smoke_detection_v2", positive_archive, positives),
            ("dfire_final_background", background_archive, backgrounds),
        ):
            for record in sorted(records, key=lambda item: item["source_image"]):
                image_data = archive.read(record["source_image"])
                if hashlib.sha256(image_data).hexdigest() != record["sha256"]:
                    raise ValueError(f"Ảnh nguồn đã thay đổi: {record['source_image']}")
                filename = Path(record["source_image"]).name
                if source_name == "dfire_final_background":
                    filename = "bg_dfire_" + filename
                output_image = f"{record['new_split']}/images/{filename}"
                output_label = f"{record['new_split']}/labels/{Path(filename).stem}.txt"
                output.writestr(output_image, image_data)
                output.writestr(output_label, archive.read(record["source_label"]))
                writer.writerow(
                    {
                        "source_dataset": source_name,
                        **record,
                        "output_image": output_image,
                        "output_label": output_label,
                        "background_original": record.get("background_original", ""),
                    }
                )
        output.writestr("DATASET_INFO.md", summary["dataset_info"])
    os.replace(temp, OUTPUT)
    (REPORT / "summary.json").write_text(json.dumps({key: value for key, value in summary.items() if key != "dataset_info"}, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    positives, corrupt = read_positive_inventory()
    if corrupt:
        raise ValueError(f"Có {len(corrupt)} ảnh giải mã lỗi; cần duyệt trước: {corrupt[:10]}")
    hashes = [record["sha256"] for record in positives]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Nguồn có ảnh trùng byte; cần xử lý trước khi chia")
    assignment = assign_groups(positives)
    for record in positives:
        record["new_split"] = assignment[record["group"]]
    backgrounds = select_backgrounds(set(hashes))
    records = positives + backgrounds
    summary = {"source_zip": str(SOURCE), "background_zip": str(BACKGROUND_SOURCE), "seed": SEED, "background_quota": BACKGROUND_QUOTA, "splits": {}}
    for split in SPLITS:
        selected = [record for record in records if record["new_split"] == split]
        categories = Counter(record["category"] for record in selected)
        summary["splits"][split] = {
            "images": len(selected),
            "categories": {category: categories[category] for category in CATEGORIES},
            "fire_present": categories["fire_only"] + categories["both"],
            "smoke_present": categories["smoke_only"] + categories["both"],
            "fire_boxes": sum(record["fire_boxes"] for record in selected),
            "smoke_boxes": sum(record["smoke_boxes"] for record in selected),
            "video_groups": len({record["group"] for record in selected if record["family"] == "video"}),
            "youtube_groups": len({record["group"] for record in selected if record["family"] == "youtube"}),
            "moved_positive_images": sum(record["old_split"] != split for record in selected if record["family"] != "dfire_background"),
        }
    group_splits = defaultdict(set)
    for record in positives:
        group_splits[record["group"]].add(record["new_split"])
    if any(len(splits) != 1 for splits in group_splits.values()):
        raise AssertionError("Nhóm nguồn bị chia qua nhiều tập")
    summary["group_count"] = len(group_splits)
    summary["positive_images"] = len(positives)
    summary["background_images_added"] = len(backgrounds)
    summary["dataset_info"] = (
        "# Fire and smoke detection v2 grouped with D-Fire backgrounds\n\n"
        "Ảnh dương tính lấy từ ZIP Roboflow v2 người dùng cung cấp. Các khung `video_N` cùng N "
        "và các biến thể `youtube-*` cùng tên gốc được giữ trong một split. Tỷ lệ mục tiêu 80/10/10. "
        "Ảnh nền lấy từ `fire_smoke_dfire_final.zip` của dự án, giữ split có sẵn và dùng nhãn YOLO rỗng. "
        "Chỉ loại trùng byte SHA-256 với ảnh nguồn. Phân nhóm theo tên không chứng minh đã loại mọi cặp cùng cảnh; "
        "ảnh nền D-Fire chưa được duyệt thủ công toàn bộ ở kích thước gốc. "
        "Dùng dữ liệu này cho thử nghiệm; cần holdout theo cảnh/video độc lập để kết luận hiệu năng.\n"
    )
    write_dataset(positives, backgrounds, summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "dataset_info"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tạo bộ dữ liệu thử nghiệm từ Roboflow v2 và ảnh nền D-Fire")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--background", type=Path, default=BACKGROUND_SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT)
    arguments = parser.parse_args()
    SOURCE = arguments.source
    BACKGROUND_SOURCE = arguments.background
    OUTPUT = arguments.output
    REPORT = arguments.report
    main()
