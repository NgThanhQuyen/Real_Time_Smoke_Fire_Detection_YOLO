"""Làm sạch D-Fire gốc và giữ nguyên cách chia tập của tác giả."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


SPLITS = ("train", "test")
IMAGE_SUFFIX = ".jpg"


def read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def validate_label(raw: bytes) -> tuple[list[int], list[str]]:
    classes: list[int] = []
    issues: list[str] = []
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError:
        return classes, ["label_not_utf8"]

    for number, line in enumerate(lines, 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            issues.append(f"line_{number}:field_count")
            continue
        try:
            values = [float(field) for field in fields]
        except ValueError:
            issues.append(f"line_{number}:non_numeric")
            continue
        class_id, x, y, width, height = values
        if not all(math.isfinite(value) for value in values):
            issues.append(f"line_{number}:non_finite")
            continue
        if class_id not in (0, 1):
            issues.append(f"line_{number}:invalid_class")
            continue
        classes.append(int(class_id))
        if width <= 0 or height <= 0:
            issues.append(f"line_{number}:nonpositive_size")
        elif (
            x - width / 2 < -1e-8
            or y - height / 2 < -1e-8
            or x + width / 2 > 1 + 1e-8
            or y + height / 2 > 1 + 1e-8
        ):
            issues.append(f"line_{number}:outside_image")
    return classes, issues


def category(classes: list[int]) -> str:
    present = set(classes)
    if present == {0, 1}:
        return "both"
    if present == {0}:
        return "smoke_only"
    if present == {1}:
        return "fire_only"
    return "background"


def count_rows(rows: list[dict[str, object]], selected_status: str | None = None) -> dict[str, dict[str, int]]:
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if selected_status is not None and row["status"] != selected_status:
            continue
        counts = result[str(row["split"])]
        counts["images"] += 1
        counts[str(row["category"])] += 1
        counts["smoke_images"] += int(row["smoke_boxes"] > 0)
        counts["fire_images"] += int(row["fire_boxes"] > 0)
        counts["smoke_boxes"] += int(row["smoke_boxes"])
        counts["fire_boxes"] += int(row["fire_boxes"])
    return {split: dict(result[split]) for split in SPLITS}


def build(source_zip: Path, source_lists: Path, output_zip: Path, report_dir: Path) -> dict[str, object]:
    report_dir.mkdir(parents=True, exist_ok=True)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    hashes: dict[str, list[dict[str, object]]] = defaultdict(list)
    image_names: dict[str, set[str]] = {}

    with zipfile.ZipFile(source_zip) as source:
        members = set(source.namelist())
        for split in SPLITS:
            names = sorted(path.rsplit("/", 1)[-1] for path in members if path.startswith(f"{split}/images/") and path.lower().endswith(IMAGE_SUFFIX))
            image_names[split] = set(names)
            label_names = {path.rsplit("/", 1)[-1] for path in members if path.startswith(f"{split}/labels/") and path.endswith(".txt")}
            if label_names != {f"{name[:-4]}.txt" for name in names}:
                raise ValueError(f"Ảnh và nhãn không ghép đủ cặp ở tập {split}.")
            expected = read_names(source_lists / f"dfire_{split}.txt")
            if len(expected) != len(set(expected)) or set(expected) != image_names[split]:
                raise ValueError(f"Danh sách tác giả không khớp tập {split} trong ZIP.")

            for index, name in enumerate(names, 1):
                image_path = f"{split}/images/{name}"
                label_path = f"{split}/labels/{name[:-4]}.txt"
                image_bytes = source.read(image_path)
                label_bytes = source.read(label_path)
                classes, issues = validate_label(label_bytes)
                try:
                    with Image.open(io.BytesIO(image_bytes)) as image:
                        image.load()
                        if image.width <= 0 or image.height <= 0:
                            raise ValueError("zero_image_dimension")
                except Exception as exc:
                    issues.append(f"image_decode:{type(exc).__name__}")
                digest = hashlib.sha256(image_bytes).hexdigest()
                row: dict[str, object] = {
                    "split": split,
                    "image": name,
                    "category": category(classes),
                    "smoke_boxes": classes.count(0),
                    "fire_boxes": classes.count(1),
                    "sha256": digest,
                    "status": "removed_invalid" if issues else "kept",
                    "reason": ";".join(issues),
                    "duplicate_of": "",
                }
                rows.append(row)
                hashes[digest].append(row)
                if index % 2000 == 0:
                    print(f"{split}: {index}/{len(names)}", flush=True)

        if image_names["train"] & image_names["test"]:
            raise ValueError("Tên ảnh xuất hiện ở cả train và test.")

        duplicate_groups = [group for group in hashes.values() if len(group) > 1]
        for group in duplicate_groups:
            splits = {str(row["split"]) for row in group if row["status"] == "kept"}
            if splits == {"train", "test"}:
                reference = next(row for row in group if row["split"] == "test" and row["status"] == "kept")
                for row in group:
                    if row["split"] == "train" and row["status"] == "kept":
                        row["status"] = "removed_cross_split_duplicate"
                        row["reason"] = "identical_image_bytes_in_test"
                        row["duplicate_of"] = f"test/{reference['image']}"

        kept = {(str(row["split"]), str(row["image"])) for row in rows if row["status"] == "kept"}
        split_lists: dict[str, list[str]] = {}
        for path in sorted(source_lists.rglob("*.txt")):
            relative = path.relative_to(source_lists).as_posix()
            names = read_names(path)
            if path.name.startswith("video_"):
                split_lists[relative] = names
                continue
            source_split = "test" if path.name == "dfire_test.txt" else "train"
            if not set(names) <= image_names[source_split]:
                raise ValueError(f"Danh sách {relative} có tên ảnh ngoài ZIP.")
            split_lists[relative] = [name for name in names if (source_split, name) in kept]

        for fold in range(1, 6):
            train_names = set(split_lists[f"5-fold cross validation/dfire_train{fold}.txt"])
            valid_names = set(split_lists[f"5-fold cross validation/dfire_valid{fold}.txt"])
            if train_names & valid_names or train_names | valid_names != set(split_lists["dfire_train.txt"]):
                raise ValueError(f"Fold {fold} không phân hoạch đúng tập train sạch.")

        temporary = output_zip.with_suffix(output_zip.suffix + ".partial")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as target:
            for row in rows:
                if row["status"] != "kept":
                    continue
                split, name = str(row["split"]), str(row["image"])
                for path in (f"{split}/images/{name}", f"{split}/labels/{name[:-4]}.txt"):
                    target.writestr(path, source.read(path))
            for relative, names in split_lists.items():
                target.writestr(f"Data splitting/{relative}", "\n".join(names) + "\n")
            target.writestr(
                "DATASET_INFO.md",
                "# D-Fire gốc sau khi làm sạch\n\n"
                "Nguồn: D-Fire.zip và Data splitting do chủ dự án cung cấp.\n"
                "Nhãn YOLO gốc: 0 = smoke, 1 = fire.\n"
                "Loại cả ảnh và nhãn khi ảnh không giải mã được hoặc nhãn có box không hợp lệ.\n"
                "Ảnh trùng byte giữa train và test được giữ ở test, loại ở train.\n"
                "Năm fold validation là các phân hoạch của train, không có thư mục val riêng.\n"
                "Không sửa nội dung ảnh/nhãn còn lại và không lọc ảnh gần trùng.\n",
            )
        temporary.replace(output_zip)

    fieldnames = ("split", "image", "category", "smoke_boxes", "fire_boxes", "sha256", "status", "reason", "duplicate_of")
    with (report_dir / "image_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary: dict[str, object] = {
        "source_zip": str(source_zip),
        "output_zip": str(output_zip),
        "class_map": {"0": "smoke", "1": "fire"},
        "before": count_rows(rows),
        "after": count_rows(rows, "kept"),
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
        "invalid_reason_counts": dict(Counter(reason.split(":", 1)[-1] for row in rows for reason in str(row["reason"]).split(";") if row["status"] == "removed_invalid" and reason)),
        "duplicate_hash_groups": len(duplicate_groups),
        "split_list_counts": {name: len(names) for name, names in split_lists.items()},
    }
    (report_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=Path(r"F:\6\KLTN\Smoke-Fire-Dataset\D-Fire.zip"))
    parser.add_argument("--source-lists", type=Path, default=Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Data splitting"))
    parser.add_argument("--output-zip", type=Path, default=Path(__file__).resolve().parents[1] / "runtime/dfire_original_clean_v1.zip")
    parser.add_argument("--report-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports/dfire_original_clean_v1")
    args = parser.parse_args()
    summary = build(args.source_zip, args.source_lists, args.output_zip, args.report_dir)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
