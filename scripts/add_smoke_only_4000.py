"""Tạo bản dữ liệu mới với 4.000 ảnh chỉ khói trong train."""

from collections import Counter
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import shutil
import zipfile

from PIL import Image
import yaml


try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = next((folder for folder in (Path.cwd(), *Path.cwd().parents)
                 if (folder / "codex.md").is_file()), None)
    if ROOT is None:
        raise FileNotFoundError("Không tìm thấy thư mục gốc dự án.")
SOURCES = Path(r"F:\6\KLTN\Smoke-Fire-Dataset")
CURRENT = ROOT / "runtime/fire_smoke_detection_yolov11_bg_v1"
DFIRE = ROOT / "runtime/dfire_clean_v2"
OUTPUT = ROOT / "runtime/fire_smoke_detection_yolov11_bg_smoke_v1"
OUTPUT_ZIP = ROOT / "runtime/fire_smoke_detection_yolov11_bg_smoke_v1.zip"
AUDIT = ROOT / "reports/smoke_candidate_audit"
QUOTAS = {"D-Fire clean": 2000, "Indoor Fire Smoke": 1500, "AIForMankind V2": 500}
SEED = 20261103


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def source_content(row: dict, archives: dict, dfire_map: dict) -> tuple[bytes, bytes]:
    if row["source"] == "D-Fire clean":
        entry = dfire_map[row["path"]]
        return (DFIRE / entry["output_image"]).read_bytes(), (DFIRE / entry["output_label"]).read_bytes()
    archive = archives[row["source"]]
    label_path = row["path"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    return archive.read(row["path"]), archive.read(label_path)


def smoke_label(raw: bytes, source: str) -> tuple[str, int]:
    expected_class = 1 if source == "Indoor Fire Smoke" else 0
    lines = []
    for line in raw.decode("utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError("Nhãn không có đúng 5 giá trị")
        values = [float(part) for part in parts]
        if values[0] != expected_class:
            raise ValueError("Nhãn không phải chỉ khói theo class map nguồn")
        x, y, width, height = values[1:]
        if not all(math.isfinite(value) for value in values) or width <= 0 or height <= 0:
            raise ValueError("Box không hữu hạn hoặc không dương")
        if min(x - width / 2, y - height / 2) < -1e-6 or max(x + width / 2, y + height / 2) > 1 + 1e-6:
            raise ValueError("Box vượt ảnh")
        lines.append("1 " + " ".join(parts[1:]))
    if not lines:
        raise ValueError("Nhãn khói rỗng")
    return "\n".join(lines) + "\n", len(lines)


def validate_inputs() -> None:
    paths = [CURRENT, DFIRE, AUDIT / "candidate_index.csv", AUDIT / "indoor_fire_probe.csv",
             AUDIT / "perceptual_pairs_locked.csv", SOURCES / "Indoor_Fire_Smoke.zip",
             SOURCES / "AIForMankind_V2_processed.zip"]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if OUTPUT.exists() or OUTPUT_ZIP.exists():
        raise FileExistsError("Đầu ra đã tồn tại; không ghi đè.")


def select_images() -> tuple[list[dict], Counter, dict]:
    index = read_csv(AUDIT / "candidate_index.csv")
    probe = {row["image"]: float(row["max_fire_conf"])
             for row in read_csv(AUDIT / "indoor_fire_probe.csv")}
    dfire_map = {row["source_image"]: row for row in read_csv(DFIRE / "metadata/manifest.csv")}
    main_manifest = [row for row in read_csv(ROOT / "runtime/fire_smoke_detection_yolov11_clean_v1/metadata/image_manifest.csv")
                     if row["status"] == "kept"]
    background_manifest = read_csv(CURRENT / "metadata/background_additions.csv")
    used_hashes = {row["sha256"] for row in main_manifest} | {row["sha256"] for row in background_manifest}
    leakage_candidates = {row["candidate"] for row in read_csv(AUDIT / "perceptual_pairs_locked.csv")}
    selected = []
    rejected = Counter()

    with zipfile.ZipFile(SOURCES / "Indoor_Fire_Smoke.zip") as indoor_zip, zipfile.ZipFile(
        SOURCES / "AIForMankind_V2_processed.zip"
    ) as aifm_zip:
        archives = {"Indoor Fire Smoke": indoor_zip, "AIForMankind V2": aifm_zip}
        for source_index, (source, target) in enumerate(QUOTAS.items()):
            pool = [row for row in index
                    if row["source"] == source and row["cat"] == "smoke_only"
                    and row["reason"] == "" and row["label_conflict_in_identical_images"] == "false"
                    and (source == "AIForMankind V2" or row["split"] == "train")]
            pool.sort(key=lambda row: row["path"])
            random.Random(SEED + source_index).shuffle(pool)
            count = 0
            for row in pool:
                if count == target:
                    break
                if row["sha"] in used_hashes:
                    rejected[(source, "exact_duplicate")] += 1
                    continue
                if row["path"] in leakage_candidates:
                    rejected[(source, "perceptual_review_against_val_test")] += 1
                    continue
                if source == "Indoor Fire Smoke" and probe[row["path"]] >= 0.1:
                    rejected[(source, "possible_unlabeled_fire")] += 1
                    continue
                try:
                    image_bytes, label_bytes = source_content(row, archives, dfire_map)
                    if hashlib.sha256(image_bytes).hexdigest() != row["sha"]:
                        raise ValueError("SHA ảnh khác chỉ mục")
                    with Image.open(BytesIO(image_bytes)) as image:
                        image.verify()
                    with Image.open(BytesIO(image_bytes)) as image:
                        image.load()
                    label_text, box_count = smoke_label(label_bytes, source)
                except Exception as error:
                    rejected[(source, f"{type(error).__name__}: {error}")] += 1
                    continue
                used_hashes.add(row["sha"])
                selected.append({"source": source, "source_image": row["path"],
                                 "source_sha256": row["sha"], "source_split": row["split"],
                                 "smoke_boxes": box_count, "label_text": label_text})
                count += 1
            if count != target:
                raise RuntimeError(f"Không đủ ảnh đạt chuẩn từ {source}: {count}/{target}")
    if len(selected) != 4000:
        raise RuntimeError(f"Số ảnh chọn không đúng: {len(selected)}")
    return selected, rejected, dfire_map


def build_dataset(selected: list[dict], dfire_map: dict) -> dict:
    staging = OUTPUT.with_name(OUTPUT.name + "_building")
    if staging.exists():
        raise FileExistsError(f"Thư mục tạm đã tồn tại: {staging}")
    shutil.copytree(CURRENT, staging)
    with zipfile.ZipFile(SOURCES / "Indoor_Fire_Smoke.zip") as indoor_zip, zipfile.ZipFile(
        SOURCES / "AIForMankind_V2_processed.zip"
    ) as aifm_zip:
        archives = {"Indoor Fire Smoke": indoor_zip, "AIForMankind V2": aifm_zip}
        for number, row in enumerate(selected, 1):
            source_row = {"source": row["source"], "path": row["source_image"]}
            content, _ = source_content(source_row, archives, dfire_map)
            if hashlib.sha256(content).hexdigest() != row["source_sha256"]:
                raise RuntimeError(f"Ảnh nguồn thay đổi: {row['source_image']}")
            suffix = Path(row["source_image"]).suffix.lower()
            image_rel = f"train/images/smoke_add_{number:05d}{suffix}"
            label_rel = f"train/labels/smoke_add_{number:05d}.txt"
            (staging / image_rel).write_bytes(content)
            (staging / label_rel).write_text(row.pop("label_text"), encoding="utf-8")
            row["output_image"] = image_rel
            row["output_label"] = label_rel

    config = yaml.safe_load((CURRENT / "data.yaml").read_text(encoding="utf-8"))
    config["path"] = str(OUTPUT)
    (staging / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    metadata = staging / "metadata"
    with (metadata / "smoke_additions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "base_images": {"train": 10653, "valid": 971, "test": 535},
        "added_smoke_only_train": 4000,
        "final_images": {"train": 14653, "valid": 971, "test": 535, "total": 16159},
        "train_categories": {"fire_only": 5965, "smoke_only": 5674, "both": 1514, "background": 1500},
        "train_images_containing_fire": 7479,
        "train_images_containing_smoke": 7188,
        "sources": dict(Counter(row["source"] for row in selected)),
        "new_smoke_boxes": sum(row["smoke_boxes"] for row in selected),
        "selection_seed": SEED,
        "rules": ["Chỉ loại trùng byte SHA-256", "Không sửa bản dữ liệu cũ", "Val/test giữ nguyên byte",
                  "Ảnh mới chỉ có lớp Smoke=1", "Ảnh gần giống trong train được giữ"],
    }
    (metadata / "smoke_addition_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for split, expected in (("train", 14653), ("valid", 971), ("test", 535)):
        images = sum(path.is_file() for path in (staging / split / "images").iterdir())
        labels = sum(path.is_file() for path in (staging / split / "labels").iterdir())
        if images != expected or labels != expected:
            raise RuntimeError(f"Sai số file {split}: ảnh={images}, nhãn={labels}, dự kiến={expected}")
    staging.rename(OUTPUT)
    return summary


def package_dataset() -> None:
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in OUTPUT.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=Path(OUTPUT.name) / path.relative_to(OUTPUT))


def main() -> None:
    validate_inputs()
    selected, rejected, dfire_map = select_images()
    print("Đã chọn:", dict(Counter(row["source"] for row in selected)), flush=True)
    print("Ứng viên bỏ qua:", {str(key): value for key, value in rejected.items()}, flush=True)
    summary = build_dataset(selected, dfire_map)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    package_dataset()
    print("Thư mục:", OUTPUT, flush=True)
    print("ZIP:", OUTPUT_ZIP, "bytes:", OUTPUT_ZIP.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
