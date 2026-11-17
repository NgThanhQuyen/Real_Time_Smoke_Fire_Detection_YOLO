"""Tạo hai bộ dữ liệu đối chứng từ 4.000 ảnh khói đã chọn."""

from collections import Counter
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

import yaml


try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = next((path for path in (Path.cwd(), *Path.cwd().parents)
                 if (path / "codex.md").is_file()), None)
    if ROOT is None:
        raise FileNotFoundError("Không tìm thấy thư mục gốc dự án.")

RUNTIME = ROOT / "runtime"
BASE = RUNTIME / "fire_smoke_detection_yolov11_bg_v1"
FULL = RUNTIME / "fire_smoke_detection_yolov11_bg_smoke_v1"
MANIFEST = FULL / "metadata/smoke_additions.csv"
VARIANTS = {
    "fire_smoke_detection_yolov11_bg_dfire2000_v1": {"D-Fire clean": 2000},
    "fire_smoke_detection_yolov11_bg_indoor_aifm2000_v1": {
        "Indoor Fire Smoke": 1500,
        "AIForMankind V2": 500,
    },
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_manifest() -> list[dict[str, str]]:
    with MANIFEST.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = Counter({"D-Fire clean": 2000, "Indoor Fire Smoke": 1500,
                        "AIForMankind V2": 500})
    actual = Counter(row["source"] for row in rows)
    if actual != expected or len(rows) != 4000:
        raise ValueError(f"Manifest không đúng số lượng nguồn: {actual}")
    if len({row["output_image"] for row in rows}) != len(rows):
        raise ValueError("Manifest có tên ảnh đầu ra trùng nhau.")
    return rows


def check_inputs(rows: list[dict[str, str]]) -> None:
    for directory in (BASE, FULL):
        if not (directory / "data.yaml").is_file():
            raise FileNotFoundError(directory / "data.yaml")
    for name in VARIANTS:
        for path in (RUNTIME / name, RUNTIME / f"{name}.zip",
                     RUNTIME / f"{name}_building", RUNTIME / f"{name}.zip.building"):
            if path.exists():
                raise FileExistsError(f"Đầu ra đã tồn tại, không ghi đè: {path}")
    for row in rows:
        for key in ("output_image", "output_label"):
            relative = Path(row[key])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Đường dẫn không hợp lệ trong manifest: {relative}")
            if not (FULL / relative).is_file():
                raise FileNotFoundError(FULL / relative)


def count_split(root: Path, split: str) -> tuple[int, int]:
    images = sum(path.suffix.lower() in IMAGE_SUFFIXES
                 for path in (root / split / "images").iterdir() if path.is_file())
    labels = sum(path.suffix.lower() == ".txt"
                 for path in (root / split / "labels").iterdir() if path.is_file())
    return images, labels


def verify_fixed_splits(root: Path) -> None:
    for split, expected in (("valid", 971), ("test", 535)):
        if count_split(root, split) != (expected, expected):
            raise RuntimeError(f"Sai số ảnh hoặc nhãn của {split}.")
        source_files = {path.relative_to(BASE) for folder in ("images", "labels")
                        for path in (BASE / split / folder).iterdir() if path.is_file()}
        target_files = {path.relative_to(root) for folder in ("images", "labels")
                        for path in (root / split / folder).iterdir() if path.is_file()}
        if source_files != target_files:
            raise RuntimeError(f"Tên file {split} khác bản gốc.")
        for relative in source_files:
            if digest(BASE / relative) != digest(root / relative):
                raise RuntimeError(f"Nội dung {split} đã thay đổi: {relative}")


def build_variant(name: str, quotas: dict[str, int], rows: list[dict[str, str]]) -> dict:
    output = RUNTIME / name
    staging = RUNTIME / f"{name}_building"
    chosen = [row for row in rows if row["source"] in quotas]
    if Counter(row["source"] for row in chosen) != Counter(quotas):
        raise ValueError(f"Số ảnh nguồn sai trong {name}.")

    shutil.copytree(BASE, staging)
    smoke_boxes = 0
    for row in chosen:
        image_relative = Path(row["output_image"])
        label_relative = Path(row["output_label"])
        image_target = staging / image_relative
        label_target = staging / label_relative
        if image_target.exists() or label_target.exists():
            raise FileExistsError(f"Tên file trùng dữ liệu gốc: {image_relative}")
        image_source = FULL / image_relative
        if digest(image_source) != row["source_sha256"]:
            raise RuntimeError(f"SHA ảnh khác manifest: {image_relative}")
        label_lines = (FULL / label_relative).read_text(encoding="utf-8").splitlines()
        if not label_lines or any(not line.split() or line.split()[0] != "1"
                                  or len(line.split()) != 5
                                  for line in label_lines):
            raise RuntimeError(f"Nhãn ảnh thêm không phải box Smoke: {label_relative}")
        if len(label_lines) != int(row["smoke_boxes"]):
            raise RuntimeError(f"Số box khác manifest: {label_relative}")
        shutil.copy2(image_source, image_target)
        shutil.copy2(FULL / label_relative, label_target)
        smoke_boxes += len(label_lines)

    config = yaml.safe_load((BASE / "data.yaml").read_text(encoding="utf-8"))
    if config["names"] != ["Fire", "Smoke"] or config["nc"] != 2:
        raise ValueError("Class map khác kỳ vọng.")
    config["path"] = str(output)
    (staging / "data.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    metadata = staging / "metadata"
    with (metadata / "smoke_additions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(chosen)
    summary = {
        "dataset": name,
        "base_dataset": BASE.name,
        "selection_manifest": str(MANIFEST.relative_to(ROOT)),
        "base_images": {"train": 10653, "valid": 971, "test": 535},
        "added_smoke_only_train": 2000,
        "sources": quotas,
        "new_smoke_boxes": smoke_boxes,
        "final_images": {"train": 12653, "valid": 971, "test": 535, "total": 14159},
        "train_categories": {"fire_only": 5965, "smoke_only": 3674,
                             "both": 1514, "background": 1500},
        "train_images_containing_fire": 7479,
        "train_images_containing_smoke": 5188,
        "rules": ["Dùng lại đúng ảnh và nhãn từ bản 4.000 ảnh đã chọn",
                  "Không sửa bộ gốc hoặc bản 4.000 ảnh", "Valid/test giữ nguyên byte"],
    }
    (metadata / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for split, expected in (("train", 12653), ("valid", 971), ("test", 535)):
        if count_split(staging, split) != (expected, expected):
            raise RuntimeError(f"Sai số ảnh hoặc nhãn của {split} trong {name}.")
    verify_fixed_splits(staging)
    staging.rename(output)
    return summary


def package_variant(name: str) -> int:
    source = RUNTIME / name
    target = RUNTIME / f"{name}.zip"
    temporary = RUNTIME / f"{name}.zip.building"
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4) as archive:
        for path in source.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=Path(name) / path.relative_to(source))
    with zipfile.ZipFile(temporary) as archive:
        bad_entry = archive.testzip()
        if bad_entry:
            raise RuntimeError(f"ZIP lỗi CRC: {bad_entry}")
        for split, expected in (("train", 12653), ("valid", 971), ("test", 535)):
            prefix = f"{name}/{split}/images/"
            count = sum(item.startswith(prefix) and Path(item).suffix.lower() in IMAGE_SUFFIXES
                        for item in archive.namelist())
            if count != expected:
                raise RuntimeError(f"ZIP sai số ảnh {split}: {count}")
    temporary.rename(target)
    return target.stat().st_size


def main() -> None:
    rows = read_manifest()
    check_inputs(rows)
    if count_split(BASE, "train") != (10653, 10653):
        raise RuntimeError("Bộ gốc không có 10.653 cặp train.")
    for name, quotas in VARIANTS.items():
        print(f"Đang tạo {name}: {quotas}", flush=True)
        summary = build_variant(name, quotas, rows)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        size = package_variant(name)
        print(f"Đã kiểm tra ZIP: {RUNTIME / (name + '.zip')} ({size} bytes)", flush=True)


if __name__ == "__main__":
    main()
