"""Bổ sung 500 ảnh nền khó vào train của bản thử nghiệm data 2."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"F:\6\KLTN\Smoke-Fire-Dataset")
BASE_ZIP = ROOT / "runtime" / "Smoke-Fire-Detection-Dataset.zip"
OUTPUT_ZIP = ROOT / "runtime" / "Smoke-Fire-Detection-Dataset-hardneg500.zip"
REPORT_DIR = ROOT / "reports" / "fire_smoke_v2_hardneg500"
REVIEW_DIR = REPORT_DIR / "review"
BASE_MANIFEST = ROOT / "reports" / "fire_smoke_v2_grouped_bg_v1" / "manifest.csv"

# Các ảnh bỏ qua được quyết định sau khi xem bảng ảnh của cả ba nguồn.
RAIN_EXCLUDE = {
    12, 13, 14, 82, 90, 105, 108, 119, 126, 134, 143, 144, 182,
    191, 194, 195, 196, 197, 198,
}
FOG_NEAR_EXCLUDE = {
    35, 38, 66, 88, 89, 105, 146, 149, 151, 162, 171, 185, 187,
    188, 212, 218, 234, 240, 242, 258, 266, 268, 279,
}
FOG_OTHER_EXCLUDE = {
    3, 8, 17, 18, 28, 34, 36, 70, 71, 72, 90, 92, 93, 148,
    203, 204, 205, 207, 209, 210, 225, 228, 229, 230, 232, 238,
    243, 244, 248, 251, 252, 253, 260, 261, 263, 271, 274, 275,
    280, 281,
}
WARM_EXCLUDE = {
    13, 14, 17, 20, 25, 36, 41, 45, 46, 48, 51, 55, 57, 61, 62, 64,
    67, 68, 72, 77, 82, 84, 86, 87, 88, 92, 94, 95, 100, 101,
    102, 107, 108, 111, 112, 113, 116, 117, 120, 123, 124, 127,
    129, 132, 133, 135, 139, 140, 141, 142, 144, 147, 148, 151,
    153, 154, 156, 157, 158, 159,
}
QUOTAS = {"rain": 180, "fog": 220, "warm_light": 100}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_candidates() -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    plan = [
        ("rain", SOURCE_ROOT / "Rain.zip", REVIEW_DIR / "rain_candidates.csv", RAIN_EXCLUDE, "index"),
        ("fog", SOURCE_ROOT / "Fog.zip", REVIEW_DIR / "fog_candidates.csv", FOG_NEAR_EXCLUDE | FOG_OTHER_EXCLUDE, "index"),
        ("warm_light", SOURCE_ROOT / "D-Fire-YOLO.zip", REVIEW_DIR / "dfire_warm_shortlist.csv", WARM_EXCLUDE, "rank"),
    ]
    review_rows: list[dict[str, str | int]] = []
    for category, archive_path, csv_path, excluded, index_column in plan:
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        archive = zipfile.ZipFile(archive_path)
        try:
            for row in rows:
                index = int(row[index_column])
                included = index not in excluded
                review_rows.append({
                    "category": category,
                    "index": index,
                    "source_member": row["source_member"],
                    "decision": "include" if included else "exclude",
                })
                if not included:
                    continue
                data = archive.read(row["source_member"])
                digest = sha256(data)
                if category != "warm_light" and digest != row["sha256"]:
                    raise ValueError(f"Hash nguồn đã thay đổi: {row['source_member']}")
                if category == "warm_light":
                    label_name = row["source_member"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                    if archive.read(label_name).strip():
                        raise ValueError(f"Ảnh D-Fire có nhãn không rỗng: {label_name}")
                selected.append({
                    "category": category,
                    "source_zip": str(archive_path),
                    "source_member": row["source_member"],
                    "sha256": digest,
                    "source_index": str(index),
                })
        finally:
            archive.close()
    counts = Counter(row["category"] for row in selected)
    if counts != QUOTAS:
        raise ValueError(f"Sai quota: {counts} != {QUOTAS}")
    with (REPORT_DIR / "review_decisions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["category", "index", "source_member", "decision"])
        writer.writeheader()
        writer.writerows(review_rows)
    return selected


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates()
    with BASE_MANIFEST.open(newline="", encoding="utf-8") as file:
        main_hashes = {row["sha256"] for row in csv.DictReader(file)}
    seen = set(main_hashes)
    for row in candidates:
        if row["sha256"] in seen:
            raise ValueError(f"Ứng viên trùng byte với bộ gốc hoặc ứng viên khác: {row['source_member']}")
        seen.add(row["sha256"])

    if OUTPUT_ZIP.exists():
        raise FileExistsError(f"Đầu ra đã tồn tại: {OUTPUT_ZIP}")
    temp_path = OUTPUT_ZIP.with_suffix(".tmp")
    if temp_path.exists():
        raise FileExistsError(f"File tạm đã tồn tại: {temp_path}")

    manifest: list[dict[str, str]] = []
    source_archives: dict[str, zipfile.ZipFile] = {}
    try:
        with zipfile.ZipFile(BASE_ZIP) as base, zipfile.ZipFile(temp_path, "w", allowZip64=True) as output:
            base_names = set(base.namelist())
            for info in base.infolist():
                output.writestr(info, base.read(info.filename))
            for number, row in enumerate(candidates, 1):
                source_path = row["source_zip"]
                if source_path not in source_archives:
                    source_archives[source_path] = zipfile.ZipFile(source_path)
                data = source_archives[source_path].read(row["source_member"])
                if sha256(data) != row["sha256"]:
                    raise ValueError(f"Hash nguồn đã đổi: {row['source_member']}")
                extension = Path(row["source_member"]).suffix.lower()
                stem = f"bg_hardneg_{row['category']}_{number:04d}"
                image_name = f"train/images/{stem}{extension}"
                label_name = f"train/labels/{stem}.txt"
                if image_name in base_names or label_name in base_names:
                    raise ValueError(f"Tên file bị trùng: {stem}")
                output.writestr(image_name, data, compress_type=zipfile.ZIP_STORED)
                output.writestr(label_name, b"", compress_type=zipfile.ZIP_STORED)
                manifest.append({**row, "output_image": image_name, "output_label": label_name})
            info_text = (
                "Bổ sung 500 ảnh nền khó vào train từ Rain, Fog (VdiAw) và D-Fire WEB.\n"
                "Mọi file thuộc bộ Smoke-Fire-Detection-Dataset.zip được giữ nguyên nội dung.\n"
                "Valid/test giữ nguyên. Ảnh mới có nhãn YOLO rỗng.\n"
                "Chi tiết nguồn và SHA-256: reports/fire_smoke_v2_hardneg500/additions.csv.\n"
                "VdiAw: https://data.mendeley.com/datasets/766ygrbt8y/1 (CC BY-NC 3.0).\n"
                "D-Fire: https://github.com/gaia-solutions-on-demand/DFireDataset.\n"
            )
            output.writestr("HARD_NEGATIVE_INFO.md", info_text.encode("utf-8"), compress_type=zipfile.ZIP_STORED)

        with zipfile.ZipFile(BASE_ZIP) as base, zipfile.ZipFile(temp_path) as result:
            bad = result.testzip()
            if bad:
                raise ValueError(f"CRC lỗi: {bad}")
            for info in base.infolist():
                changed = result.getinfo(info.filename)
                if (changed.file_size, changed.CRC) != (info.file_size, info.CRC):
                    raise ValueError(f"File gốc thay đổi: {info.filename}")
            for row in manifest:
                if sha256(result.read(row["output_image"])) != row["sha256"]:
                    raise ValueError(f"Ảnh thêm sai hash: {row['output_image']}")
                if result.read(row["output_label"]) != b"":
                    raise ValueError(f"Nhãn nền không rỗng: {row['output_label']}")
            image_counts = Counter(
                name.split("/", 1)[0] for name in result.namelist()
                if "/images/" in name
            )
            if image_counts != {"train": 14269, "valid": 1684, "test": 1684}:
                raise ValueError(f"Số ảnh đầu ra sai: {image_counts}")

        with (REPORT_DIR / "additions.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
        summary = {
            "base_zip": str(BASE_ZIP),
            "output_zip": str(OUTPUT_ZIP),
            "added_by_category": dict(Counter(row["category"] for row in manifest)),
            "image_counts": dict(image_counts),
            "original_member_count": len(base_names),
            "original_members_content_unchanged": True,
            "new_labels_empty": True,
            "exact_duplicate_candidates": 0,
            "crc_ok": True,
            "review_note": "Ảnh ứng viên đã được xem qua bảng ảnh; chưa duyệt 500 ảnh ở kích thước gốc.",
        }
        (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.move(temp_path, OUTPUT_ZIP)
        print(json.dumps(summary, ensure_ascii=True, indent=2))
    finally:
        for archive in source_archives.values():
            archive.close()


if __name__ == "__main__":
    main()
