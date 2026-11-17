"""Tạo bản D-Fire chỉ thêm ảnh lửa từ DBA-Fire và Indoor vào train."""

import csv
import hashlib
import json
import os
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "runtime/dfire_original_clean_split_v1.zip"
OUTPUT = ROOT / "runtime/dfire_original_clean_split_fire2591_v1.zip"
REPORT_DIR = ROOT / "reports/dfire_fire2591_v1"
SCREEN = ROOT / "reports/dba_indoor_relaxed_review/manifest.csv"
SIZES = ROOT / "reports/dba_indoor_relaxed_review/fire_only_imgsz640.csv"
SOURCES = {
    "DBA-Fire": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\DBA-YOLO-Dataset.zip"),
    "Indoor": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Indoor_Fire_Smoke.zip"),
}


def load_selection():
    with SCREEN.open(encoding="utf-8", newline="") as handle:
        manifest = {(row["source"], row["image"]): row for row in csv.DictReader(handle)}
    with SIZES.open(encoding="utf-8", newline="") as handle:
        size_rows = list(csv.DictReader(handle))
    chosen = []
    for row in size_rows:
        key = (row["source"], row["image"])
        review = manifest[key]
        if row["long_edge_ge_640"] != "True":
            continue
        if review["status"] != "eligible" or review["category"] != "fire_only":
            raise ValueError(f"Ảnh không còn đủ điều kiện: {key}")
        if int(row["long_edge"]) < 640 or row["width"] != review["width"] or row["height"] != review["height"]:
            raise ValueError(f"Kích thước không khớp: {key}")
        chosen.append((row, review))
    chosen.sort(key=lambda pair: (pair[0]["source"], pair[0]["image"]))
    if len(chosen) != 2591 or Counter(row["source"] for row, _ in chosen) != {"DBA-Fire": 786, "Indoor": 1805}:
        raise ValueError("Số lượng ứng viên khác kết quả đã duyệt")
    return chosen


def convert_label(data):
    """Đảo lớp lửa 0 thành 1 và ghim box nhô mép dưới dung sai làm tròn."""
    lines = []
    clipped_boxes = 0
    for line in data.decode("utf-8-sig").splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[0] != "0":
            raise ValueError("Ứng viên không còn là ảnh chỉ lửa đúng định dạng YOLO")
        x, y, width, height = map(float, fields[1:])
        if width <= 0 or height <= 0:
            raise ValueError("Box lửa có kích thước không dương")
        left, top = x - width / 2, y - height / 2
        right, bottom = x + width / 2, y + height / 2
        if min(left, top) < -0.001 or max(right, bottom) > 1.001:
            raise ValueError("Box lửa vượt mép quá dung sai")
        new_left, new_top = max(0.0, left), max(0.0, top)
        new_right, new_bottom = min(1.0, right), min(1.0, bottom)
        if min(left, top) < -1e-9 or max(right, bottom) > 1 + 1e-9:
            clipped_boxes += 1
            lines.append(f"1 {(new_left + new_right) / 2:.12f} {(new_top + new_bottom) / 2:.12f} {new_right - new_left:.12f} {new_bottom - new_top:.12f}")
        else:
            lines.append("1 " + " ".join(fields[1:]))
    if not lines:
        raise ValueError("Ứng viên chỉ lửa không có box")
    return ("\n".join(lines) + "\n").encode("utf-8"), len(lines), clipped_boxes


def write_member(output, name, data):
    output.writestr(name, data, compress_type=zipfile.ZIP_STORED)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Đầu ra đã tồn tại: {OUTPUT}")
    chosen = load_selection()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".zip.partial")
    if temporary.exists():
        raise FileExistsError(f"File tạm đã tồn tại: {temporary}")
    additions = []
    locked_input_hash = {}
    try:
        with zipfile.ZipFile(INPUT) as old_zip, zipfile.ZipFile(SOURCES["DBA-Fire"]) as dba_zip, zipfile.ZipFile(SOURCES["Indoor"]) as indoor_zip, zipfile.ZipFile(temporary, "w", allowZip64=True) as new_zip:
            source_zips = {"DBA-Fire": dba_zip, "Indoor": indoor_zip}
            old_names = set(old_zip.namelist())
            for count, entry in enumerate(old_zip.infolist(), 1):
                name = entry.filename
                if name in ("splits/train.txt", "DATASET_INFO.md"):
                    continue
                data = old_zip.read(name)
                write_member(new_zip, name, data)
                if name.startswith(("val/", "test/")):
                    locked_input_hash[name] = hashlib.sha256(data).hexdigest()
                if count % 10000 == 0:
                    print("Đã sao chép", count, "thành viên D-Fire", flush=True)
            original_train = old_zip.read("splits/train.txt").decode("utf-8").splitlines()
            if len(original_train) != 12699:
                raise ValueError("Train D-Fire đầu vào không có đúng 12.699 ảnh")
            names_added = []
            source_counters = Counter()
            for index, (size_row, screen_row) in enumerate(chosen, 1):
                source = size_row["source"]
                archive = source_zips[source]
                data = archive.read(size_row["image"])
                digest = hashlib.sha256(data).hexdigest()
                if digest != screen_row["sha256"]:
                    raise ValueError(f"Ảnh nguồn đã đổi: {size_row['image']}")
                label_data, fire_boxes, clipped = convert_label(archive.read(screen_row["label"]))
                source_counters[source] += 1
                prefix = "dba" if source == "DBA-Fire" else "indoor"
                stem = f"add_{prefix}_{source_counters[source]:05d}"
                image_suffix = Path(size_row["image"]).suffix.lower()
                image_name = f"train/images/{stem}{image_suffix}"
                label_name = f"train/labels/{stem}.txt"
                if image_name in old_names or label_name in old_names:
                    raise ValueError(f"Tên file bị trùng: {stem}")
                write_member(new_zip, image_name, data)
                write_member(new_zip, label_name, label_data)
                names_added.append(f"{stem}{image_suffix}")
                additions.append({
                    "output_image": image_name,
                    "output_label": label_name,
                    "source": source,
                    "source_image": size_row["image"],
                    "source_label": screen_row["label"],
                    "source_sha256": digest,
                    "width": size_row["width"],
                    "height": size_row["height"],
                    "fire_boxes": fire_boxes,
                    "clipped_boxes": clipped,
                })
                if index % 500 == 0:
                    print("Đã thêm", index, "ảnh chỉ lửa", flush=True)
            write_member(new_zip, "splits/train.txt", ("\n".join(original_train + names_added) + "\n").encode("utf-8"))
            info = (
                "# D-Fire sạch: train bổ sung 2.591 ảnh chỉ lửa\n\n"
                "Nền: dfire_original_clean_split_v1.zip. Val/test giữ nguyên nội dung.\n"
                "Train thêm 786 ảnh DBA-Fire và 1.805 ảnh Indoor có cạnh dài gốc >=640 px.\n"
                "Chỉ nhận ảnh đã qua sàng lọc lỗi và giống D-Fire val/test <=90% theo báo cáo.\n"
                "Nhãn hai nguồn được đảo từ 0=fire sang 1=fire của D-Fire.\n"
                "Một số box nhô mép dưới một pixel được ghim đúng mép khi xuất.\n"
                "Không kiểm duyệt ngữ nghĩa toàn bộ ảnh/nhãn; tham khảo manifest và báo cáo.\n"
            )
            write_member(new_zip, "DATASET_INFO.md", info.encode("utf-8"))
        with zipfile.ZipFile(temporary) as result:
            bad = result.testzip()
            if bad is not None:
                raise ValueError(f"ZIP đầu ra lỗi CRC: {bad}")
            names = result.namelist()
            if len(names) != len(set(names)) or len(names) != len(old_names) + 2 * len(chosen):
                raise ValueError("Số file ZIP đầu ra không đúng hoặc trùng tên")
            if result.read("data.yaml") != zipfile.ZipFile(INPUT).read("data.yaml"):
                raise ValueError("data.yaml đã đổi")
            if len(result.read("splits/train.txt").decode("utf-8").splitlines()) != 15290:
                raise ValueError("Danh sách train sau gộp không đúng")
            for split in ("val", "test"):
                if len([name for name in names if name.startswith(f"{split}/images/")]) != 4244:
                    raise ValueError(f"Số ảnh {split} thay đổi")
            for index, (name, digest) in enumerate(locked_input_hash.items(), 1):
                if hashlib.sha256(result.read(name)).hexdigest() != digest:
                    raise ValueError(f"Val/test khác byte: {name}")
                if index % 4000 == 0:
                    print("Đã xác minh", index, "file val/test", flush=True)
            for addition in additions:
                if hashlib.sha256(result.read(addition["output_image"])).hexdigest() != addition["source_sha256"]:
                    raise ValueError(f"Ảnh mới khác nguồn: {addition['output_image']}")
                rows = result.read(addition["output_label"]).decode("utf-8").splitlines()
                if len(rows) != addition["fire_boxes"] or any(row.split()[0] != "1" for row in rows):
                    raise ValueError(f"Nhãn mới sai lớp: {addition['output_label']}")
        with (REPORT_DIR / "addition_manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=additions[0].keys())
            writer.writeheader()
            writer.writerows(additions)
        summary = {
            "output_zip": str(OUTPUT),
            "source_images": dict(source_counters),
            "added_images": len(additions),
            "added_fire_boxes": sum(row["fire_boxes"] for row in additions),
            "clipped_boxes": sum(row["clipped_boxes"] for row in additions),
            "clipped_images": sum(row["clipped_boxes"] > 0 for row in additions),
            "splits": {"train": 15290, "val": 4244, "test": 4244},
            "train_categories": {"fire_only": 3277, "smoke_only": 3464, "both": 2664, "background": 5885},
            "val_test_members_verified_sha256": len(locked_input_hash),
            "archive_crc_verified": True,
        }
        (REPORT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, OUTPUT)
        print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


if __name__ == "__main__":
    main()
