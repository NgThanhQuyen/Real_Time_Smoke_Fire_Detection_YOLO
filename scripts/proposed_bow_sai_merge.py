"""Tuyển ảnh BoWFire và SAINetset đã xem, rồi thêm riêng vào train hiện tại."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

from catargiu_audit import hashes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
BASE = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1"
OUTPUT = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_bow_sai_v1"
BOW_FIRE = (7, 15, 32, 37, 52, 62)
BOW_BOX_OVERRIDES = {15: (100, 180, 1024, 503), 37: (0, 225, 1024, 706)}
BOW_NEGATIVE = (7, 12, 16, 30, 34, 35, 36, 42, 43, 57, 67, 78, 87, 90)
SAI_NEGATIVE = (
    ("val", "407608.jpg", "cam2_mist"),
    ("train", "397418.jpg", "cam3_night_lights"),
    ("train", "400326.jpg", "cam1_haze"),
    ("train", "305517.jpg", "cam4_night_lights"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def fingerprint(data: bytes) -> tuple[str, str, str, int, int]:
    with Image.open(BytesIO(data)) as image:
        image.load()
        width, height = image.size
        dhash, phash = hashes(image)
    return hashlib.sha256(data).hexdigest(), dhash, phash, width, height


def candidate_rows() -> list[dict]:
    results = []
    with zipfile.ZipFile(SOURCE / "BoWFireDataset.zip") as archive:
        for category, numbers in (("fire", BOW_FIRE), ("negative", BOW_NEGATIVE)):
            for number in numbers:
                stem = f"fire{number:03d}" if category == "fire" else f"not_fire{number:03d}"
                image_path = f"BoWFireDataset/dataset/img/{stem}.png"
                data = archive.read(image_path)
                sha, dhash, phash, width, height = fingerprint(data)
                label = ""
                if category == "fire":
                    mask_path = f"BoWFireDataset/dataset/gt/{stem}_gt.png"
                    with Image.open(BytesIO(archive.read(mask_path))) as mask:
                        mask = mask.convert("L").point(lambda value: 255 if value > 127 else 0)
                        assert mask.size == (width, height)
                        box = mask.getbbox()
                    if box is None:
                        raise ValueError(f"Mask lửa rỗng: {stem}")
                    box = BOW_BOX_OVERRIDES.get(number, box)
                    x1, y1, x2, y2 = box
                    x = (x1 + x2) / (2 * width)
                    y = (y1 + y2) / (2 * height)
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    label = f"1 {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n"
                results.append({"source": "BoWFire", "source_split": "unsplit", "source_image": image_path,
                                "scene_key": stem, "category": category, "image_name": f"BoWFire_{stem}.png",
                                "data": data, "label": label, "sha256": sha, "dhash": dhash, "phash": phash,
                                "width": width, "height": height})

    conflicts = {row["sha256"] for row in read_csv(SOURCE / "sainet_exact_label_conflicts.csv")}
    image_index = read_csv(SOURCE / "sainet_image_index.csv")
    for split, filename, scene_key in SAI_NEGATIVE:
        source_image = f"data/{split}/images/{filename}"
        original = SOURCE / "sainet_sample" / source_image
        label_path = SOURCE / "sainet_sample" / f"data/{split}/labels/{Path(filename).stem}.txt"
        if label_path.read_text(encoding="utf-8-sig").strip():
            raise ValueError(f"SAINetset không có nhãn rỗng: {source_image}")
        data = original.read_bytes()
        sha, dhash, phash, width, height = fingerprint(data)
        if sha in conflicts:
            raise ValueError(f"SAINetset có nhãn mâu thuẫn: {source_image}")
        same_bytes = [row for row in image_index if row["sha256"] == sha]
        if not same_bytes:
            raise ValueError(f"Không tìm thấy ảnh trong chỉ mục SAINetset: {source_image}")
        for row in same_bytes:
            label_split = "train" if row["split"] == "train" else "val"
            peer_label = SOURCE / "sainet_labels_repo2" / "data" / label_split / "labels" / f"{Path(row['image_path']).stem}.txt"
            if peer_label.read_text(encoding="utf-8-sig").strip():
                raise ValueError(f"SAINetset có ảnh trùng byte gắn khói: {source_image}")
        results.append({"source": "SAINetset_v8", "source_split": split, "source_image": source_image,
                        "scene_key": scene_key, "category": "negative", "image_name": f"SAINet_{filename}",
                        "data": data, "label": "", "sha256": sha, "dhash": dhash, "phash": phash,
                        "width": width, "height": height})
    return results


def existing_rows() -> list[dict]:
    manifest = read_csv(BASE / "metadata/dfire_manifest.csv")
    active = {row["source_sha256"]: row for row in manifest if row["output_split"] in ("train", "val", "test")}
    result = []
    for row in read_csv(ROOT / "runtime/dfire_raw_review/image_index.csv"):
        if row["sha256"] in active:
            output = active[row["sha256"]]
            result.append({"image": output["output_image"], "split": output["output_split"],
                           "sha256": row["sha256"], "dhash": row["dhash"], "phash": row["phash"]})
    for path in sorted((BASE / "train/images").glob("*.jpg")):
        if not path.name.startswith(("PyroSDIS_", "Catargiu_")):
            continue
        sha, dhash, phash, _, _ = fingerprint(path.read_bytes())
        result.append({"image": f"train/images/{path.name}", "split": "train",
                       "sha256": sha, "dhash": dhash, "phash": phash})
    return result


def overlap_audit(candidates: list[dict], baseline: list[dict]) -> list[dict]:
    checks = []
    for item in candidates:
        exact = [row for row in baseline if row["sha256"] == item["sha256"]]
        near = []
        d = int(item["dhash"], 16)
        p = int(item["phash"], 16)
        for row in baseline:
            dd = (d ^ int(row["dhash"], 16)).bit_count()
            if dd <= 6:
                pp = (p ^ int(row["phash"], 16)).bit_count()
                if pp <= 12:
                    near.append((dd, pp, row["split"], row["image"]))
        near.sort()
        peers = [row for row in candidates if row is not item]
        internal = [row["image_name"] for row in peers if row["sha256"] == item["sha256"] or
                    ((d ^ int(row["dhash"], 16)).bit_count() <= 6 and
                     (p ^ int(row["phash"], 16)).bit_count() <= 12)]
        checks.append({"image_name": item["image_name"], "exact_existing": len(exact),
                       "hash_existing": len(near), "closest_existing": str(near[:3]),
                       "hash_internal": len(internal), "internal_peers": ";".join(internal)})
    return checks


def sha_split(folder: Path) -> dict[str, str]:
    return {str(path.relative_to(folder)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(folder.rglob("*")) if path.is_file()}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Bộ dữ liệu đã tồn tại: {OUTPUT}")
    candidates = candidate_rows()
    assert len(candidates) == 24
    checks = overlap_audit(candidates, existing_rows())
    if any(row["exact_existing"] or row["hash_existing"] or row["hash_internal"] for row in checks):
        audit_path = SOURCE / "bow_sai_overlap_blocked.csv"
        write_csv(audit_path, checks)
        raise ValueError(f"Có cặp nghi trùng cần xem bằng mắt: {audit_path}")

    val_before = sha_split(BASE / "val")
    test_before = sha_split(BASE / "test")
    shutil.copytree(BASE, OUTPUT)
    selected = []
    for item in candidates:
        image = OUTPUT / "train/images" / item["image_name"]
        label = OUTPUT / "train/labels" / f"{Path(item['image_name']).stem}.txt"
        if image.exists() or label.exists():
            raise FileExistsError(image)
        image.write_bytes(item["data"])
        label.write_text(item["label"], encoding="utf-8")
        selected.append({key: value for key, value in item.items() if key not in ("data", "label")})
        selected[-1]["output_image"] = str(image.relative_to(OUTPUT)).replace("\\", "/")
        selected[-1]["output_label"] = str(label.relative_to(OUTPUT)).replace("\\", "/")
        selected[-1]["review"] = "full_image_and_bow_mask" if item["source"] == "BoWFire" else "full_image_empty_label"
    write_csv(OUTPUT / "metadata/bow_sai_selected_manifest.csv", selected)
    write_csv(OUTPUT / "metadata/bow_sai_overlap_audit.csv", checks)
    if sha_split(OUTPUT / "val") != val_before or sha_split(OUTPUT / "test") != test_before:
        raise ValueError("Validation hoặc test thay đổi")
    counts = Counter(item["category"] for item in candidates)
    image_counts = {split: len(list((OUTPUT / split / "images").iterdir())) for split in ("train", "val", "test")}
    assert image_counts == {"train": 8011, "val": 1371, "test": 4249}
    summary = {"added": dict(counts), "by_source": dict(Counter(item["source"] for item in candidates)),
               "split_images": image_counts, "val_test_unchanged_sha256": True,
               "exact_or_hash_candidates": 0}
    (OUTPUT / "metadata/bow_sai_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUTPUT / "DATASET_CARD.md").write_text(
        "# D-Fire sạch v2 + Pyro-SDIS + Catargiu + BoWFire + SAINetset v1\n\n"
        "Train 8.011, val 1.371, test 4.249 ảnh. Đợt này thêm 6 ảnh lửa BoWFire, "
        "14 ảnh âm tính BoWFire và 4 ảnh âm tính SAINetset vào train. "
        "Thứ tự lớp: `0 = smoke`, `1 = fire`.\n\n"
        "Ảnh BoWFire được xem toàn bộ trên bảng ảnh; ảnh nhận và mask được xem ở kích thước lớn. "
        "Chỉ dùng ảnh lửa không thấy vùng khói cần gắn nhãn. Box lửa tạo từ mask nhị phân "
        "sau khi kiểm tra bằng mắt; mở rộng thủ công box fire015 và fire037 để ôm đủ đầu ngọn lửa. "
        "Ảnh SAINetset được xem ảnh gốc, xác nhận nhãn rỗng "
        "và không thuộc nhóm trùng byte có nhãn mâu thuẫn; mỗi ảnh thuộc một camera/cảnh khác nhau.\n\n"
        "Tất cả ảnh được so SHA-256, dHash ≤ 6 và pHash ≤ 12 với toàn bộ train/val/test "
        "hiện dùng và với nhau, không có cặp ứng viên theo ngưỡng đó. Hash không chứng minh "
        "không còn trùng cảnh khi góc nhìn hoặc ánh sáng thay đổi. Validation và test giữ nguyên byte.\n\n"
        "Đợt bổ sung rất nhỏ và chưa cân bằng lửa/khói. Chưa nên dùng riêng đợt này để kết luận "
        "mức cải thiện sau một lần train. Truy nguyên ở `metadata/bow_sai_selected_manifest.csv`; "
        "báo cáo kiểm tra nguồn ở `reports/proposed_sources/ASSESSMENT.md`.\n",
        encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
