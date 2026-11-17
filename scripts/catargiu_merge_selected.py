"""Gộp ảnh Catargiu đã duyệt vào train, giữ nguyên val và test."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path("F:/6/KLTN/Smoke-Fire-Dataset")
BASE = ROOT / "runtime/dfire_clean_v2_plus_pyro_v1"
OUTPUT = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1"
REVIEW = ROOT / "runtime/catargiu_audit/review_pages/visual_review.csv"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_label(image_name: str) -> str:
    return image_name.replace("/images/", "/labels/").removesuffix(".jpg") + ".txt"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def verify_base_copy() -> dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        base_images = sorted((BASE / split / "images").iterdir())
        out_images = sorted((OUTPUT / split / "images").iterdir())
        if [path.name for path in base_images] != [path.name for path in out_images[:len(base_images)]] and split != "train":
            raise RuntimeError(f"Sai danh sách ảnh {split}")
        if split in ("val", "test"):
            for subdir in ("images", "labels"):
                original = sorted((BASE / split / subdir).iterdir())
                copied = sorted((OUTPUT / split / subdir).iterdir())
                if [p.name for p in original] != [p.name for p in copied]:
                    raise RuntimeError(f"Sai tên file {split}/{subdir}")
                for first, second in zip(original, copied):
                    if digest(first.read_bytes()) != digest(second.read_bytes()):
                        raise RuntimeError(f"Val/test đã thay đổi: {first}")
        counts[split] = len(out_images)
    return counts


def main() -> None:
    if OUTPUT.exists():
        raise SystemExit(f"Thư mục đích đã tồn tại: {OUTPUT}")
    selected = [row for row in read_rows(REVIEW) if row["decision"] == "accept"]
    if not selected:
        raise RuntimeError("Chưa có ảnh được duyệt")
    index = {(row["source"], row["image"]): row for row in read_rows(ROOT / "runtime/catargiu_audit/image_index.csv")}
    for row in selected:
        info = index[row["source"], row["image"]]
        expected = "fire" if row["review_id"].startswith("fire_only") else "other"
        if info["category"] != expected or info["sha256"] != row["sha256"]:
            raise RuntimeError(f"Sai loại hoặc hash: {row['review_id']}")

    shutil.copytree(BASE, OUTPUT)
    manifests = []
    source_names = sorted({row["source"] for row in selected})
    with zipfile.ZipFile(SOURCE_ROOT / source_names[0]) as first, zipfile.ZipFile(SOURCE_ROOT / source_names[-1]) as second:
        archives = {source_names[0]: first, source_names[-1]: second}
        for row in selected:
            archive = archives[row["source"]]
            image_bytes = archive.read(row["image"])
            if digest(image_bytes) != row["sha256"]:
                raise RuntimeError(f"Ảnh nguồn đổi sau kiểm duyệt: {row['review_id']}")
            label_text = archive.read(source_label(row["image"])).decode("utf-8-sig")
            fire = row["review_id"].startswith("fire_only")
            lines = [line.strip().split() for line in label_text.splitlines() if line.strip()]
            if not lines or any(len(parts) != 5 or parts[0] != ("0" if fire else "1") for parts in lines):
                raise RuntimeError(f"Nhãn nguồn không khớp: {row['review_id']}")
            output_name = f"Catargiu_{row['sha256'][:24]}"
            image_out = OUTPUT / "train/images" / f"{output_name}.jpg"
            label_out = OUTPUT / "train/labels" / f"{output_name}.txt"
            if image_out.exists() or label_out.exists():
                raise RuntimeError(f"Trùng tên đầu ra: {output_name}")
            image_out.write_bytes(image_bytes)
            label_out.write_text("".join("1 " + " ".join(parts[1:]) + "\n" for parts in lines) if fire else "", encoding="utf-8")
            manifests.append({**row, "output_image": f"train/images/{output_name}.jpg",
                              "output_label": f"train/labels/{output_name}.txt",
                              "output_class": "fire" if fire else "background",
                              "output_boxes": len(lines) if fire else 0})

    counts = verify_base_copy()
    expected = {"train": 7883 + len(selected), "val": 1371, "test": 4249}
    if counts != expected:
        raise RuntimeError(f"Sai số lượng ảnh: {counts} != {expected}")
    shutil.copy2(REVIEW, OUTPUT / "metadata/catargiu_visual_review.csv")
    write_rows(OUTPUT / "metadata/catargiu_selected_manifest.csv", manifests)
    summary = {
        "base": BASE.name,
        "source": "Catargiu FireAndSmoke part1 v2 and part2 v1 supplied by user",
        "splits": counts,
        "added": dict(Counter(row["output_class"] for row in manifests)),
        "added_fire_boxes": sum(int(row["output_boxes"]) for row in manifests),
        "policy": "Chỉ nhận ảnh đã xem đủ 640 px; chỉ thêm vào train; val/test được so SHA-256 từng file với bản gốc.",
    }
    (OUTPUT / "metadata/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "DATASET_CARD.md").write_text(
        "# D-Fire clean v2 + Pyro-SDIS + Catargiu tuyển chọn v1\n\n"
        f"Train {counts['train']}, val {counts['val']}, test {counts['test']} ảnh. "
        f"Đợt này thêm {summary['added'].get('fire', 0)} ảnh chỉ lửa và "
        f"{summary['added'].get('background', 0)} ảnh âm tính khó vào train, "
        f"{summary['added_fire_boxes']} box lửa.\n\n"
        "Ảnh đã qua kiểm tra nhãn, ảnh ghép độ phân giải gốc, loại cảnh lặp theo tên nguồn, "
        "đối chiếu SHA/dHash/pHash với D-Fire và rà soát trùng trong nhóm nhận. "
        "Nhãn gốc `fire=0` được đổi thành `fire=1`; ảnh `other` thành nhãn rỗng. "
        "Val/test được giữ nguyên byte so với bản D-Fire + Pyro. "
        "Hash là tín hiệu kiểm duyệt, không chứng minh sạch tuyệt đối về nguồn cảnh.\n\n"
        "Chỉ dùng bản này để thử nghiệm so sánh trên cùng val/test. "
        "Đợt bổ sung nhỏ nên chưa cân bằng lửa và khói. "
        "Tập test D-Fire đã được dùng để định hướng dữ liệu, vì vậy vẫn cần holdout mới chưa xem.\n\n"
        "Truy nguyên tại `metadata/catargiu_selected_manifest.csv` và "
        "`metadata/catargiu_visual_review.csv`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
