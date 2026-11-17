"""Xuất D-Fire sạch v2 cộng ảnh Pyro-SDIS đã kiểm duyệt, giữ nguyên val/test."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("runtime")
BASE = ROOT / "dfire_clean_v2"
SOURCE = ROOT / "pyro_sdis_v2_source" / "data"
SELECTION = Path("reports/pyro_sdis_v2_audit/accepted_provisional.csv")
OUTPUT = ROOT / "dfire_clean_v2_plus_pyro_v1"


def copy_base() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            source_dir = BASE / split / kind
            output_dir = OUTPUT / split / kind
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in source_dir.iterdir():
                destination = output_dir / path.name
                if destination.exists():
                    continue
                try:
                    os.link(path, destination)
                except OSError:
                    shutil.copy2(path, destination)
    (OUTPUT / "metadata").mkdir(exist_ok=True)
    shutil.copy2(BASE / "metadata" / "manifest.csv", OUTPUT / "metadata" / "dfire_manifest.csv")


def selected_rows() -> list[dict[str, str]]:
    with SELECTION.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len({row["sha256"] for row in rows}) != len(rows):
        raise ValueError("Trùng SHA trong ảnh Pyro-SDIS đã chọn")
    if any(row["issue"] or row["dfire_exact_match"] != "0" or row["dfire_strong_hash_candidate"] != "0" for row in rows):
        raise ValueError("Một ảnh đã chọn còn lỗi nhãn hoặc ứng viên trùng D-Fire")
    return rows


def add_pyro(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_file: dict[str, dict[int, dict[str, str]]] = {}
    for row in rows:
        by_file.setdefault(row["parquet_file"], {})[int(row["source_row"])] = row
    manifest = []
    for parquet_name, wanted in by_file.items():
        table = pq.read_table(SOURCE / parquet_name, columns=["image", "annotations", "image_name"])
        for source_row, row in wanted.items():
            if table["image_name"][source_row].as_py() != row["image_name"]:
                raise ValueError(f"Sai dòng nguồn: {parquet_name}:{source_row}")
            image = table["image"][source_row].as_py()["bytes"]
            label = table["annotations"][source_row].as_py() or ""
            if hashlib.sha256(image).hexdigest() != row["sha256"]:
                raise ValueError(f"Sai SHA: {row['image_name']}")
            lines = [line.strip() for line in label.splitlines() if line.strip()]
            if row["decision"] == "accept_smoke":
                if not lines or any(line.split()[0] != "0" or len(line.split()) != 5 for line in lines):
                    raise ValueError(f"Nhãn khói sai: {row['image_name']}")
            elif row["decision"] == "accept_negative":
                if lines:
                    raise ValueError(f"Ảnh nền có box: {row['image_name']}")
            else:
                raise ValueError(f"Quyết định lạ: {row['decision']}")
            stem = "PyroSDIS_v2_" + row["sha256"][:20]
            image_path = OUTPUT / "train" / "images" / (stem + ".jpg")
            label_path = OUTPUT / "train" / "labels" / (stem + ".txt")
            image_path.write_bytes(image)
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            manifest.append({
                "output_image": image_path.relative_to(OUTPUT).as_posix(),
                "output_label": label_path.relative_to(OUTPUT).as_posix(),
                "source_image": row["image_name"],
                "source_parquet": parquet_name,
                "source_row": source_row,
                "source_sha256": row["sha256"],
                "camera_key": row["camera_key"],
                "date": row["date"],
                "decision": row["decision"],
                "box_count": len(lines),
                "nearest_dfire_dhash": row["nearest_dfire_dhash"],
                "nearest_dfire_phash": row["nearest_dfire_phash"],
            })
        print(f"Extracted {len(wanted)} images from {parquet_name}", flush=True)
    return sorted(manifest, key=lambda row: row["output_image"])


def write_metadata(rows: list[dict[str, str]], manifest: list[dict[str, str]]) -> None:
    with (OUTPUT / "metadata" / "pyro_selected_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    (OUTPUT / "data.yaml").write_text(
        "train: train/images\nval: val/images\ntest: test/images\nnc: 2\nnames:\n  0: smoke\n  1: fire\n",
        encoding="utf-8",
    )
    counts = Counter(row["decision"] for row in rows)
    summary = {
        "base": "dfire_clean_v2",
        "source": "pyronear/pyro-sdis v2.0",
        "source_revision": "ebfc02d0cba7f7dfe0623d13bcf0f0aa8a38e3ee",
        "pyro_images": len(rows),
        "pyro_smoke": counts["accept_smoke"],
        "pyro_background": counts["accept_negative"],
        "pyro_cameras": len({row["camera_key"] for row in rows}),
        "pyro_smoke_boxes": sum(row["box_count"] for row in manifest),
        "splits": {
            split: len(list((OUTPUT / split / "images").iterdir()))
            for split in ("train", "val", "test")
        },
        "policy": "Chỉ thêm ảnh vào train; val/test D-Fire sạch v2 không đổi; ảnh Pyro được chọn theo camera và kiểm duyệt trực quan; ảnh không rõ loại khỏi bản này.",
    }
    (OUTPUT / "metadata" / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "DATASET_CARD.md").write_text(
        "# D-Fire clean v2 + Pyro-SDIS v2 (đợt 1)\n\n"
        "Nguồn gốc: D-Fire sạch v2 và Pyro-SDIS v2.0, revision `ebfc02d0cba7f7dfe0623d13bcf0f0aa8a38e3ee`. "
        "Pyro-SDIS công bố giấy phép Apache-2.0.\n\n"
        f"Bổ sung {counts['accept_smoke']} ảnh khói nhỏ/xa và {counts['accept_negative']} ảnh âm tính khó "
        f"từ {summary['pyro_cameras']} camera. Train có {summary['splits']['train']} ảnh; "
        f"val {summary['splits']['val']}, test {summary['splits']['test']} không đổi.\n\n"
        "Ảnh và nhãn Pyro-SDIS được kiểm tra trực quan trên contact sheet, loại ảnh khói mờ/không chắc, "
        "ảnh có lỗi box và ảnh trùng byte. Toàn bộ 12.378 ảnh nguồn được đối chiếu SHA với D-Fire gốc; "
        "73 ảnh chọn không có cặp dHash≤6 và pHash≤12 với D-Fire gốc. Hash chỉ là tín hiệu rà soát, "
        "không chứng minh hết rò rỉ cảnh. Nguồn Pyro-SDIS có camera/date để kiểm soát nhóm. "
        "Tất cả ảnh Pyro đã chọn nằm ở train, nên không có camera Pyro bắc qua split.\n\n"
        "Giới hạn: Pyro-SDIS chỉ có lớp khói; đợt này chưa giải quyết thiếu lửa, khói đêm và bối cảnh trong nhà. "
        "Ảnh âm tính được nguồn ghi không có khói và được xem trên contact sheet, nhưng ảnh mơ hồ vẫn có "
        "thể cần kiểm tra lại ở độ phân giải gốc. Bộ test D-Fire đã dùng để phân tích lỗi nên cần holdout mới "
        "chưa xem cho đánh giá độc lập cuối cùng. Không dùng video kiểm thử của người dùng để train.\n\n"
        "Xem `metadata/pyro_selected_manifest.csv`, `metadata/dfire_manifest.csv` và "
        "`reports/pyro_sdis_v2_audit/` trong repository để truy nguyên kiểm duyệt.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


def make_zip() -> None:
    destination = OUTPUT.with_suffix(".zip")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(OUTPUT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT).as_posix())
    os.replace(temporary, destination)
    print(f"Kaggle package: {destination} ({destination.stat().st_size} bytes)", flush=True)


def main() -> None:
    rows = selected_rows()
    copy_base()
    manifest = add_pyro(rows)
    write_metadata(rows, manifest)
    make_zip()


if __name__ == "__main__":
    main()
