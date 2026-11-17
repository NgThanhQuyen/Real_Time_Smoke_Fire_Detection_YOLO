"""Tạo bộ YOLO cuối từ đúng hai ZIP nguồn và manifest đã kiểm duyệt."""

from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import json
import random
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runtime"
OLD = RUNTIME / "fire_smoke_detection_yolov11_bg_dfire2000_v1"
DFIRE_CLEAN = RUNTIME / "dfire_clean_v2"
REVIEW = ROOT / "reports/dfire_remaining_for_main/reviewed_candidates.csv"
REPORT = ROOT / "reports/final_two_source_dataset"
MAIN_ZIP = Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Fire-Smoke-Detection-Yolov11.zip")
DFIRE_ZIP = Path(r"F:\6\KLTN\Smoke-Fire-Dataset\D-Fire-YOLO.zip")
FINAL_ZIP = RUNTIME / "fire_smoke_dfire_final.zip"
SEED = 20261108

QUOTAS = {
    "train": {"smoke_only": 176, "background": 796},
    "valid": {"fire_only": 70, "smoke_only": 276, "both": 121, "background": 228},
    "test": {"fire_only": 179, "smoke_only": 110, "both": 400, "background": 442},
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def category(label_text: str) -> str:
    classes = {int(line.split()[0]) for line in label_text.splitlines() if line.strip()}
    if classes == {0}:
        return "fire_only"
    if classes == {1}:
        return "smoke_only"
    if classes == {0, 1}:
        return "both"
    if not classes:
        return "background"
    raise ValueError(f"Nhãn có lớp lạ: {classes}")


def remap_dfire(label_text: str) -> str:
    lines = []
    for line in label_text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if parts[0] not in {"0", "1"} or len(parts) != 5:
            raise ValueError(f"Nhãn D-Fire không hợp lệ: {line}")
        parts[0] = "1" if parts[0] == "0" else "0"
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")


def make_row(source: str, source_image: str, source_hash: str, split: str,
             output_image: str, label_text: str, reason: str) -> dict[str, str]:
    output_label = output_image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    return {
        "source_dataset": source,
        "source_image": source_image,
        "source_sha256": source_hash,
        "split": split,
        "output_image": output_image,
        "output_label": output_label,
        "category": category(label_text),
        "label_text": label_text,
        "selection_reason": reason,
    }


def existing_rows() -> list[dict[str, str]]:
    rows = []
    clean_source = {item["output_image"]: item["source_image"]
                    for item in read_csv(DFIRE_CLEAN / "metadata/manifest.csv")
                    if item["output_split"] != "excluded"}
    main = read_csv(OLD / "metadata/image_manifest.csv")
    for item in main:
        if item["status"] != "kept":
            continue
        path = item["image"]
        label = (OLD / item["label"]).read_text(encoding="utf-8-sig")
        rows.append(make_row("Fire-Smoke-Detection-Yolov11", path,
                             item["sha256"], item["split"], path, label,
                             "main_kept_after_exact_dedup"))

    for item in read_csv(OLD / "metadata/background_additions.csv"):
        if item["source"] != "D-Fire":
            continue
        raw_path = clean_source[item["source_path"]]
        rows.append(make_row("D-Fire", raw_path, item["sha256"],
                             item["split"], item["output_image"], "",
                             "existing_dfire_background"))

    for item in read_csv(OLD / "metadata/smoke_additions.csv"):
        label = (OLD / item["output_label"]).read_text(encoding="utf-8-sig")
        rows.append(make_row("D-Fire", item["source_image"],
                             item["source_sha256"], "train", item["output_image"],
                             label, "existing_dfire_smoke"))
    return rows


def candidates_train(archive: zipfile.ZipFile) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in read_csv(REVIEW):
        if item["review_decision"] != "available_after_review":
            continue
        if item["category"] not in QUOTAS["train"]:
            continue
        if item["pool"] == "clean_train":
            label = (DFIRE_CLEAN / item["label"]).read_text(encoding="utf-8-sig")
        else:
            label = archive.read(item["label"]).decode("utf-8-sig")
        label = remap_dfire(label)
        if category(label) != item["category"]:
            raise ValueError(f"Sai lớp ảnh train: {item['source_image']}")
        result[item["category"]].append({**item, "source_sha256": item["sha256"],
                                         "label_text": label})
    return result


def candidates_holdout() -> dict[str, dict[str, list[dict[str, str]]]]:
    result: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for item in read_csv(DFIRE_CLEAN / "metadata/manifest.csv"):
        if item["output_split"] not in {"val", "test"}:
            continue
        split = "valid" if item["output_split"] == "val" else "test"
        result[split][item["output_category"]].append(item)
    return result


def sample_items(items: list[dict[str, str]], count: int, used: set[str],
                 salt: str) -> list[dict[str, str]]:
    eligible = [item for item in items if item["source_sha256"] not in used]
    eligible.sort(key=lambda item: hashlib.sha256(
        f"{SEED}|{salt}|{item['source_sha256']}".encode()).hexdigest())
    chosen = []
    for item in eligible:
        if item["source_sha256"] in used:
            continue
        chosen.append(item)
        used.add(item["source_sha256"])
        if len(chosen) == count:
            break
    if len(chosen) != count:
        raise ValueError(f"Thiếu ảnh cho {salt}: cần {count}, có {len(chosen)}")
    return chosen


def build_manifest(dfire_archive: zipfile.ZipFile) -> list[dict[str, str]]:
    rows = existing_rows()
    hashes: dict[str, str] = {}
    paths: set[str] = set()
    for row in rows:
        if row["source_sha256"] in hashes:
            raise ValueError(f"Ảnh trùng byte ở bản hiện tại: {row['output_image']}")
        hashes[row["source_sha256"]] = row["split"]
        paths.add(row["output_image"])
    used = set(hashes)

    train = candidates_train(dfire_archive)
    smoke = sample_items(train["smoke_only"], QUOTAS["train"]["smoke_only"],
                         used, "train_smoke")
    scores = {item["source_image"]: float(item["max_conf"])
              for item in read_csv(RUNTIME / "dfire_background_scores_for_main_v1.csv")}
    backgrounds = [item for item in train["background"] if item["source_sha256"] not in used]
    backgrounds.sort(key=lambda item: (-scores.get(item["source_image"], 0.0),
                                       item["source_sha256"]))
    hard = backgrounds[:400]
    for item in hard:
        used.add(item["source_sha256"])
    other = sample_items(backgrounds[400:], 396, used, "train_background_diverse")
    for group, items in (("smoke_only", smoke), ("background", hard + other)):
        for number, item in enumerate(items, 1):
            name = f"train/images/dfire_final_{group}_{number:04d}.jpg"
            rows.append(make_row("D-Fire", item["source_image"], item["sha256"],
                                 "train", name, item["label_text"],
                                 "new_train_" + group))

    holdouts = candidates_holdout()
    for split in ("valid", "test"):
        for group, count in QUOTAS[split].items():
            items = sample_items(holdouts[split][group], count, used, f"{split}_{group}")
            for number, item in enumerate(items, 1):
                label = (DFIRE_CLEAN / item["output_label"]).read_text(encoding="utf-8-sig")
                label = remap_dfire(label)
                if category(label) != group:
                    raise ValueError(f"Sai lớp ảnh holdout: {item['source_image']}")
                name = f"{split}/images/dfire_final_{group}_{number:04d}.jpg"
                rows.append(make_row("D-Fire", item["source_image"],
                                     item["source_sha256"], split, name, label,
                                     "new_" + split + "_" + group))

    for row in rows:
        if row["output_image"] in paths and row["selection_reason"].startswith("new_"):
            raise ValueError(f"Trùng tên đầu ra: {row['output_image']}")
        paths.add(row["output_image"])
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise ValueError("Có ảnh trùng byte trong manifest cuối")
    expected = {"train": 13329, "valid": 1666, "test": 1666}
    actual = Counter(row["split"] for row in rows)
    if dict(actual) != expected:
        raise ValueError(f"Sai kích thước split: {actual}")
    return rows


def save_manifest(rows: list[dict[str, str]]) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    path = REPORT / "selection_manifest.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def verify_image(data: bytes, path: str) -> None:
    with Image.open(BytesIO(data)) as image:
        image.verify()
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"Ảnh sai kích thước: {path}")


def write_zip(rows: list[dict[str, str]], main: zipfile.ZipFile,
              dfire: zipfile.ZipFile) -> None:
    if FINAL_ZIP.exists():
        raise FileExistsError(f"Đã có ZIP cuối, không ghi đè: {FINAL_ZIP}")
    building = FINAL_ZIP.with_suffix(".zip.building")
    if building.exists():
        raise FileExistsError(f"Có bản đang tạo: {building}")
    info = (
        "# Bộ dữ liệu Fire và Smoke cho khóa luận\n\n"
        "Bộ YOLO này được tổng hợp từ hai nguồn ảnh: Fire-Smoke-Detection-Yolov11 "
        "và D-Fire. Nhãn đầu ra: 0 = Fire, 1 = Smoke. Ảnh nền có file nhãn rỗng.\n\n"
        "Ảnh trùng byte được kiểm tra bằng SHA-256. Danh sách nguồn, split, hash "
        "và nhãn cuối nằm trong metadata/selection_manifest.jsonl. "
        "Quy trình tái lập và phân tích nằm trong notebook của dự án.\n"
    )
    yaml = "train: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames:\n  0: Fire\n  1: Smoke\n"
    try:
        with zipfile.ZipFile(building, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=4, allowZip64=True) as output:
            output.writestr("data.yaml", yaml)
            output.writestr("DATASET_INFO.md", info)
            for index, row in enumerate(rows, 1):
                source = main if row["source_dataset"] == "Fire-Smoke-Detection-Yolov11" else dfire
                data = source.read(row["source_image"])
                if sha(data) != row["source_sha256"]:
                    raise ValueError(f"Sai SHA-256 nguồn: {row['source_image']}")
                verify_image(data, row["source_image"])
                output.writestr(row["output_image"], data)
                output.writestr(row["output_label"], row["label_text"].encode("utf-8"))
                if index % 2000 == 0:
                    print(f"Verified {index}/{len(rows)} images", flush=True)
            manifest = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            output.writestr("metadata/selection_manifest.jsonl", manifest.encode("utf-8"))
        with zipfile.ZipFile(building) as archive:
            bad = archive.testzip()
            if bad:
                raise ValueError(f"ZIP lỗi CRC: {bad}")
        building.replace(FINAL_ZIP)
    except BaseException:
        building.unlink(missing_ok=True)
        raise


def summarize(rows: list[dict[str, str]]) -> dict:
    splits = {split: dict(Counter(row["category"] for row in rows if row["split"] == split))
              for split in ("train", "valid", "test")}
    sources = {split: dict(Counter(row["source_dataset"] for row in rows if row["split"] == split))
               for split in ("train", "valid", "test")}
    counts = Counter(row["split"] for row in rows)
    return {"zip": str(FINAL_ZIP), "total_images": len(rows), "splits": dict(counts),
            "categories": splits, "sources": sources,
            "sha256_unique": len({row["source_sha256"] for row in rows}) == len(rows),
            "label_mapping": {"0": "Fire", "1": "Smoke"},
            "selection_seed": SEED, "quota_new": QUOTAS}


def main() -> None:
    with zipfile.ZipFile(MAIN_ZIP) as main_archive, zipfile.ZipFile(DFIRE_ZIP) as dfire_archive:
        frozen = REPORT / "selection_manifest.jsonl"
        if OLD.exists() and DFIRE_CLEAN.exists() and REVIEW.exists():
            rows = build_manifest(dfire_archive)
            save_manifest(rows)
        elif frozen.is_file():
            rows = [json.loads(line) for line in frozen.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        else:
            raise FileNotFoundError("Thiếu cả dữ liệu trung gian và manifest cuối")
        summary = summarize(rows)
        (REPORT / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)
        write_zip(rows, main_archive, dfire_archive)
    print(f"Completed: {FINAL_ZIP} ({FINAL_ZIP.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
