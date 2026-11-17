"""Khảo sát ảnh D-Fire còn có thể bổ sung vào train dữ liệu chính."""

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import csv
import hashlib
import json
import math
import re
import zipfile

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "runtime/fire_smoke_detection_yolov11_bg_dfire2000_v1"
DFIRE = ROOT / "runtime/dfire_clean_v2"
RAW_ZIP = Path(r"F:\6\KLTN\Smoke-Fire-Dataset\D-Fire-YOLO.zip")
REPORT = ROOT / "reports/dfire_remaining_for_main"
CATEGORIES = {"smoke_only", "both", "background"}
SOURCE_ID = re.compile(r"(?i)(WEB|AoF|PublicDataset)0*(\d+)")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hashes(image: Image.Image) -> tuple[int, int]:
    gray = ImageOps.exif_transpose(image).convert("L")
    pixels = np.asarray(gray.resize((9, 8), Image.Resampling.BILINEAR))
    d_hash = int("".join(map(str, (pixels[:, 1:] > pixels[:, :-1]).astype(int).ravel())), 2)
    low = dctn(np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS),
                           dtype=np.float32), norm="ortho")[:8, :8]
    bits = low.ravel() > np.median(low.ravel()[1:])
    p_hash = int("".join(map(str, bits.astype(int))), 2)
    return d_hash, p_hash


def source_id(path: str) -> tuple[str, int] | None:
    match = SOURCE_ID.search(Path(path).stem)
    return (match.group(1).lower(), int(match.group(2))) if match else None


def validate_raw_near(row: dict[str, str], archive: zipfile.ZipFile) -> str:
    image_path = row["source_image"]
    label_path = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
    try:
        content = archive.read(image_path)
        if hashlib.sha256(content).hexdigest() != row["source_sha256"]:
            return "sha_mismatch"
        with Image.open(BytesIO(content)) as image:
            image.verify()
        with Image.open(BytesIO(content)) as image:
            image.load()
        label = archive.read(label_path).decode("utf-8-sig")
    except Exception as error:
        return f"read_error:{type(error).__name__}"
    classes = set()
    for line in label.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            return "invalid_label_row"
        try:
            values = [float(part) for part in parts]
        except ValueError:
            return "invalid_label_number"
        if any(not math.isfinite(value) for value in values):
            return "nonfinite_label"
        cls, x, y, width, height = values
        if cls not in (0, 1) or width <= 0 or height <= 0:
            return "invalid_label_geometry"
        if min(x - width / 2, y - height / 2) < -1e-6 or max(
            x + width / 2, y + height / 2
        ) > 1 + 1e-6:
            return "box_outside_image"
        classes.add(int(cls))
    category = "both" if classes == {0, 1} else "smoke_only" if classes == {0} else (
        "fire_only" if classes == {1} else "background"
    )
    return "ok" if category == row["original_category"] else "category_mismatch"


def locked_hashes() -> list[dict]:
    rows = []
    for split in ("valid", "test"):
        for image_path in sorted((CURRENT / split / "images").iterdir()):
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                continue
            with Image.open(image_path) as image:
                d_hash, p_hash = perceptual_hashes(image)
            rows.append({"split": split, "image": str(image_path.relative_to(CURRENT)).replace("\\", "/"),
                         "dhash": d_hash, "phash": p_hash, "source_id": source_id(image_path.name)})
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = read_csv(DFIRE / "metadata/manifest.csv")
    raw_index = {row["image"]: row for row in read_csv(ROOT / "runtime/dfire_raw_review/image_index.csv")}
    original = read_csv(CURRENT / "metadata/image_manifest.csv")
    backgrounds = read_csv(CURRENT / "metadata/background_additions.csv")
    smoke = read_csv(CURRENT / "metadata/smoke_additions.csv")
    used_hashes = {row["sha256"] for row in original if row["status"] == "kept"}
    used_hashes |= {row["sha256"] for row in backgrounds}
    used_hashes |= {row["source_sha256"] for row in smoke}
    source_locked = {row["source_path"] for row in backgrounds
                     if row["source"] == "D-Fire" and row["split"] in ("valid", "test")}

    candidates = []
    with zipfile.ZipFile(RAW_ZIP) as archive:
        for row in manifest:
            if row["output_split"] == "train" and row["output_category"] in CATEGORIES:
                category, pool, path, label = (row["output_category"], "clean_train",
                                               row["output_image"], row["output_label"])
                label_status = "ok"
            elif (row["original_split"] == "train" and row["output_split"] == "excluded"
                  and row["original_category"] in CATEGORIES
                  and row["exclusion_reason"].startswith("near_identical_train_frame_of:")):
                category, pool, path, label = (row["original_category"], "raw_near_train",
                                               row["source_image"],
                                               row["source_image"].replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt")
                label_status = validate_raw_near(row, archive)
            else:
                continue
            if row["source_image"] not in raw_index:
                raise RuntimeError(f"Không có hash nguồn: {row['source_image']}")
            indexed = raw_index[row["source_image"]]
            candidates.append({
                "category": category, "pool": pool, "source_image": row["source_image"],
                "image": path, "label": label, "sha256": row["source_sha256"],
                "dhash": indexed["dhash"], "phash": indexed["phash"],
                "label_status": label_status,
                "exact_duplicate_current": row["source_sha256"] in used_hashes,
                "source_locked_val_test": path in source_locked,
                "source_id": source_id(row["source_image"]),
            })

    locked = locked_hashes()
    locked_ids = defaultdict(list)
    for row in locked:
        if row["source_id"]:
            locked_ids[row["source_id"]].append(row)
    pairs = []
    for candidate in candidates:
        if candidate["exact_duplicate_current"] or candidate["label_status"] != "ok":
            continue
        d_hash, p_hash = int(candidate["dhash"], 16), int(candidate["phash"], 16)
        for other in locked:
            d_distance = (d_hash ^ other["dhash"]).bit_count()
            if d_distance > 6:
                continue
            p_distance = (p_hash ^ other["phash"]).bit_count()
            if p_distance <= 12:
                pairs.append({"candidate": candidate["source_image"], "category": candidate["category"],
                              "locked_image": other["image"], "locked_split": other["split"],
                              "dhash_distance": d_distance, "phash_distance": p_distance})
    matched = {row["candidate"] for row in pairs}
    same_ids = []
    for candidate in candidates:
        if candidate["exact_duplicate_current"] or candidate["label_status"] != "ok":
            continue
        for other in locked_ids.get(candidate["source_id"], []):
            same_ids.append({"candidate": candidate["source_image"], "category": candidate["category"],
                             "locked_image": other["image"], "locked_split": other["split"]})
    same_id_candidates = {row["candidate"] for row in same_ids}

    seen_sha = set()
    candidate_rows = []
    for candidate in candidates:
        duplicate_pool = candidate["sha256"] in seen_sha
        seen_sha.add(candidate["sha256"])
        candidate_rows.append({
            "category": candidate["category"], "pool": candidate["pool"],
            "source_image": candidate["source_image"], "image": candidate["image"],
            "label": candidate["label"], "sha256": candidate["sha256"],
            "label_status": candidate["label_status"],
            "exact_duplicate_current": int(candidate["exact_duplicate_current"]),
            "exact_duplicate_pool": int(duplicate_pool),
            "near_hash_locked": int(candidate["source_image"] in matched),
            "same_source_id_locked": int(candidate["source_image"] in same_id_candidates),
            "source_locked_val_test": int(candidate["source_locked_val_test"]),
        })
    write_csv(REPORT / "candidates.csv", candidate_rows, list(candidate_rows[0]))
    write_csv(REPORT / "perceptual_pairs_locked.csv", pairs,
              ["candidate", "category", "locked_image", "locked_split", "dhash_distance", "phash_distance"])
    write_csv(REPORT / "same_source_id_locked.csv", same_ids,
              ["candidate", "category", "locked_image", "locked_split"])
    summary = {
        "current_dataset": CURRENT.name,
        "locked_images": len(locked),
        "counts_by_category_pool": {
            category: {pool: sum(row["category"] == category and row["pool"] == pool
                                 for row in candidate_rows)
                       for pool in ("clean_train", "raw_near_train")}
            for category in sorted(CATEGORIES)
        },
        "exact_duplicate_current": dict(Counter(row["category"] for row in candidate_rows
                                                if row["exact_duplicate_current"])),
        "exact_duplicate_pool": dict(Counter(row["category"] for row in candidate_rows
                                             if row["exact_duplicate_pool"])),
        "raw_label_issues": dict(Counter(row["label_status"] for row in candidate_rows
                                         if row["pool"] == "raw_near_train" and row["label_status"] != "ok")),
        "near_hash_candidates": dict(Counter(row["category"] for row in candidate_rows
                                             if row["near_hash_locked"])),
        "same_source_id_candidates": dict(Counter(row["category"] for row in candidate_rows
                                                  if row["same_source_id_locked"])),
        "method": ["Chỉ xét train D-Fire sạch v2 và khung train từng bị loại chỉ do gần giống trong train",
                   "Không lấy val/test D-Fire hoặc ảnh bị cách ly vì nguy cơ cùng cảnh với val/test",
                   "Trùng 100% dựa SHA-256 của byte ảnh; dHash<=6 và pHash<=12 chỉ tạo cặp cần duyệt",
                   "Không sửa hoặc ghép dữ liệu trong bước khảo sát"],
    }
    (REPORT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
