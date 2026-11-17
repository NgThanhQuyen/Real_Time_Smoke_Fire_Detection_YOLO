"""Sàng lọc hai nguồn ảnh để chỉ bổ sung vào D-Fire train."""

import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/dba_indoor_relaxed_review"
DFIRE = ROOT / "runtime/dfire_original_clean_split_v1.zip"
SOURCES = {
    "DBA-Fire": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\DBA-YOLO-Dataset.zip"),
    "Indoor": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Indoor_Fire_Smoke.zip"),
}
BITS = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)


def signature(data):
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError("image_decode")
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dh = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    dhash = np.packbits((dh[:, 1:] > dh[:, :-1]).astype(np.uint8)).tobytes()
    ph = cv2.dct(np.float32(cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)))[:8, :8]
    phash = np.packbits((ph > np.median(ph[1:, :])).astype(np.uint8)).tobytes()
    return image, width, height, dhash, phash


def ssim(left, right):
    """Đo độ giống cấu trúc trên ảnh xám đã cùng cỡ 128x128."""
    a = cv2.resize(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), (128, 128)).astype(np.float32)
    b = cv2.resize(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), (128, 128)).astype(np.float32)
    ma = cv2.GaussianBlur(a, (11, 11), 1.5)
    mb = cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - ma * ma
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mb * mb
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - ma * mb
    score = ((2 * ma * mb + 6.5025) * (2 * cov + 58.5225)) / ((ma * ma + mb * mb + 6.5025) * (va + vb + 58.5225))
    return float(np.mean(score))


def label_class(text, source):
    classes = set()
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            raise ValueError("label_columns")
        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError as error:
            raise ValueError("label_parse") from error
        if cls not in (0, 1):
            raise ValueError("label_class")
        # Roboflow có sai số làm tròn tọa độ nhỏ hơn một pixel; giữ nguyên nhãn nguồn.
        if not np.isfinite((x, y, w, h)).all() or w <= 0 or h <= 0 or x - w / 2 < -0.001 or y - h / 2 < -0.001 or x + w / 2 > 1.001 or y + h / 2 > 1.001:
            raise ValueError("label_geometry")
        classes.add(cls)
    if not classes:
        return "background"
    fire_id = 0
    if classes == {fire_id}:
        return "fire_only"
    if classes == {1}:
        return "smoke_only"
    return "both"


def source_pairs(archive, source):
    names = set(archive.namelist())
    for name in archive.namelist():
        if not name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        if source == "DBA-Fire":
            label = name.replace("/JPEGImages/", "/txt/").rsplit(".", 1)[0] + ".txt"
        else:
            label = name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        yield name, label if label in names else None


def load_locked(archive):
    entries = []
    for name in archive.namelist():
        if not name.startswith(("val/images/", "test/images/")) or not name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        data = archive.read(name)
        _, _, _, dh, ph = signature(data)
        entries.append((name, hashlib.sha256(data).hexdigest(), dh, ph))
        if len(entries) % 1000 == 0:
            print("D-Fire locked", len(entries), flush=True)
    return entries


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(DFIRE) as locked_zip:
        locked = load_locked(locked_zip)
        dbytes = np.frombuffer(b"".join(row[2] for row in locked), dtype=np.uint8).reshape(-1, 8)
        pbytes = np.frombuffer(b"".join(row[3] for row in locked), dtype=np.uint8).reshape(-1, 8)
        sha_map = {row[1]: row[0] for row in locked}
        print("D-Fire locked total", len(locked), flush=True)
        all_rows = []
        summary = {}
        for source, path in SOURCES.items():
            rows = []
            counts = Counter()
            with zipfile.ZipFile(path) as archive:
                bad_crc = archive.testzip()
                if bad_crc:
                    raise RuntimeError(f"CRC lỗi: {path}: {bad_crc}")
                for number, (image_path, label_path) in enumerate(source_pairs(archive, source), 1):
                    row = {"source": source, "image": image_path, "label": label_path or "", "category": "", "status": "eligible", "reason": "", "matched_locked": "", "ssim": "", "dhash_distance": "", "phash_distance": "", "width": "", "height": "", "sha256": ""}
                    try:
                        if label_path is None:
                            raise ValueError("label_missing")
                        row["category"] = label_class(archive.read(label_path).decode("utf-8-sig"), source)
                        data = archive.read(image_path)
                        row["sha256"] = hashlib.sha256(data).hexdigest()
                        image, row["width"], row["height"], dh, ph = signature(data)
                        if row["sha256"] in sha_map:
                            row.update(status="excluded", reason="exact_locked", matched_locked=sha_map[row["sha256"]], ssim=1.0)
                        else:
                            dd = BITS[np.bitwise_xor(dbytes, np.frombuffer(dh, np.uint8)).astype(np.uint8)].sum(axis=1)
                            pd = BITS[np.bitwise_xor(pbytes, np.frombuffer(ph, np.uint8)).astype(np.uint8)].sum(axis=1)
                            indices = np.flatnonzero((dd <= 12) | (pd <= 12))
                            if len(indices):
                                indices = sorted(indices, key=lambda i: (int(dd[i]) + int(pd[i]), int(dd[i])))[:30]
                                best = (-1.0, None)
                                for index in indices:
                                    other = cv2.imdecode(np.frombuffer(locked_zip.read(locked[index][0]), np.uint8), cv2.IMREAD_COLOR)
                                    value = ssim(image, other)
                                    if value > best[0]:
                                        best = (value, index)
                                    if value > 0.90:
                                        break
                                if best[1] is not None:
                                    row.update(matched_locked=locked[best[1]][0], ssim=round(best[0], 5), dhash_distance=int(dd[best[1]]), phash_distance=int(pd[best[1]]))
                                    if best[0] > 0.90:
                                        row.update(status="excluded", reason="ssim_over_90_locked")
                    except (ValueError, UnicodeDecodeError) as error:
                        row.update(status="excluded", reason=str(error))
                    rows.append(row)
                    counts[(row["status"], row["category"], row["reason"])] += 1
                    if number % 500 == 0:
                        print(source, number, "processed", flush=True)
            all_rows.extend(rows)
            summary[source] = {"source_images": len(rows), "eligible": sum(r["status"] == "eligible" for r in rows), "eligible_by_category": dict(Counter(r["category"] for r in rows if r["status"] == "eligible")), "excluded_by_reason": dict(Counter(r["reason"] for r in rows if r["status"] == "excluded")), "source_by_category": dict(Counter(r["category"] for r in rows))}
            print(source, summary[source], flush=True)
        with (OUT / "manifest.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
