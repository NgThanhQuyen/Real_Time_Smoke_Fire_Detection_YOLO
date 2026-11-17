"""So ảnh đề xuất với D-Fire đang dùng; hash chỉ tạo ứng viên cần xem ảnh."""

import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from catargiu_audit import hashes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
REPORT = ROOT / "reports/proposed_sources"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def new_rows():
    rows = []
    for path in sorted(SOURCE.glob("home_index_*.csv")):
        for row in read(path):
            rows.append({"source": "home", "split": row["split"], "image": row["image"],
                         "sha256": row["sha256"], "dhash": row["dhash"], "phash": row["phash"]})
    with zipfile.ZipFile(SOURCE / "BoWFireDataset.zip") as archive:
        for name in archive.namelist():
            if not name.startswith("BoWFireDataset/dataset/img/") or not name.lower().endswith(".png"):
                continue
            data = archive.read(name)
            with Image.open(BytesIO(data)) as image:
                d, p = hashes(image)
            rows.append({"source": "bowfire", "split": "unsplit", "image": name,
                         "sha256": hashlib.sha256(data).hexdigest(), "dhash": d, "phash": p})
    full = SOURCE / "sainet_image_index.csv"
    if full.exists() and (SOURCE / "sainet_image_summary.json").exists():
        for record in read(full):
            rows.append({"source": "sainet_full", "split": record["split"], "image": record["image_path"],
                         "sha256": record["sha256"], "dhash": record["dhash"], "phash": record["phash"]})
    else:
        sample = SOURCE / "sainet_sample/sample_index.csv"
        for record in read(sample):
            path = SOURCE / "sainet_sample" / record["image"]
            data = path.read_bytes()
            with Image.open(BytesIO(data)) as image:
                d, p = hashes(image)
            rows.append({"source": "sainet_sample", "split": record["image"].split("/")[1], "image": record["image"],
                         "sha256": hashlib.sha256(data).hexdigest(), "dhash": d, "phash": p})
    return rows


def main():
    manifest = read(ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1/metadata/dfire_manifest.csv")
    active = {r["source_sha256"]: (r["output_split"], r["output_image"])
              for r in manifest if r["output_split"] in ("train", "val", "test")}
    dfire = [r for r in read(ROOT / "runtime/dfire_raw_review/image_index.csv") if r["sha256"] in active]
    buckets = defaultdict(list)
    exact = defaultdict(list)
    for index, row in enumerate(dfire):
        d = int(row["dhash"], 16)
        exact[row["sha256"]].append(index)
        for band in range(8):
            buckets[band, (d >> (band * 8)) & 255].append(index)
    rows = new_rows()
    results = []
    summary = defaultdict(Counter)
    for row in rows:
        source = row["source"]
        summary[source]["images"] += 1
        exact_splits = {active[dfire[i]["sha256"]][0] for i in exact[row["sha256"]]}
        for split in exact_splits:
            summary[source]["exact_" + split] += 1
        d = int(row["dhash"], 16)
        p = int(row["phash"], 16)
        ids = set()
        for band in range(8):
            ids.update(buckets[band, (d >> (band * 8)) & 255])
        matches = []
        for index in ids:
            other = dfire[index]
            dd = (d ^ int(other["dhash"], 16)).bit_count()
            if dd <= 6:
                pp = (p ^ int(other["phash"], 16)).bit_count()
                if pp <= 12:
                    matches.append((dd, pp, other))
        if matches:
            matches.sort(key=lambda item: (item[0], item[1]))
            for split in {active[m[2]["sha256"]][0] for m in matches}:
                summary[source]["hash_candidate_" + split] += 1
            best = matches[0]
            best_split, best_image = active[best[2]["sha256"]]
            results.append({**row, "exact_splits": ";".join(sorted(exact_splits)),
                            "nearest_split": best_split, "nearest_dfire": best_image,
                            "dhash_distance": best[0], "phash_distance": best[1],
                            "candidate_count": len(matches)})
    with (SOURCE / "overlap_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, list(results[0]) if results else ["source", "image"])
        writer.writeheader()
        writer.writerows(results)
    (SOURCE / "overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
