"""Tìm ứng viên trùng ảnh Home Fire với bộ Indoor đã được người dùng cung cấp."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


home = []
for path in sorted((ROOT / "runtime/proposed_sources").glob("home_index_*.csv")):
    home.extend(read(path))
indoor = [r for r in read(ROOT / "runtime/new_sources_audit/image_index.csv") if r["source"] == "Indoor_Fire_Smoke.zip"]
buckets = defaultdict(list)
exact = {r["sha256"] for r in indoor}
for index, row in enumerate(indoor):
    d = int(row["dhash"], 16)
    for band in range(8):
        buckets[band, (d >> (band * 8)) & 255].append(index)
summary = Counter()
results = []
for row in home:
    summary["home_images"] += 1
    summary["exact_indoor"] += row["sha256"] in exact
    d = int(row["dhash"], 16)
    p = int(row["phash"], 16)
    ids = set()
    for band in range(8):
        ids.update(buckets[band, (d >> (band * 8)) & 255])
    matches = []
    for index in ids:
        other = indoor[index]
        dd = (d ^ int(other["dhash"], 16)).bit_count()
        if dd <= 6:
            pp = (p ^ int(other["phash"], 16)).bit_count()
            if pp <= 12:
                matches.append((dd, pp, other))
    if matches:
        matches.sort(key=lambda item: (item[0], item[1]))
        summary["hash_candidate_indoor"] += 1
        best = matches[0]
        results.append({"home_split": row["split"], "home_image": row["image"],
                        "indoor_image": best[2]["image"], "dhash_distance": best[0],
                        "phash_distance": best[1], "candidate_count": len(matches)})
with (ROOT / "runtime/proposed_sources/home_indoor_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, ["home_split", "home_image", "indoor_image", "dhash_distance", "phash_distance", "candidate_count"])
    writer.writeheader()
    writer.writerows(results)
print(json.dumps(summary, indent=2))
