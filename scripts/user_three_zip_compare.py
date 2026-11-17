"""Read-only overlap audit for the three proposed sources against fixed D-Fire splits."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "runtime/new_sources_audit"
BASE = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_v1/metadata"


def read(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    current = read(AUDIT / "image_index.csv")
    active = {r["source_sha256"]: r["output_split"] for r in read(BASE / "dfire_manifest.csv")
              if r["output_split"] in ("train", "val", "test")}
    raw = [r for r in read(ROOT / "runtime/dfire_raw_review/image_index.csv") if r["sha256"] in active]
    exact = defaultdict(list)
    buckets = defaultdict(list)
    for index, row in enumerate(raw):
        exact[row["sha256"]].append(row)
        d = int(row["dhash"], 16)
        for band in range(8):
            buckets[band, (d >> (band * 8)) & 255].append(index)
    rows = []
    summary = defaultdict(Counter)
    for row in current:
        source = row["source"]
        summary[source]["images"] += 1
        exact_rows = exact[row["sha256"]]
        exact_splits = sorted({active[r["sha256"]] for r in exact_rows})
        if exact_splits:
            summary[source]["exact_active_any"] += 1
            for split in exact_splits:
                summary[source]["exact_active_" + split] += 1
        d = int(row["dhash"], 16)
        p = int(row["phash"], 16)
        candidate_ids = set()
        for band in range(8):
            candidate_ids.update(buckets[band, (d >> (band * 8)) & 255])
        matches = []
        for index in candidate_ids:
            candidate = raw[index]
            dd = (d ^ int(candidate["dhash"], 16)).bit_count()
            if dd <= 6:
                pp = (p ^ int(candidate["phash"], 16)).bit_count()
                if pp <= 12:
                    matches.append((dd, pp, candidate))
        if matches:
            matches.sort(key=lambda item: (item[0], item[1]))
            for split in {active[m[2]["sha256"]] for m in matches}:
                summary[source]["hash_candidate_active_" + split] += 1
            best = matches[0]
            rows.append({"source": source, "image": row["image"], "exact_splits": ";".join(exact_splits),
                         "nearest_split": active[best[2]["sha256"]], "nearest_dfire": best[2]["image"],
                         "dhash_distance": best[0], "phash_distance": best[1], "candidate_count": len(matches)})
        elif exact_splits:
            rows.append({"source": source, "image": row["image"], "exact_splits": ";".join(exact_splits),
                         "nearest_split": "", "nearest_dfire": "", "dhash_distance": "", "phash_distance": "", "candidate_count": 0})
    with (AUDIT / "dfire_overlap_candidates.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, ["source", "image", "exact_splits", "nearest_split", "nearest_dfire",
                                         "dhash_distance", "phash_distance", "candidate_count"])
        writer.writeheader()
        writer.writerows(rows)
    (AUDIT / "dfire_overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
