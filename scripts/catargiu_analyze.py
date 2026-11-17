"""Phân tích chỉ mục FireAndSmoke và ứng viên trùng với D-Fire đã khóa."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("runtime/catargiu_audit")
DFIRE = Path("runtime/dfire_raw_review/image_index.csv")
DFIRE_MANIFEST = Path("runtime/dfire_clean_v2/metadata/manifest.csv")
PYRO = Path("runtime/dfire_clean_v2_plus_pyro_v1/metadata/pyro_selected_manifest.csv")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    source = rows(ROOT / "image_index.csv")
    dfire = rows(DFIRE)
    active = {row["source_sha256"]: row["output_split"] for row in rows(DFIRE_MANIFEST) if row["output_split"] in ("train", "val", "test")}
    pyro = {row["source_sha256"] for row in rows(PYRO)}
    known = []
    for row in dfire:
        split = active.get(row["sha256"])
        if split:
            known.append({"sha256": row["sha256"], "image": row["image"], "split": split, "dhash": row["dhash"], "phash": row["phash"]})
    by_sha = defaultdict(list)
    by_band = defaultdict(list)
    for index, row in enumerate(known):
        by_sha[row["sha256"]].append(index)
        value = int(row["dhash"], 16)
        for band in range(8):
            by_band[(band, (value >> (8 * band)) & 255)].append(index)
    matches = []
    for number, row in enumerate(source, 1):
        if not row["dhash"]:
            continue
        value = int(row["dhash"], 16)
        candidates = set()
        for band in range(8):
            candidates.update(by_band[(band, (value >> (8 * band)) & 255)])
        near = []
        for index in candidates:
            other = known[index]
            distance = (value ^ int(other["dhash"], 16)).bit_count()
            if distance <= 6:
                near.append((distance, (int(row["phash"], 16) ^ int(other["phash"], 16)).bit_count(), other))
        if near:
            near.sort(key=lambda item: (item[0], item[1]))
            nearest_by_split = {}
            for distance, phash_distance, other in near:
                for threshold in (64, 12, 8):
                    if phash_distance <= threshold:
                        nearest_by_split.setdefault((other["split"], threshold), (distance, phash_distance, other))
            emitted = set()
            for distance, phash_distance, other in nearest_by_split.values():
                match_key = (other["image"], distance, phash_distance)
                if match_key in emitted:
                    continue
                emitted.add(match_key)
                matches.append({"source": row["source"], "image": row["image"], "category": row["category"], "scene_key": row["scene_key"],
                                "dfire_image": other["image"], "dfire_split": other["split"], "dhash_distance": distance,
                                "phash_distance": phash_distance, "sha_equal": int(row["sha256"] == other["sha256"])})
        if number % 5000 == 0:
            print(f"matched {number}/{len(source)}", flush=True)
    with (ROOT / "dfire_near_matches.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "image", "category", "scene_key", "dfire_image", "dfire_split", "dhash_distance", "phash_distance", "sha_equal"))
        writer.writeheader()
        writer.writerows(matches)

    groups = defaultdict(list)
    sha_groups = defaultdict(list)
    for row in source:
        groups[(row["source"], row["scene_key"])].append(row)
        sha_groups[row["sha256"]].append(row)
    with (ROOT / "scene_groups.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("source", "scene_key", "count", "splits", "fire_images", "smoke_images", "other_only_images", "empty_images"))
        writer.writeheader()
        for (source_name, scene), group in sorted(groups.items()):
            writer.writerow({"source": source_name, "scene_key": scene, "count": len(group),
                             "splits": "+".join(sorted({r["split"] for r in group})),
                             "fire_images": sum(int(r["fire_boxes"]) > 0 for r in group),
                             "smoke_images": sum(int(r["smoke_boxes"]) > 0 for r in group),
                             "other_only_images": sum(r["category"] == "other" for r in group),
                             "empty_images": sum(r["category"] == "empty" for r in group)})
    cross = defaultdict(set)
    for match in matches:
        cross[(match["source"], match["image"])].add(match["dfire_split"])
    summary = {
        "source_images": len(source), "by_source": dict(Counter(row["source"] for row in source)),
        "category": dict(Counter(row["category"] for row in source)),
        "images_with_fire": sum(int(row["fire_boxes"]) > 0 for row in source),
        "images_with_smoke": sum(int(row["smoke_boxes"]) > 0 for row in source),
        "images_other_only": sum(row["category"] == "other" for row in source),
        "images_empty": sum(row["category"] == "empty" for row in source),
        "issues": dict(Counter(row["issue"] for row in source if row["issue"])),
        "scene_groups": len(groups),
        "groups_crossing_source_splits": sum(len({r["split"] for r in group}) > 1 for group in groups.values()),
        "images_in_cross_split_groups": sum(len(group) for group in groups.values() if len({r["split"] for r in group}) > 1),
        "duplicate_sha_groups": sum(len(group) > 1 for group in sha_groups.values()),
        "duplicate_sha_extra_images": sum(len(group) - 1 for group in sha_groups.values()),
        "exact_dfire_images": sum(row["sha256"] in by_sha for row in source),
        "exact_pyro_selected_images": sum(row["sha256"] in pyro for row in source),
        "near_dfire_images_dhash_le6": len(cross),
        "near_dfire_split_images": dict(Counter(split for splits in cross.values() for split in splits)),
        "near_dfire_images_dhash_le6_phash_le12": len({m["image"] for m in matches if int(m["phash_distance"]) <= 12}),
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
