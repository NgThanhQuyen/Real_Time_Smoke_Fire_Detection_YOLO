"""Tạo hàng đợi duyệt thủ công, không tự nhận ảnh vào tập huấn luyện."""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("runtime/catargiu_audit")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    images = read_csv(ROOT / "image_index.csv")
    matches = read_csv(ROOT / "dfire_near_matches.csv")
    dfire = read_csv(Path("runtime/dfire_clean_v2/metadata/manifest.csv"))
    known_names = {Path(row["source_image"]).stem for row in dfire}
    groups = defaultdict(list)
    blocked = defaultdict(set)
    for row in images:
        key = (row["source"], row["scene_key"])
        groups[key].append(row)
        stem = Path(row["image"]).stem.split("_jpg.rf.")[0]
        if stem in known_names:
            blocked[key].add("dfire_filename")
        if stem.startswith(("WEB", "AoF", "PublicDataset", "MirrorWEB")):
            blocked[key].add("dfire_lineage_prefix")
        if re.match(r"(?i)^frame[_-]?\d", stem) or stem.startswith(("FrameVideoManual", "image_")):
            blocked[key].add("unknown_frame_sequence")
        if row["issue"]:
            blocked[key].add("file_or_label_issue")
    for row in matches:
        key = (row["source"], row["scene_key"])
        dhash = int(row["dhash_distance"])
        phash = int(row["phash_distance"])
        if dhash <= 3 and phash <= 8:
            blocked[key].add("strong_dfire_hash_candidate")
        elif row["dfire_split"] in ("val", "test") and dhash <= 6 and phash <= 12:
            blocked[key].add("val_test_hash_candidate")
    queue = []
    seen_sha = set()
    for key, group in sorted(groups.items()):
        if key in blocked:
            continue
        fire = sorted((row for row in group if row["category"] == "fire"), key=lambda row: row["image"])
        positive_group = any(int(row["fire_boxes"]) or int(row["smoke_boxes"]) for row in group)
        negative = [] if positive_group else sorted((row for row in group if row["category"] == "other"), key=lambda row: row["image"])
        for kind, candidates in (("fire_only", fire), ("hard_negative", negative)):
            if not candidates:
                continue
            row = candidates[len(candidates) // 2]
            if row["sha256"] in seen_sha:
                continue
            seen_sha.add(row["sha256"])
            queue.append({"priority": kind, "source": row["source"], "image": row["image"],
                          "scene_key": row["scene_key"], "source_split": row["split"],
                          "scene_frames": len(group), "sha256": row["sha256"],
                          "review_status": "pending_visual_review"})
    path = ROOT / "review_queue.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("priority", "source", "image", "scene_key", "source_split", "scene_frames", "sha256", "review_status"))
        writer.writeheader()
        writer.writerows(queue)
    print("queue", len(queue), dict(Counter(row["priority"] for row in queue)))
    print("blocked groups", len(blocked), "reasons", dict(Counter(reason for reasons in blocked.values() for reason in reasons)))


if __name__ == "__main__":
    main()
