"""Chốt mẫu đã xem đủ kích thước và kiểm tra trùng trong đợt nhận."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


ROOT = Path("runtime/catargiu_audit")
REVIEW = ROOT / "review_pages" / "visual_review.csv"

FIRE_REJECT = {
    3: "visible_unlabeled_smoke", 7: "visible_unlabeled_smoke",
    17: "visible_unlabeled_smoke", 23: "visible_unlabeled_smoke",
    25: "visible_unlabeled_smoke", 43: "visible_unlabeled_smoke",
    64: "blurred_fire", 87: "unlabeled_extra_fire",
    90: "possible_unlabeled_smoke", 92: "unlabeled_extra_fire",
    95: "stock_watermark", 105: "possible_unlabeled_smoke",
    151: "unlabeled_extra_fire", 165: "stock_watermark",
    170: "possible_unlabeled_smoke", 195: "possible_unlabeled_smoke",
    210: "visible_unlabeled_smoke", 222: "unlabeled_extra_fire",
    224: "unlabeled_extra_fire", 230: "text_overlay",
    237: "unlabeled_extra_fire", 239: "ambiguous_staged_fire",
}

NEGATIVE_REJECT = {
    17: "ambiguous_roof_plume", 19: "google_watermark",
    25: "source_watermark", 29: "google_watermark",
    30: "source_watermark", 40: "video_text_overlay",
    41: "google_watermark", 42: "google_watermark",
    45: "source_watermark", 61: "source_watermark",
    62: "google_watermark", 79: "active_fire_response_ambiguous",
    100: "source_watermark_and_illustration",
    101: "active_fire_response_ambiguous",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rows = read_csv(REVIEW)
    for row in rows:
        if row["decision"] != "provisional":
            continue
        kind, number_text = row["review_id"].split(":")
        number = int(number_text)
        rejection = (FIRE_REJECT if kind == "fire_only" else NEGATIVE_REJECT).get(number)
        row["decision"] = "reject" if rejection else "accept"
        row["reason"] = rejection or "manual_full_resolution_review"

    index = {(row["source"], row["image"]): row for row in read_csv(ROOT / "image_index.csv")}
    accepted = [row for row in rows if row["decision"] == "accept"]
    seen_sha: dict[str, str] = {}
    for row in accepted:
        digest = row["sha256"]
        if digest in seen_sha:
            row["decision"] = "reject"
            row["reason"] = f"same_sha_as_{seen_sha[digest]}"
        else:
            seen_sha[digest] = row["review_id"]

    accepted = [row for row in rows if row["decision"] == "accept"]
    by_image = {(row["source"], row["image"]): row for row in accepted}
    for match in read_csv(ROOT / "dfire_near_matches.csv"):
        row = by_image.get((match["source"], match["image"]))
        if row is None or row["decision"] != "accept":
            continue
        d = int(match["dhash_distance"])
        p = int(match["phash_distance"])
        split = match["dfire_split"]
        if (d <= 6 and p <= 12) or (split in ("val", "test") and d <= 3 and p <= 20) or (d <= 2 and p <= 18):
            row["decision"] = "reject"
            row["reason"] = f"conservative_dfire_{split}_hash_candidate_d{d}_p{p}"

    accepted = [row for row in rows if row["decision"] == "accept"]
    pairs = []
    for i, first in enumerate(accepted):
        a = index[first["source"], first["image"]]
        for second in accepted[i + 1:]:
            b = index[second["source"], second["image"]]
            d = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            p = (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count()
            if d <= 8 or p <= 12:
                pairs.append({"first": first["review_id"], "second": second["review_id"],
                              "dhash_distance": d, "phash_distance": p,
                              "first_scene": first["scene_key"], "second_scene": second["scene_key"]})
    write_csv(REVIEW, rows)
    if pairs:
        write_csv(ROOT / "review_pages" / "accepted_hash_pairs.csv", pairs)
    print("review", dict(Counter((r["review_id"].split(":")[0], r["decision"]) for r in rows)))
    print("accepted hash candidates", len(pairs))
    for pair in pairs[:40]:
        print(pair)


if __name__ == "__main__":
    main()
