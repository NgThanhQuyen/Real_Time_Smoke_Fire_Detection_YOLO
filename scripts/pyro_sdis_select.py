"""Ghi quyết định tuyển chọn Pyro-SDIS và đối chiếu hash với D-Fire gốc."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


SMOKE_ACCEPT = {1, 2, 3, 5, 6, 7, 8, 10, 12, 13, 14, 18, 19, 20, 24, 25, 27, 28, 29, 31, 32, 35}
NEGATIVE_ACCEPT = set(range(38, 46)) | set(range(50, 59)) | set(range(60, 68)) | {70, 73, 74, 75} | set(range(76, 98))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path("reports/pyro_sdis_v2_audit")
    candidates = read_rows(root / "candidate_review_1.csv")
    dfire = read_rows(Path("runtime/dfire_raw_review/image_index.csv"))
    dfire_dh = [int(row["dhash"], 16) for row in dfire]
    dfire_ph = [int(row["phash"], 16) for row in dfire]
    accepted = []
    for position, row in enumerate(candidates, 1):
        if position not in SMOKE_ACCEPT | NEGATIVE_ACCEPT:
            continue
        if row["issue"] or row["dfire_exact_match"] != "0":
            raise ValueError(f"Ứng viên có lỗi nhãn/ảnh hoặc trùng D-Fire: {row['image_name']}")
        dh = int(row["dhash"], 16)
        ph = int(row["phash"], 16)
        dh_dist = [(value ^ dh).bit_count() for value in dfire_dh]
        ph_dist = [(value ^ ph).bit_count() for value in dfire_ph]
        nearest = np.argsort([a + b / 4 for a, b in zip(dh_dist, ph_dist)])[:3]
        best = nearest[0]
        row = dict(row)
        row["candidate_number"] = position
        row["decision"] = "accept_smoke" if position in SMOKE_ACCEPT else "accept_negative"
        row["nearest_dfire_image"] = dfire[best]["image"]
        row["nearest_dfire_dhash"] = int(dh_dist[best])
        row["nearest_dfire_phash"] = int(ph_dist[best])
        row["dfire_strong_hash_candidate"] = int(any(dh_dist[i] <= 6 and ph_dist[i] <= 12 for i in nearest))
        accepted.append(row)
    fields = list(accepted[0])
    with (root / "accepted_provisional.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(accepted)
    print(f"accepted={len(accepted)}; strong_hash_candidates={sum(int(r['dfire_strong_hash_candidate']) for r in accepted)}")


if __name__ == "__main__":
    main()
