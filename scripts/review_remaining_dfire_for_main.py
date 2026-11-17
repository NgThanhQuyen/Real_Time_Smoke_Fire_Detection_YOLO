"""Tổng hợp ảnh D-Fire còn lại sau khi duyệt cặp nghi rò rỉ."""

from collections import Counter
from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/dfire_remaining_for_main"

# Các ảnh cùng cảnh với valid/test hiện tại, xác nhận bằng ảnh ghép kiểm duyệt.
SAME_SCENE = set("""
PublicDataset01008 WEB04033 WEB01452 WEB01588 WEB01595 WEB01687 WEB01949
WEB02003 WEB02006 WEB02008 WEB03654 WEB03665
WEB00292 WEB00294 WEB01370 WEB01373 WEB01372 WEB01374 WEB04038 WEB01439
WEB00245 WEB00235 WEB01283 WEB01280 WEB01281 WEB00241 WEB02035 WEB04063
WEB02045 WEB01593 WEB02032 WEB06038 WEB01592 WEB02029 WEB02018 WEB02007
WEB02004 WEB03667 WEB02034 WEB00239 WEB00240 WEB01967 WEB01586 WEB01587
WEB01641 WEB00521 WEB00507 WEB00359 WEB01058 WEB00503 WEB00504 WEB01240
WEB00516 WEB00602 WEB00603 WEB00523 WEB00599
""".split())


def main() -> None:
    with (REPORT / "candidates.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    found = {Path(row["source_image"]).stem for row in rows}
    missing = SAME_SCENE - found
    if missing:
        raise RuntimeError(f"Thiếu ảnh đã duyệt: {sorted(missing)}")

    for row in rows:
        if row["exact_duplicate_current"] == "1" or row["exact_duplicate_pool"] == "1":
            decision = "already_present_exact"
        elif Path(row["source_image"]).stem in SAME_SCENE:
            decision = "exclude_same_scene_locked_split"
        elif row["label_status"] != "ok":
            decision = "label_needs_correction"
        else:
            decision = "available_after_review"
        row["review_decision"] = decision

    with (REPORT / "reviewed_candidates.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    counts = {category: dict(Counter(row["review_decision"] for row in rows
                                     if row["category"] == category))
              for category in ("smoke_only", "both", "background")}
    summary = {
        "starting_dataset": "runtime/fire_smoke_detection_yolov11_bg_dfire2000_v1",
        "destination_split_if_selected": "train",
        "counts": counts,
        "notes": [
            "Ảnh trùng 100% được xác định bằng SHA-256 trên byte ảnh.",
            "Ảnh cùng cảnh valid/test được loại sau khi duyệt ảnh ghép bằng mắt, kể cả khác byte.",
            "Chỉ khảo sát D-Fire clean v2 train và 51 khung train bị loại trước vì gần giống trong train.",
            "Một nhãn của ảnh cả lửa lẫn khói có box vượt biên, cần sửa trước khi dùng.",
            "Sàng lọc perceptual hash không chứng minh tuyệt đối rằng mọi cảnh trùng đã được phát hiện.",
            "Chưa thay đổi bộ dữ liệu hoặc chọn số ảnh để bổ sung.",
        ],
    }
    (REPORT / "review_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
