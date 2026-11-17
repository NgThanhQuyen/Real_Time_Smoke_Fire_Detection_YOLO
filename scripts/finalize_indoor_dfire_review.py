"""Chốt danh sách Indoor đã xem bằng mắt để chủ dự án quyết định bổ sung."""

import csv
import json
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/indoor_dfire_original_review"
SOURCE = Path("F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip")

PRIMARY = {"I0626", "I0810", "I1748"}
OPTIONAL_CONTROLLED = {"I1091", "I1173"}
NEEDS_REVIEW = {
    "I0007": "khói hoặc vùng mờ chưa gắn nhãn",
    "I0087": "box lửa chưa bao hết vùng cháy",
    "I0095": "khói nhìn thấy nhưng nguồn chỉ gắn lửa",
    "I0100": "box lửa phân mảnh, cần duyệt lại toàn ảnh",
    "I0161": "khói/haze trong phòng chưa gắn nhãn",
    "I0359": "cảnh thử nghiệm và có thể thiếu nhãn khói",
    "I0481": "khói/haze trong phòng chưa gắn nhãn",
    "I0608": "khói/haze rõ, cần gắn nhãn khói",
    "I1019": "lửa cháy sáng quá mức, cần xác minh nhãn trên khung gốc",
    "I1038": "box lửa bỏ sót phần chóp ngọn lửa",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def box_error(data: bytes) -> bool:
    tolerance = 1e-5
    for line in data.decode("utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) != 5:
            return True
        try:
            label, x, y, width, height = map(float, parts)
        except ValueError:
            return True
        if label != 0 or width <= 0 or height <= 0:
            return True
        if x - width / 2 < -tolerance or y - height / 2 < -tolerance:
            return True
        if x + width / 2 > 1 + tolerance or y + height / 2 > 1 + tolerance:
            return True
    return False


def main() -> None:
    rows = read_csv(REPORT / "full_review/index.csv")
    nearest = {row["indoor"]: row for row in read_csv(REPORT / "nearest_dfire.csv")}
    overlap = {row["indoor"] for row in read_csv(REPORT / "perceptual_candidates.csv")}
    reviewed_ids = PRIMARY | OPTIONAL_CONTROLLED | NEEDS_REVIEW.keys()
    assert len(rows) == 2165
    assert reviewed_ids <= {row["review_id"] for row in rows}
    assert not (reviewed_ids & {row["review_id"] for row in rows if row["image"] in overlap})

    output = []
    with zipfile.ZipFile(SOURCE) as archive:
        for row in rows:
            review_id = row["review_id"]
            geometry_error = box_error(archive.read(row["label"]))
            near = nearest[row["image"]]
            if geometry_error:
                status, reason = "invalid_box", "box nguồn lỗi hình học"
            elif row["image"] in overlap:
                status, reason = "dfire_same_frame", "cùng khung hình với D-Fire hiện tại"
            elif review_id in PRIMARY:
                status, reason = "recommended_realistic", "cảnh cháy camera/hiện trường, box hợp lệ"
            elif review_id in OPTIONAL_CONTROLLED:
                status, reason = "optional_controlled", "camera thật nhưng lửa đốt trong phòng thí nghiệm"
            elif review_id in NEEDS_REVIEW:
                status, reason = "needs_relabel_or_review", NEEDS_REVIEW[review_id]
            else:
                status, reason = "not_prioritized_after_overview", "không nằm trong danh sách ưu tiên sau xem toàn bộ bảng ảnh"
            output.append({
                "review_id": review_id,
                "image": row["image"],
                "label": row["label"],
                "source_split": row["source_split"],
                "width": row["width"],
                "height": row["height"],
                "fire_boxes": row["fire_boxes"],
                "sha256": row["sha256"],
                "dhash": row["dhash"],
                "phash": row["phash"],
                "nearest_dfire": near["nearest_dfire"],
                "nearest_dfire_split": near["nearest_split"],
                "nearest_dhash": near["nearest_dhash"],
                "nearest_phash": near["nearest_phash"],
                "status": status,
                "reason": reason,
            })

    fieldnames = list(output[0])
    with (REPORT / "curation_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)
    proposed = [row for row in output if row["status"] in {"recommended_realistic", "optional_controlled"}]
    with (REPORT / "proposed_images.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(proposed)

    counts = dict(Counter(row["status"] for row in output))
    summary = {
        "source_fire_only": len(output),
        "counts": counts,
        "proposed_total_including_controlled": len(proposed),
        "proposed_fire_boxes": sum(int(row["fire_boxes"]) for row in proposed),
        "proposed_source_splits": dict(Counter(row["source_split"] for row in proposed)),
        "dfire_train_fire_only_before": 686,
        "dfire_train_fire_only_if_all_proposed_added": 686 + len(proposed),
        "dfire_val_test_unchanged": True,
    }
    (REPORT / "curation_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
