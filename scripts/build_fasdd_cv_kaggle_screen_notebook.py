"""Tạo notebook tự chứa để sàng lọc FASDD_CV trên Kaggle."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/data_preparation/05_audit_fasdd_cv.ipynb"
TARGET = ROOT / "notebooks/data_preparation/20_screen_fasdd_cv_for_dfire_kaggle.ipynb"


def cell(kind, content):
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update({"execution_count": None, "outputs": []})
    return result


notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
for existing in notebook["cells"]:
    if existing["cell_type"] == "code":
        existing["execution_count"] = None
        existing["outputs"] = []

notebook["cells"][0] = cell("markdown", """# Sàng lọc FASDD_CV để cân nhắc bổ sung D-Fire

Notebook **chỉ đọc** dataset FASDD_CV đã gắn trong Kaggle Input. Không train, không sửa hay gộp dữ liệu. Mục tiêu là xác định pool ảnh chỉ-lửa có thể kiểm duyệt tiếp cho D-Fire (`0=smoke, 1=fire`, `imgsz=640`). Số ảnh được gắn nhãn chỉ-lửa **không phải** số ảnh đã được duyệt.

## Cách chạy

1. Tạo notebook Kaggle, tải file `.ipynb` này lên và Add Input dataset FASDD_CV trong ảnh bạn gửi. CPU và Internet off là đủ; không cần gắn D-Fire.
2. Chạy Run All. Việc đọc header khoảng 95 nghìn ảnh, đối chiếu COCO/YOLO và băm pool lửa có thể mất thời gian. Không xóa Kaggle Input.
3. Tải **`/kaggle/working/fasdd_cv_dfire_screen.zip`** ở Output và gửi lại. Gói chỉ gồm bảng thống kê, lỗi nhãn, ảnh ghép mẫu và danh sách cần duyệt; không chứa 12 GB ảnh gốc.

Các ảnh ghép chỉ giúp sàng lọc; sau khi xem kết quả, nếu nguồn có triển vọng mới chọn ảnh gốc và đối chiếu trùng/cùng cảnh với D-Fire train/val/test. Split của FASDD không được dùng thay val/test D-Fire. Một số báo cáo trung gian chuẩn hóa lớp thành `0=fire, 1=smoke` để đối chiếu nhãn FASDD; notebook không xuất nhãn train và khi gộp về sau phải đổi sang thứ tự D-Fire.
""")

setup = "".join(notebook["cells"][2]["source"])
setup = setup.replace('OUTPUT_ROOT = Path("/kaggle/working/fasdd_cv_audit")', 'OUTPUT_ROOT = Path("/kaggle/working/fasdd_cv_dfire_screen")')
notebook["cells"][2]["source"] = setup.splitlines(keepends=True)

strata = "".join(notebook["cells"][10]["source"])
strata = strata.replace("SAMPLES_PER_STRATUM=12", "SAMPLES_PER_STRATUM=24")
notebook["cells"][10]["source"] = strata.splitlines(keepends=True)

final = "".join(notebook["cells"][12]["source"])
final = final[: final.index('report_archive=Path("/kaggle/working/fasdd_cv_audit_05.zip")')]
final = final.replace(
    "deterministic sample in notebook 05; all images during notebook 06 hashing",
    "deterministic sample in this notebook; fire-only pool SHA-256 in the last cell",
)
notebook["cells"][12]["source"] = final.splitlines(keepends=True)

notebook["cells"].append(cell("markdown", """## 6. Mức phù hợp với D-Fire và ảnh chỉ-lửa

Thống kê dưới đây dùng box COCO hợp lệ sau kiểm tra hình học. Box được quy đổi bằng **tỉ lệ cạnh dài về 640**; đây là kích thước hình học trước padding, không khẳng định vật thể nhìn rõ hay nhãn đúng. Pool chỉ-lửa được phân theo split nguồn để phát hiện tập trung/lặp, nhưng split nguồn không quyết định tập đích.

Phần SHA-256 chỉ băm tối đa 20.000 ảnh chỉ-lửa theo mẫu cố định, để tránh đọc lại toàn bộ 12 GB. Báo cáo ghi rõ đã băm toàn bộ pool hay chỉ một phần. Trùng byte và tên gốc chuẩn hóa chỉ là tín hiệu; cùng cảnh khác khung/hình cắt vẫn phải duyệt bằng mắt.
"""))

notebook["cells"].append(cell("code", r'''import re
import statistics

def quantile(values, probability):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)

def normalized_stem(name):
    stem = Path(str(name)).stem.lower()
    stem = re.sub(r"\.rf\.[0-9a-f]{8,}$", "", stem)
    stem = re.sub(r"(?:_jpg|_jpeg|_png)$", "", stem)
    return stem

fire_id = name_to_id["fire"]
smoke_id = name_to_id["smoke"]
category_rows = []
fire_pool = []
fire_box_short_640 = []
fire_box_area_fraction = []
size_counts = collections.Counter()
stem_groups = collections.defaultdict(list)
issue_keys = {(issue.get("split"), issue.get("image_id")) for issue in issues}
for split, (images, annotations_by_image, _) in valid_by_split.items():
    counts = collections.Counter()
    for image_id, image_info in images.items():
        key = (split, image_id)
        header = image_headers.get(key, {})
        annotations = annotations_by_image.get(image_id, [])
        class_ids = {int(annotation["category_id"]) for annotation in annotations}
        has_issue = key in issue_keys
        if not header.get("width"):
            category = "unreadable_or_missing"
        elif has_issue:
            category = "annotation_issue"
        elif class_ids == {fire_id}:
            category = "fire_only"
        elif class_ids == {smoke_id}:
            category = "smoke_only"
        elif class_ids == {fire_id, smoke_id}:
            category = "fire_and_smoke"
        elif not class_ids:
            category = "background"
        else:
            category = "other_class"
        counts[category] += 1
        filename = str(image_info.get("file_name", ""))
        stem_groups[normalized_stem(filename)].append((split, image_id, filename))
        if category == "fire_only":
            width, height = int(header["width"]), int(header["height"])
            fire_pool.append({"split": split, "image_id": image_id, "file_name": filename,
                              "width": width, "height": height, "path": str(header["path"]),
                              "fire_boxes": len(annotations)})
            size_counts[f"{width}x{height}"] += 1
        if header.get("width") and header.get("height"):
            width, height = int(header["width"]), int(header["height"])
            factor = 640 / max(width, height)
            for annotation in annotations:
                if int(annotation["category_id"]) == fire_id:
                    _, _, box_width, box_height = map(float, annotation["bbox"])
                    fire_box_short_640.append(min(box_width, box_height) * factor)
                    fire_box_area_fraction.append(box_width * box_height / (width * height))
    category_rows.append({"split": split, **{name: counts[name] for name in
                         ("fire_only", "smoke_only", "fire_and_smoke", "background",
                          "annotation_issue", "unreadable_or_missing", "other_class")},
                          "total": len(images)})

fire_stems = collections.defaultdict(list)
for row in fire_pool:
    fire_stems[normalized_stem(row["file_name"])].append(row)
repeated_fire_stems = [{"normalized_stem": stem, "count": len(rows),
                        "splits": ",".join(sorted({row["split"] for row in rows})),
                        "example_names": " | ".join(row["file_name"] for row in rows[:5])}
                       for stem, rows in fire_stems.items() if len(rows) > 1]
repeated_fire_stems.sort(key=lambda row: (-row["count"], row["normalized_stem"]))

fire_summary = {
    "category_counts_by_source_split": category_rows,
    "fire_only_labeled_images": len(fire_pool),
    "fire_only_normalized_stems": len(fire_stems),
    "fire_only_repeated_stem_groups": len(repeated_fire_stems),
    "fire_only_repeated_stem_images": sum(row["count"] for row in repeated_fire_stems),
    "fire_only_top_dimensions": size_counts.most_common(25),
    "fire_only_long_side_below_640": sum(max(row["width"], row["height"]) < 640 for row in fire_pool),
    "fire_only_short_side_below_360": sum(min(row["width"], row["height"]) < 360 for row in fire_pool),
    "fire_box_count": len(fire_box_short_640),
    "fire_box_short_side_at_640_p10_p25_p50_p75_p90":
        [quantile(fire_box_short_640, p) for p in (0.10, 0.25, 0.50, 0.75, 0.90)],
    "fire_box_short_side_at_640_under_16px": sum(value < 16 for value in fire_box_short_640),
    "fire_box_short_side_at_640_under_32px": sum(value < 32 for value in fire_box_short_640),
    "fire_box_area_fraction_p10_p50_p90":
        [quantile(fire_box_area_fraction, p) for p in (0.10, 0.50, 0.90)],
    "meaning": "Thống kê pool nguồn, chưa phải ảnh được duyệt để bổ sung D-Fire."
}
write_csv(OUTPUT_ROOT / "dfire_fit_category_counts.csv", category_rows)
write_csv(OUTPUT_ROOT / "fire_only_pool.csv", [{key: value for key, value in row.items() if key != "path"} for row in fire_pool])
write_csv(OUTPUT_ROOT / "fire_only_repeated_stems.csv", repeated_fire_stems)
(OUTPUT_ROOT / "dfire_fit_summary.json").write_text(
    json.dumps(fire_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(fire_summary, ensure_ascii=False, indent=2))
'''))

notebook["cells"].append(cell("code", r'''MAX_FIRE_HASH_IMAGES = 20000
ordered_fire_pool = sorted(fire_pool, key=lambda row: (row["split"], row["file_name"], row["image_id"]))
if len(ordered_fire_pool) > MAX_FIRE_HASH_IMAGES:
    randomizer = random.Random(SEED)
    hashed_pool = sorted(randomizer.sample(ordered_fire_pool, MAX_FIRE_HASH_IMAGES),
                         key=lambda row: (row["split"], row["file_name"]))
else:
    hashed_pool = ordered_fire_pool

hash_groups = collections.defaultdict(list)
hash_errors = []
for row in tqdm(hashed_pool, desc="SHA-256 pool chỉ-lửa", unit="ảnh"):
    try:
        digest = sha256_file(Path(row["path"]))
        hash_groups[digest].append(row)
    except Exception as error:
        hash_errors.append({"split": row["split"], "file_name": row["file_name"],
                            "error": f"{type(error).__name__}: {error}"})

duplicate_hash_rows = []
for digest, rows in hash_groups.items():
    if len(rows) > 1:
        duplicate_hash_rows.append({"sha256": digest, "count": len(rows),
                                    "splits": ",".join(sorted({row["split"] for row in rows})),
                                    "names": " | ".join(row["file_name"] for row in rows[:10])})
duplicate_hash_rows.sort(key=lambda row: -row["count"])
write_csv(OUTPUT_ROOT / "fire_only_exact_duplicate_groups.csv", duplicate_hash_rows)
write_csv(OUTPUT_ROOT / "fire_only_hash_errors.csv", hash_errors)
hash_summary = {
    "fire_only_pool_images": len(fire_pool),
    "hashed_images": len(hashed_pool),
    "coverage": "complete" if len(hashed_pool) == len(fire_pool) else "partial_sample",
    "exact_duplicate_groups_in_hashed_scope": len(duplicate_hash_rows),
    "exact_duplicate_images_in_hashed_scope": sum(row["count"] for row in duplicate_hash_rows),
    "hash_errors": len(hash_errors),
    "limitation": "Chỉ tìm trùng byte trong pool chỉ-lửa đã băm; không đối chiếu D-Fire, không phát hiện ảnh tái mã hóa/cùng cảnh."
}
(OUTPUT_ROOT / "fire_only_hash_summary.json").write_text(
    json.dumps(hash_summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(hash_summary, ensure_ascii=False, indent=2))

readme = """FASDD_CV -> D-Fire screening (read-only)
=========================================
Open dfire_fit_summary.json for counts and 640-pixel box geometry.
Open fire_only_hash_summary.json to see whether exact hashing was complete or partial.
Open manual_review_template.csv and contact sheets to review image realism, smoke-label completeness and repeated scenes.
No images were approved, merged or trained. FASDD source splits are not D-Fire val/test.
Exact or same-scene overlap with D-Fire has not been checked because D-Fire was not attached.
The source Kaggle page in the user's screenshot displays License: Unknown; verify upstream rights before redistribution.
"""
(OUTPUT_ROOT / "README.txt").write_text(readme, encoding="utf-8")
report_archive = Path("/kaggle/working/fasdd_cv_dfire_screen.zip")
if report_archive.exists():
    report_archive.unlink()
with zipfile.ZipFile(report_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for file_path in sorted(OUTPUT_ROOT.rglob("*")):
        if file_path.is_file():
            archive.write(file_path, file_path.relative_to(OUTPUT_ROOT))
with zipfile.ZipFile(report_archive) as archive:
    failed_member = archive.testzip()
if failed_member:
    raise RuntimeError(f"ZIP báo cáo lỗi CRC: {failed_member}")
print("Gói kết quả cần gửi:", report_archive)
print("Dung lượng:", round(report_archive.stat().st_size / 1024 / 1024, 2), "MB")
'''))

notebook["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python", "version": "3.11"}}
notebook["nbformat_minor"] = 5
for index, existing in enumerate(notebook["cells"]):
    existing["id"] = f"fasdd-screen-{index:02d}"
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
