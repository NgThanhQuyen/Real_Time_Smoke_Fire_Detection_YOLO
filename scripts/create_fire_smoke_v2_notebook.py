"""Tạo notebook minh chứng quy trình xử lý data 2."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "data_preparation" / "17_build_fire_smoke_v2_grouped_bg.ipynb"


def main() -> None:
    pipeline_source = (ROOT / "scripts" / "build_fire_smoke_v2_grouped_bg.py").read_text(encoding="utf-8")
    pipeline_source = pipeline_source.split('\nif __name__ == "__main__":', 1)[0]
    cells = [
        nbf.v4.new_markdown_cell(
            """# Xử lý Fire and Smoke Detection v2: chia theo nguồn và thêm ảnh nền

Notebook này là chứng cứ cho bộ `runtime/Smoke-Fire-Detection-Dataset.zip` dùng để thử nghiệm YOLO11 detection. Nó cho thấy dữ liệu nguồn, quy tắc phân nhóm, cách chạy pipeline, thống kê trước/sau và kiểm tra ZIP đầu ra. Mã xử lý được chép nguyên từ `scripts/build_fire_smoke_v2_grouped_bg.py` vào ô bên dưới, nên có thể xem và chạy lại ngay trong notebook.

**Nguồn:** ZIP Roboflow v2 người dùng cung cấp và ảnh nền đã gắn nhãn rỗng trong `runtime/fire_smoke_dfire_final.zip`. Lớp `0 = Fire`, `1 = Smoke`. ZIP gốc không bị sửa. Notebook này không huấn luyện mô hình.

Để tái lập, đặt hai ZIP nguồn đúng đường dẫn ở ô cấu hình và đổi `RUN_BUILD = True`. Chọn đường dẫn ZIP đầu ra mới để không ghi đè bản đã kiểm chứng. Nếu chỉ cần xem chứng cứ, chạy với `RUN_BUILD = False`; các ô sau sẽ kiểm tra bản ZIP hiện có."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
from zipfile import ZipFile
from collections import Counter
from hashlib import sha256
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as NotebookImage, display


def find_project_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "codex.md").is_file() and (candidate / "runtime").is_dir():
            return candidate
    raise FileNotFoundError("Không tìm thấy thư mục gốc dự án; hãy đặt notebook trong repository")


ROOT = find_project_root(Path.cwd().resolve())
SOURCE_ZIP = Path(r"F:\\6\\KLTN\\Smoke-Fire-Dataset\\fire and smoke detection.v2i.yolov11.zip")
BACKGROUND_ZIP = ROOT / "runtime" / "fire_smoke_dfire_final.zip"
DATASET_ZIP = ROOT / "runtime" / "Smoke-Fire-Detection-Dataset.zip"
REPORT_DIR = ROOT / "reports" / "fire_smoke_v2_grouped_bg_v1"
SCRIPT = ROOT / "scripts" / "build_fire_smoke_v2_grouped_bg.py"
RUN_BUILD = False
__file__ = str(SCRIPT)

print("Nguồn dương tính:", SOURCE_ZIP)
print("Nguồn ảnh nền:", BACKGROUND_ZIP)
print("ZIP đầu ra:", DATASET_ZIP)
assert SOURCE_ZIP.is_file() and BACKGROUND_ZIP.is_file()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Đọc dữ liệu gốc

Đếm ảnh, ảnh chỉ lửa, chỉ khói, cả hai và ảnh nền trực tiếp từ từng file nhãn YOLO trong ZIP gốc. Bước tạo dữ liệu của script còn giải mã toàn bộ ảnh, kiểm tra tọa độ, CRC và SHA-256."""
        ),
        nbf.v4.new_code_cell(
            """source_rows = []
with ZipFile(SOURCE_ZIP) as archive:
    names = set(archive.namelist())
    image_paths = sorted(
        name for name in names
        if "/images/" in name and Path(name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    for image_path in image_paths:
        label_path = image_path.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        assert label_path in names, image_path
        classes = {
            int(line.split()[0])
            for line in archive.read(label_path).decode("utf-8-sig").splitlines()
            if line.strip()
        }
        category = (
            "Chỉ lửa" if classes == {0} else
            "Chỉ khói" if classes == {1} else
            "Cả hai" if classes == {0, 1} else "Nền"
        )
        source_rows.append({"image": image_path, "split": image_path.split("/", 1)[0], "category": category})

source_df = pd.DataFrame(source_rows)
category_order = ["Chỉ lửa", "Chỉ khói", "Cả hai", "Nền"]
split_order = ["train", "valid", "test"]
before = pd.crosstab(source_df["split"], source_df["category"]).reindex(index=split_order, columns=category_order, fill_value=0)
before["Tổng ảnh"] = before.sum(axis=1)
display(before)
print("Tổng ảnh gốc:", len(source_df))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Quy tắc chống rò rỉ theo tên nguồn

Mã bên dưới gom mọi khung có cùng mã `video_N` vào một nhóm. Với tên `youtube-*`, mã bỏ hậu tố Roboflow `.rf.<hash>` rồi gom các biến thể có cùng tên gốc. Bộ chia tối ưu tỷ lệ gần 80:10:10 và cân đối bốn loại ảnh. Đây là quy tắc có thể kiểm tra lại từ manifest; nó không chứng minh đã phát hiện hết các ảnh cùng cảnh mang tên khác.

Ô mã này chứa toàn bộ các hàm đọc, kiểm tra, chia nhóm, chọn ảnh nền và xuất ZIP. Nội dung được đồng bộ từ script dòng lệnh khi tạo notebook."""
        ),
        nbf.v4.new_code_cell(pipeline_source),
        nbf.v4.new_code_cell(
            """if RUN_BUILD:
    rebuild_zip = ROOT / "runtime" / "fire_smoke_v2_grouped_bg_rebuild.zip"
    rebuild_report = ROOT / "reports" / "fire_smoke_v2_grouped_bg_rebuild"
    assert not rebuild_zip.exists(), f"Đầu ra tái lập đã tồn tại: {rebuild_zip}"
    SOURCE, BACKGROUND_SOURCE = SOURCE_ZIP, BACKGROUND_ZIP
    OUTPUT, REPORT = rebuild_zip, rebuild_report
    main()
    DATASET_ZIP, REPORT_DIR = rebuild_zip, rebuild_report
    print("Đã tạo lại:", DATASET_ZIP)
else:
    print("Đang kiểm tra bản đã tạo. Đặt RUN_BUILD = True để chạy lại pipeline vào ZIP riêng.")

manifest_path = REPORT_DIR / "manifest.csv"
summary_path = REPORT_DIR / "summary.json"
assert DATASET_ZIP.is_file() and manifest_path.is_file() and summary_path.is_file()
manifest = pd.read_csv(manifest_path, keep_default_na=False)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("Số dòng manifest:", len(manifest), "| Seed:", summary["seed"])
display(manifest.head(5))"""
        ),
        nbf.v4.new_code_cell(
            """positive_rows = manifest[manifest["source_dataset"] == "fire_and_smoke_detection_v2"].copy()
background_rows = manifest[manifest["source_dataset"] == "dfire_final_background"].copy()
assert set(positive_rows["source_image"]) == set(source_df["image"])

old_cross = positive_rows.groupby("group")["old_split"].nunique()
new_cross = positive_rows.groupby("group")["new_split"].nunique()
crossing_before = int((old_cross > 1).sum())
crossing_after = int((new_cross > 1).sum())
assert crossing_after == 0
assert positive_rows["group"].nunique() == summary["group_count"]
assert len(background_rows) == summary["background_images_added"]

display(pd.DataFrame({
    "Chỉ tiêu": ["Nhóm nguồn", "Nhóm xuyên tập trước", "Nhóm xuyên tập sau", "Ảnh dương tính đổi tập", "Ảnh nền được thêm"],
    "Giá trị": [len(old_cross), crossing_before, crossing_after,
                int((positive_rows["old_split"] != positive_rows["new_split"]).sum()), len(background_rows)],
}))
display(pd.crosstab(positive_rows["family"], positive_rows["new_split"]))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Chọn ảnh nền

Script chọn theo seed cố định 1.500 ảnh nền train, 150 valid và 150 test từ bản D-Fire final. Mọi ảnh được thêm phải có nhãn rỗng, giải mã được và không trùng SHA-256 với ảnh đã chọn. Hai ảnh nền vốn có trong data 2 vẫn được giữ trong nhóm nguồn của chúng."""
        ),
        nbf.v4.new_code_cell(
            """background_added = background_rows["new_split"].value_counts().reindex(split_order, fill_value=0)
original_background = positive_rows.loc[positive_rows["category"] == "background", "new_split"].value_counts().reindex(split_order, fill_value=0)
background_table = pd.DataFrame({
    "Nền thêm từ D-Fire": background_added,
    "Nền gốc data 2": original_background,
})
background_table["Tổng nền"] = background_table.sum(axis=1)
display(background_table)
assert background_added.to_dict() == summary["background_quota"]"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Thống kê bộ dữ liệu mới

Ảnh có cả lửa và khói được tính một lần ở cột “Cả hai”, nhưng được tính vào cả “Ảnh chứa lửa” và “Ảnh chứa khói”."""
        ),
        nbf.v4.new_code_cell(
            """category_names = {"fire_only": "Chỉ lửa", "smoke_only": "Chỉ khói", "both": "Cả hai", "background": "Nền"}
after = pd.crosstab(manifest["new_split"], manifest["category"]).reindex(
    index=split_order, columns=list(category_names), fill_value=0
).rename(columns=category_names)
after["Tổng ảnh"] = after.sum(axis=1)
after["Ảnh chứa lửa"] = after["Chỉ lửa"] + after["Cả hai"]
after["Ảnh chứa khói"] = after["Chỉ khói"] + after["Cả hai"]
after["Box lửa"] = manifest.groupby("new_split")["fire_boxes"].sum().reindex(split_order)
after["Box khói"] = manifest.groupby("new_split")["smoke_boxes"].sum().reindex(split_order)
display(after)
display(after.sum().to_frame("Tổng ba tập").T)
for split in split_order:
    assert int(after.loc[split, "Tổng ảnh"]) == summary["splits"][split]["images"]
    assert int(after.loc[split, "Nền"]) == summary["splits"][split]["categories"]["background"]"""
        ),
        nbf.v4.new_code_cell(
            """colors = ["#d95f02", "#4c78a8", "#7a5195", "#7f8c8d"]
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
before[category_order].plot.bar(stacked=True, ax=axes[0], color=colors, rot=0)
after[category_order].plot.bar(stacked=True, ax=axes[1], color=colors, rot=0)
axes[0].set_title("Original split")
axes[1].set_title("Grouped split with backgrounds")
for axis in axes:
    axis.set_xlabel("Dataset split")
    axis.set_ylabel("Number of images")
    axis.legend(["Fire only", "Smoke only", "Both", "Background"], loc="upper right")
fig.tight_layout()
display(fig)
plt.close(fig)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 5. Kiểm tra ZIP cuối

Đối chiếu từng ảnh trong ZIP với SHA-256 ở manifest, xác nhận nhãn đủ cặp và ảnh nền mới có nhãn rỗng. Kiểm tra CRC toàn bộ thành viên ZIP. Đây là kiểm tra cấu trúc và byte, không thay cho kiểm duyệt ngữ nghĩa mọi ảnh."""
        ),
        nbf.v4.new_code_cell(
            """assert manifest["output_image"].is_unique
assert manifest["sha256"].is_unique
missing_pairs = 0
wrong_hashes = 0
nonempty_new_background_labels = 0
with ZipFile(DATASET_ZIP) as archive:
    members = set(archive.namelist())
    bad_crc_member = archive.testzip()
    for row in manifest.itertuples(index=False):
        if row.output_image not in members or row.output_label not in members:
            missing_pairs += 1
            continue
        if sha256(archive.read(row.output_image)).hexdigest() != row.sha256:
            wrong_hashes += 1
        if row.source_dataset == "dfire_final_background" and archive.getinfo(row.output_label).file_size != 0:
            nonempty_new_background_labels += 1

quality = pd.DataFrame({
    "Kiểm tra": ["Ảnh trong manifest", "Thiếu cặp ảnh–nhãn", "Hash ảnh sai", "Nhãn nền thêm không rỗng", "Nhóm nguồn xuyên tập", "Thành viên ZIP lỗi CRC"],
    "Kết quả": [len(manifest), missing_pairs, wrong_hashes, nonempty_new_background_labels, crossing_after, str(bad_crc_member)],
})
display(quality)
assert missing_pairs == wrong_hashes == nonempty_new_background_labels == crossing_after == 0
assert bad_crc_member is None"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. Mẫu ảnh nền và giới hạn kết luận

Contact sheet dưới đây là mẫu 20 ảnh mỗi tập đã xem trực quan. Việc gom theo `video_N` và tên gốc `youtube-*` xử lý rò rỉ nhận biết được từ tên file. Các ảnh cùng cảnh mang tên khác, hoặc ảnh nền D-Fire gần giống xuyên tập, chưa được kiểm duyệt toàn diện. Bộ này phù hợp để **train thử và xem chỉ số**, chưa phải test độc lập theo cảnh/video. Cần đánh giá thêm trên video/camera độc lập trước khi kết luận hiệu năng triển khai."""
        ),
        nbf.v4.new_code_cell(
            """contact_sheet = REPORT_DIR / "background_sample.jpg"
if contact_sheet.is_file():
    display(NotebookImage(filename=str(contact_sheet)))
else:
    print("Chưa có contact sheet tại", contact_sheet)"""
        ),
    ]
    notebook = nbf.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
    )
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
