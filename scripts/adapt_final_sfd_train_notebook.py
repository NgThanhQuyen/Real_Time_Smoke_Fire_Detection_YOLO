"""Create the Kaggle training notebook for FinalSFD from the earlier D-Fire run."""

import json
import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "notebooks/model_training/train_yolo11n_dfire_fire2591.ipynb"
TARGET = ROOT / "notebooks/model_training/train_yolo11n_final_sfd.ipynb"
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))


def get(index):
    return "".join(notebook["cells"][index]["source"])


def put(index, source):
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)


put(0, """# Huấn luyện YOLO11n trên FinalSFD

Notebook đọc trực tiếp thư mục Kaggle Dataset `/kaggle/input/datasets/quynnguynthanh/finalsfd-yolo-dataset/FinalSFD_YOLO_Dataset` mà bạn đã cung cấp. Nếu dữ liệu nằm trong một thư mục con, notebook tự tìm gốc chứa `data.yaml`, `images/{train,val,test}` và `labels/{train,val,test}`. Không dùng output cũ dưới `/kaggle/input/notebooks/`. Số ảnh và nhãn được kiểm tra, thống kê trực tiếp; không có số lượng cố định.

Chạy tuần tự các ô trên Kaggle với GPU và Internet bật. ZIP nguồn được giải nén vào `/kaggle/working`; cần đủ dung lượng cho dữ liệu giải nén và kết quả huấn luyện. Class 0 là smoke, class 1 là fire. Notebook dùng `imgsz=640`, YOLO11n pretrained và giữ các tham số train của bản trước để so sánh. Dự đoán ảnh nền và ảnh minh họa chạy trên CPU, mỗi lần một ảnh để tránh lỗi CUDA out of memory sau train/eval.

Kết quả gồm `best.pt`, metrics val/test, thống kê dữ liệu, dự đoán ảnh nền và ảnh minh họa. Tập test chỉ được đánh giá sau khi checkpoint đã được chọn từ validation.
""")

cell = get(2).replace("import hashlib\n", "import gc\nimport hashlib\n")
put(2, cell)

cell = get(4)
cell = cell.replace('dfire_original_clean_split_fire2591_v1.zip', 'FinalSFD_YOLO_Dataset.zip')
cell = cell.replace('DATASET_OVERRIDE = None  # Đặt đường dẫn ZIP hoặc thư mục đã giải nén nếu Kaggle đổi tên Dataset.',
                    'DATASET_OVERRIDE = "/kaggle/input/datasets/quynnguynthanh/finalsfd-yolo-dataset/FinalSFD_YOLO_Dataset"  # Kaggle Dataset bạn đã tải lên.')
cell = cell.replace('DATASET_HINT = "dfire-original-clean-split-fire2591"', 'DATASET_HINT = "finalsfd"')
cell = cell.replace('DATASET_HINT = "finalsfd"\n', '')
cell = cell.replace('dfire_fire2591_yolo11n_', 'final_sfd_yolo11n_')
start = cell.index('EXPECTED_COUNTS = ')
end = cell.index('\n\ndef has_dataset_layout', start)
cell = cell[:start] + cell[end:]
cell = cell.replace('(folder / split / "images").is_dir() and (folder / split / "labels").is_dir()',
                    '(folder / "images" / split).is_dir() and (folder / "labels" / split).is_dir()')
cell = cell.replace('thư mục D-Fire hợp lệ', 'thư mục FinalSFD hợp lệ')
start = cell.index('def find_dataset_source()')
end = cell.index('\n\ndef extract_dataset_zip(', start)
cell = cell[:start] + '''def find_dataset_source() -> Path:
    # Dữ liệu do người dùng tải lên nằm ngoài nhánh /kaggle/input/notebooks/.
    # Không tự chọn output cũ của Data_Preprocessing_YOLO khi cả hai cùng tên.
    def uploaded(path: Path) -> bool:
        return "notebooks" not in path.relative_to(INPUT_ROOT).parts

    if DATASET_OVERRIDE is not None:
        candidate = Path(DATASET_OVERRIDE).expanduser().resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return candidate
        if candidate.is_dir():
            roots = [candidate] if has_dataset_layout(candidate) else [
                path.parent for path in candidate.rglob("data.yaml") if has_dataset_layout(path.parent)
            ]
            if len(roots) == 1:
                return roots[0].resolve()
            if not roots:
                archives = sorted(candidate.rglob(DATASET_ZIP_NAME))
                if len(archives) == 1:
                    return archives[0].resolve()
        raise RuntimeError(f"DATASET_OVERRIDE không trỏ đến một ZIP hoặc bộ FinalSFD hợp lệ: {candidate}")

    archives = sorted(path.resolve() for path in INPUT_ROOT.rglob(DATASET_ZIP_NAME)
                      if path.is_file() and uploaded(path))
    directories = sorted({path.parent.resolve() for path in INPUT_ROOT.rglob("data.yaml")
                          if uploaded(path) and has_dataset_layout(path.parent)})
    candidates = archives + directories
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Không thấy {DATASET_ZIP_NAME} hoặc thư mục images/labels/data.yaml "
            f"trong Kaggle Dataset đã tải lên. Kiểm tra Kaggle Input hoặc đặt DATASET_OVERRIDE."
        )
    raise RuntimeError(
        f"Có nhiều nguồn FinalSFD được tải lên: {candidates}. "
        "Đặt DATASET_OVERRIDE thành đường dẫn ZIP hoặc thư mục cần train."
    )
''' + cell[end:]
cell = cell.replace('any(name.startswith(f"{split}/images/") for name in names)',
                    'any(name.startswith(f"images/{split}/") for name in names) and\n            any(name.startswith(f"labels/{split}/") for name in names)')
cell = cell.replace('ZIP không có cấu trúc D-Fire train/val/test như dự kiến',
                    'ZIP không có images/{train,val,test}, labels/{train,val,test} và data.yaml ở thư mục gốc')
start = cell.index('def extract_dataset_zip(')
end = cell.index('\n\nDATASET_SOURCE = ', start)
cell = cell[:start] + '''def extract_dataset_zip(source_zip: Path, destination: Path) -> tuple[int, int, dict]:
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(source_zip) as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        names = {info.filename for info in infos}
        # Một số notebook Kaggle nén cả đường dẫn kaggle/working/<dataset>/.
        # Chọn tiền tố duy nhất chứa data.yaml và đủ ba tập, không giả định gốc ZIP.
        prefixes = sorted({name[:-len("data.yaml")] for name in names if name.endswith("data.yaml")})
        valid_prefixes = [prefix for prefix in prefixes if all(
            any(name.startswith(prefix + f"images/{split}/") for name in names)
            and any(name.startswith(prefix + f"labels/{split}/") for name in names)
            for split in ("train", "val", "test")
        )]
        if len(valid_prefixes) != 1:
            raise RuntimeError(
                f"Cần đúng một gốc dữ liệu trong ZIP; tìm thấy {valid_prefixes}. "
                f"data.yaml ứng viên: {prefixes[:10]}"
            )
        prefix = valid_prefixes[0]
        members = [info for info in infos if info.filename.startswith(prefix)]
        source_image_counts = {
            split: sum(
                info.filename.startswith(prefix + f"images/{split}/")
                and Path(info.filename).suffix.lower() in IMAGE_EXTENSIONS
                for info in members
            )
            for split in ("train", "val", "test")
        }
        total_bytes = sum(info.file_size for info in members)
        reserve_bytes = 2 * 1024**3
        free_bytes = shutil.disk_usage(WORK_ROOT).free
        if free_bytes < total_bytes + reserve_bytes:
            raise RuntimeError(
                f"Không đủ dung lượng giải nén: cần {(total_bytes + reserve_bytes) / 1024**3:.2f} GB, "
                f"còn {free_bytes / 1024**3:.2f} GB"
            )
        for info in members:
            original = Path(info.filename)
            relative = Path(info.filename[len(prefix):])
            target = (root / relative).resolve()
            if (original.is_absolute() or ".." in original.parts or relative.is_absolute()
                    or target == root or not target.is_relative_to(root)
                    or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError(f"Đường dẫn không an toàn trong ZIP: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    print("Thư mục gốc trong ZIP:", prefix or "(gốc ZIP)")
    return total_bytes, len(members), source_image_counts
''' + cell[end:]
cell = cell.replace('"train": "train/images",\n    "val": "val/images",\n    "test": "test/images",',
                    '"train": "images/train",\n    "val": "images/val",\n    "test": "images/test",')
cell = cell.replace('if [str(name).lower() for name in names or []] != ["smoke", "fire"]:',
                    'if source_config.get("nc") != 2 or [str(name).lower() for name in names or []] != ["smoke", "fire"]:')
put(4, cell)

cell = get(4)
cell = cell.replace('source_bytes, copied_files = extract_dataset_zip(DATASET_SOURCE, LOCAL_DATASET_ROOT)',
                    'source_bytes, copied_files, source_image_counts = extract_dataset_zip(DATASET_SOURCE, LOCAL_DATASET_ROOT)')
cell = cell.replace('    copy_method = "copy_directory"',
                    '    copy_method = "copy_directory"\n    source_image_counts = {split: sum(\n        path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS\n        for path in (DATASET_SOURCE / "images" / split).iterdir()\n    ) for split in ("train", "val", "test")}')
cell = cell.replace('print("Dataset Kaggle:", DATASET_SOURCE)',
                    'print("Dataset Kaggle:", DATASET_SOURCE)\n'
                    'print("Dung lượng ZIP/tập nguồn (GiB):", round(DATASET_SOURCE.stat().st_size / 1024**3, 3) if DATASET_SOURCE.is_file() else "thư mục")\n'
                    'print("Số ảnh trong nguồn:", source_image_counts)')
put(4, cell)

put(5, """## 3. Kiểm tra ảnh, nhãn và thống kê trước train

ZIP FinalSFD còn một số box vượt mép ảnh và box có kích thước bằng 0. Notebook chỉ sửa bản nhãn đã sao chép vào `/kaggle/working`: cắt box về biên ảnh, bỏ box không còn diện tích và ghi `reports/label_fixes.csv`. Ảnh, nhãn nguồn trong Kaggle Input không đổi. Thống kê và metrics dùng bản nhãn đã sửa; số ảnh ở mỗi tập giữ nguyên.
""")

cell = get(6)
cell = cell.replace('LOCAL_DATASET_ROOT / split / "images"', 'LOCAL_DATASET_ROOT / "images" / split')
cell = cell.replace('LOCAL_DATASET_ROOT / split / "labels"', 'LOCAL_DATASET_ROOT / "labels" / split')
cell = cell.replace('    orphan_labels = [path.name for path in labels if path.stem not in image_stems]',
                    '    if not images:\n        raise RuntimeError(f"Tập {split} không có ảnh")\n    if len(image_stems) != len(images):\n        raise RuntimeError(f"Tập {split} có ảnh trùng tên gốc ở nhiều định dạng")\n    orphan_labels = [path.name for path in labels if path.stem not in image_stems]')
start = cell.index('    if len(images) != EXPECTED_COUNTS[split]')
end = cell.index('    return {"images":', start)
cell = cell[:start] + cell[end:]
repair_code = '''def repair_local_labels() -> pd.DataFrame:
    fixes = []
    for split in ("train", "val", "test"):
        for label_path in sorted((LOCAL_DATASET_ROOT / "labels" / split).glob("*.txt")):
            original_lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            corrected_lines = []
            changed = False
            for line_number, line in enumerate(original_lines, 1):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5 or fields[0] not in {"0", "1"}:
                    raise RuntimeError(f"Nhãn không đúng YOLO box: {label_path}:{line_number}")
                try:
                    x, y, width, height = (float(value) for value in fields[1:])
                except ValueError as error:
                    raise RuntimeError(f"Không đọc được tọa độ: {label_path}:{line_number}") from error
                if not all(np.isfinite(value) for value in (x, y, width, height)):
                    raise RuntimeError(f"Tọa độ không hữu hạn: {label_path}:{line_number}")

                left, right = x - width / 2, x + width / 2
                top, bottom = y - height / 2, y + height / 2
                clipped_left, clipped_right = max(0.0, left), min(1.0, right)
                clipped_top, clipped_bottom = max(0.0, top), min(1.0, bottom)
                if (width <= 0 or height <= 0 or clipped_right <= clipped_left
                        or clipped_bottom <= clipped_top):
                    action, corrected = "drop_zero_area", ""
                elif (any(value < 0 or value > 1 for value in (x, y, width, height))
                      or max(abs(clipped_left - left), abs(clipped_right - right),
                             abs(clipped_top - top), abs(clipped_bottom - bottom)) > 1e-6):
                    new_x = (clipped_left + clipped_right) / 2
                    new_y = (clipped_top + clipped_bottom) / 2
                    new_width = clipped_right - clipped_left
                    new_height = clipped_bottom - clipped_top
                    action = "clip_to_image"
                    corrected = f"{fields[0]} {new_x:.12g} {new_y:.12g} {new_width:.12g} {new_height:.12g}"
                else:
                    action, corrected = "keep", line

                if action != "keep":
                    changed = True
                    fixes.append({
                        "split": split,
                        "label": label_path.relative_to(LOCAL_DATASET_ROOT).as_posix(),
                        "line": line_number,
                        "action": action,
                        "before": line,
                        "after": corrected,
                    })
                if corrected:
                    corrected_lines.append(corrected)
            if changed:
                label_path.write_text("\\n".join(corrected_lines) + ("\\n" if corrected_lines else ""), encoding="utf-8")
    return pd.DataFrame(fixes, columns=["split", "label", "line", "action", "before", "after"])


label_fixes = repair_local_labels()
label_fixes.to_csv(RUN_DIR / "reports" / "label_fixes.csv", index=False)
label_fix_summary = {
    "clip_to_image": int((label_fixes["action"] == "clip_to_image").sum()),
    "drop_zero_area": int((label_fixes["action"] == "drop_zero_area").sum()),
    "affected_labels": int(label_fixes["label"].nunique()),
}
(RUN_DIR / "reports" / "label_fix_summary.json").write_text(
    json.dumps(label_fix_summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Sửa nhãn cục bộ:", label_fix_summary)


'''
cell = repair_code + cell
cell = cell.replace('audit = {split: inspect_split(split) for split in ("train", "val", "test")}',
                    'audit = {split: inspect_split(split) for split in ("train", "val", "test")}\n'
                    'actual_image_counts = {split: item["images"] for split, item in audit.items()}\n'
                    'if actual_image_counts != source_image_counts:\n'
                    '    raise RuntimeError(f"Số ảnh sau giải nén/sao chép khác nguồn: nguồn={source_image_counts}, thực tế={actual_image_counts}")\n'
                    'print("Đối chiếu nguồn và bản giải nén:", actual_image_counts, "(khớp)")')
put(6, cell)

cell = get(8)
cell += '\n# Giải phóng model/trainer trước khi nạp best.pt để đánh giá.\ndel train_model\ngc.collect()\ntorch.cuda.empty_cache()\n'
put(8, cell)

cell = get(10)
cell = cell.replace('    evaluation[split_name] = summarize_metrics(metrics, split_name)',
                    '    evaluation[split_name] = summarize_metrics(metrics, split_name)\n    del metrics\n    gc.collect()\n    torch.cuda.empty_cache()')
cell += '\n# Phần báo cáo ảnh dùng CPU; không giữ model đánh giá trên GPU.\ndel best_model\ngc.collect()\ntorch.cuda.empty_cache()\n'
put(10, cell)

put(11, """## 6. Ảnh nền test và dự đoán minh họa

Dự đoán dùng CPU, `batch=1` và xử lý từng ảnh. Cách này chậm hơn GPU nhưng tránh lỗi CUDA out of memory khi bộ nhớ GPU còn bị chiếm sau train/eval. Nếu tập test không có ảnh nền, báo cáo ghi 0 ảnh và tỉ lệ false positive là `null`.
""")

put(12, """test_records = image_records[image_records["split"] == "test"]
background_paths = test_records.loc[test_records["category"] == "background", "image"].tolist()
REPORT_DEVICE = "cpu"
REPORT_BATCH_SIZE = 1
report_model = YOLO(str(BEST_PT))


def predict_one(path: str):
    # Danh sách ảnh rất dài với batch=train đã gây CUDA OOM; xử lý một ảnh trên CPU.
    return report_model.predict(
        source=path, batch=REPORT_BATCH_SIZE, conf=VIS_CONF,
        iou=NMS_IOU, imgsz=IMAGE_SIZE, device=REPORT_DEVICE, verbose=False,
    )[0]


background_predictions = []
for index, path in enumerate(background_paths, 1):
    result = predict_one(path)
    classes = result.boxes.cls.cpu().numpy().astype(int).tolist()
    confidences = result.boxes.conf.cpu().numpy().tolist()
    background_predictions.append({
        "image": Path(result.path).name,
        "detections": len(classes),
        "fire_detections": classes.count(1),
        "smoke_detections": classes.count(0),
        "max_confidence": max(confidences, default=0.0),
    })
    del result
    if index % 100 == 0 or index == len(background_paths):
        print(f"Đã dự đoán ảnh nền: {index}/{len(background_paths)}")

background_table = pd.DataFrame(background_predictions, columns=[
    "image", "detections", "fire_detections", "smoke_detections", "max_confidence"
])
background_table.to_csv(RUN_DIR / "reports" / "background_test_predictions.csv", index=False)
background_false_positive_images = int((background_table["detections"] > 0).sum())
background_report = {
    "split": "test", "confidence": VIS_CONF,
    "device": REPORT_DEVICE, "batch": REPORT_BATCH_SIZE,
    "background_images": len(background_table),
    "images_with_detection": background_false_positive_images,
    "image_false_positive_rate": (
        background_false_positive_images / len(background_table) if len(background_table) else None
    ),
}
(RUN_DIR / "reports" / "background_test_summary.json").write_text(
    json.dumps(background_report, ensure_ascii=False, indent=2), encoding="utf-8"
)
display(pd.DataFrame([background_report]))

sample_rng = random.Random(SEED)
sample_paths = []
for category in ("fire_only", "smoke_only", "both", "background"):
    candidates = sorted(test_records.loc[test_records["category"] == category, "image"].tolist())
    sample_paths.extend(sample_rng.sample(candidates, min(3, len(candidates))))

sample_dir = RUN_DIR / "reports" / "sample_predictions"
sample_dir.mkdir(parents=True, exist_ok=True)
category_by_path = dict(zip(test_records["image"], test_records["category"]))
sample_rows = []
for path in sample_paths:
    result = predict_one(path)
    source_path = str(Path(result.path))
    output_path = sample_dir / Path(source_path).name
    result.save(filename=str(output_path))
    sample_rows.append({
        "image": Path(source_path).name,
        "ground_truth_category": category_by_path[source_path],
        "detections": len(result.boxes),
        "annotated_file": output_path.name,
    })
    del result
sample_table = pd.DataFrame(sample_rows, columns=[
    "image", "ground_truth_category", "detections", "annotated_file"
])
sample_table.to_csv(RUN_DIR / "reports" / "sample_predictions.csv", index=False)
display(sample_table)
for output_path in sorted(sample_dir.iterdir())[:4]:
    display(NotebookImage(filename=str(output_path), width=500))
del report_model
gc.collect()
""")

cell = get(14).replace('# Kết quả YOLO11n D-Fire smoke/fire', '# Kết quả YOLO11n FinalSFD smoke/fire')
cell = cell.replace('    "dataset_images": dataset_rows,',
                    '    "dataset_images": dataset_rows,\n    "local_label_fixes": label_fix_summary,')
cell = cell.replace('"thông số train, metrics valid/test và SHA-256 checkpoint. `reports/labelmap.json`',
                    '"thông số train, metrics valid/test và SHA-256 checkpoint. `reports/label_fixes.csv` ghi nhãn đã sửa cục bộ. `reports/labelmap.json`')
put(14, cell)
cell = get(16).replace('dfire_fire2591_yolo11n_', 'final_sfd_yolo11n_')
put(16, cell)

for cell in notebook["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
runpy.run_path(str(ROOT / "scripts/extend_final_sfd_train_notebook.py"))
print(TARGET)
