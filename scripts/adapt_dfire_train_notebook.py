"""Chuyển notebook train tham khảo sang bản D-Fire thêm 2.591 ảnh lửa."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"F:\6\KLTN\Notebook\train-yolo11n-smoke-fire-detection-v2.ipynb")
TARGET = ROOT / "notebooks/model_training/train_yolo11n_dfire_fire2591.ipynb"


def set_cell(notebook, index, content):
    notebook["cells"][index]["source"] = content.splitlines(keepends=True)
    notebook["cells"][index]["outputs"] = [] if notebook["cells"][index]["cell_type"] == "code" else notebook["cells"][index].get("outputs", [])
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None


def replace_once(value, old, new):
    if value.count(old) != 1:
        raise ValueError(f"Không thấy đúng một đoạn cần thay: {old[:80]}")
    return value.replace(old, new)


def main():
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    if len(notebook["cells"]) != 17:
        raise ValueError("Notebook nguồn đã thay đổi cấu trúc")
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    set_cell(notebook, 0, """# Huấn luyện YOLO11n trên D-Fire đã thêm 2.591 ảnh chỉ lửa

Notebook này dùng `dfire_original_clean_split_fire2591_v1.zip` (15.290 train, 4.244 val, 4.244 test; nhãn `0=smoke`, `1=fire`). Chỉ train được tăng ảnh; val/test D-Fire giữ nguyên. Tải ZIP này lên Kaggle dưới dạng Dataset, rồi thêm Dataset đó vào Kaggle Input của notebook. Có thể dùng bản Kaggle đã giải nén cùng cấu trúc.

Recipe giữ từ notebook tham khảo: YOLO11n pretrained, tối đa 250 epoch, `imgsz=640`, `batch=32`, `patience=20`. Notebook kiểm tra class map và nhãn, chọn `best.pt` theo val, đánh giá val/test, đo báo nhầm trên ảnh nền test và đóng gói kết quả. Không dùng test để chọn epoch hoặc chỉnh ngưỡng.

**Trước khi chạy:** chọn GPU trong Kaggle Settings và bật Internet nếu cần cài Ultralytics hoặc tải `yolo11n.pt`. Nếu Kaggle Input chứa ZIP, notebook giải nén trực tiếp vào `/kaggle/working`; nếu là thư mục ảnh, notebook sao chép một lần. Cần khoảng 5–6 GB dung lượng trống ngoài đầu ra train. Cuối notebook chỉ giữ ZIP kết quả sau khi kiểm tra CRC; Kaggle Input không bị sửa. Nếu train hoặc đóng gói lỗi, thư mục run được giữ để kiểm tra.

Ảnh mới đã qua sàng lọc kỹ thuật theo quy tắc của chủ dự án, nhưng chưa được kiểm duyệt toàn bộ nhãn ngữ nghĩa. Metrics ảnh cần kiểm tra thêm bằng video/camera độc lập trước khi dùng để cảnh báo.
""")

    set_cell(notebook, 4, """INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
DATASET_OVERRIDE = None  # Đặt đường dẫn ZIP hoặc thư mục đã giải nén nếu Kaggle đổi tên Dataset.
DATASET_ZIP_NAME = "dfire_original_clean_split_fire2591_v1.zip"
DATASET_HINT = "dfire-original-clean-split-fire2591"
RUN_ID = "dfire_fire2591_yolo11n_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
RUN_DIR = WORK_ROOT / RUN_ID
RESULT_ZIP = WORK_ROOT / f"{RUN_ID}_results.zip"

IMAGE_SIZE = 640
BATCH_SIZE = 32
EPOCHS = 250
PATIENCE = 20
WORKERS = 4
SEED = 0
EVAL_CONF = 0.001
VIS_CONF = 0.25
NMS_IOU = 0.7
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_COUNTS = {"train": 15290, "val": 4244, "test": 4244}
EXPECTED_CATEGORIES = {
    "train": {"fire_only": 3277, "smoke_only": 3464, "both": 2664, "background": 5885},
    "val": {"fire_only": 225, "smoke_only": 1176, "both": 895, "background": 1948},
    "test": {"fire_only": 208, "smoke_only": 1174, "both": 857, "background": 2005},
}


def has_dataset_layout(folder: Path) -> bool:
    return (folder / "data.yaml").is_file() and all(
        (folder / split / "images").is_dir() and (folder / split / "labels").is_dir()
        for split in ("train", "val", "test")
    )


def find_dataset_source() -> Path:
    if DATASET_OVERRIDE is not None:
        candidate = Path(DATASET_OVERRIDE).expanduser().resolve()
        if candidate.is_file() and candidate.suffix.lower() == ".zip":
            return candidate
        if candidate.is_dir() and has_dataset_layout(candidate):
            return candidate
        raise RuntimeError(f"DATASET_OVERRIDE không phải ZIP hoặc thư mục D-Fire hợp lệ: {candidate}")

    archives = sorted(path for path in INPUT_ROOT.rglob(DATASET_ZIP_NAME) if path.is_file())
    if len(archives) == 1:
        return archives[0].resolve()
    if len(archives) > 1:
        raise RuntimeError(f"Có nhiều ZIP cùng tên trong Kaggle Input: {archives}. Hãy đặt DATASET_OVERRIDE.")
    directories = sorted({path.parent.resolve() for path in INPUT_ROOT.rglob("data.yaml")
                          if DATASET_HINT in str(path.parent).lower().replace("_", "-")
                          and has_dataset_layout(path.parent)})
    if len(directories) == 1:
        return directories[0]
    raise FileNotFoundError(
        f"Không tìm thấy {DATASET_ZIP_NAME} hoặc thư mục đã giải nén trong {INPUT_ROOT}. "
        "Hãy thêm đúng Kaggle Dataset hoặc đặt DATASET_OVERRIDE."
    )


def extract_dataset_zip(source_zip: Path, destination: Path) -> tuple[int, int]:
    destination.mkdir(parents=True)
    root = destination.resolve()
    with zipfile.ZipFile(source_zip) as archive:
        members = [info for info in archive.infolist() if not info.is_dir()]
        names = {info.filename for info in members}
        if "data.yaml" not in names or not all(
            any(name.startswith(f"{split}/images/") for name in names)
            for split in ("train", "val", "test")
        ):
            raise RuntimeError("ZIP không có cấu trúc D-Fire train/val/test như dự kiến")
        total_bytes = sum(info.file_size for info in members)
        reserve_bytes = 2 * 1024**3
        free_bytes = shutil.disk_usage(WORK_ROOT).free
        if free_bytes < total_bytes + reserve_bytes:
            raise RuntimeError(f"Không đủ dung lượng giải nén: cần {(total_bytes + reserve_bytes) / 1024**3:.2f} GB, còn {free_bytes / 1024**3:.2f} GB")
        for info in members:
            relative = Path(info.filename)
            target = (root / relative).resolve()
            if (relative.is_absolute() or ".." in relative.parts or target == root
                    or not target.is_relative_to(root) or (info.external_attr >> 16) & 0o170000 == 0o120000):
                raise RuntimeError(f"Đường dẫn không an toàn trong ZIP: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return total_bytes, len(members)


DATASET_SOURCE = find_dataset_source()
if RUN_DIR.exists() or RESULT_ZIP.exists():
    raise FileExistsError(f"Run đã tồn tại: {RUN_DIR} hoặc {RESULT_ZIP}. Hãy chạy lại ô cấu hình để tạo RUN_ID mới.")
RUN_DIR.mkdir(parents=True)
(RUN_DIR / "reports").mkdir()
LOCAL_DATASET_ROOT = RUN_DIR / "dataset_local"

copy_started = time.perf_counter()
if DATASET_SOURCE.is_file():
    source_bytes, copied_files = extract_dataset_zip(DATASET_SOURCE, LOCAL_DATASET_ROOT)
    copy_method = "extract_zip"
else:
    source_files = [path for path in DATASET_SOURCE.rglob("*") if path.is_file()]
    source_bytes = sum(path.stat().st_size for path in source_files)
    reserve_bytes = 2 * 1024**3
    free_bytes = shutil.disk_usage(WORK_ROOT).free
    if free_bytes < source_bytes + reserve_bytes:
        raise RuntimeError(f"Không đủ dung lượng sao chép: cần {(source_bytes + reserve_bytes) / 1024**3:.2f} GB, còn {free_bytes / 1024**3:.2f} GB")
    shutil.copytree(DATASET_SOURCE, LOCAL_DATASET_ROOT)
    copied_files = len(source_files)
    copy_method = "copy_directory"
copy_seconds = time.perf_counter() - copy_started
if not has_dataset_layout(LOCAL_DATASET_ROOT):
    raise RuntimeError(f"Bản dữ liệu cục bộ thiếu ảnh hoặc nhãn: {LOCAL_DATASET_ROOT}")

SOURCE_YAML = LOCAL_DATASET_ROOT / "data.yaml"
source_config = yaml.safe_load(SOURCE_YAML.read_text(encoding="utf-8"))
names = source_config.get("names")
if isinstance(names, dict):
    names = [names[key] for key in sorted(names, key=lambda value: int(value))]
if [str(name).lower() for name in names or []] != ["smoke", "fire"]:
    raise RuntimeError(f"Class map phải là 0=smoke, 1=fire; hiện tại: {names}")

TRAIN_YAML = RUN_DIR / "data_for_training.yaml"
TRAIN_YAML.write_text(yaml.safe_dump({
    "path": str(LOCAL_DATASET_ROOT),
    "train": "train/images",
    "val": "val/images",
    "test": "test/images",
    "nc": 2,
    "names": {0: "smoke", 1: "fire"},
}, sort_keys=False, allow_unicode=True), encoding="utf-8")

print("Dataset Kaggle:", DATASET_SOURCE)
print("Bản dữ liệu cục bộ:", LOCAL_DATASET_ROOT)
print("Phương pháp:", copy_method, "số file:", copied_files)
print("Dung lượng dữ liệu (GB):", round(source_bytes / 1024**3, 2))
print("Thời gian chuẩn bị (giây):", round(copy_seconds, 1))
print("YAML train:", TRAIN_YAML)
print("Thư mục run:", RUN_DIR)
print("ZIP kết quả sẽ tạo:", RESULT_ZIP)
""")

    inspect_code = "".join(notebook["cells"][6]["source"])
    inspect_code = replace_once(inspect_code, '"fire_only" if present == {0} else\n            "smoke_only" if present == {1}', '"fire_only" if present == {1} else\n            "smoke_only" if present == {0}')
    inspect_code = replace_once(inspect_code, 'for split in ("train", "valid", "test")', 'for split in ("train", "val", "test")')
    inspect_code = replace_once(inspect_code, '"fire_boxes": item["boxes"][0], "smoke_boxes": item["boxes"][1]', '"fire_boxes": item["boxes"][1], "smoke_boxes": item["boxes"][0]')
    set_cell(notebook, 6, inspect_code)
    set_cell(notebook, 9, """## 5. Đánh giá `best.pt` trên val và test

Hai lần đánh giá dùng cùng `imgsz`, `batch`, confidence thấp để tính mAP và NMS IoU. Test chỉ được đọc sau khi `best.pt` đã được chọn bằng val.
""")
    evaluation_code = "".join(notebook["cells"][10]["source"])
    evaluation_code = replace_once(evaluation_code, '(("valid", "val"), ("test", "test"))', '(("val", "val"), ("test", "test"))')
    set_cell(notebook, 10, evaluation_code)
    prediction_code = "".join(notebook["cells"][12]["source"])
    prediction_code = replace_once(prediction_code, '"fire_detections": classes.count(0),\n        "smoke_detections": classes.count(1)', '"fire_detections": classes.count(1),\n        "smoke_detections": classes.count(0)')
    set_cell(notebook, 12, prediction_code)
    report_code = "".join(notebook["cells"][14]["source"])
    report_code = replace_once(report_code, '"evaluation_valid/PR_curve.png"', '"evaluation_val/PR_curve.png"')
    report_code = replace_once(report_code, '"dataset": str(DATASET_ROOT)', '"dataset": str(DATASET_SOURCE)')
    report_code = replace_once(report_code, '"class_map": {"0": "Fire", "1": "Smoke"}', '"class_map": {"0": "smoke", "1": "fire"}')
    report_code = replace_once(report_code, '"dataset_copy_seconds": round(copy_seconds, 2)', '"dataset_copy_seconds": round(copy_seconds, 2),\n        "dataset_copy_method": copy_method')
    report_code = replace_once(report_code, 'json.dumps({"0": "Fire", "1": "Smoke"}', 'json.dumps({"0": "smoke", "1": "fire"}')
    report_code = replace_once(report_code, '"# Kết quả YOLO11n Fire/Smoke\\n\\n"', '"# Kết quả YOLO11n D-Fire smoke/fire\\n\\n"')
    report_code = replace_once(report_code, '"0=Fire, 1=Smoke. Các file `evaluation_*` là biểu đồ đánh giá. "', '"0=smoke, 1=fire. Các file `evaluation_*` là biểu đồ đánh giá. "')
    set_cell(notebook, 14, report_code)
    package_code = "".join(notebook["cells"][16]["source"])
    package_code = replace_once(package_code, '"smoke_fire_v2_yolo11n_"', '"dfire_fire2591_yolo11n_"')
    set_cell(notebook, 16, package_code)

    serialized = json.dumps(notebook, ensure_ascii=False, indent=1) + "\n"
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(serialized, encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
