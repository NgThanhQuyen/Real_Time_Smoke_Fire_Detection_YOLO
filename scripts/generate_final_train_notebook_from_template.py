"""Tạo notebook train bộ final theo recipe của notebook D-Fire mẫu."""

from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = Path(r"F:\6\KLTN\Notebook\03-train-yolo11n-dfire.ipynb")
OUTPUT = ROOT / "tmp/04-train-yolo11n-fire-smoke-dfire-final.ipynb"


INTRO = """# Huấn luyện YOLO11n trên bộ Fire–Smoke + D-Fire final

Notebook này dùng **một Kaggle Input** là bộ `fire_smoke_dfire_final.zip` đã tổng hợp từ Fire-Smoke-Detection-Yolov11 và D-Fire. Bộ final có 13.329 ảnh train, 1.666 ảnh valid và 1.666 ảnh test; nhãn `0 = Fire`, `1 = Smoke`. Không thêm các bộ dữ liệu khác.

## Chuẩn bị trên Kaggle

1. Tải `fire_smoke_dfire_final.zip` lên Kaggle Dataset và gắn Dataset đó vào notebook.
2. Bật GPU và Internet để cài Ultralytics đã khóa phiên bản, đồng thời tải `yolo11n.pt` nếu chưa có cache.
3. Notebook tự tìm thư mục đã giải nén; nếu Input chỉ chứa ZIP, notebook kiểm tra và giải nén vào `/kaggle/working/fire_smoke_dfire_final_data`.
4. Bật lưu Output để tải gói kết quả sau khi Run All hoàn tất.

## Quy trình

- Giữ recipe hai giai đoạn của notebook D-Fire mẫu: YOLO11n pretrained COCO, giai đoạn 1 freeze 10 lớp trong tối đa 20 epoch, giai đoạn 2 mở toàn bộ model trong tối đa 180 epoch. Theo yêu cầu mới, dùng `imgsz=640`; `batch=32`, optimizer AdamW, seed 42 và các tham số augmentation được giữ nguyên.
- Chọn checkpoint theo valid. Chỉ đánh giá test sau khi đã chọn checkpoint; gói kết quả lưu `best.pt`, `last.pt`, thông số train, các bảng metrics và hình Ultralytics.
- Bộ final có test khác test D-Fire cũ; không so trực tiếp các chỉ số giữa hai notebook như tác động riêng của dữ liệu. Nếu cần đối chứng, dùng cùng tập test và cùng recipe.

Notebook không thay đổi dữ liệu nguồn, chỉ tạo `data_for_training.yaml` trong thư mục làm việc để trỏ đúng ba split.
"""


DATA_SETUP = """# Cấu hình train giữ nguyên recipe notebook mẫu; chỉ đổi dữ liệu đầu vào và tên đầu ra.
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
DATASET_ARCHIVE_NAME = "fire_smoke_dfire_final.zip"
DATASET_DIR_OVERRIDE = ""  # Điền đường dẫn thư mục đã giải nén nếu Kaggle Input có nhiều dataset hợp lệ.
DATASET_LABEL = "Fire-Smoke-Detection-Yolov11 + D-Fire final"
ARTIFACT_ZIP = WORK_ROOT / "fire_smoke_dfire_final_yolo11n_transfer_results.zip"
RUN_ROOT = WORK_ROOT / "train_fire_smoke_dfire_final_yolo11n"
ARTIFACT_ROOT = WORK_ROOT / "fire_smoke_dfire_final_yolo11n_artifacts"

# Mỗi lần chạy bắt đầu trong thư mục kết quả sạch để tránh dùng nhầm checkpoint cũ.
if RUN_ROOT.exists():
    shutil.rmtree(RUN_ROOT)
if ARTIFACT_ROOT.exists():
    shutil.rmtree(ARTIFACT_ROOT)
RUN_ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_NAMES = ["Fire", "Smoke"]
EXPECTED_SPLIT_COUNTS = {"train": 13329, "val": 1666, "test": 1666}
SEED = 42
IMAGE_SIZE = 640
BATCH_SIZE = 32
WORKERS = 4
NMS_IOU = 0.6
TOTAL_EPOCHS = 200
FROZEN_EPOCHS = 20
FINETUNE_EPOCHS = TOTAL_EPOCHS - FROZEN_EPOCHS
PRETRAINED_WEIGHTS = "yolo11n.pt"


def is_dataset_root(root: Path) -> bool:
    return (root / "data.yaml").is_file() and all(
        (root / split / "images").is_dir() and (root / split / "labels").is_dir()
        for split in ("train", "valid", "test")
    )


def find_dataset_root(input_root: Path) -> Path:
    '''Tìm dataset đã giải nén hoặc giải nén đúng ZIP final từ Kaggle Input.'''
    if DATASET_DIR_OVERRIDE:
        root = Path(DATASET_DIR_OVERRIDE).expanduser().resolve()
        if not (root / "data.yaml").is_file():
            raise RuntimeError(f"Thư mục override không có data.yaml: {root}")
        return root

    # Kaggle thường hiển thị nội dung ZIP đã giải nén và không giữ file ZIP trong Input.
    direct = [input_root / name / "data.yaml"
              for name in ("fire_smoke_dfire_final", "fire-smoke-dfire-final")]
    yaml_files = sorted(set(input_root.rglob("data.yaml")) | {path for path in direct if path.is_file()})
    matching = [path for path in yaml_files
                if "fire_smoke_dfire_final" in str(path.parent).lower().replace("-", "_")]
    if len(matching) == 1:
        return matching[0].parent
    if len(yaml_files) == 1:
        return yaml_files[0].parent
    if yaml_files:
        raise RuntimeError(f"Có nhiều data.yaml trong Kaggle Input: {yaml_files}. Hãy đặt DATASET_DIR_OVERRIDE.")

    archives = sorted(input_root.rglob(DATASET_ARCHIVE_NAME))
    if len(archives) != 1:
        raise RuntimeError(
            f"Không tìm thấy data.yaml hoặc {DATASET_ARCHIVE_NAME} trong {input_root}. "
            f"Các mục cấp đầu: {[path.name for path in input_root.iterdir()]}"
        )
    extracted = WORK_ROOT / "fire_smoke_dfire_final_data"
    if is_dataset_root(extracted):
        return extracted
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP dataset lỗi CRC; hãy tải lại Kaggle Input.")
        base = extracted.resolve()
        for item in archive.infolist():
            target = (base / item.filename).resolve()
            if not target.is_relative_to(base):
                raise RuntimeError(f"Đường dẫn nguy hiểm trong ZIP: {item.filename}")
        archive.extractall(extracted)
    if not is_dataset_root(extracted):
        raise RuntimeError(f"ZIP giải nén xong nhưng thiếu thư mục ảnh/nhãn: {extracted}")
    return extracted


def count_images(folder: Path) -> int:
    return sum(1 for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


DATASET_ROOT = find_dataset_root(INPUT_ROOT)
SOURCE_YAML = DATASET_ROOT / "data.yaml"
source_config = yaml.safe_load(SOURCE_YAML.read_text(encoding="utf-8"))
source_names = source_config.get("names")
if isinstance(source_names, dict):
    source_names = [source_names[key] for key in sorted(source_names, key=lambda value: int(value))]
normalized_names = [str(name).strip().lower() for name in source_names or []]
if normalized_names != [name.lower() for name in EXPECTED_NAMES]:
    raise RuntimeError(f"Class map phải là 0=Fire, 1=Smoke; hiện tại: {source_names}")

split_counts = {}
for split, folder_name in (("train", "train"), ("val", "valid"), ("test", "test")):
    image_dir = DATASET_ROOT / folder_name / "images"
    label_dir = DATASET_ROOT / folder_name / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        split_root = DATASET_ROOT / folder_name
        raise RuntimeError(
            f"Tập {split} cần {image_dir} và {label_dir}; "
            f"các mục hiện có: {[path.name for path in split_root.iterdir()] if split_root.is_dir() else []}"
        )
    image_files = [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
    missing_labels = [path.name for path in image_files if not (label_dir / f"{path.stem}.txt").is_file()]
    if missing_labels:
        raise RuntimeError(f"Thiếu {len(missing_labels)} nhãn trong {split}; ví dụ: {missing_labels[:3]}")
    split_counts[split] = len(image_files)
if split_counts != EXPECTED_SPLIT_COUNTS:
    raise RuntimeError(f"Sai số ảnh bộ final: {split_counts}; mong đợi {EXPECTED_SPLIT_COUNTS}")

# Ultralytics dùng data.yaml riêng với đường dẫn tuyệt đối trên Kaggle.
TRAIN_DATA_YAML = RUN_ROOT / "data_for_training.yaml"
training_config = {
    "path": str(DATASET_ROOT),
    "train": "train/images",
    "val": "valid/images",
    "test": "test/images",
    "nc": 2,
    "names": EXPECTED_NAMES,
}
TRAIN_DATA_YAML.write_text(yaml.safe_dump(training_config, sort_keys=False, allow_unicode=True), encoding="utf-8")

print("Dataset:", DATASET_LABEL)
print("Đường dẫn:", DATASET_ROOT)
print("Số ảnh theo split:", split_counts)
print("Class map:", dict(enumerate(EXPECTED_NAMES)))
print("data.yaml huấn luyện:", TRAIN_DATA_YAML)
"""


def main() -> None:
    notebook = nbf.read(TEMPLATE, as_version=4)
    if len(notebook.cells) != 8:
        raise RuntimeError("Notebook mẫu đã thay đổi cấu trúc, cần kiểm tra lại.")
    notebook.cells[0].source = INTRO
    notebook.cells[2].source = DATA_SETUP
    summary = notebook.cells[7].source
    summary = summary.replace('"class_map": {"0": "khoi", "1": "lua"}',
                              '"class_map": {"0": "Fire", "1": "Smoke"}')
    if '"class_map": {"0": "Fire", "1": "Smoke"}' not in summary:
        raise RuntimeError("Không thay được class map trong báo cáo.")
    notebook.cells[7].source = summary
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            compile(cell.source, "notebook_cell", "exec")
    nbf.validate(notebook)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
