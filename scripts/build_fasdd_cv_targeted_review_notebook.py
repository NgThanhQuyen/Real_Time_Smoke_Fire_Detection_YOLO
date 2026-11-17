"""Tạo notebook Kaggle lấy mẫu ảnh lửa FASDD_CV để kiểm duyệt."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "notebooks/data_preparation/21_review_fasdd_cv_fire_candidates_kaggle.ipynb"


def make_cell(kind, source, index):
    item = {"cell_type": kind, "metadata": {}, "id": f"fasdd-review-{index:02d}",
            "source": source.splitlines(keepends=True)}
    if kind == "code":
        item.update({"execution_count": None, "outputs": []})
    return item


cells = []
cells.append(make_cell("markdown", """# Xem mẫu ảnh chỉ lửa FASDD_CV cho D-Fire

Notebook này chạy **sau báo cáo sàng lọc đầu tiên**. Nó chỉ đọc ba file COCO và khoảng 300 ảnh mẫu từ FASDD_CV trên Kaggle; không quét header 95 nghìn ảnh, không sửa ảnh/nhãn, không train. Mục tiêu là xem tỷ lệ cảnh cháy thật có thể dùng trong 5.898 ảnh chỉ-lửa có cạnh dài ít nhất 640 px.

1. Add Input cùng dataset FASDD_CV và Run All trên CPU. Không cần gắn D-Fire hay báo cáo ZIP cũ.
2. Tải `/kaggle/working/fasdd_cv_fire_candidate_review.zip` ở Output và gửi lại để kiểm duyệt ảnh mẫu.

Kết quả này **chưa** kiểm tra trùng/cùng cảnh với D-Fire và chưa chứng minh nhãn khói đầy đủ. Không dùng mẫu hay split nguồn làm train/val/test ngay.
""", 0))

cells.append(make_cell("code", r'''from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageOps
from tqdm.auto import tqdm
import csv, json, random, hashlib, zipfile, math, time

SEED = 20261111
OUTPUT = Path("/kaggle/working/fasdd_cv_fire_candidate_review")
OUTPUT.mkdir(parents=True, exist_ok=True)

def find_root():
    for path in Path("/kaggle/input").rglob("train.json"):
        if path.parent.name != "Annotations" or path.parent.parent.name != "COCO_CV":
            continue
        root = path.parents[3]
        if (root / "images").is_dir() and all((path.parent / f"{split}.json").is_file()
                                              for split in ("train", "val", "test")):
            return root
    raise FileNotFoundError("Không tìm thấy FASDD_CV/images và COCO_CV/Annotations/{train,val,test}.json")

ROOT = find_root()
ANNOTATIONS = ROOT / "annotations" / "COCO_CV" / "Annotations"
print("FASDD_CV:", ROOT)
''', 1))

cells.append(make_cell("markdown", """## Lọc pool bằng nhãn COCO

Chỉ nhận vào **pool xem mẫu** nếu COCO có lửa, không có khói, mọi box hợp lệ theo kích thước metadata và cạnh dài ảnh ít nhất 640 px. Đây là lọc cấu trúc; ảnh bị gắn nhãn thiếu khói vẫn có thể lọt vào và sẽ được nhận diện khi xem ảnh ghép.
""", 2))

cells.append(make_cell("code", r'''pool = []
source_summary = []
source_hashes = {}
for split in ("train", "val", "test"):
    path = ANNOTATIONS / f"{split}.json"
    raw = path.read_bytes()
    source_hashes[split] = hashlib.sha256(raw).hexdigest()
    data = json.loads(raw)
    class_map = {int(category["id"]): str(category["name"]).lower().strip()
                 for category in data["categories"]}
    name_to_id = {name: class_id for class_id, name in class_map.items()}
    if not {"fire", "smoke"}.issubset(name_to_id):
        raise RuntimeError(f"Class map không có fire/smoke: {class_map}")
    fire_id, smoke_id = name_to_id["fire"], name_to_id["smoke"]
    images = {int(image["id"]): image for image in data["images"]}
    by_image = defaultdict(list)
    invalid_image_ids = set()
    for annotation in data["annotations"]:
        image_id = int(annotation["image_id"])
        image = images.get(image_id)
        box = annotation.get("bbox", [])
        if image is None or len(box) != 4:
            invalid_image_ids.add(image_id)
            continue
        try:
            x, y, width, height = map(float, box)
            image_width, image_height = float(image["width"]), float(image["height"])
            valid = (all(math.isfinite(value) for value in (x,y,width,height,image_width,image_height))
                     and width > 0 and height > 0 and image_width > 0 and image_height > 0
                     and x >= 0 and y >= 0 and x+width <= image_width+1e-6
                     and y+height <= image_height+1e-6)
        except (ValueError, TypeError, KeyError):
            valid = False
        if not valid or int(annotation["category_id"]) not in (fire_id, smoke_id):
            invalid_image_ids.add(image_id)
            continue
        by_image[image_id].append(annotation)
    all_fire_only = 0
    high_resolution = 0
    for image_id, image in images.items():
        if image_id in invalid_image_ids:
            continue
        annotations = by_image.get(image_id, [])
        if {int(item["category_id"]) for item in annotations} != {fire_id}:
            continue
        all_fire_only += 1
        width, height = int(image["width"]), int(image["height"])
        if max(width, height) < 640:
            continue
        high_resolution += 1
        fire_boxes = [list(map(float, item["bbox"])) for item in annotations]
        shortest_box = min(min(box[2], box[3]) * 640 / max(width, height)
                           for box in fire_boxes)
        pool.append({"split": split, "image_id": image_id,
                     "file_name": str(image["file_name"]), "width": width, "height": height,
                     "fire_boxes": fire_boxes, "smallest_fire_box_short_side_640": shortest_box})
    source_summary.append({"split": split, "images": len(images),
                           "fire_only_valid_metadata": all_fire_only,
                           "fire_only_long_side_at_least_640": high_resolution,
                           "invalid_annotation_images": len(invalid_image_ids)})

print("Pool:", len(pool), "ảnh")
print(source_summary)
if not pool:
    raise RuntimeError("Không có ảnh ứng viên; kiểm tra phiên bản dataset Kaggle")
''', 3))

cells.append(make_cell("markdown", """## Lấy mẫu phân tầng

Lấy tối đa 240 ảnh ngẫu nhiên theo tỷ lệ split nguồn và thêm tối đa 60 ảnh có box lửa nhỏ hơn 16 px sau khi quy cạnh dài về 640. Seed cố định để có thể chạy lại. Lấy mẫu theo ảnh, **không** tuyên bố ảnh/cảnh độc lập. Ảnh có tiền tố trùng nhưng ID khác hoặc cùng cảnh dưới tên khác vẫn có thể xuất hiện.
""", 4))

cells.append(make_cell("code", r'''rng = random.Random(SEED)
by_split = defaultdict(list)
for item in pool:
    by_split[item["split"]].append(item)

selected = []
general_quota = {"train": 110, "val": 80, "test": 50}
for split, quota in general_quota.items():
    selected.extend(rng.sample(by_split[split], min(quota, len(by_split[split]))))
selected_keys = {(item["split"], item["image_id"]) for item in selected}
small_pool = [item for item in pool if item["smallest_fire_box_short_side_640"] < 16
              and (item["split"], item["image_id"]) not in selected_keys]
selected.extend(rng.sample(small_pool, min(60, len(small_pool))))

def resolve_image(file_name):
    relative = Path(str(file_name).replace("\\", "/"))
    candidates = [ROOT / "images" / relative, ROOT / relative]
    for index, part in enumerate(relative.parts):
        if part.lower() == "images" and index + 1 < len(relative.parts):
            candidates.append(ROOT / "images" / Path(*relative.parts[index+1:]))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]

def make_tile(item):
    path = resolve_image(item["file_name"])
    with Image.open(path) as opened:
        original_size = opened.size
        if original_size != (item["width"], item["height"]):
            raise ValueError(f"COCO size {item['width']}x{item['height']} != header {original_size}")
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((310, 205), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (330, 250), "white")
    x_offset = (330 - image.width) // 2
    y_offset = 28 + (205 - image.height) // 2
    tile.paste(image, (x_offset, y_offset))
    draw = ImageDraw.Draw(tile)
    scale_x = image.width / item["width"]
    scale_y = image.height / item["height"]
    for x, y, width, height in item["fire_boxes"]:
        draw.rectangle((x_offset+x*scale_x, y_offset+y*scale_y,
                        x_offset+(x+width)*scale_x, y_offset+(y+height)*scale_y),
                       outline="red", width=2)
    draw.text((5, 3), f"{item['split']} | {Path(item['file_name']).name}", fill="black")
    draw.text((5, 235), f"{item['width']}x{item['height']} | min fire {item['smallest_fire_box_short_side_640']:.1f}px @640", fill="black")
    return tile

rendered = []
errors = []
for item in tqdm(selected, desc="Tạo ảnh ghép", unit="ảnh"):
    try:
        rendered.append((item, make_tile(item)))
    except Exception as error:
        errors.append({"split": item["split"], "file_name": item["file_name"],
                       "error": f"{type(error).__name__}: {error}"})

sheet_dir = OUTPUT / "contact_sheets"
sheet_dir.mkdir(exist_ok=True)
SHEET_SIZE = 20
for page_index in range(0, len(rendered), SHEET_SIZE):
    page = Image.new("RGB", (1320, 1250), "white")
    for offset, (_, tile) in enumerate(rendered[page_index:page_index+SHEET_SIZE]):
        page.paste(tile, ((offset % 4)*330, (offset // 4)*250))
    page.save(sheet_dir / f"fire_candidates_{page_index//SHEET_SIZE+1:03d}.jpg",
              format="JPEG", quality=88, optimize=True)

with (OUTPUT / "review_manifest.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.writer(stream)
    writer.writerow(["split", "image_id", "file_name", "width", "height", "fire_box_count",
                     "smallest_fire_box_short_side_640", "review_real_incident",
                     "review_smoke_complete", "review_box_correct", "review_same_scene_group", "notes"])
    for item in selected:
        writer.writerow([item["split"], item["image_id"], item["file_name"], item["width"],
                         item["height"], len(item["fire_boxes"]),
                         round(item["smallest_fire_box_short_side_640"], 2), "", "", "", "", ""])

with (OUTPUT / "image_errors.csv").open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=["split", "file_name", "error"])
    writer.writeheader()
    writer.writerows(errors)

summary = {"dataset_root": str(ROOT), "coco_json_sha256": source_hashes,
           "source_summary": source_summary, "high_resolution_fire_only_pool": len(pool),
           "general_random_sample": sum(min(quota, len(by_split[split])) for split, quota in general_quota.items()),
           "extra_small_box_sample": len(selected)-sum(min(quota, len(by_split[split])) for split, quota in general_quota.items()),
           "selected_total": len(selected), "rendered_total": len(rendered),
           "image_errors": len(errors), "seed": SEED,
           "scope": "Mẫu phân tầng để quyết định có tiếp tục tuyển ảnh; chưa duyệt toàn pool, chưa đối chiếu D-Fire."}
(OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

archive_path = Path("/kaggle/working/fasdd_cv_fire_candidate_review.zip")
with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT))
with zipfile.ZipFile(archive_path) as archive:
    failed = archive.testzip()
if failed:
    raise RuntimeError(f"ZIP CRC lỗi: {failed}")
print("Gửi lại file:", archive_path)
print("Dung lượng MiB:", round(archive_path.stat().st_size / 2**20, 2))
''', 5))

notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                           "language_info": {"name": "python", "version": "3.11"}},
            "nbformat": 4, "nbformat_minor": 5}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
