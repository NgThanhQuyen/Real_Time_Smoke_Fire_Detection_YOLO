"""Đọc BoWFire gốc và tạo ảnh ghép cùng mask để xem khả năng dùng cho YOLO."""

import random
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "runtime/proposed_sources/BoWFireDataset.zip"
with zipfile.ZipFile(ZIP) as archive:
    names = set(archive.namelist())
    images = sorted(n for n in names if n.startswith("BoWFireDataset/dataset/img/") and n.lower().endswith(".png"))
    fire = [n for n in images if Path(n).stem.startswith("fire")]
    negative = [n for n in images if Path(n).stem.startswith("not_fire")]
    chosen = random.Random(29).sample(fire, min(20, len(fire))) + random.Random(17).sample(negative, min(16, len(negative)))
    width, height, columns = 300, 245, 6
    canvas = Image.new("RGB", (width * columns, height * ((len(chosen) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(canvas)
    sizes = Counter()
    missing_mask = []
    for index, name in enumerate(chosen):
        mask_name = name.replace("/img/", "/gt/").replace(".png", "_gt.png")
        if mask_name not in names:
            missing_mask.append(name)
        with Image.open(BytesIO(archive.read(name))) as image:
            sizes[image.size] += 1
            thumb = ImageOps.contain(image.convert("RGB"), (width, height - 25))
        x, y = index % columns * width, index // columns * height
        canvas.paste(thumb, (x + (width - thumb.width) // 2, y + 25))
        draw.text((x + 3, y + 3), Path(name).name, fill="black")
    target = ROOT / "reports/proposed_sources/bowfire_sample.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=88)
    print(f"images={len(images)} fire={len(fire)} negative={len(negative)} sampled_sizes={sizes} missing_mask={len(missing_mask)}")
    print(target)
