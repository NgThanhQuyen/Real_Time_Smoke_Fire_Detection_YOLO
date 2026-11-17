"""Tạo ảnh ghép có box từ bản Home Fire để kiểm tra lớp và nhãn."""

import random
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "runtime/proposed_sources/HomeFire_train.zip"
REPORT = ROOT / "reports/proposed_sources/home_fire_sample.jpg"
REPORT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as archive:
    images = sorted(n for n in archive.namelist() if n.startswith("images/") and n.lower().endswith(".jpg"))
    rng = random.Random(39)
    selected = rng.sample(images, min(48, len(images)))
    width, height, columns = 280, 220, 6
    canvas = Image.new("RGB", (width * columns, height * ((len(selected) + columns - 1) // columns)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, name in enumerate(selected):
        label_name = name.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
        labels = archive.read(label_name).decode("utf-8-sig").splitlines()
        with Image.open(BytesIO(archive.read(name))) as image:
            img = image.convert("RGB")
        box_draw = ImageDraw.Draw(img)
        for line in labels:
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = parts
            cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
            x1, y1 = (cx - bw / 2) * img.width, (cy - bh / 2) * img.height
            x2, y2 = (cx + bw / 2) * img.width, (cy + bh / 2) * img.height
            box_draw.rectangle((x1, y1, x2, y2), outline="lime" if cls == "0" else "red", width=4)
        thumb = ImageOps.contain(img, (width, height - 24))
        x, y = index % columns * width, index // columns * height
        canvas.paste(thumb, (x + (width - thumb.width) // 2, y + 24))
        draw.text((x + 3, y + 3), f"{Path(name).stem} {'/'.join(l.split()[0] for l in labels if l.strip())}", fill="black")
    canvas.save(REPORT, quality=88)
    print(len(images), REPORT)
