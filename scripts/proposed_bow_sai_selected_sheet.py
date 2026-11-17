"""Tạo ảnh ghép của toàn bộ mẫu vừa thêm để kiểm tra nhãn lần cuối."""

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_bow_sai_v1"
REPORT = ROOT / "reports/proposed_sources/bow_sai_selected.jpg"
with (DATA / "metadata/bow_sai_selected_manifest.csv").open(newline="", encoding="utf-8") as handle:
    selected = list(csv.DictReader(handle))

width, height, columns = 350, 250, 4
canvas = Image.new("RGB", (width * columns, height * 6), "white")
draw = ImageDraw.Draw(canvas)
for index, row in enumerate(selected):
    with Image.open(DATA / row["output_image"]) as source:
        image = source.convert("RGB")
    for line in (DATA / row["output_label"]).read_text(encoding="utf-8").splitlines():
        cls, x, y, w, h = line.split()
        x, y, w, h = map(float, (x, y, w, h))
        ImageDraw.Draw(image).rectangle(((x-w/2)*image.width, (y-h/2)*image.height,
                                         (x+w/2)*image.width, (y+h/2)*image.height),
                                        outline="lime" if cls == "1" else "red", width=5)
    thumb = ImageOps.contain(image, (width-8, height-30))
    left = index % columns * width
    top = index // columns * height
    canvas.paste(thumb, (left+(width-thumb.width)//2, top+25))
    draw.text((left+4, top+5), f"{row['image_name']} {row['category']}", fill="black")
canvas.save(REPORT, quality=92)
print(REPORT)
