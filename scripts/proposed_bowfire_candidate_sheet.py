"""Xem vùng mask và toàn cảnh các ảnh BoWFire có thể tuyển chọn."""

import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources/BoWFireDataset.zip"
REPORT = ROOT / "reports/proposed_sources/bowfire_candidates"
REPORT.mkdir(parents=True, exist_ok=True)
FIRE_IDS = (7, 14, 15, 19, 32, 37, 38, 39, 52, 53, 57, 58, 61, 62, 63, 65, 68, 69, 83, 84, 85, 87)

with zipfile.ZipFile(SOURCE) as archive:
    for number in FIRE_IDS:
        name = f"fire{number:03d}"
        image_path = f"BoWFireDataset/dataset/img/{name}.png"
        mask_path = f"BoWFireDataset/dataset/gt/{name}_gt.png"
        with Image.open(BytesIO(archive.read(image_path))) as image:
            image = image.convert("RGB")
        with Image.open(BytesIO(archive.read(mask_path))) as mask:
            box = mask.convert("L").point(lambda value: 255 if value > 127 else 0).getbbox()
        overlay = image.copy()
        if box:
            draw = ImageDraw.Draw(overlay)
            draw.rectangle(box, outline="red", width=max(3, image.width // 150))
        canvas = Image.new("RGB", (1200, 900), "white")
        photo = ImageOps.contain(image, (590, 820))
        annotated = ImageOps.contain(overlay, (590, 820))
        canvas.paste(photo, ((600 - photo.width) // 2, 40))
        canvas.paste(annotated, (600 + (600 - annotated.width) // 2, 40))
        ImageDraw.Draw(canvas).text((10, 8), f"{name}: image left, mask bbox right; original={image.size}, bbox={box}", fill="black")
        canvas.save(REPORT / f"{name}.jpg", quality=92)
    print(REPORT)
