"""Create complete visual review sheets for the user-supplied Fog archive."""

from io import BytesIO
from pathlib import Path
import csv
import hashlib
import zipfile

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('F:/6/KLTN/Smoke-Fire-Dataset/Fog.zip')
OUT = ROOT / 'reports/fog_cloud_audit'
OUT.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(SOURCE) as archive:
    names = sorted(n for n in archive.namelist() if n.lower().endswith('.jpg'))
    rows = []
    for page_no, start in enumerate(range(0, len(names), 50), 1):
        canvas = Image.new('RGB', (1200, 10 * 125), 'white')
        draw = ImageDraw.Draw(canvas)
        for k, name in enumerate(names[start:start+50]):
            data = archive.read(name)
            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
                thumb = ImageOps.contain(image.convert('RGB'), (232, 100))
            x, y = (k % 5) * 240, (k // 5) * 125
            canvas.paste(thumb, (x, y + 17))
            draw.text((x + 2, y + 2), Path(name).name, fill='black')
            rows.append((name, width, height, hashlib.sha256(data).hexdigest(), page_no))
        canvas.save(OUT / f'fog_page_{page_no:02}.jpg', quality=88)

with (OUT / 'fog_image_index.csv').open('w', newline='', encoding='utf-8') as handle:
    writer = csv.writer(handle)
    writer.writerow(['image', 'width', 'height', 'sha256', 'review_page'])
    writer.writerows(rows)
print(f'{len(rows)} images; {page_no} review sheets')
