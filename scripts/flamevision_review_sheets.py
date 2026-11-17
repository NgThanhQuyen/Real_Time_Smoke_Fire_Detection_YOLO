"""Vẽ mẫu ảnh Detection FlameVision cùng box nguồn để duyệt bằng mắt."""

import argparse
import random
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('F:/6/KLTN/Smoke-Fire-Dataset/FlameVision.zip')
OUT = ROOT / 'reports/flamevision_audit'
parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=180)
args = parser.parse_args()

index = pd.read_csv(OUT / 'detection_index.csv').fillna('')
rng = random.Random(42)
sample = []
for split, group in index.groupby('split'):
    sample.extend(rng.sample(list(group.itertuples()), round(args.count * len(group) / len(index))))
rng.shuffle(sample)
pd.DataFrame([r._asdict() for r in sample]).to_csv(OUT / 'visual_sample.csv', index=False, encoding='utf-8-sig')

with ZipFile(SOURCE) as archive:
    for page, start in enumerate(range(0, len(sample), 30), 1):
        canvas = Image.new('RGB', (1200, 6 * 205), 'white')
        draw = ImageDraw.Draw(canvas)
        for cell, row in enumerate(sample[start:start + 30]):
            with Image.open(BytesIO(archive.read(row.image))) as original:
                image = original.convert('RGB')
            root = ET.fromstring(archive.read(row.xml))
            pen = ImageDraw.Draw(image)
            for obj in root.findall('object'):
                box = obj.find('bndbox')
                x1, y1, x2, y2 = (float(box.findtext(k)) for k in ('xmin', 'ymin', 'xmax', 'ymax'))
                pen.rectangle((max(0, x1), max(0, y1), min(image.width - 1, x2), min(image.height - 1, y2)), outline='lime', width=2)
            thumb = ImageOps.contain(image, (236, 170))
            x, y = (cell % 5) * 240, (cell // 5) * 205
            canvas.paste(thumb, (x, y + 28))
            draw.text((x + 2, y + 2), f'V{start + cell + 1:03} {row.raw_stem[:18]}', fill='black')
        canvas.save(OUT / f'visual_page_{page:02}.jpg', quality=88)
print('sampled', len(sample))
