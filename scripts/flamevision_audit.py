"""Kiểm kê ảnh và nhãn Detection của FlameVision mà không giải nén nguồn."""

import csv
import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image
from catargiu_audit import hashes

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('F:/6/KLTN/Smoke-Fire-Dataset/FlameVision.zip')
OUT = ROOT / 'reports/flamevision_audit'
OUT.mkdir(parents=True, exist_ok=True)

rows = []
with ZipFile(SOURCE) as archive:
    names = set(archive.namelist())
    images = sorted(n for n in names if '/Detection/' in n and '/images/' in n and n.lower().endswith('.jpg'))
    for index, name in enumerate(images, 1):
        label = name.replace('/images/', '/annotations/').rsplit('.', 1)[0] + '.xml'
        issues = []
        counts = Counter()
        areas = []
        if label not in names:
            issues.append('missing_xml')
        else:
            root = ET.fromstring(archive.read(label))
            size = root.find('size')
            xml_width = int(size.findtext('width', default='0')) if size is not None else 0
            xml_height = int(size.findtext('height', default='0')) if size is not None else 0
            for obj in root.findall('object'):
                cls = (obj.findtext('name') or '').strip().lower()
                counts[cls] += 1
                box = obj.find('bndbox')
                try:
                    xmin = float(box.findtext('xmin'))
                    ymin = float(box.findtext('ymin'))
                    xmax = float(box.findtext('xmax'))
                    ymax = float(box.findtext('ymax'))
                    if not (0 <= xmin < xmax <= xml_width and 0 <= ymin < ymax <= xml_height):
                        issues.append('invalid_box')
                    elif cls == 'fire':
                        areas.append((xmax - xmin) * (ymax - ymin) / (xml_width * xml_height))
                except (AttributeError, TypeError, ValueError, ZeroDivisionError):
                    issues.append('malformed_box')
        data = archive.read(name)
        try:
            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
                dhash, phash = hashes(image)
        except Exception as exc:
            issues.append('decode_error:' + type(exc).__name__)
            width = height = 0
            dhash = phash = ''
        if label in names and (width, height) != (xml_width, xml_height):
            issues.append('size_mismatch')
        raw_stem = re.sub(r'_png\.rf\.[0-9a-f]+$', '', Path(name).stem, flags=re.IGNORECASE)
        rows.append({
            'image': name, 'xml': label, 'split': name.split('/Detection/')[1].split('/')[0],
            'raw_stem': raw_stem, 'width': width, 'height': height,
            'sha256': hashlib.sha256(data).hexdigest(), 'dhash': dhash, 'phash': phash,
            'fire_boxes': counts['fire'], 'smoke_boxes': counts['smoke'],
            'other_boxes': sum(v for k, v in counts.items() if k not in ('fire', 'smoke')),
            'max_fire_area': max(areas, default=0), 'issues': '|'.join(sorted(set(issues))),
        })
        if index % 500 == 0:
            print('indexed', index, '/', len(images), flush=True)

with (OUT / 'detection_index.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print('images', len(rows))
print('splits', Counter(row['split'] for row in rows))
print('classes', Counter((row['fire_boxes'] > 0, row['smoke_boxes'] > 0, row['other_boxes'] > 0) for row in rows))
print('issues', Counter(row['issues'] for row in rows))
print('raw_stems', len({row['raw_stem'] for row in rows}))
print('sha_unique', len({row['sha256'] for row in rows}))
