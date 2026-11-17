"""Index fog and cloud negatives against the locked D-Fire val/test split."""

import csv
import json
import zipfile
from collections import Counter
from io import BytesIO, StringIO
from pathlib import Path

from PIL import Image
from catargiu_audit import hashes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fog_cloud_audit'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = Path('F:/6/KLTN/Smoke-Fire-Dataset')

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

manifest = read_csv(ROOT / 'runtime/dfire_clean_v2/metadata/manifest.csv')
locked_sha = {r['source_sha256']: r['output_split'] for r in manifest if r['output_split'] in ('val', 'test')}
raw = {r['sha256']: r for r in read_csv(ROOT / 'runtime/dfire_raw_review/image_index.csv')}
locked = [(r['image'], int(r['dhash'], 16), int(r['phash'], 16), locked_sha[r['sha256']])
          for sha, r in raw.items() if sha in locked_sha]
print('locked reference:', len(locked), flush=True)

fog = []
with zipfile.ZipFile(SOURCE / 'Fog.zip') as archive:
    for name in sorted(n for n in archive.namelist() if n.lower().endswith('.jpg')):
        with Image.open(BytesIO(archive.read(name))) as image:
            image.load()
            dh, ph = hashes(image)
        fog.append({'source': 'Fog', 'image': name, 'camera': '', 'timestamp': '',
                    'dhash': dh, 'phash': ph})

cloud = []
index = {r['image']: r for r in read_csv(ROOT / 'runtime/new_sources_audit/image_index.csv')
         if r['source'] == 'AIForMankind_Cloud_processed.zip'}
with zipfile.ZipFile(SOURCE / 'AIForMankind_Cloud_processed.zip') as archive:
    metadata_name = next(n for n in archive.namelist() if n.endswith('/metadata/metadata.csv'))
    metadata = list(csv.DictReader(StringIO(archive.read(metadata_name).decode('utf-8-sig'))))
    for row in metadata:
        name = 'AIForMankind_Cloud_processed/images/' + row['image']
        info = index[name]
        stem = Path(row['original_path']).stem
        camera, stamp = stem.rsplit('_', 1)
        cloud.append({'source': 'Cloud', 'image': name, 'camera': camera, 'timestamp': stamp,
                      'dhash': info['dhash'], 'phash': info['phash']})

rows = fog + cloud
for i, row in enumerate(rows, 1):
    dh, ph = int(row['dhash'], 16), int(row['phash'], 16)
    matches = sorted(((dh ^ d).bit_count(), (ph ^ p).bit_count(), name, split)
                     for name, d, p, split in locked)
    best = matches[0]
    joint = [m for m in matches if m[0] <= 6 and m[1] <= 12]
    row.update({'min_dhash_locked': best[0], 'phash_at_min_dhash': best[1],
                'nearest_locked': best[2], 'nearest_locked_split': best[3],
                'joint_hash_candidates': len(joint),
                'joint_hash_nearest': joint[0][2] if joint else ''})
    if i % 300 == 0:
        print('compared', i, '/', len(rows), flush=True)

with (OUT / 'negative_index.csv').open('w', encoding='utf-8', newline='') as handle:
    writer = csv.DictWriter(handle, rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {'fog': len(fog), 'cloud': len(cloud),
           'cloud_cameras': dict(Counter(r['camera'] for r in cloud)),
           'exact_hash_locked_candidates': sum(r['min_dhash_locked'] == 0 and r['phash_at_min_dhash'] == 0 for r in rows),
           'joint_hash_locked_candidates': sum(r['joint_hash_candidates'] > 0 for r in rows)}
(OUT / 'index_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(summary)
