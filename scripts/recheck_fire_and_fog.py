"""Recount candidate images under the revised within-train frame policy."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fog_cloud_audit'

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

negative = rows(OUT / 'negative_index.csv')
fog = [r for r in negative if r['source'] == 'Fog']
fog_sha = {r['image']: r['sha256'] for r in rows(OUT / 'fog_image_index.csv')}
parent = list(range(len(fog)))
def root(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i

pair_count = Counter()
for i in range(len(fog)):
    for j in range(i):
        a, b = fog[i], fog[j]
        dh = (int(a['dhash'], 16) ^ int(b['dhash'], 16)).bit_count()
        ph = (int(a['phash'], 16) ^ int(b['phash'], 16)).bit_count()
        if fog_sha[a['image']] == fog_sha[b['image']]:
            pair_count['same_byte_pairs'] += 1
            parent[root(i)] = root(j)
        elif dh <= 3 and ph <= 8:
            pair_count['same_frame_hash_candidates'] += 1
            parent[root(i)] = root(j)
        elif dh <= 6 and ph <= 12:
            pair_count['near_frame_review_candidates'] += 1
groups = defaultdict(list)
for i, r in enumerate(fog):
    groups[root(i)].append(r['image'])

fire = rows(ROOT / 'reports/source_cqu_audit/fire_only_index.csv')
manifest = rows(ROOT / 'runtime/dfire_clean_v2/metadata/manifest.csv')
locked_sha = {r['source_sha256']: r['output_split'] for r in manifest if r['output_split'] in ('val', 'test')}
raw = rows(ROOT / 'runtime/dfire_raw_review/image_index.csv')
locked = [(r['image'], int(r['dhash'],16), int(r['phash'],16), locked_sha[r['sha256']])
          for r in raw if r['sha256'] in locked_sha]
out = []
for k, r in enumerate(fire, 1):
    d, p = int(r['dhash'],16), int(r['phash'],16)
    best = min(locked, key=lambda q:(d^q[1]).bit_count())
    dh = (d^best[1]).bit_count()
    ph = (p^best[2]).bit_count()
    joint = [q for q in locked if (d^q[1]).bit_count()<=6 and (p^q[2]).bit_count()<=12]
    out.append({'image':r['image'], 'sha256':r['sha256'], 'min_dhash_locked':dh,
                'phash_at_min_dhash':ph, 'nearest_locked':best[0], 'joint_hash_candidates':len(joint),
                'joint_hash_nearest':joint[0][0] if joint else ''})
    if k % 1000 == 0:
        print('fire compared',k,'/',len(fire),flush=True)
with (OUT / 'fire_cross_locked.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,out[0].keys());writer.writeheader();writer.writerows(out)
summary = {'fog_total':len(fog),'fog_exact_extra_images':len(fog)-len(set(fog_sha.values())),
           'fog_strong_hash_groups':len(groups),'fog_pair_counts':dict(pair_count),
           'fog_joint_locked_candidates':sum(int(r['joint_hash_candidates'])>0 for r in fog),
           'fire_source_labeled_only':len(fire),
           'fire_joint_locked_candidates':sum(r['joint_hash_candidates']>0 for r in out),
           'fire_dhash_le6_locked_candidates':sum(r['min_dhash_locked']<=6 for r in out)}
(OUT / 'revised_counts.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(summary)
