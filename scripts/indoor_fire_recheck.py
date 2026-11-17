"""Recheck all Indoor fire-only labels against locked D-Fire val/test."""

import csv
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/indoor_fire_audit'
OUT.mkdir(parents=True,exist_ok=True)
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))
index=[r for r in read(ROOT/'runtime/new_sources_audit/image_index.csv') if r['source']=='Indoor_Fire_Smoke.zip']
fire=[r for r in index if int(r['class_0_boxes'])>0 and int(r['class_1_boxes'])==0 and not r['issue']]
manifest=read(ROOT/'runtime/dfire_clean_v2/metadata/manifest.csv')
locked_sha={r['source_sha256']:r['output_split'] for r in manifest if r['output_split'] in ('val','test')}
locked=[r for r in read(ROOT/'runtime/dfire_raw_review/image_index.csv') if r['sha256'] in locked_sha]
locked_hash=[(r['image'],int(r['dhash'],16),int(r['phash'],16),locked_sha[r['sha256']]) for r in locked]
out=[]
for k,r in enumerate(fire,1):
    d,p=int(r['dhash'],16),int(r['phash'],16)
    best=min(locked_hash,key=lambda q:(d^q[1]).bit_count())
    nearest_d=(d^best[1]).bit_count();nearest_p=(p^best[2]).bit_count()
    joint=[q for q in locked_hash if (d^q[1]).bit_count()<=6 and (p^q[2]).bit_count()<=12]
    out.append({'image':r['image'],'label':r['label'],'scene_key':r['scene_key'],
                'source_split':r['split'],'width':r['width'],'height':r['height'],
                'fire_boxes':r['class_0_boxes'],'sha256':r['sha256'],'dhash':r['dhash'],'phash':r['phash'],
                'min_dhash_locked':nearest_d,'phash_at_min_dhash':nearest_p,'nearest_locked':best[0],
                'joint_hash_candidates':len(joint),'joint_hash_nearest':joint[0][0] if joint else ''})
    if k%600==0:print('crosschecked',k,'/',len(fire),flush=True)
with (OUT/'fire_only_cross_locked.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,out[0].keys());writer.writeheader();writer.writerows(out)
print('source fire-only',len(fire),'unique scene_key',len(set(r['scene_key'] for r in fire)),
      'duplicate sha extra',len(fire)-len(set(r['sha256'] for r in fire)),
      'joint locked',sum(r['joint_hash_candidates']>0 for r in out),
      'dhash<=6 locked',sum(r['min_dhash_locked']<=6 for r in out))
