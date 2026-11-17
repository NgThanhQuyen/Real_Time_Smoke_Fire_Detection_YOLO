"""Find same-photo candidates between the two fire sources."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fire_source_overlap'
OUT.mkdir(parents=True,exist_ok=True)
def read(path):
    with path.open(encoding='utf-8',newline='') as handle:return list(csv.DictReader(handle))
cqu=read(ROOT/'reports/source_cqu_audit/fire_only_index.csv')
indoor=read(ROOT/'reports/indoor_fire_audit/fire_only_cross_locked.csv')
ih=[(r['image'],r['scene_key'],int(r['dhash'],16),int(r['phash'],16)) for r in indoor]
pairs=[]
for k,r in enumerate(cqu,1):
    dh,ph=int(r['dhash'],16),int(r['phash'],16)
    for name,scene,d,p in ih:
        hd=(dh^d).bit_count()
        if hd>6:continue
        hp=(ph^p).bit_count()
        if hp<=12:pairs.append((r['image'],name,scene,hd,hp))
    if k%1000==0:print(k,'/',len(cqu),flush=True)
with (OUT/'cross_source_hash_pairs.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.writer(handle);writer.writerow(['cqu_image','indoor_image','indoor_scene_key','dhash_distance','phash_distance']);writer.writerows(pairs)
print('joint hash pairs',len(pairs),'cqu images',len(set(p[0] for p in pairs)),
      'indoor images',len(set(p[1] for p in pairs)))
