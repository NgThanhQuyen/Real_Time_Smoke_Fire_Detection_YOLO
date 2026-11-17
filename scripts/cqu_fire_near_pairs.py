"""Flag re-encoded same frames inside CQUniversity fire-only labels."""
import csv
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/source_cqu_audit'
with (OUT/'fire_only_index.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
h=[(int(r['dhash'],16),int(r['phash'],16)) for r in rows]
counts=Counter();pairs=[]
for i in range(len(rows)):
    for j in range(i):
        dh=(h[i][0]^h[j][0]).bit_count()
        if dh>3:continue
        ph=(h[i][1]^h[j][1]).bit_count()
        if ph>8:continue
        level='very_strong' if dh<=1 and ph<=4 else 'strong'
        counts[level]+=1
        pairs.append((rows[i]['image'],rows[j]['image'],dh,ph,level))
    if (i+1)%1000==0:print(i+1,'/',len(rows),flush=True)
with (OUT/'within_source_near_pairs.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.writer(handle);writer.writerow(['image_a','image_b','dhash_distance','phash_distance','level']);writer.writerows(pairs)
print(counts,'images in very strong pairs',len(set(x for p in pairs if p[4]=='very_strong' for x in p[:2])))
