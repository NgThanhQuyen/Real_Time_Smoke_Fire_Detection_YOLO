"""Measure perceptual concentration in current visual review queue."""
import csv
from collections import Counter,defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fire_source_overlap'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:return list(csv.DictReader(handle))
queue=read(OUT/'fire_review_queue_1500.csv')
index={r['image']:(int(r['dhash'],16),int(r['phash'],16)) for r in read(ROOT/'reports/source_cqu_audit/fire_only_index.csv')}
index.update({r['image']:(int(r['dhash'],16),int(r['phash'],16)) for r in read(ROOT/'reports/indoor_fire_audit/fire_only_cross_locked.csv')})
parent=list(range(len(queue)))
def root(i):
    while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
    return i
for i,a in enumerate(queue):
    d,p=index[a['image']]
    for j in range(i):
        x,y=index[queue[j]['image']]
        if (d^x).bit_count()<=3 and (p^y).bit_count()<=8:parent[root(i)]=root(j)
groups=defaultdict(list)
for i,r in enumerate(queue):groups[root(i)].append(r)
sizes=Counter(len(g) for g in groups.values())
print('groups',len(groups),'size distribution',sizes,'top',sorted((len(g),g[0]['image']) for g in groups.values())[-15:])
