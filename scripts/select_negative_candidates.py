"""Record reviewed fog/cloud candidates without creating a merged dataset."""

import csv
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fog_cloud_audit'
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as handle:
        return list(csv.DictReader(handle))

index=read(OUT/'negative_index.csv')
image_info={r['image']:r for r in read(OUT/'fog_image_index.csv')}
probe={r['image']:r for r in read(OUT/'negative_model_probe.csv')}
fog=[r for r in index if r['source']=='Fog']
parent=list(range(len(fog)))
def root(i):
    while parent[i]!=i:
        parent[i]=parent[parent[i]];i=parent[i]
    return i
for i,a in enumerate(fog):
    for j in range(i):
        b=fog[j]
        if image_info[a['image']]['sha256']==image_info[b['image']]['sha256'] or (
           (int(a['dhash'],16)^int(b['dhash'],16)).bit_count()<=3 and
           (int(a['phash'],16)^int(b['phash'],16)).bit_count()<=8):
            parent[root(i)]=root(j)
groups=defaultdict(list)
for i,r in enumerate(fog):groups[root(i)].append(r)

selected=set()
for members in groups.values():
    members.sort(key=lambda r:(int(image_info[r['image']]['width'])*int(image_info[r['image']]['height']),
               max(float(probe[r['image']]['smoke_max_conf']),float(probe[r['image']]['fire_max_conf']))),reverse=True)
    r=members[0]
    info=image_info[r['image']]
    if max(int(info['width']),int(info['height']))>=400:
        selected.add(r['image'])
cloud=[r for r in index if r['source']=='Cloud']
selected.update(probe[r['image']]['image'] for r in cloud if r['image'] in probe)

out=[]
for r in index:
    name=r['image'];flag=name in selected
    reason='visual_and_hash_review' if flag else ('low_resolution' if r['source']=='Fog' and max(int(image_info[name]['width']),int(image_info[name]['height']))<400 else 'same_frame_representative_selected')
    row={'source':r['source'],'image':name,'camera':r['camera'],'timestamp':r['timestamp'],
         'status':'selected_for_final_crosscheck' if flag else 'excluded',
         'reason':reason,'nearest_locked':r['nearest_locked'],
         'joint_hash_candidates':r['joint_hash_candidates'],
         'smoke_max_conf':probe[name]['smoke_max_conf'] if name in probe else '',
         'fire_max_conf':probe[name]['fire_max_conf'] if name in probe else ''}
    out.append(row)
with (OUT/'negative_candidate_manifest.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,out[0].keys());writer.writeheader();writer.writerows(out)
print('Fog',sum(r['source']=='Fog' and r['image'] in selected for r in index),
      'Cloud',sum(r['source']=='Cloud' and r['image'] in selected for r in index))
