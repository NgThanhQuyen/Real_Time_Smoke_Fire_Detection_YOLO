"""Measure fire-box visibility in two source-labeled fire-only pools."""
import csv
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCES=[('CQUniversity','F:/6/KLTN/Smoke-Fire-Dataset/MyData_Fire.zip',
          ROOT/'reports/source_cqu_audit/fire_only_index.csv'),
         ('Indoor','F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip',
          ROOT/'reports/indoor_fire_audit/fire_only_cross_locked.csv')]
OUT=ROOT/'reports/fire_source_overlap/fire_geometry.csv'
rows=[]
for source,zip_path,index_path in SOURCES:
    with index_path.open(encoding='utf-8',newline='') as handle:
        index=list(csv.DictReader(handle))
    with zipfile.ZipFile(zip_path) as archive:
        for i,r in enumerate(index,1):
            name=r['image'];label=name.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            boxes=[];issues=[]
            for line in archive.read(label).decode('utf-8-sig').splitlines():
                if not line.strip():continue
                parts=line.split()
                try:
                    cls,x,y,w,h=map(float,parts)
                    if cls!=0 or w<=0 or h<=0 or min(x-w/2,y-h/2)<-1e-4 or max(x+w/2,y+h/2)>1+1e-4:
                        issues.append('unexpected_box')
                    else:boxes.append((w*h,w,h))
                except ValueError:issues.append('malformed_box')
            rows.append({'source':source,'image':name,'fire_boxes':len(boxes),
                         'max_box_area':max((b[0] for b in boxes),default=0),
                         'sum_box_area':sum(b[0] for b in boxes),
                         'issue':';'.join(issues)})
            if i%1000==0:print(source,i,'/',len(index),flush=True)
with OUT.open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,rows[0].keys());writer.writeheader();writer.writerows(rows)
print('measured',len(rows),'images')
