"""Review evenly spaced cloud frames and probe negative photos with the baseline."""

import csv
import os
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fog_cloud_audit'
CONFIG = ROOT / 'runtime/ultralytics_config'
CONFIG.mkdir(parents=True, exist_ok=True)
os.environ['YOLO_CONFIG_DIR'] = str(CONFIG)
from ultralytics import YOLO

with (OUT / 'negative_index.csv').open(encoding='utf-8',newline='') as handle:
    rows=list(csv.DictReader(handle))
fog=[r for r in rows if r['source']=='Fog']
groups=defaultdict(list)
for r in rows:
    if r['source']=='Cloud':
        groups[r['camera']].append(r)
cloud=[]
for camera, frames in sorted(groups.items()):
    frames.sort(key=lambda r:int(r['timestamp']))
    cloud.extend(frames[round(k*(len(frames)-1)/5)] for k in range(6))

with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/AIForMankind_Cloud_processed.zip') as archive:
    canvas=Image.new('RGB',(1200, 8*125),'white')
    draw=ImageDraw.Draw(canvas)
    for k,r in enumerate(cloud):
        with Image.open(BytesIO(archive.read(r['image']))) as image:
            thumb=ImageOps.contain(image.convert('RGB'),(194,100))
        x,y=(k%6)*200,(k//6)*125
        canvas.paste(thumb,(x,y+20))
        draw.text((x+2,y+2),f"{r['camera'][:9]} #{k%6+1}",fill='black')
    canvas.save(OUT / 'cloud_36_spaced.jpg',quality=90)

model=YOLO('C:/Users/User/Downloads/dfire_clean_v2_yolo11n_best.pt')
output=[]
for source, selected, zip_path in [
    ('Fog',fog,'F:/6/KLTN/Smoke-Fire-Dataset/Fog.zip'),
    ('Cloud',cloud,'F:/6/KLTN/Smoke-Fire-Dataset/AIForMankind_Cloud_processed.zip')]:
    with zipfile.ZipFile(zip_path) as archive:
        for start in range(0,len(selected),12):
            batch=selected[start:start+12]
            images=[]
            for r in batch:
                with Image.open(BytesIO(archive.read(r['image']))) as image:
                    # Ultralytics treats ndarray inputs as BGR, unlike PIL inputs.
                    images.append(np.asarray(image.convert('RGB'))[:, :, ::-1].copy())
            predictions=model.predict(images, imgsz=640, conf=.05, verbose=False, device='cpu')
            for r, pred in zip(batch,predictions):
                scores={0:0.0,1:0.0}
                for box in pred.boxes:
                    cls=int(box.cls.item());scores[cls]=max(scores[cls],float(box.conf.item()))
                output.append({'source':source,'image':r['image'],'camera':r['camera'],
                               'smoke_max_conf':round(scores[0],5),'fire_max_conf':round(scores[1],5)})
            print(source,start+len(batch),'/',len(selected),flush=True)
with (OUT / 'negative_model_probe.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,output[0].keys());writer.writeheader();writer.writerows(output)
