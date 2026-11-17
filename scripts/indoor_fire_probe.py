"""Prioritize manual review of Indoor fire-only images with the D-Fire model."""
import csv
import os
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/indoor_fire_audit'
CONFIG=ROOT/'runtime/ultralytics_config'
CONFIG.mkdir(parents=True,exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(CONFIG)
from ultralytics import YOLO

with (OUT/'fire_only_cross_locked.csv').open(encoding='utf-8',newline='') as handle:
    rows=list(csv.DictReader(handle))
model=YOLO('C:/Users/User/Downloads/dfire_clean_v2_yolo11n_best.pt')
out=[]
with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip') as archive:
    for start in range(0,len(rows),16):
        batch=rows[start:start+16]
        images=[]
        for r in batch:
            with Image.open(BytesIO(archive.read(r['image']))) as image:
                # Ultralytics treats ndarray inputs as BGR, unlike PIL inputs.
                images.append(np.asarray(image.convert('RGB'))[:, :, ::-1].copy())
        pred=model.predict(images,imgsz=640,conf=.05,verbose=False,device='cpu')
        for r,result in zip(batch,pred):
            scores={0:0.0,1:0.0};counts={0:0,1:0}
            for box in result.boxes:
                cls=int(box.cls.item());conf=float(box.conf.item())
                counts[cls]+=1;scores[cls]=max(scores[cls],conf)
            out.append({'image':r['image'],'scene_key':r['scene_key'],
                        'smoke_max_conf':round(scores[0],5),'fire_max_conf':round(scores[1],5),
                        'smoke_boxes':counts[0],'fire_boxes':counts[1]})
        if (start+len(batch))%320==0 or start+len(batch)==len(rows):
            print(start+len(batch),'/',len(rows),flush=True)
with (OUT/'model_probe.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.DictWriter(handle,out[0].keys());writer.writeheader();writer.writerows(out)
