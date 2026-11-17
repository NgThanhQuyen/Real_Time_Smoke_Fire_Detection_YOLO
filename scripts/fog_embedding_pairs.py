"""Find same-photo candidates in Fog using pretrained visual embeddings."""

import csv
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import resnet18, ResNet18_Weights

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fog_cloud_audit'
with (OUT/'negative_index.csv').open(encoding='utf-8',newline='') as handle:
    names=[r['image'] for r in csv.DictReader(handle) if r['source']=='Fog']
weights=ResNet18_Weights.IMAGENET1K_V1
model=resnet18(weights=weights)
model.fc=torch.nn.Identity()
model.eval()
transform=weights.transforms()
features=[]
with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/Fog.zip') as archive, torch.no_grad():
    for start in range(0,len(names),24):
        batch=[]
        for name in names[start:start+24]:
            with Image.open(BytesIO(archive.read(name))) as image:
                batch.append(transform(image.convert('RGB')))
        features.append(model(torch.stack(batch)).numpy())
        print(start+len(batch),'/',len(names),flush=True)
array=np.concatenate(features)
array=array/np.linalg.norm(array,axis=1,keepdims=True)
sim=array@array.T
pairs=[]
for i in range(len(names)):
    for j in range(i):
        if sim[i,j]>=.96:
            pairs.append((names[i],names[j],round(float(sim[i,j]),5)))
with (OUT/'fog_embedding_pairs.csv').open('w',encoding='utf-8',newline='') as handle:
    writer=csv.writer(handle);writer.writerow(['image_a','image_b','cosine']);writer.writerows(sorted(pairs,key=lambda r:-r[2]))
print('pairs >=.96:',len(pairs))
