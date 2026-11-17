"""Show suspected CQUniversity vs Indoor duplicate photos."""
import csv
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/fire_source_overlap'
with (OUT/'cross_source_hash_pairs.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
rows.sort(key=lambda r:(int(r['dhash_distance']),int(r['phash_distance'])))
chosen=[];seen=set()
for r in rows:
    if r['cqu_image'] in seen:continue
    chosen.append(r);seen.add(r['cqu_image'])
    if len(chosen)==24:break
canvas=Image.new('RGB',(1200,8*175),'white');draw=ImageDraw.Draw(canvas)
with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/MyData_Fire.zip') as cqu,zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip') as indoor:
    for k,r in enumerate(chosen):
        x,y=(k%3)*400,(k//3)*175
        draw.text((x+2,y+2),f"d{r['dhash_distance']} p{r['phash_distance']} {r['indoor_scene_key'][:18]}",fill='black')
        for off,archive,key in ((0,cqu,'cqu_image'),(200,indoor,'indoor_image')):
            with Image.open(BytesIO(archive.read(r[key]))) as image:
                thumb=ImageOps.contain(image.convert('RGB'),(196,145))
            canvas.paste(thumb,(x+off,y+22))
canvas.save(OUT/'cross_source_pairs_24.jpg',quality=88)
