"""Inspect hash-near Indoor pairs before deciding whether frames differ."""
import csv
import random
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/indoor_fire_audit'
with (OUT/'within_source_near_pairs.csv').open(encoding='utf-8',newline='') as handle:
    rows=list(csv.DictReader(handle))
random.seed(7)
for level in ('very_strong','strong'):
    pool=[r for r in rows if r['level']==level and r['scene_a']==r['scene_b']]
    chosen=random.sample(pool,min(24,len(pool)))
    canvas=Image.new('RGB',(1200,8*175),'white');draw=ImageDraw.Draw(canvas)
    with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip') as archive:
        for k,r in enumerate(chosen):
            x,y=(k%3)*400,(k//3)*175
            draw.text((x+2,y+2),f"{r['scene_a'][:22]} d{r['dhash_distance']} p{r['phash_distance']}",fill='black')
            for offset,key in ((0,'image_a'),(200,'image_b')):
                with Image.open(BytesIO(archive.read(r[key]))) as image:
                    thumb=ImageOps.contain(image.convert('RGB'),(196,145))
                canvas.paste(thumb,(x+offset,y+22))
    canvas.save(OUT/f'{level}_pair_sheet.jpg',quality=88)
