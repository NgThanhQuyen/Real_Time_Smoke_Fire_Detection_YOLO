"""Show within-CQUniversity near pairs for same-frame review."""
import csv
import random
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/source_cqu_audit'
with (OUT/'within_source_near_pairs.csv').open(encoding='utf-8',newline='') as handle:rows=list(csv.DictReader(handle))
random.seed(9)
for level in ('very_strong','strong'):
    pool=[r for r in rows if r['level']==level]
    chosen=random.sample(pool,min(24,len(pool)))
    canvas=Image.new('RGB',(1200,8*175),'white');draw=ImageDraw.Draw(canvas)
    with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/MyData_Fire.zip') as archive:
        for k,r in enumerate(chosen):
            x,y=(k%3)*400,(k//3)*175
            draw.text((x+2,y+2),f"d{r['dhash_distance']} p{r['phash_distance']} {Path(r['image_a']).stem[:18]}",fill='black')
            for offset,key in ((0,'image_a'),(200,'image_b')):
                with Image.open(BytesIO(archive.read(r[key]))) as image:
                    thumb=ImageOps.contain(image.convert('RGB'),(196,145))
                canvas.paste(thumb,(x+offset,y+22))
    canvas.save(OUT/f'{level}_pair_sheet.jpg',quality=88)
