"""Render every queued fire-only image with source fire boxes for visual QA."""
import csv
import argparse
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw

parser=argparse.ArgumentParser()
parser.add_argument('--queue',default='reports/fire_source_overlap/fire_review_queue_1500.csv')
parser.add_argument('--output',default='reports/fire_source_overlap/review_pages')
parser.add_argument('--prefix',default='')
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/args.output
OUT.mkdir(parents=True,exist_ok=True)
with (ROOT/args.queue).open(encoding='utf-8-sig',newline='') as handle:
    rows=list(csv.DictReader(handle))
sources={'CQUniversity':'F:/6/KLTN/Smoke-Fire-Dataset/MyData_Fire.zip',
         'Indoor':'F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip'}
with zipfile.ZipFile(sources['CQUniversity']) as cqu,zipfile.ZipFile(sources['Indoor']) as indoor:
    archives={'CQUniversity':cqu,'Indoor':indoor}
    for page,start in enumerate(range(0,len(rows),30),1):
        canvas=Image.new('RGB',(1200,6*205),'white');draw=ImageDraw.Draw(canvas)
        for k,row in enumerate(rows[start:start+30]):
            archive=archives[row['source']]
            with Image.open(BytesIO(archive.read(row['image']))) as original:
                image=original.convert('RGB')
            label=row['image'].replace('/images/','/labels/').rsplit('.',1)[0]+'.txt'
            boxes=archive.read(label).decode('utf-8-sig')
            boxdraw=ImageDraw.Draw(image)
            for line in boxes.splitlines():
                if not line.strip():continue
                cls,x,y,w,h=map(float,line.split())
                if cls==0:
                    W,H=image.size
                    boxdraw.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),outline='lime',width=max(2,W//250))
            thumb=ImageOps.contain(image,(236,170))
            x,y=(k%5)*240,(k//5)*205
            canvas.paste(thumb,(x,y+28))
            uid=args.prefix+('C' if row['source']=='CQUniversity' else 'I')+f"{start+k+1:04}"
            draw.text((x+2,y+2),uid+' '+row['raw_stem'][:18],fill='black')
        canvas.save(OUT/f'page_{page:02}.jpg',quality=86)
        if page%10==0:print(page,'/',(len(rows)+29)//30,flush=True)
