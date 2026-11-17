"""Sample Indoor fire-only source images across scenes and model flags."""
import csv
import random
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageDraw,ImageOps

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/indoor_fire_audit'
def read(path):
    with path.open(encoding='utf-8',newline='') as handle:
        return list(csv.DictReader(handle))
cross=read(OUT/'fire_only_cross_locked.csv')
score={r['image']:r for r in read(OUT/'model_probe.csv')}
random.seed(42)
groups=defaultdict(list)
for r in cross:
    if not int(r['joint_hash_candidates']):groups[r['scene_key']].append(r)
one_per_group=[random.choice(g) for g in groups.values()]
random.shuffle(one_per_group)
pages=[('scene_sample',one_per_group[:40]),
       ('smoke_flag',sorted(one_per_group,key=lambda r:-float(score[r['image']]['smoke_max_conf']))[:40]),
       ('no_smoke_flag',[r for r in one_per_group if float(score[r['image']]['smoke_max_conf'])==0][:40]),
       ('video_frames',[r for r in cross if len(groups[r['scene_key']])>=10 and not int(r['joint_hash_candidates'])][:40])]
with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/Indoor_Fire_Smoke.zip') as archive:
    for title,items in pages:
        canvas=Image.new('RGB',(1200,10*220),'white');draw=ImageDraw.Draw(canvas)
        for k,r in enumerate(items):
            with Image.open(BytesIO(archive.read(r['image']))) as im:
                image=im.convert('RGB')
            # Class 0 is fire in this source; show every source fire box.
            label=archive.read(r['label']).decode('utf-8-sig')
            boxdraw=ImageDraw.Draw(image)
            for line in label.splitlines():
                cls,x,y,w,h=map(float,line.split())
                if cls==0:
                    W,H=image.size
                    boxdraw.rectangle(((x-w/2)*W,(y-h/2)*H,(x+w/2)*W,(y+h/2)*H),outline='lime',width=max(2,W//250))
            thumb=ImageOps.contain(image,(294,190))
            x,y=(k%4)*300,(k//4)*220
            canvas.paste(thumb,(x,y+24))
            s=score[r['image']]
            desc=f"{Path(r['image']).stem[:17]} S{float(s['smoke_max_conf']):.2f} F{float(s['fire_max_conf']):.2f}"
            draw.text((x+2,y+2),desc,fill='black')
        canvas.save(OUT/f'{title}.jpg',quality=88)
