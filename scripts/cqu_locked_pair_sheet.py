"""Inspect CQUniversity fire-only hash candidates against locked D-Fire."""
import csv
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'reports/source_cqu_audit'
with (ROOT/'reports/fog_cloud_audit/fire_cross_locked.csv').open(encoding='utf-8',newline='') as handle:
    rows=[r for r in csv.DictReader(handle) if int(r['joint_hash_candidates'])>0]
with (ROOT/'runtime/dfire_clean_v2/metadata/manifest.csv').open(encoding='utf-8-sig',newline='') as handle:
    locked_path={r['source_image']:r['output_image'] for r in csv.DictReader(handle)
                 if r['output_split'] in ('val','test')}
for page,start in enumerate(range(0,len(rows),15),1):
    items=rows[start:start+15]
    canvas=Image.new('RGB',(1200,5*200),'white');draw=ImageDraw.Draw(canvas)
    with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/MyData_Fire.zip') as archive:
        for k,r in enumerate(items):
            x,y=(k%3)*400,(k//3)*200
            draw.text((x+2,y+2),Path(r['image']).stem[:22],fill='black')
            for off,source in ((0,r['image']),(200,r['joint_hash_nearest'])):
                if off==0:
                    image=Image.open(BytesIO(archive.read(source)))
                else:
                    image=Image.open(ROOT/'runtime/dfire_clean_v2'/locked_path[source])
                with image:
                    thumb=ImageOps.contain(image.convert('RGB'),(196,170))
                canvas.paste(thumb,(x+off,y+22))
    canvas.save(OUT/f'locked_pair_page_{page:02}.jpg',quality=90)
