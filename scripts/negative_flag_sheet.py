"""Show high-confidence false-positive candidates at review size."""
import csv
import zipfile
from io import BytesIO
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fog_cloud_audit'
with (OUT/'negative_model_probe.csv').open(encoding='utf-8',newline='') as handle:
    rows=list(csv.DictReader(handle))
rows.sort(key=lambda r:max(float(r['smoke_max_conf']),float(r['fire_max_conf'])),reverse=True)
for page,start in enumerate(range(0,min(len(rows),48),24),1):
    subset=rows[start:start+24]
    canvas=Image.new('RGB',(1200,6*225),'white');draw=ImageDraw.Draw(canvas)
    with zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/Fog.zip') as fog,zipfile.ZipFile('F:/6/KLTN/Smoke-Fire-Dataset/AIForMankind_Cloud_processed.zip') as cloud:
        for k,row in enumerate(subset):
            archive=fog if row['source']=='Fog' else cloud
            with Image.open(BytesIO(archive.read(row['image']))) as im:
                thumb=ImageOps.contain(im.convert('RGB'),(294,190))
            x,y=(k%4)*300,(k//4)*225
            canvas.paste(thumb,(x,y+28))
            label=f"{Path(row['image']).name} S{float(row['smoke_max_conf']):.2f} F{float(row['fire_max_conf']):.2f}"
            draw.text((x+2,y+3),label,fill='black')
    canvas.save(OUT/f'negative_flag_page_{page:02}.jpg',quality=92)
