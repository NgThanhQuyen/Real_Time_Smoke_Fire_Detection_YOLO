"""Giới hạn mức tập trung cảnh Indoor trước khi duyệt từng ảnh lửa."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fire_source_overlap'
queue = pd.read_csv(OUT / 'fire_review_queue_1500.csv').fillna('')
cqu = queue[queue.source == 'CQUniversity'].copy()
cqu = cqu[~cqu.raw_stem.str.startswith('rdr2_', na=False)]
indoor = queue[queue.source == 'Indoor'].copy()
indoor = indoor[~indoor.raw_stem.str.contains('fireai|gan|synthetic|datacluster|firefire', case=False, na=False)]

families = (
    'FireVid29', 'FireVid36', 'FireVid27', 'part1', 'part7', 'part8a',
    'part8b', '3_mp4', '6_mp4', 'firefold', 'barbecue', 'flame2_frames',
    'Inside-a-burning-house', 'FrontRoomFire', 'ClosingYourBedroom',
    '20240310_192502', '20240310_192253', '20240310_192535', 'mergedfire',
)
caps = {
    'fire_tlm': 70, 'mergedfire': 10, 'FrontRoomFire': 5,
    'barbecue': 5, 'firefold': 5,
}
caps.update({name: 3 for name in families if name not in caps})

def family(stem):
    if stem.startswith('fire_tlm'):
        return 'fire_tlm'
    for prefix in families:
        if stem.startswith(prefix):
            return prefix
    return stem

indoor['scene_family'] = indoor.raw_stem.map(family)
indoor['family_order'] = indoor.groupby('scene_family').cumcount()
indoor = indoor[indoor.apply(lambda row: row.family_order < caps.get(row.scene_family, 1), axis=1)]
indoor = indoor.drop(columns=['family_order'])
cqu['scene_family'] = cqu.group
shortlist = pd.concat([cqu, indoor], ignore_index=True)
shortlist['status'] = 'pending_full_visual_review'
shortlist.to_csv(OUT / 'fire_shortlist.csv', index=False, encoding='utf-8-sig')
print(shortlist.source.value_counts().to_dict())
print('indoor scene families', indoor.scene_family.nunique())
