"""Create a balanced fire-only visual review queue, never a train dataset."""

from collections import Counter
from pathlib import Path
import re
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/fire_source_overlap'
OUT.mkdir(parents=True,exist_ok=True)

geom=pd.read_csv(OUT/'fire_geometry.csv')[['image','max_box_area']]
cqu=pd.read_csv(ROOT/'reports/source_cqu_audit/fire_only_cross_dfire.csv')[['image','stem','same_dfire_stem']]
cqu=cqu.merge(pd.read_csv(ROOT/'reports/source_cqu_audit/model_probe_full.csv'),on='image')
cqu=cqu.merge(pd.read_csv(ROOT/'reports/fog_cloud_audit/fire_cross_locked.csv')[['image','joint_hash_candidates']],on='image')
cqu=cqu.merge(geom,on='image')
cqu=cqu.merge(pd.read_csv(ROOT/'reports/source_cqu_audit/fire_only_index.csv')[['image','dhash','phash']],on='image')
cqu['source']='CQUniversity'
cqu['raw_stem']=cqu['stem']
cqu['group']=cqu['stem'].str.extract(r'^([a-z]+)')[0].fillna('numeric')
cqu['width']=640;cqu['height']=640
cqu['reason']=''
cqu.loc[cqu.stem.str.startswith('img_20231008'),'reason']='tire_burn_series_with_unlabeled_smoke'
cqu.loc[(cqu.joint_hash_candidates>0)|(cqu.same_dfire_stem==1),'reason']='locked_dfire_overlap_review'
cqu.loc[(cqu.reason=='')&(cqu.max_box_area<.005),'reason']='fire_box_too_small_for_this_round'
cqu.loc[(cqu.reason=='')&(cqu.smoke_max_conf>=.25),'reason']='possible_unlabeled_smoke_review'

indoor=pd.read_csv(ROOT/'reports/indoor_fire_audit/fire_only_cross_locked.csv')
indoor=indoor.merge(pd.read_csv(ROOT/'reports/indoor_fire_audit/model_probe.csv')[['image','smoke_max_conf','fire_max_conf']],on='image')
indoor=indoor.merge(geom,on='image')
indoor['source']='Indoor'
indoor['raw_stem']=indoor.image.str.split('/').str[-1].str.replace(r'_(?:png_)?jpg.rf.*','',regex=True)
indoor['group']=indoor.scene_key
indoor.loc[indoor.group.isin(['img','Img','frame']),'group']=indoor.raw_stem
indoor['reason']=''
overlap=pd.read_csv(OUT/'cross_source_hash_pairs.csv')
indoor.loc[indoor.image.isin(overlap.indoor_image),'reason']='cross_source_same_photo_review'
indoor.loc[indoor.joint_hash_candidates>0,'reason']='locked_dfire_overlap_review'
indoor.loc[(indoor.reason=='')&(indoor[['width','height']].min(axis=1)<320),'reason']='low_resolution_review'
indoor.loc[(indoor.reason=='')&(indoor.max_box_area<.005),'reason']='fire_box_too_small_for_this_round'
indoor.loc[(indoor.reason=='')&(indoor.smoke_max_conf>=.25),'reason']='possible_unlabeled_smoke_review'

def pick(source,max_count,group_cap=None):
    eligible=source[source.reason==''].copy()
    eligible['rank']=eligible.fire_max_conf-0.6*eligible.smoke_max_conf
    eligible=eligible.sort_values(['rank','max_box_area'],ascending=[False,False])
    eligible=eligible.drop_duplicates('raw_stem',keep='first')
    items=list(eligible.itertuples())
    parent=list(range(len(items)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    hashes=[(int(r.dhash,16),int(r.phash,16)) for r in items]
    for i,(d,p) in enumerate(hashes):
        for j in range(i):
            x,y=hashes[j]
            if (d^x).bit_count()<=3 and (p^y).bit_count()<=8:
                parent[root(i)]=root(j)
    group_counts=Counter();hash_counts=Counter();picked=[]
    for i,row in enumerate(items):
        if group_cap and group_counts[row.group]>=group_cap:continue
        cluster=root(i)
        if hash_counts[cluster]>=5:continue
        group_counts[row.group]+=1;hash_counts[cluster]+=1
        picked.append(row.image)
        if len(picked)>=max_count:break
    return picked,len(eligible)

cqu_picked,cqu_unique=pick(cqu,750)
indoor_picked,indoor_unique=pick(indoor,750,20)
selected=set(cqu_picked+indoor_picked)
columns=['source','image','raw_stem','group','width','height','max_box_area',
         'smoke_max_conf','fire_max_conf','joint_hash_candidates','reason']
both=pd.concat([cqu[columns],indoor[columns]],ignore_index=True)
both['status']='visual_review_queue'
both.loc[~both.image.isin(selected),'status']='reserve_or_excluded'
both.loc[(both.reason=='')&~both.image.isin(selected),'reason']='not_in_first_review_batch_or_variant'
both.to_csv(OUT/'fire_review_manifest.csv',index=False,encoding='utf-8-sig')
order={name:i for i,name in enumerate(cqu_picked+indoor_picked)}
queue=both[both.image.isin(selected)].copy()
queue['review_order']=queue.image.map(order)
queue=queue.sort_values('review_order')
queue.to_csv(OUT/'fire_review_queue_1500.csv',index=False,encoding='utf-8-sig')
print('CQU eligible unique stems',cqu_unique,'queued',len(cqu_picked))
print('Indoor eligible unique stems',indoor_unique,'queued',len(indoor_picked))
print('reason counts',both.reason.value_counts().to_dict())
