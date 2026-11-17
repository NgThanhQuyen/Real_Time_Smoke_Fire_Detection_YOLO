"""Chốt ảnh âm tính đã xem trực quan và kiểm tra trùng với D-Fire."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fog_cloud_audit'
candidate = pd.read_csv(OUT / 'negative_candidate_manifest.csv').fillna('')
fog = pd.read_csv(OUT / 'fog_image_index.csv')[['image', 'sha256']]
cloud = pd.read_csv(ROOT / 'runtime/new_sources_audit/image_index.csv')
cloud = cloud[cloud.source == 'AIForMankind_Cloud_processed.zip'][['image', 'sha256']]
source_hashes = pd.concat([fog, cloud], ignore_index=True)
dfire_hashes = set(pd.read_csv(ROOT / 'runtime/dfire_raw_review/image_index.csv').sha256)
candidate = candidate.merge(source_hashes, on='image', validate='one_to_one')
selected = candidate.status == 'selected_for_final_crosscheck'
assert len(candidate[selected]) == 296
assert not candidate.loc[selected, 'sha256'].isin(dfire_hashes).any()
assert candidate.loc[selected, 'sha256'].is_unique
assert candidate.loc[selected & (candidate.joint_hash_candidates.astype(int) > 0), 'image'].tolist() == ['Fog/haze-007.jpg']

# Cặp haze-007/WEB10832 đã được mở ở ảnh gốc: hai cảnh khác nhau.
candidate.loc[selected, 'status'] = 'approved_for_next_train_if_fire_gate_met'
candidate.loc[selected, 'reason'] = 'manual_contact_sheet_and_cross_split_review'
candidate.loc[candidate.image == 'Fog/haze-007.jpg', 'reason'] = 'hash_collision_visually_distinct_from_locked_test'
candidate.to_csv(OUT / 'negative_final_manifest.csv', index=False, encoding='utf-8-sig')
print(candidate.status.value_counts().to_dict())
