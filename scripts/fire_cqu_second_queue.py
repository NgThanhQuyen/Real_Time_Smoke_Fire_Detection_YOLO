"""Prepare additional CQUniversity images for manual review without merging them."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/fire_source_overlap'
manifest = pd.read_csv(OUT / 'fire_review_manifest.csv').fillna('')
cqu = manifest[manifest.source == 'CQUniversity'].copy()
first = set(pd.read_csv(OUT / 'fire_review_queue_1500.csv').image)

reserve = cqu[cqu.reason == 'not_in_first_review_batch_or_variant'].copy()
reserve = reserve[~reserve.raw_stem.isin(cqu[cqu.image.isin(first)].raw_stem)]
reserve = reserve.sort_values(['smoke_max_conf', 'fire_max_conf'], ascending=[True, False])
reserve = reserve.drop_duplicates('raw_stem')
reserve['review_order'] = range(len(reserve))
reserve.to_csv(OUT / 'cqu_clean_reserve_queue.csv', index=False, encoding='utf-8-sig')

flagged = cqu[cqu.reason == 'possible_unlabeled_smoke_review'].copy()
flagged = flagged[~flagged.raw_stem.isin(cqu[cqu.image.isin(first)].raw_stem)]
flagged = flagged.sort_values(['smoke_max_conf', 'fire_max_conf'], ascending=[True, False])
flagged = flagged.drop_duplicates('raw_stem').head(500)
flagged['review_order'] = range(len(flagged))
flagged.to_csv(OUT / 'cqu_smoke_flag_review_queue.csv', index=False, encoding='utf-8-sig')
print('Reserve unique stems:', len(reserve))
print('Smoke-flagged review queue:', len(flagged))
