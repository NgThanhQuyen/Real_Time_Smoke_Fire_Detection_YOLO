"""Summarize source scene repetition and dimensions; no dataset mutations."""

import csv
import json
import zipfile
from collections import Counter, defaultdict
from io import TextIOWrapper
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("F:/6/KLTN/Smoke-Fire-Dataset")
rows = list(csv.DictReader((ROOT / "runtime/new_sources_audit/image_index.csv").open(encoding="utf-8")))
out = {}
for source in sorted({r["source"] for r in rows}):
    items = [r for r in rows if r["source"] == source]
    scenes = defaultdict(list)
    dimensions = Counter()
    categories = Counter()
    for r in items:
        scene = Path(r["image"]).stem.split("_jpg.rf.")[0].split("_png_jpg.rf.")[0] if source == "Indoor_Fire_Smoke.zip" else r["scene_key"]
        scenes[scene].append(r)
        dimensions[(r["width"], r["height"])] += 1
        a, b = int(r["class_0_boxes"]), int(r["class_1_boxes"])
        categories["both" if a and b else "class0_only" if a else "class1_only" if b else "empty"] += 1
    split_cross = sum(len({r["split"] for r in group}) > 1 for group in scenes.values())
    split_cross_images = sum(len(group) for group in scenes.values() if len({r["split"] for r in group}) > 1)
    out[source] = {"images": len(items), "scene_keys": len(scenes), "scenes_repeated": sum(len(v) > 1 for v in scenes.values()),
                   "cross_source_split_scene_keys": split_cross, "images_in_cross_split_scenes": split_cross_images,
                   "largest_scene": max(map(len, scenes.values())), "categories": categories,
                   "top_dimensions": [(list(k), v) for k, v in dimensions.most_common(8)]}

with zipfile.ZipFile(SOURCE / "AIForMankind_V2_processed.zip") as archive:
    pairs = list(csv.DictReader(TextIOWrapper(archive.open("AIForMankind_V2_processed/metadata/near_duplicate_review.csv"), encoding="utf-8-sig")))
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for pair in pairs:
        a, b = Path(pair["image_1"]).name, Path(pair["image_2"]).name
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    size = Counter(find(x) for x in parent)
    out["AIForMankind_V2_processed.zip"]["near_dup_review"] = {
        "pairs": len(pairs), "images_in_pairs": len(parent), "components": len(size),
        "largest_component": max(size.values()), "top_components": sorted(size.values(), reverse=True)[:10],
        "distance_counts": dict(Counter(p["distance"] for p in pairs))}

target = ROOT / "runtime/new_sources_audit/source_stats.json"
target.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
