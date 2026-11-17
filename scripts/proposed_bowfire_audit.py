"""Kiểm tra mask BoWFire gốc và khả năng chuyển thành box phát hiện lửa."""

import csv
import hashlib
import json
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
rows = []
counts = Counter()
with zipfile.ZipFile(SOURCE / "BoWFireDataset.zip") as archive:
    names = set(archive.namelist())
    images = sorted(n for n in names if n.startswith("BoWFireDataset/dataset/img/") and n.lower().endswith(".png"))
    for name in images:
        mask_name = name.replace("/img/", "/gt/").replace(".png", "_gt.png")
        issue = []
        data = archive.read(name)
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            image.load()
        if mask_name not in names:
            issue.append("missing_mask")
            mask_bbox = None
        else:
            with Image.open(BytesIO(archive.read(mask_name))) as mask:
                if mask.size != (width, height):
                    issue.append("dimension_mismatch")
                gray = mask.convert("L")
                colors = {value for _, value in gray.getcolors(maxcolors=256) or []}
                if not colors.issubset({0, 255}):
                    issue.append("nonbinary_mask")
                mask_bbox = gray.point(lambda value: 255 if value > 127 else 0).getbbox()
        category = "fire" if Path(name).stem.startswith("fire") else "negative"
        if category == "fire" and mask_bbox is None:
            issue.append("fire_empty_mask")
        if category == "negative" and mask_bbox is not None:
            issue.append("negative_nonempty_mask")
        counts[category] += 1
        counts["issues"] += bool(issue)
        rows.append({"image": name, "category": category, "width": width, "height": height,
                     "sha256": hashlib.sha256(data).hexdigest(), "mask_bbox": mask_bbox, "issue": ";".join(issue)})
with (SOURCE / "bowfire_index.csv").open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
counts["exact_duplicate_extra"] = len(rows) - len({r["sha256"] for r in rows})
(SOURCE / "bowfire_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
print(json.dumps(counts, indent=2))
