"""Kiểm kê toàn bộ ảnh SAINetset từ bản Parquet do Hugging Face xuất."""

import csv
import hashlib
import json
from collections import Counter
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from catargiu_audit import hashes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources/sainet_parquet"
OUTPUT = ROOT / "runtime/proposed_sources/sainet_image_index.csv"
fields = ("split", "parquet", "row_index", "image_path", "width", "height", "sha256", "dhash", "phash", "issue")
counts = Counter()
with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fields)
    writer.writeheader()
    for parquet_path in sorted(SOURCE.glob("*/*.parquet")):
        part = pq.ParquetFile(parquet_path)
        row_index = 0
        for batch in part.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                item = row["image"]
                data = item["bytes"]
                issue = []
                if not data:
                    issue.append("missing_bytes")
                width = height = 0
                d = p = ""
                if data:
                    try:
                        with Image.open(BytesIO(data)) as image:
                            width, height = image.size
                            d, p = hashes(image)
                    except Exception as error:
                        issue.append(type(error).__name__)
                writer.writerow({"split": parquet_path.parent.name, "parquet": parquet_path.name,
                                 "row_index": row_index, "image_path": item["path"],
                                 "width": width, "height": height,
                                 "sha256": hashlib.sha256(data or b"").hexdigest(),
                                 "dhash": d, "phash": p, "issue": ";".join(issue)})
                row_index += 1
                counts[parquet_path.parent.name] += 1
                counts["issues"] += bool(issue)
        handle.flush()
        print(parquet_path.parent.name, parquet_path.name, row_index, flush=True)
(ROOT / "runtime/proposed_sources/sainet_image_summary.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
print(json.dumps(counts, indent=2))
