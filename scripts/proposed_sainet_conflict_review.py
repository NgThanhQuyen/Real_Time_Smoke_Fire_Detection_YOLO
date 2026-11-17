"""Ghi nhận ảnh SAINetset trùng byte nhưng nhãn bất đồng giữa train và val."""

import csv
import json
from collections import defaultdict
from io import BytesIO
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
LABELS = SOURCE / "sainet_labels_repo2/data"
rows = list(csv.DictReader((SOURCE / "sainet_image_index.csv").open(encoding="utf-8")))
groups = defaultdict(list)
for row in rows:
    groups[row["sha256"]].append(row)
conflicts = []
for sha, group in groups.items():
    if len({r["split"] for r in group}) < 2:
        continue
    labels = []
    for row in group:
        split = "val" if row["split"] == "validation" else row["split"]
        path = LABELS / split / "labels" / (Path(row["image_path"]).stem + ".txt")
        labels.append((split, path.read_text(encoding="utf-8-sig").strip(), row))
    if len({label for _, label, _ in labels}) > 1:
        conflicts.append({"sha256": sha, "image": group[0]["image_path"], "labels": labels})
target_csv = SOURCE / "sainet_exact_label_conflicts.csv"
with target_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, ["sha256", "image", "train_labels", "val_labels"])
    writer.writeheader()
    for conflict in conflicts:
        writer.writerow({"sha256": conflict["sha256"], "image": conflict["image"],
                         "train_labels": " | ".join(label for split, label, _ in conflict["labels"] if split == "train"),
                         "val_labels": " | ".join(label for split, label, _ in conflict["labels"] if split == "val")})


def image_bytes(row):
    path = SOURCE / "sainet_parquet" / row["split"] / row["parquet"]
    offset = int(row["row_index"])
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=32):
        if index + len(batch) > offset:
            return batch.to_pylist()[offset - index]["image"]["bytes"]
        index += len(batch)
    raise IndexError(offset)


selected = conflicts[:8]
width, height = 350, 250
canvas = Image.new("RGB", (width * 2, height * len(selected)), "white")
draw = ImageDraw.Draw(canvas)
for index, conflict in enumerate(selected):
    row = conflict["labels"][0][2]
    with Image.open(BytesIO(image_bytes(row))) as image:
        raw = image.convert("RGB")
    for offset, split in ((0, "train"), (width, "val")):
        frame = raw.copy()
        box_draw = ImageDraw.Draw(frame)
        label = next((text for s, text, _ in conflict["labels"] if s == split), "")
        for line in label.splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cx, cy, bw, bh = map(float, parts[1:])
            box_draw.rectangle(((cx - bw/2) * frame.width, (cy - bh/2) * frame.height,
                                (cx + bw/2) * frame.width, (cy + bh/2) * frame.height), outline="red", width=6)
        thumb = ImageOps.contain(frame, (width, height - 25))
        x, y = offset, index * height
        canvas.paste(thumb, (x + (width - thumb.width)//2, y + 25))
        draw.text((x + 3, y + 3), f"{split} {conflict['image']} {'box' if label else 'empty'}", fill="black")
target = ROOT / "reports/proposed_sources/sainet_exact_label_conflicts.jpg"
canvas.save(target, quality=88)
print(json.dumps({"exact_cross_split_conflict_groups": len(conflicts), "sheet": str(target)}))
