"""Kiểm kê ảnh và nhãn Home Fire, tạo chỉ mục để rà trùng với D-Fire."""

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

from catargiu_audit import hashes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runtime/proposed_sources"
FIELDS = ("split", "image", "width", "height", "fire_boxes", "smoke_boxes", "sha256", "dhash", "phash", "issue")


def main() -> None:
    summary = {}
    splits = tuple(sys.argv[1:]) or ("train", "val", "test")
    target = SOURCE / (f"home_index_{splits[0]}.csv" if len(splits) == 1 else "home_index.csv")
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        for split in splits:
            counts = Counter()
            path = SOURCE / f"HomeFire_{split}.zip"
            with zipfile.ZipFile(path) as archive:
                names = set(archive.namelist())
                images = sorted(n for n in names if n.startswith("images/") and n.lower().endswith((".jpg", ".jpeg", ".png")))
                for index, name in enumerate(images, 1):
                    label = name.replace("images/", "labels/").rsplit(".", 1)[0] + ".txt"
                    issue = []
                    classes = Counter()
                    if label not in names:
                        issue.append("missing_label")
                    else:
                        for line in archive.read(label).decode("utf-8-sig").splitlines():
                            if not line.strip():
                                continue
                            parts = line.split()
                            if len(parts) != 5 or parts[0] not in ("0", "1"):
                                issue.append("bad_label_syntax")
                                continue
                            try:
                                cx, cy, bw, bh = map(float, parts[1:])
                                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
                                    issue.append("bad_box_range")
                            except ValueError:
                                issue.append("bad_box_value")
                            classes[parts[0]] += 1
                    data = archive.read(name)
                    sha = hashlib.sha256(data).hexdigest()
                    with Image.open(BytesIO(data)) as image:
                        width, height = image.size
                        d, p = hashes(image)
                    writer.writerow({"split": split, "image": name, "width": width, "height": height,
                                     "fire_boxes": classes["0"], "smoke_boxes": classes["1"],
                                     "sha256": sha, "dhash": d, "phash": p, "issue": ";".join(issue)})
                    counts["images"] += 1
                    counts["fire_boxes"] += classes["0"]
                    counts["smoke_boxes"] += classes["1"]
                    counts["empty"] += not classes
                    counts["issues"] += bool(issue)
                    if index % 500 == 0:
                        handle.flush()
                        print(f"{split}: {index}/{len(images)}", flush=True)
            summary[split] = dict(counts)
    (SOURCE / (f"home_summary_{splits[0]}.json" if len(splits) == 1 else "home_summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
