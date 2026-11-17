"""Kiểm kê ba ZIP người dùng gợi ý, không nhận ảnh vào bộ huấn luyện."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from catargiu_audit import hashes


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("F:/6/KLTN/Smoke-Fire-Dataset")
OUTPUT = ROOT / "runtime/new_sources_audit"
NAMES = ("AIForMankind_V2_processed.zip", "Indoor_Fire_Smoke.zip", "AIForMankind_Cloud_processed.zip")
FIELDS = ("source", "image", "label", "split", "scene_key", "class_0_boxes", "class_1_boxes",
          "width", "height", "sha256", "dhash", "phash", "issue")


def scene_key(source: str, name: str) -> str:
    stem = Path(name).stem
    if source == "Indoor_Fire_Smoke.zip":
        stem = stem.split("_jpg.rf.")[0].split("_png_jpg.rf.")[0]
        return stem
    if source == "AIForMankind_Cloud_processed.zip":
        return "cloud_camera_pending_metadata"
    return "smoke_camera_pending_visual_review"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {}
    with (OUTPUT / "image_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, FIELDS)
        writer.writeheader()
        for source in NAMES:
            counts = Counter()
            sha_groups = defaultdict(list)
            with zipfile.ZipFile(SOURCE / source) as archive:
                names = set(archive.namelist())
                images = sorted(name for name in names if "/images/" in f"/{name}" and name.lower().endswith((".jpg", ".jpeg", ".png")))
                for index, name in enumerate(images, 1):
                    label = f"/{name}".replace("/images/", "/labels/").lstrip("/").rsplit(".", 1)[0] + ".txt"
                    issue = []
                    class_counts = Counter()
                    if label not in names:
                        issue.append("missing_label")
                    else:
                        for line in archive.read(label).decode("utf-8-sig").splitlines():
                            if not line.strip():
                                continue
                            parts = line.split()
                            if len(parts) != 5 or parts[0] not in ("0", "1"):
                                issue.append("malformed_label")
                            else:
                                class_counts[parts[0]] += 1
                    data = archive.read(name)
                    sha = hashlib.sha256(data).hexdigest()
                    width = height = 0
                    dhash = phash = ""
                    try:
                        with Image.open(BytesIO(data)) as image:
                            width, height = image.size
                            dhash, phash = hashes(image)
                    except Exception as error:
                        issue.append("image_" + type(error).__name__)
                    split = name.split("/")[1] if source == "Indoor_Fire_Smoke.zip" else "unsplit"
                    writer.writerow({"source": source, "image": name, "label": label, "split": split,
                                     "scene_key": scene_key(source, name), "class_0_boxes": class_counts["0"],
                                     "class_1_boxes": class_counts["1"], "width": width, "height": height,
                                     "sha256": sha, "dhash": dhash, "phash": phash, "issue": ";".join(issue)})
                    counts["images"] += 1
                    counts["class_0_boxes"] += class_counts["0"]
                    counts["class_1_boxes"] += class_counts["1"]
                    counts["empty_labels"] += not class_counts
                    counts["issues"] += bool(issue)
                    sha_groups[sha].append(name)
                    if index % 500 == 0:
                        handle.flush()
                        print(f"{source} {index}/{len(images)}", flush=True)
            counts["exact_duplicate_extra_images"] = sum(len(group) - 1 for group in sha_groups.values())
            summary[source] = dict(counts)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == "__main__":
    main()
