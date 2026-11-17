"""Lấy mẫu ảnh và nhãn SAINetset từ nguồn chính thức để kiểm duyệt sơ bộ."""

from __future__ import annotations

import csv
import random
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi


REPO = "SAINetset/SAINetset_v8.0"
ROOT = Path(__file__).resolve().parents[1] / "runtime/proposed_sources/sainet_sample"


def fetch(path: str) -> tuple[str, str]:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{urllib.parse.quote(path)}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as response:
                    target.write_bytes(response.read())
                break
            except (OSError, urllib.error.HTTPError):
                if attempt == 2:
                    raise
    return path, str(target)


def main() -> None:
    api = HfApi()
    rng = random.Random(42)
    chosen = []
    for split, count in (("train", 96), ("val", 24)):
        names = [item.path for item in api.list_repo_tree(REPO, repo_type="dataset",
                                                            path_in_repo=f"data/{split}/images", recursive=False)
                 if item.path.lower().endswith((".jpg", ".jpeg", ".png"))]
        chosen += rng.sample(names, min(count, len(names)))
        print(f"{split}: {len(names)} images, sample {min(count, len(names))}", flush=True)
    paths = chosen + [p.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt" for p in chosen]
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch, path) for path in paths]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 40 == 0:
                print(f"Downloaded {index}/{len(paths)} files", flush=True)
    rows = []
    for image in chosen:
        label = image.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
        text = (ROOT / label).read_text(encoding="utf-8-sig")
        rows.append({"image": image, "label": label, "boxes": len([line for line in text.splitlines() if line.strip()])})
    with (ROOT / "sample_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, ["image", "label", "boxes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Done: {len(rows)} images; {sum(not r['boxes'] for r in rows)} empty labels", flush=True)


if __name__ == "__main__":
    main()
