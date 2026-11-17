"""Tải bản Parquet ảnh SAINetset để kiểm kê trùng cảnh trên toàn bộ nguồn."""

import json
import subprocess
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "runtime/proposed_sources/sainet_parquet"
OUTPUT.mkdir(parents=True, exist_ok=True)
url = "https://datasets-server.huggingface.co/parquet?dataset=SAINetset/SAINetset_v8.0"
with urllib.request.urlopen(url, timeout=30) as response:
    files = json.load(response)["parquet_files"]
for item in files:
    split = item["split"]
    name = item["filename"].split("/")[-1]
    target = OUTPUT / split / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == item["size"]:
        print(f"skip {split}/{name}", flush=True)
        continue
    print(f"download {split}/{name} {item['size']}", flush=True)
    subprocess.run(["curl.exe", "-L", "--fail", "--retry", "3", "--connect-timeout", "15",
                    "--max-time", "240", "-sS", "-o", str(target), item["url"]], check=True)
    if target.stat().st_size != item["size"]:
        raise RuntimeError(f"Kích thước tệp không đúng: {target}")
print("done", flush=True)
