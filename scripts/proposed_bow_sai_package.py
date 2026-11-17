"""Đóng gói bản D-Fire đã kiểm định để tải lên Kaggle."""

import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_bow_sai_v1"
ARCHIVE = ROOT / "runtime/dfire_clean_v2_plus_pyro_catargiu_bow_sai_v1.zip"

gate = json.loads((DATA / "metadata/quality_gate.json").read_text(encoding="utf-8"))
if not gate["val_test_byte_identical"] or gate["selected"] != 24:
    raise ValueError("Quality gate chưa đạt")
if ARCHIVE.exists():
    raise FileExistsError(ARCHIVE)
files = sorted(path for path in DATA.rglob("*") if path.is_file())
with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
    for index, path in enumerate(files, 1):
        output.write(path, path.relative_to(DATA).as_posix())
        if index % 5000 == 0:
            print(f"{index}/{len(files)}", flush=True)
with zipfile.ZipFile(ARCHIVE) as check:
    names = set(check.namelist())
    if len(names) != len(files) or "data.yaml" not in names or "metadata/quality_gate.json" not in names:
        raise ValueError("ZIP thiếu file")
print(f"{ARCHIVE} files={len(files)} bytes={ARCHIVE.stat().st_size}")
