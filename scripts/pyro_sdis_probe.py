"""Tạo ảnh ghép kiểm duyệt mẫu Pyro-SDIS, không xuất dữ liệu huấn luyện."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


SOURCE = Path("runtime/pyro_sdis_v2_source/data")
MANIFEST = Path("runtime/pyro_sdis_v2_audit/manifest.csv")
OUTPUT = Path("reports/pyro_sdis_v2_audit")
TILE_WIDTH = 420
TILE_HEIGHT = 270


def choose_rows(frame: pd.DataFrame, positive: bool, count: int = 48) -> pd.DataFrame:
    subset = frame[(frame.positive == int(positive)) & frame.issue.eq("")].copy()
    if positive:
        subset["area"] = pd.to_numeric(subset.min_box_area)
        subset = subset[subset.area.between(0.002, 0.015)]
        subset["distance"] = (subset.area - 0.0065).abs()
        subset = subset.sort_values(["camera_key", "distance", "date"])
    else:
        subset = subset.sort_values(["camera_key", "date"])
    subset = subset.drop_duplicates("camera_key")
    subset["order"] = subset.camera_key.map(
        lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    )
    return subset.sort_values(["order", "camera_key"]).head(count)


def load_selected(rows: pd.DataFrame) -> dict[tuple[str, int], tuple[bytes, str]]:
    selected = {(row.parquet_file, int(row.source_row)) for row in rows.itertuples()}
    by_file: dict[str, set[int]] = defaultdict(set)
    for filename, number in selected:
        by_file[filename].add(number)
    output: dict[tuple[str, int], tuple[bytes, str]] = {}
    for filename, indices in by_file.items():
        parquet = pq.ParquetFile(SOURCE / filename)
        offset = 0
        for batch in parquet.iter_batches(batch_size=64, columns=["image", "annotations"]):
            needed = [number for number in indices if offset <= number < offset + batch.num_rows]
            if needed:
                data = batch.to_pydict()
                for number in needed:
                    index = number - offset
                    output[(filename, number)] = (
                        data["image"][index]["bytes"],
                        data["annotations"][index] or "",
                    )
            offset += batch.num_rows
        if not indices.issubset({number for name, number in output if name == filename}):
            raise RuntimeError(f"Không trích đủ ảnh từ {filename}")
    return output


def make_tile(row: pd.Series, image_bytes: bytes, annotation: str) -> np.ndarray:
    frame = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Không đọc được {row.image_name}")
    height, width = frame.shape[:2]
    boxes: list[tuple[int, int, int, int]] = []
    for line in annotation.splitlines():
        if not line.strip():
            continue
        _, center_x, center_y, box_width, box_height = line.split()
        cx, cy, bw, bh = map(float, (center_x, center_y, box_width, box_height))
        x1, y1 = round((cx - bw / 2) * width), round((cy - bh / 2) * height)
        x2, y2 = round((cx + bw / 2) * width), round((cy + bh / 2) * height)
        boxes.append((x1, y1, x2, y2))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 40, 230), 4)
    scaled = cv2.resize(frame, (TILE_WIDTH, 236), interpolation=cv2.INTER_AREA)
    tile = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 245, dtype=np.uint8)
    tile[34:270] = scaled
    title = f"{row.partner}/{row.camera} {row.day}"
    cv2.putText(tile, title[:43], (5, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
    if boxes:
        x1, y1, x2, y2 = boxes[0]
        margin = max(30, round(max(x2 - x1, y2 - y1) * 1.2))
        xa, ya = max(0, x1 - margin), max(0, y1 - margin)
        xb, yb = min(width, x2 + margin), min(height, y2 + margin)
        crop = frame[ya:yb, xa:xb]
        if crop.size:
            crop = cv2.resize(crop, (155, 110), interpolation=cv2.INTER_AREA)
            tile[155:265, 260:415] = crop
            cv2.rectangle(tile, (259, 154), (415, 265), (20, 20, 20), 2)
    return tile


def write_sheet(rows: pd.DataFrame, payloads: dict, path: Path) -> None:
    tiles = [
        make_tile(row, *payloads[(row.parquet_file, int(row.source_row))])
        for row in rows.itertuples()
    ]
    blank = np.full((TILE_HEIGHT, TILE_WIDTH, 3), 245, dtype=np.uint8)
    while len(tiles) % 6:
        tiles.append(blank.copy())
    sheet = np.vstack([np.hstack(tiles[index : index + 6]) for index in range(0, len(tiles), 6)])
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Không ghi được {path}")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST, keep_default_na=False)
    positives = choose_rows(manifest, True)
    negatives = choose_rows(manifest, False)
    selected = pd.concat([positives, negatives], ignore_index=True)
    payloads = load_selected(selected)
    write_sheet(positives, payloads, OUTPUT / "smoke_probe.jpg")
    write_sheet(negatives, payloads, OUTPUT / "negative_probe.jpg")
    selected.to_csv(OUTPUT / "probe_manifest.csv", index=False)
    print(f"Smoke: {len(positives)}, negative: {len(negatives)}")


if __name__ == "__main__":
    main()
