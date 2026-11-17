"""Thống kê độ phân giải của ảnh chỉ lửa đã qua sàng lọc val/test."""

import csv
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports/dba_indoor_relaxed_review"
MANIFEST = REPORT_DIR / "manifest.csv"
SOURCES = {
    "DBA-Fire": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\DBA-YOLO-Dataset.zip"),
    "Indoor": Path(r"F:\6\KLTN\Smoke-Fire-Dataset\Indoor_Fire_Smoke.zip"),
}


def size_band(long_edge):
    if long_edge < 320:
        return "<320"
    if long_edge < 640:
        return "320-639"
    if long_edge < 960:
        return "640-959"
    if long_edge < 1280:
        return "960-1279"
    return ">=1280"


def main():
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        candidates = [row for row in csv.DictReader(handle) if row["status"] == "eligible" and row["category"] == "fire_only"]
    archives = {name: zipfile.ZipFile(path) for name, path in SOURCES.items()}
    output = []
    try:
        for row in candidates:
            width = int(row["width"])
            height = int(row["height"])
            long_edge = max(width, height)
            short_edge = min(width, height)
            scale = 640 / long_edge
            label = archives[row["source"]].read(row["label"]).decode("utf-8-sig")
            fire_box_sizes = []
            for line in label.splitlines():
                parts = line.split()
                if parts and int(parts[0]) == 0:
                    box_w = float(parts[3]) * width * scale
                    box_h = float(parts[4]) * height * scale
                    fire_box_sizes.append(min(box_w, box_h))
            if not fire_box_sizes:
                raise ValueError(f"Thiếu box lửa: {row['image']}")
            output.append({
                "source": row["source"],
                "image": row["image"],
                "width": width,
                "height": height,
                "long_edge": long_edge,
                "short_edge": short_edge,
                "long_edge_band": size_band(long_edge),
                "long_edge_ge_640": long_edge >= 640,
                "both_edges_ge_640": short_edge >= 640,
                "exact_640_square": width == 640 and height == 640,
                "upscale_factor_to_640": round(scale, 4),
                "fire_boxes": len(fire_box_sizes),
                "largest_fire_box_short_edge_at_640": round(max(fire_box_sizes), 2),
                "any_fire_box_short_edge_ge_16": any(value >= 16 for value in fire_box_sizes),
                "any_fire_box_short_edge_ge_32": any(value >= 32 for value in fire_box_sizes),
            })
    finally:
        for archive in archives.values():
            archive.close()
    with (REPORT_DIR / "fire_only_imgsz640.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    summary = {}
    for source in (*SOURCES, "Tổng"):
        subset = output if source == "Tổng" else [row for row in output if row["source"] == source]
        summary[source] = {
            "images": len(subset),
            "long_edge_ge_640": sum(row["long_edge_ge_640"] for row in subset),
            "long_edge_lt_640": sum(not row["long_edge_ge_640"] for row in subset),
            "both_edges_ge_640": sum(row["both_edges_ge_640"] for row in subset),
            "exact_640_square": sum(row["exact_640_square"] for row in subset),
            "long_edge_bands": dict(Counter(row["long_edge_band"] for row in subset)),
            "common_resolutions": [{"resolution": f"{w}x{h}", "images": count} for (w, h), count in Counter((row["width"], row["height"]) for row in subset).most_common(12)],
            "any_fire_box_short_edge_ge_16": sum(row["any_fire_box_short_edge_ge_16"] for row in subset),
            "any_fire_box_short_edge_ge_32": sum(row["any_fire_box_short_edge_ge_32"] for row in subset),
        }
    (REPORT_DIR / "fire_only_imgsz640_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
