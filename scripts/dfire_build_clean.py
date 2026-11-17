"""Tạo bộ D-Fire YOLO bảo thủ từ ZIP gốc và kiểm tra rò rỉ sau khi xuất."""

from __future__ import annotations

from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import re
import zipfile

import numpy as np
import pandas as pd
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import GroupShuffleSplit


SOURCE_SHA256 = "b0dd9fec2b37f319b79f7880f15fc87bc90bd4efb917c0311298209536f62f0b"
STEAM_AS_BACKGROUND = {f"test/images/WEB{number:05d}.jpg" for number in (10030, 10032, 10038, 10039, 10040, 10041)}
MANUAL_SUNSET_TRAIN_RANGE = range(2130, 2180)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_id(name: str) -> int:
    return int(re.search(r"(\d+)\.jpg$", name).group(1))


def graph_components(keys: list[str], edges: list[tuple[str, str]]) -> tuple[np.ndarray, dict[str, int]]:
    lookup = {key: number for number, key in enumerate(keys)}
    valid = [(lookup[a], lookup[b]) for a, b in edges if a in lookup and b in lookup]
    if valid:
        left, right = np.asarray(valid, dtype=np.int32).T
        graph = coo_matrix((np.ones(2 * len(left), dtype=np.uint8),
                            (np.r_[left, right], np.r_[right, left])), shape=(len(keys), len(keys)))
    else:
        graph = coo_matrix((len(keys), len(keys)), dtype=np.uint8)
    _, components = connected_components(graph, directed=False)
    return components, lookup


def deduplicate(
    keys: set[str], pairs: pd.DataFrame, row_index: pd.DataFrame, archive: zipfile.ZipFile,
    max_dhash: int, max_phash: int, min_color: float,
) -> dict[str, str]:
    eligible = pairs[(pairs.image_a.isin(keys)) & (pairs.image_b.isin(keys)) &
                     (pairs.prefix_a == pairs.prefix_b) & (pairs.category_a == pairs.category_b) &
                     (pairs.dhash_distance <= max_dhash) & (pairs.phash_distance <= max_phash) &
                     (pairs.color_overlap >= min_color)].copy()
    eligible = eligible[[abs(numeric_id(a) - numeric_id(b)) <= 2
                         for a, b in zip(eligible.image_a, eligible.image_b)]]
    ordered = sorted(keys)
    components, _ = graph_components(ordered, list(zip(eligible.image_a, eligible.image_b)))
    groups = defaultdict(list)
    for key, group_id in zip(ordered, components):
        groups[int(group_id)].append(key)
    exclusions = {}
    for members in groups.values():
        if len(members) == 1:
            continue
        representative = min(members, key=lambda key: (
            -archive.getinfo(key).file_size, key
        ))
        for key in members:
            if key != representative:
                exclusions[key] = representative
    return exclusions


def build_label(raw: bytes, image: str, background_override: bool) -> tuple[str, int]:
    if background_override:
        return "", 0
    lines = []
    clipped = 0
    for line in raw.decode("utf-8-sig").splitlines():
        if not line.strip():
            continue
        cls, x, y, width, height = map(float, line.split())
        if width <= 0 or height <= 0:
            raise ValueError(f"Box không dương chưa được loại khỏi tập: {image}")
        x1, x2 = x - width / 2, x + width / 2
        y1, y2 = y - height / 2, y + height / 2
        nx1, nx2 = max(0.0, x1), min(1.0, x2)
        ny1, ny2 = max(0.0, y1), min(1.0, y2)
        if nx1 >= nx2 or ny1 >= ny2:
            raise ValueError(f"Box nằm ngoài ảnh: {image}")
        retained = (nx2 - nx1) * (ny2 - ny1) / (width * height)
        if retained < 0.8:
            raise ValueError(f"Box cần duyệt thủ công: {image}")
        if nx1 != x1 or nx2 != x2 or ny1 != y1 or ny2 != y2:
            clipped += 1
        cx, cy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        new_width, new_height = nx2 - nx1, ny2 - ny1
        lines.append(f"{int(cls)} {cx:.10f} {cy:.10f} {new_width:.10f} {new_height:.10f}")
    return ("\n".join(lines) + "\n" if lines else ""), clipped


def split_train_val(
    train_keys: set[str], images: pd.DataFrame, pairs: pd.DataFrame,
) -> tuple[set[str], set[str], pd.DataFrame]:
    index = images.set_index("image")
    ordered = sorted(train_keys)
    internal = pairs[(pairs.pair_type == "within_train") &
                     pairs.image_a.isin(train_keys) & pairs.image_b.isin(train_keys)]
    edges = list(zip(internal.image_a, internal.image_b))
    by_source = images[images.image.isin(train_keys)].copy()
    by_source["number"] = by_source.image.map(numeric_id)
    for _, group in by_source.groupby("prefix"):
        members = list(group.sort_values("number").itertuples())
        for first, second in zip(members, members[1:]):
            if second.number - first.number > 2:
                continue
            ph = (int(first.phash, 16) ^ int(second.phash, 16)).bit_count()
            dh = (int(first.dhash, 16) ^ int(second.dhash, 16)).bit_count()
            if ph <= 20 and dh <= 24:
                edges.append((first.image, second.image))
    components, _ = graph_components(ordered, edges)
    rows = index.loc[ordered]
    category = rows.category.to_numpy()
    source = rows.prefix.to_numpy()
    target_share = 0.15
    best = None
    splitter = GroupShuffleSplit(n_splits=160, test_size=target_share, random_state=20261029)
    dummy = np.zeros(len(ordered))
    for train_index, val_index in splitter.split(dummy, category, components):
        val_mask = np.zeros(len(ordered), dtype=bool)
        val_mask[val_index] = True
        score = 4 * abs(val_mask.mean() - target_share)
        for labels, weight in ((category, 1.0), (source, 0.7)):
            for value in np.unique(labels):
                total = np.sum(labels == value)
                if total:
                    score += weight * abs(np.sum(val_mask & (labels == value)) / total - target_share)
        if best is None or score < best[0]:
            best = (score, train_index, val_index)
    assert best is not None
    train = {ordered[number] for number in best[1]}
    val = {ordered[number] for number in best[2]}
    group_table = pd.DataFrame({"image": ordered, "scene_candidate_group": components,
                                "assigned_split": ["val" if key in val else "train" for key in ordered]})
    return train, val, group_table


def residual_hash_pairs(left_keys: set[str], right_keys: set[str], index: pd.DataFrame) -> list[tuple[str, str, int, int]]:
    """Tìm thêm cặp rất giống bất kể kích thước để cách ly bảo thủ."""
    left = index.set_index("image").loc[sorted(left_keys)]
    right = index.set_index("image").loc[sorted(right_keys)]
    left_hashes = np.array([int(value, 16) for value in left.phash], dtype=np.uint64)
    right_hashes = np.array([int(value, 16) for value in right.phash], dtype=np.uint64)
    lookup = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(axis=1).astype(np.uint8)
    pairs = []
    for start in range(0, len(left_hashes), 128):
        xor = left_hashes[start:start + 128, None] ^ right_hashes[None, :]
        distances = lookup[xor.view(np.uint8).reshape(xor.shape + (8,))].sum(axis=2)
        left_numbers, right_numbers = np.where(distances <= 4)
        for offset, right_number in zip(left_numbers, right_numbers):
            first = left.iloc[start + offset]
            second = right.iloc[right_number]
            dh = (int(first.dhash, 16) ^ int(second.dhash, 16)).bit_count()
            if dh <= 6:
                pairs.append((first.name, second.name, int(distances[offset, right_number]), dh))
    return pairs


def build(source: Path, audit: Path, review: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Đầu ra đã tồn tại: {output}")
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError("ZIP nguồn không đúng bản D-Fire đã kiểm tra.")
    index = pd.read_csv(audit / "image_index.csv", encoding="utf-8-sig")
    pairs = pd.read_csv(audit / "candidate_pairs.csv", encoding="utf-8-sig")
    cross_dim = pd.read_csv(audit / "cross_dimension_visual_review.csv", encoding="utf-8-sig")
    preview = pd.read_csv(review / "image_decision_manifest.csv", encoding="utf-8-sig")
    if len(index) != 21527 or len(preview) != len(index) or set(index.image) != set(preview.image):
        raise ValueError("Bảng khảo sát không khớp 21.527 ảnh nguồn.")
    by_key = index.set_index("image")
    test = set(index.loc[index.split == "test", "image"])
    train = set(index.loc[index.split == "train", "image"])
    reasons = defaultdict(list)

    def exclude(keys: set[str], reason: str) -> None:
        for key in keys:
            reasons[key].append(reason)

    nonpositive = set(pd.read_csv(review / "label_fix_proposals.csv").query("reason == 'nonpositive'").image)
    exclude(nonpositive, "nonpositive_label_box")
    exclude({key for key in train if by_key.loc[key, "prefix"] == "AoF"}, "aof_camera_overlap_with_test")
    cross_candidates = set(preview.loc[(preview.original_split == "train") &
                                       (preview.cross_split_candidate_count > 0), "image"])
    exclude(cross_candidates, "same_dimension_test_scene_candidate")
    confirmed_cross_dim = set(cross_dim.loc[cross_dim.visual_review == "same_scene", "train_image"])
    exclude(confirmed_cross_dim, "visually_confirmed_cross_dimension_test_scene")
    sunset_train = {key for key in train if by_key.loc[key, "prefix"] == "WEB" and
                    numeric_id(key) in MANUAL_SUNSET_TRAIN_RANGE}
    exclude(sunset_train, "manually_identified_sunset_steam_source_sequence")
    remaining_train = train - set(reasons)
    inconsistent = pairs[(pairs.pair_type == "within_train") & (pairs.category_a != pairs.category_b) &
                         pairs.image_a.isin(remaining_train) & pairs.image_b.isin(remaining_train)]
    exclude(set(inconsistent.image_a) | set(inconsistent.image_b), "inconsistent_category_near_same_train_scene")

    with zipfile.ZipFile(source) as archive:
        test -= set(reasons)
        test_dups = deduplicate(test, pairs[pairs.pair_type == "within_test"], by_key, archive, 0, 0, 0.995)
        for key, representative in test_dups.items():
            reasons[key].append(f"near_identical_test_frame_of:{representative}")
        test -= set(test_dups)
        remaining_train = train - set(reasons)
        train_dups = deduplicate(remaining_train, pairs[pairs.pair_type == "within_train"],
                                 by_key, archive, 1, 4, 0.95)
        for key, representative in train_dups.items():
            reasons[key].append(f"near_identical_train_frame_of:{representative}")
        remaining_train -= set(train_dups)
        final_train, val, group_table = split_train_val(remaining_train, index, pairs)
        residual_test_train = residual_hash_pairs(final_train, test, index)
        residual_test_val = residual_hash_pairs(val, test, index)
        for key in {first for first, _, _, _ in residual_test_train}:
            reasons[key].append("residual_cross_test_hash_candidate")
            final_train.remove(key)
        for key in {first for first, _, _, _ in residual_test_val}:
            reasons[key].append("residual_cross_test_hash_candidate")
            val.remove(key)
        residual_train_val = residual_hash_pairs(final_train, val, index)
        for key in {second for _, second, _, _ in residual_train_val}:
            reasons[key].append("residual_train_val_hash_candidate")
            val.remove(key)
        group_table["assigned_split"] = group_table.image.map(
            lambda key: "train" if key in final_train else "val" if key in val else "excluded_after_residual_audit"
        )
        if residual_hash_pairs(final_train, test, index) or residual_hash_pairs(val, test, index) or residual_hash_pairs(final_train, val, index):
            raise ValueError("Còn cặp hash nghi vấn vượt split sau cách ly.")
        if not final_train.isdisjoint(val) or not final_train.isdisjoint(test) or not val.isdisjoint(test):
            raise ValueError("Ảnh xuất hiện ở nhiều split.")
        for pair in pairs.itertuples():
            a, b = pair.image_a, pair.image_b
            if a in final_train and b in val or a in val and b in final_train:
                raise ValueError("Ứng viên gần trùng vượt train/val.")
            if a in test and (b in final_train or b in val) or b in test and (a in final_train or a in val):
                raise ValueError("Ứng viên gần trùng vượt train/test hoặc val/test.")
        for pair in cross_dim[cross_dim.visual_review == "same_scene"].itertuples():
            if pair.test_image in test and (pair.train_image in final_train or pair.train_image in val):
                raise ValueError("Cặp khác kích thước cùng cảnh vượt split.")

        building = output.with_name(output.name + "_building")
        if building.exists():
            raise FileExistsError(f"Thư mục đang dựng đã tồn tại: {building}")
        building.mkdir(parents=True)
        for split in ("train", "val", "test"):
            (building / split / "images").mkdir(parents=True)
            (building / split / "labels").mkdir(parents=True)
        (building / "metadata").mkdir()
        group_table.to_csv(building / "metadata" / "train_val_candidate_groups.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [dict(image_a=a, image_b=b, phash_distance=ph, dhash_distance=dh,
                  reason="cross_test" if kind != "train_val" else "train_val")
             for kind, rows in (("train_test", residual_test_train), ("val_test", residual_test_val),
                                ("train_val", residual_train_val)) for a, b, ph, dh in rows]
        ).to_csv(building / "metadata" / "residual_hash_quarantine.csv", index=False, encoding="utf-8-sig")
        manifest = []
        fixed_boxes = 0
        for number, row in enumerate(index.itertuples(), 1):
            key = row.image
            role = "test" if key in test else "val" if key in val else "train" if key in final_train else "excluded"
            record = dict(source_image=key, source_sha256=row.sha256, original_split=row.split,
                          source_prefix=row.prefix, original_category=row.category, output_split=role,
                          output_category="background" if key in STEAM_AS_BACKGROUND and role == "test" else row.category,
                          output_image="", output_label="", exclusion_reason="|".join(sorted(set(reasons[key]))),
                          semantic_correction="steam_or_sunset_not_fire" if key in STEAM_AS_BACKGROUND and role == "test" else "",
                          clipped_boxes=0)
            if role != "excluded":
                data = archive.read(key)
                if hashlib.sha256(data).hexdigest() != row.sha256:
                    raise ValueError(f"SHA-256 ảnh nguồn thay đổi: {key}")
                extension = ".png" if row.format == "PNG" else ".jpg"
                stem = Path(key).stem
                output_image = Path(role) / "images" / f"{stem}{extension}"
                output_label = Path(role) / "labels" / f"{stem}.txt"
                label_key = key.replace("/images/", "/labels/")[:-4] + ".txt"
                label, clipped = build_label(archive.read(label_key), key, key in STEAM_AS_BACKGROUND)
                (building / output_image).write_bytes(data)
                (building / output_label).write_text(label, encoding="utf-8")
                record["output_image"] = output_image.as_posix()
                record["output_label"] = output_label.as_posix()
                record["clipped_boxes"] = clipped
                fixed_boxes += clipped
            manifest.append(record)
            if number % 5000 == 0:
                print(f"processed={number}", flush=True)
        frame = pd.DataFrame(manifest)
        frame.to_csv(building / "metadata" / "manifest.csv", index=False, encoding="utf-8-sig")
        frame[frame.output_split == "excluded"].to_csv(
            building / "metadata" / "excluded_images.csv", index=False, encoding="utf-8-sig"
        )
        source_counts = frame[frame.output_split != "excluded"].groupby(
            ["output_split", "source_prefix", "output_category"]
        ).size().reset_index(name="images")
        source_counts.to_csv(building / "metadata" / "distribution.csv", index=False, encoding="utf-8-sig")
        for split in ("train", "val", "test"):
            count_images = len(list((building / split / "images").iterdir()))
            count_labels = len(list((building / split / "labels").iterdir()))
            if count_images != count_labels or count_images != sum(frame.output_split == split):
                raise ValueError(f"Thiếu ảnh hoặc nhãn ở {split}.")
        (building / "data.yaml").write_text(
            "train: train/images\nval: val/images\ntest: test/images\nnc: 2\nnames:\n  0: smoke\n  1: fire\n",
            encoding="utf-8",
        )
        stats = dict(source_zip_sha256=SOURCE_SHA256, train=len(final_train), val=len(val), test=len(test),
                     excluded=len(index) - len(final_train) - len(val) - len(test),
                     clipped_boxes=fixed_boxes, semantic_background_corrections=len(STEAM_AS_BACKGROUND & test),
                     status="BUILT_WITH_DOCUMENTED_RESIDUAL_SEMANTIC_REVIEW")
        (building / "metadata" / "summary.json").write_text(
            json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        building.rename(output)
        print(json.dumps(stats, ensure_ascii=True, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.audit, args.review, args.output)


if __name__ == "__main__":
    main()
