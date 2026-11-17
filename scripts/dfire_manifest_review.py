"""Lập manifest kiểm duyệt D-Fire từ ZIP gốc, chưa khóa test hoặc xuất dataset."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
from pathlib import Path
import argparse
import hashlib
import json
import math
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_audit(source: Path, audit: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary = json.loads((audit / "summary.json").read_text(encoding="utf-8"))
    if source_sha256(source) != summary["source_zip_sha256"]:
        raise ValueError("ZIP D-Fire không khớp bản dùng để khảo sát.")
    images = pd.read_csv(audit / "image_index.csv", encoding="utf-8-sig")
    pairs = pd.read_csv(audit / "candidate_pairs.csv", encoding="utf-8-sig")
    issues = pd.read_csv(audit / "label_issues.csv", encoding="utf-8-sig")
    if len(images) != 21527 or images.image.duplicated().any():
        raise ValueError("Manifest ảnh thiếu hoặc trùng khóa.")
    with zipfile.ZipFile(source) as archive:
        source_images = {name for name in archive.namelist() if name.endswith(".jpg")}
        if source_images != set(images.image):
            raise ValueError("Danh sách ảnh nguồn khác khảo sát.")
    if not set(pairs.image_a).issubset(source_images) or not set(pairs.image_b).issubset(source_images):
        raise ValueError("Cặp ứng viên tham chiếu ảnh ngoài nguồn.")
    return images, pairs, issues


def image_candidate_counts(images: pd.DataFrame, pairs: pd.DataFrame, issues: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cross = pairs[pairs.pair_type == "cross_split"].copy()
    cross["test_image"] = np.where(cross.split_a == "test", cross.image_a, cross.image_b)
    cross["train_image"] = np.where(cross.split_a == "train", cross.image_a, cross.image_b)
    cross["test_category"] = np.where(cross.split_a == "test", cross.category_a, cross.category_b)
    cross["train_category"] = np.where(cross.split_a == "train", cross.category_a, cross.category_b)
    cross["test_source"] = np.where(cross.split_a == "test", cross.prefix_a, cross.prefix_b)
    cross["train_source"] = np.where(cross.split_a == "train", cross.prefix_a, cross.prefix_b)
    cross["category_mismatch"] = cross.test_category != cross.train_category
    cross["priority_score"] = cross.dhash_distance + cross.phash_distance / 4 + (1 - cross.color_overlap) * 10
    nearest = cross.sort_values(["priority_score", "test_image", "train_image"]).drop_duplicates("test_image")
    representative = cross.sort_values(
        ["category_mismatch", "priority_score", "test_image", "train_image"],
        ascending=[False, True, True, True],
    ).drop_duplicates("test_image")
    test_counts = cross.groupby("test_image").size()
    train_counts = cross.groupby("train_image").size()
    mismatch_counts = cross[cross.category_mismatch].groupby("test_image").size()
    repeat = pairs[pairs.pair_type == "within_test"]
    repeat_counts = pd.concat([repeat.image_a, repeat.image_b]).value_counts()
    issue_counts = issues.image.value_counts()
    manifest = images.rename(columns={"split": "original_split", "prefix": "source_prefix"}).copy()
    manifest["cross_split_candidate_count"] = manifest.image.map(
        pd.concat([test_counts, train_counts])
    ).fillna(0).astype(int)
    manifest["test_repetition_candidate_count"] = manifest.image.map(repeat_counts).fillna(0).astype(int)
    manifest["label_issue_count"] = manifest.image.map(issue_counts).fillna(0).astype(int)
    manifest["nearest_train_candidate"] = manifest.image.map(nearest.set_index("test_image").train_image).fillna("")
    manifest["category_mismatch_candidate_count"] = manifest.image.map(mismatch_counts).fillna(0).astype(int)
    manifest["proposed_role"] = np.where(manifest.original_split == "test", "test_candidate", "train_pool")
    manifest["provisional_action"] = np.select(
        [manifest.original_split.eq("test"), manifest.cross_split_candidate_count.gt(0)],
        ["retain_test_candidate", "quarantine_train_pending_scene_review"],
        default="eligible_train_val_pending_review",
    )
    manifest["provisional_reason"] = np.select(
        [manifest.original_split.eq("test"), manifest.cross_split_candidate_count.gt(0)],
        ["original_test_starting_pool", "possible_same_scene_as_original_test"],
        default="no_cross_split_candidate_in_current_filter",
    )
    manifest["review_flags"] = manifest.apply(
        lambda row: "|".join(flag for flag, active in (
            ("label", row.label_issue_count > 0),
            ("cross_split_scene", row.cross_split_candidate_count > 0),
            ("test_repetition", row.test_repetition_candidate_count > 0),
        ) if active), axis=1
    )
    manifest["decision"] = "pending_review"
    manifest["decision_reason"] = ""
    queue = representative[["test_image", "train_image", "test_category", "train_category", "test_source", "train_source",
                     "dhash_distance", "phash_distance", "color_overlap", "priority_score"]].copy()
    queue["category_mismatch"] = queue.test_category != queue.train_category
    queue["cross_split_candidate_count"] = queue.test_image.map(test_counts)
    queue["category_mismatch_candidate_count"] = queue.test_image.map(mismatch_counts).fillna(0).astype(int)
    queue["review_outcome"] = "pending_visual_review"
    queue["review_notes"] = ""
    queue = queue.sort_values(["category_mismatch", "cross_split_candidate_count", "priority_score"],
                              ascending=[False, False, True])
    return manifest, queue


def label_proposals(issues: pd.DataFrame) -> pd.DataFrame:
    proposals = []
    for row in issues.itertuples():
        parts = row.raw.split()
        if len(parts) != 5:
            raise ValueError(f"Dòng nhãn không có đủ 5 giá trị: {row.image}")
        cls, x, y, w, h = map(float, parts)
        record = dict(image=row.image, original_split=row.image.split("/")[0], line=row.line,
                      reason=row.reason, original_box=row.raw, proposed_box="", retained_area_fraction=np.nan,
                      proposal_status="pending_visual_review")
        if row.reason == "outside" and w > 0 and h > 0:
            x1, x2 = max(0, x - w / 2), min(1, x + w / 2)
            y1, y2 = max(0, y - h / 2), min(1, y + h / 2)
            fraction = (x2 - x1) * (y2 - y1) / (w * h)
            record["retained_area_fraction"] = fraction
            if fraction >= 0.8:
                record["proposed_box"] = f"{int(cls)} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}"
                record["proposal_status"] = "clip_after_visual_review"
            else:
                record["proposal_status"] = "manual_relabel_or_exclude"
        elif row.reason == "nonpositive":
            record["proposal_status"] = "drop_box_then_review_whole_image"
        proposals.append(record)
    return pd.DataFrame(proposals)


def candidate_components(images: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    strong = pairs[(pairs.dhash_distance <= 1) & (pairs.phash_distance <= 4) &
                   (pairs.color_overlap >= 0.95)].copy()
    lookup = pd.Series(np.arange(len(images)), index=images.image)
    left = strong.image_a.map(lookup).to_numpy(dtype=np.int32)
    right = strong.image_b.map(lookup).to_numpy(dtype=np.int32)
    graph = coo_matrix((np.ones(2 * len(left), dtype=np.uint8),
                        (np.r_[left, right], np.r_[right, left])), shape=(len(images), len(images)))
    _, numbers = connected_components(graph, directed=False)
    members = images[["image", "split", "prefix", "category"]].copy()
    members["component_number"] = numbers
    counts = members.component_number.value_counts()
    members = members[members.component_number.map(counts) > 1].copy()
    rows = []
    for number, group in members.groupby("component_number"):
        splits = group.split.value_counts()
        rows.append(dict(component_number=number, images=len(group), train=int(splits.get("train", 0)),
                         test=int(splits.get("test", 0)), prefixes="|".join(sorted(group.prefix.unique())),
                         categories="|".join(sorted(group.category.unique())),
                         review_status="candidate_group_not_confirmed"))
    groups = pd.DataFrame(rows).sort_values(["test", "images"], ascending=False).reset_index(drop=True)
    group_ids = {row.component_number: f"C{number:04d}" for number, row in enumerate(groups.itertuples(), 1)}
    groups["candidate_group_id"] = groups.component_number.map(group_ids)
    members["candidate_group_id"] = members.component_number.map(group_ids)
    return groups, members


def render_group_sheets(source: Path, groups: pd.DataFrame, members: pd.DataFrame, output: Path) -> None:
    output.mkdir(exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        for row in groups[(groups.train > 0) & (groups.test > 0)].head(24).itertuples():
            group = members[members.component_number == row.component_number]
            selected = []
            for split in ("test", "train"):
                subset = group[group.split == split]
                selected.extend(subset.head(6).image.tolist())
            sheet = Image.new("RGB", (960, 4 * 230 + 35), "#202020")
            draw = ImageDraw.Draw(sheet)
            draw.text((8, 8), f"{row.candidate_group_id}: {row.images} images, test={row.test}, train={row.train}", fill="white")
            for index, key in enumerate(selected[:12]):
                with Image.open(BytesIO(archive.read(key))) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    image.thumbnail((310, 196))
                tile_x = (index % 3) * 320 + (320 - image.width) // 2
                tile_y = 35 + (index // 3) * 230
                sheet.paste(image, (tile_x, tile_y))
                draw.text(((index % 3) * 320 + 5, tile_y + 201), key[:46], fill="white")
            sheet.save(output / f"{row.candidate_group_id}.jpg", quality=84)


def render_label_sheets(source: Path, proposals: pd.DataFrame, output: Path) -> None:
    output.mkdir(exist_ok=True)
    affected = proposals.image.drop_duplicates().tolist()
    with zipfile.ZipFile(source) as archive:
        for page, start in enumerate(range(0, len(affected), 8), 1):
            sheet = Image.new("RGB", (1000, 4 * 275), "#202020")
            draw = ImageDraw.Draw(sheet)
            for position, key in enumerate(affected[start:start + 8]):
                column, row = position % 2, position // 2
                with Image.open(BytesIO(archive.read(key))) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
                    image.thumbnail((490, 240))
                x0 = column * 500 + (500 - image.width) // 2
                y0 = row * 275
                sheet.paste(image, (x0, y0))
                for issue in proposals[proposals.image == key].itertuples():
                    _, x, y, width, height = map(float, issue.original_box.split())
                    x1 = x0 + max(0, x - width / 2) * image.width
                    y1 = y0 + max(0, y - height / 2) * image.height
                    x2 = x0 + min(1, x + width / 2) * image.width
                    y2 = y0 + min(1, y + height / 2) * image.height
                    if x2 > x1 and y2 > y1:
                        draw.rectangle((x1, y1, x2, y2), outline="#ffff00", width=3)
                    else:
                        cx, cy = x0 + x * image.width, y0 + y * image.height
                        draw.line((cx - 8, cy - 8, cx + 8, cy + 8), fill="#ff4040", width=3)
                        draw.line((cx - 8, cy + 8, cx + 8, cy - 8), fill="#ff4040", width=3)
                reason = "|".join(proposals.loc[proposals.image == key, "reason"].unique())
                draw.text((column * 500 + 5, y0 + 244), f"{key} | {reason}"[:74], fill="white")
            sheet.save(output / f"labels_{page:02d}.jpg", quality=86)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    images, pairs, issues = read_audit(args.source, args.audit)
    manifest, scene_queue = image_candidate_counts(images, pairs, issues)
    labels = label_proposals(issues)
    groups, group_members = candidate_components(images, pairs)
    manifest.to_csv(args.output / "image_decision_manifest.csv", index=False, encoding="utf-8-sig")
    manifest[["image", "original_split", "source_prefix", "category", "provisional_action",
              "provisional_reason", "cross_split_candidate_count", "test_repetition_candidate_count",
              "label_issue_count", "nearest_train_candidate", "decision"]].to_csv(
        args.output / "decision_preview.csv", index=False, encoding="utf-8-sig"
    )
    manifest[manifest.original_split == "test"].to_csv(
        args.output / "test_candidate_manifest.csv", index=False, encoding="utf-8-sig"
    )
    manifest[manifest.original_split == "train"].to_csv(
        args.output / "train_pool_manifest.csv", index=False, encoding="utf-8-sig"
    )
    scene_queue.to_csv(args.output / "scene_review_queue.csv", index=False, encoding="utf-8-sig")
    labels.to_csv(args.output / "label_fix_proposals.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(args.output / "candidate_groups.csv", index=False, encoding="utf-8-sig")
    group_members.to_csv(args.output / "candidate_group_members.csv", index=False, encoding="utf-8-sig")
    render_group_sheets(args.source, groups, group_members, args.output / "group_sheets")
    render_label_sheets(args.source, labels, args.output / "label_sheets")
    summary = dict(status="PENDING_REVIEW_TEST_NOT_LOCKED", source_zip_sha256=source_sha256(args.source),
                   images=len(manifest), test_candidates=int((manifest.original_split == "test").sum()),
                   train_pool=int((manifest.original_split == "train").sum()),
                   test_with_cross_split_candidates=int(((manifest.original_split == "test") &
                                                         (manifest.cross_split_candidate_count > 0)).sum()),
                   scene_review_rows=len(scene_queue), label_proposal_rows=len(labels),
                   candidate_groups=len(groups), cross_split_candidate_groups=int(((groups.train > 0) &
                                                                                    (groups.test > 0)).sum()),
                   max_candidate_group_size=int(groups.images.max()),
                   train_provisionally_quarantined=int(((manifest.original_split == "train") &
                                                        manifest.cross_split_candidate_count.gt(0)).sum()),
                   note="Không có quyết định giữ/loại hoặc split mới; nhóm hash có thể nối sai cảnh.")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
