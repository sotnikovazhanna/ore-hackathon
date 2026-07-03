from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


def stable_split(groups: list[str], seed: int = 42) -> dict[str, str]:
    """Assign groups approximately 70/15/15 using a deterministic hash order."""
    unique = sorted(set(str(group) for group in groups))
    unique.sort(key=lambda value: hashlib.md5(f"{seed}:{value}".encode("utf-8")).hexdigest())
    n = len(unique)
    n_val = max(1, round(n * 0.15)) if n >= 3 else 0
    n_test = max(1, round(n * 0.15)) if n >= 3 else 0
    n_train = max(1, n - n_val - n_test)

    mapping: dict[str, str] = {}
    for index, group in enumerate(unique):
        if index < n_train:
            mapping[group] = "train"
        elif index < n_train + n_val:
            mapping[group] = "validation"
        else:
            mapping[group] = "test"
    return mapping


def resolve_inventory_path(raw_path: str, relative_path: str, data_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.exists():
        return path
    candidate = data_root / Path(str(relative_path))
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Image not found: {raw_path} / {candidate}")


def image_shape(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        return image.height, image.width


def main() -> None:
    parser = argparse.ArgumentParser(description="Build segmentation manifest from manual positives and hard negatives.")
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--annotations", type=Path, default=Path("data/manual_talc/merged_annotations.csv"))
    parser.add_argument("--inventory", type=Path, default=Path("outputs/audit/dataset_inventory.csv"))
    parser.add_argument("--negative-count", type=int, default=60)
    parser.add_argument("--output", type=Path, default=Path("data/manual_talc/talc_segmentation_manifest.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    annotations = pd.read_csv(args.annotations, encoding="utf-8-sig")
    inventory = pd.read_csv(args.inventory, encoding="utf-8-sig")

    positive_rows: list[dict] = []
    for row in annotations.itertuples():
        image_path = data_root / Path(row.original_relative)
        mask_path = Path(str(row.mask_path))
        if not image_path.exists():
            raise FileNotFoundError(f"Positive image not found: {image_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Manual mask not found: {mask_path}")
        positive_rows.append(
            {
                "image_path": str(image_path.resolve()),
                "mask_path": str(mask_path.resolve()),
                "target_presence": 1,
                "source_type": "manual_positive",
                "label": "talc",
                "sample_group": str(row.task_id),
                "filename": row.filename,
            }
        )

    negatives = inventory[inventory["label"].isin(["ordinary", "fine"])].copy()
    if "phash" in negatives:
        negatives = negatives.drop_duplicates("phash", keep="first")

    # Half of the negatives are deliberately selected from the darkest images;
    # the other half is sampled across the remaining distribution.
    dark_count = min(args.negative_count // 2, len(negatives))
    negatives = negatives.sort_values("brightness_mean", ascending=True, na_position="last")
    dark = negatives.head(dark_count)
    remaining = negatives.drop(dark.index)
    random_count = min(args.negative_count - len(dark), len(remaining))
    random_part = remaining.sample(n=random_count, random_state=args.seed) if random_count else remaining.head(0)
    selected_negatives = pd.concat([dark, random_part], ignore_index=True)

    negative_rows: list[dict] = []
    for row in selected_negatives.itertuples():
        image_path = resolve_inventory_path(row.absolute_path, row.relative_path, data_root)
        negative_rows.append(
            {
                "image_path": str(image_path.resolve()),
                "mask_path": "",
                "target_presence": 0,
                "source_type": "hard_negative",
                "label": row.label,
                "sample_group": str(row.sample_group),
                "filename": row.filename,
            }
        )

    frame = pd.DataFrame(positive_rows + negative_rows)

    # Split positive and negative groups separately so every split gets both classes.
    split_column = pd.Series(index=frame.index, dtype="object")
    for target in [0, 1]:
        subset = frame[frame["target_presence"] == target]
        mapping = stable_split(subset["sample_group"].astype(str).tolist(), seed=args.seed + target)
        split_column.loc[subset.index] = subset["sample_group"].astype(str).map(mapping)
    frame["split"] = split_column

    # Final safety check: one sample group must never appear in multiple splits.
    leakage = frame.groupby("sample_group")["split"].nunique().gt(1).sum()
    if leakage:
        raise RuntimeError(f"Group leakage detected: {leakage}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"Manifest written: {args.output.resolve()}")
    print(frame.groupby(["split", "target_presence"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
