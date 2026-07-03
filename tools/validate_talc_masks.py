from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    if mask.shape != shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {shape}: {path}")
    return mask > 127


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = rgb.astype(np.float32).copy()
    result[mask] = 0.55 * result[mask] + 0.45 * np.array([0, 90, 255], dtype=np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def thumb(rgb: np.ndarray, size=(420, 315)) -> Image.Image:
    image = Image.fromarray(rgb)
    image.thumbnail(size)
    canvas = Image.new("RGB", size, "black")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate manual talc masks and make review pages.")
    parser.add_argument("--annotations", type=Path, default=Path("data/manual_talc/merged_annotations.csv"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/manual_talc/review"))
    parser.add_argument("--rows-per-page", type=int, default=6)
    args = parser.parse_args()

    frame = pd.read_csv(args.annotations, encoding="utf-8-sig")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in frame.itertuples():
        image_path = args.data_root / Path(row.original_relative)
        mask_path = Path(str(row.mask_path))
        rgb = load_rgb(image_path)
        mask = load_mask(mask_path, rgb.shape[:2])
        records.append(
            {
                "task_id": row.task_id,
                "filename": row.filename,
                "image_path": str(image_path),
                "mask_path": str(mask_path),
                "mask_share_percent": float(mask.mean() * 100),
                "empty": bool(not mask.any()),
                "full": bool(mask.all()),
            }
        )

    stats = pd.DataFrame(records)
    stats.to_csv(args.output_dir / "mask_validation.csv", index=False, encoding="utf-8-sig")

    page_count = math.ceil(len(stats) / args.rows_per_page)
    tile_size = (420, 315)
    row_height = tile_size[1] + 34

    for page_index in range(page_count):
        rows = stats.iloc[page_index * args.rows_per_page : (page_index + 1) * args.rows_per_page]
        canvas = Image.new("RGB", (tile_size[0] * 2, 45 + row_height * len(rows)), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 14), f"Manual talc masks — page {page_index + 1}/{page_count}", fill="black")

        for local_index, row in enumerate(rows.itertuples()):
            rgb = load_rgb(Path(row.image_path))
            mask = load_mask(Path(row.mask_path), rgb.shape[:2])
            y = 45 + local_index * row_height
            canvas.paste(thumb(rgb, tile_size), (0, y))
            canvas.paste(thumb(overlay(rgb, mask), tile_size), (tile_size[0], y))
            caption = f"{row.filename} | area={row.mask_share_percent:.1f}% | empty={row.empty} | full={row.full}"
            draw.text((8, y + tile_size[1] + 8), caption, fill="red" if row.empty or row.full else "black")

        canvas.save(args.output_dir / f"review_{page_index + 1:02d}.jpg", quality=92)

    print(f"Validated masks: {len(stats)}")
    print(f"Empty masks: {int(stats['empty'].sum())}")
    print(f"Full masks: {int(stats['full'].sum())}")
    print(f"Review pages: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
