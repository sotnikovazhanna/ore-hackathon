from __future__ import annotations

import io
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import cv2
import numpy as np
from PIL import Image, ImageOps

# The case includes very large microscopy panoramas. Pillow's default safeguard
# rejects them before we can make a reduced working copy. We therefore disable
# that check, then enforce our own explicit upper bound below.
Image.MAX_IMAGE_PIXELS = None

MAX_SOURCE_PIXELS = 700_000_000


@dataclass(frozen=True)
class LoadedImage:
    rgb: np.ndarray
    original_width: int
    original_height: int
    analysis_width: int
    analysis_height: int
    scale: float
    decoder: str
    file_size_bytes: int

    @property
    def original_megapixels(self) -> float:
        return self.original_width * self.original_height / 1_000_000

    @property
    def analysis_megapixels(self) -> float:
        return self.analysis_width * self.analysis_height / 1_000_000


def _target_size(width: int, height: int, max_pixels: int) -> tuple[int, int, float]:
    pixels = width * height
    if pixels <= max_pixels:
        return width, height, 1.0
    scale = math.sqrt(max_pixels / pixels)
    return max(1, int(round(width * scale))), max(1, int(round(height * scale))), scale


def _to_uint8_rgb(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)

    # Common TIFF layouts: YX, YXS, SYX.
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    elif array.ndim == 3 and array.shape[0] in (3, 4) and array.shape[-1] not in (3, 4):
        array = np.moveaxis(array, 0, -1)

    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape: {array.shape}")

    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    elif array.shape[2] >= 4:
        array = array[..., :3]

    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        if info.max > 255:
            array = (array.astype(np.float32) / info.max * 255.0)
        return np.clip(array, 0, 255).astype(np.uint8)

    array = array.astype(np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros((*array.shape[:2], 3), dtype=np.uint8)
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        high = low + 1.0
    array = (array - low) / (high - low) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def _load_tiff(data: bytes, max_pixels: int) -> LoadedImage:
    import tifffile

    with tempfile.NamedTemporaryFile(suffix=".tiff", delete=True) as temporary:
        temporary.write(data)
        temporary.flush()

        with tifffile.TiffFile(temporary.name) as tif:
            series = tif.series[0]
            levels = list(getattr(series, "levels", [series]))

            def dimensions(level) -> tuple[int, int]:
                shape = tuple(level.shape)
                axes = getattr(level, "axes", "")
                if "Y" in axes and "X" in axes:
                    return int(shape[axes.index("X")]), int(shape[axes.index("Y")])
                return int(shape[-1]), int(shape[-2])

            original_width, original_height = dimensions(levels[0])
            if original_width * original_height > MAX_SOURCE_PIXELS:
                raise ValueError(
                    f"Image has {original_width * original_height:,} pixels; "
                    f"the safety limit is {MAX_SOURCE_PIXELS:,}."
                )

            # Prefer an existing pyramid level. It avoids decoding the full level.
            chosen = levels[-1]
            for level in levels:
                width, height = dimensions(level)
                if width * height <= max_pixels:
                    chosen = level
                    break

            array = chosen.asarray()
            rgb = _to_uint8_rgb(array)
            target_width, target_height, _ = _target_size(rgb.shape[1], rgb.shape[0], max_pixels)
            if (target_width, target_height) != (rgb.shape[1], rgb.shape[0]):
                rgb = cv2.resize(rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)

            return LoadedImage(
                rgb=np.ascontiguousarray(rgb),
                original_width=original_width,
                original_height=original_height,
                analysis_width=rgb.shape[1],
                analysis_height=rgb.shape[0],
                scale=rgb.shape[1] / original_width,
                decoder="tifffile-pyramid" if len(levels) > 1 else "tifffile",
                file_size_bytes=len(data),
            )


def _load_pillow(data: bytes, max_pixels: int) -> LoadedImage:
    with Image.open(io.BytesIO(data)) as opened:
        image_format = (opened.format or "").upper()
        original_width, original_height = opened.size
        pixels = original_width * original_height
        if pixels > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"Image has {pixels:,} pixels; the safety limit is {MAX_SOURCE_PIXELS:,}."
            )

        target_width, target_height, scale = _target_size(
            original_width, original_height, max_pixels
        )

        # JPEG draft asks libjpeg to decode directly at a reduced resolution.
        # This is the crucial path for 100–350 MP panoramas.
        decoder = "pillow"
        if image_format in {"JPEG", "JPG"} and scale < 1.0:
            opened.draft("RGB", (target_width, target_height))
            decoder = "jpeg-draft"

        source = ImageOps.exif_transpose(opened)
        source.thumbnail(
            (target_width, target_height),
            resample=Image.Resampling.LANCZOS,
            reducing_gap=3.0,
        )
        rgb = np.asarray(source.convert("RGB"), dtype=np.uint8)

    return LoadedImage(
        rgb=np.ascontiguousarray(rgb),
        original_width=original_width,
        original_height=original_height,
        analysis_width=rgb.shape[1],
        analysis_height=rgb.shape[0],
        scale=rgb.shape[1] / original_width,
        decoder=decoder,
        file_size_bytes=len(data),
    )


def load_uploaded_image(
    uploaded_file: BinaryIO,
    filename: str,
    max_analysis_megapixels: float,
) -> LoadedImage:
    """Decode a possibly huge panorama into a bounded working-resolution RGB array.

    Percent-area estimates are computed on the uniformly reduced image, so the
    percentage remains comparable while memory and inference time stay bounded.
    """

    data = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    if not data:
        raise ValueError("The uploaded file is empty.")

    max_pixels = max(1_000_000, int(max_analysis_megapixels * 1_000_000))
    suffix = Path(filename).suffix.lower()

    if suffix in {".tif", ".tiff"}:
        try:
            return _load_tiff(data, max_pixels)
        except Exception:
            # Some vendor TIFF variants are better handled by Pillow.
            return _load_pillow(data, max_pixels)

    return _load_pillow(data, max_pixels)


def resize_for_display(rgb: np.ndarray, max_side: int = 1800) -> np.ndarray:
    height, width = rgb.shape[:2]
    if max(height, width) <= max_side:
        return rgb
    scale = max_side / max(height, width)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, target, interpolation=cv2.INTER_AREA)
