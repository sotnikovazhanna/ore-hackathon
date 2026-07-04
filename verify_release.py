from __future__ import annotations

import io
import inspect
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED_VERSION = "orescope-final-v1"
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.large_image import load_uploaded_image
from src.ore_pipeline import OreAnalyzer, PIPELINE_API_VERSION


def fail(message: str) -> None:
    raise SystemExit(f"VERIFY FAILED: {message}")


if PIPELINE_API_VERSION != EXPECTED_VERSION:
    fail(
        f"pipeline version is {PIPELINE_API_VERSION!r}, "
        f"expected {EXPECTED_VERSION!r}"
    )

if not hasattr(OreAnalyzer, "model_status"):
    fail("OreAnalyzer.model_status is missing")

pipeline_path = Path(inspect.getfile(OreAnalyzer)).resolve()
expected_pipeline = (PROJECT_ROOT / "src" / "ore_pipeline.py").resolve()
if pipeline_path != expected_pipeline:
    fail(f"imported pipeline from {pipeline_path}, expected {expected_pipeline}")

rgb = np.zeros((1200, 2000, 3), dtype=np.uint8)
rgb[..., 0] = np.linspace(0, 255, rgb.shape[1], dtype=np.uint8)
buffer = io.BytesIO()
Image.fromarray(rgb).save(buffer, format="JPEG", quality=85)


class Upload:
    def getvalue(self) -> bytes:
        return buffer.getvalue()


loaded = load_uploaded_image(
    Upload(),
    "verify.jpg",
    max_analysis_megapixels=1.0,
)
if loaded.analysis_width * loaded.analysis_height > 1_050_000:
    fail("large-image loader did not respect the megapixel budget")
if loaded.rgb.ndim != 3 or loaded.rgb.shape[2] != 3:
    fail(f"unexpected loaded image shape {loaded.rgb.shape}")


empty_models = PROJECT_ROOT / "_verify_empty_models"
empty_models.mkdir(exist_ok=True)
analyzer = OreAnalyzer(model_dir=empty_models, device="cpu")
status = analyzer.model_status()
if status.get("api_version") != EXPECTED_VERSION:
    fail(f"model_status api_version is {status.get('api_version')!r}")
if "talc_gate_loaded" in status:
    fail("legacy talc gate is still exposed in the final API")

small = np.full((384, 512, 3), 128, dtype=np.uint8)
result = analyzer.analyze(
    small,
    tile_size=320,
    overlap=16,
    batch_size=1,
)
if result.overlay.shape != small.shape:
    fail("analysis overlay shape mismatch")
if not isinstance(result.ore_class, str):
    fail("analysis did not return a class")
if not hasattr(result, "quality_label"):
    fail("quality assessment is missing")

try:
    empty_models.rmdir()
except OSError:
    pass

print("VERIFY OK")
print(f"Version: {EXPECTED_VERSION}")
print(f"Pipeline: {pipeline_path}")
print(
    "Large image loader: "
    f"{loaded.original_width}x{loaded.original_height} -> "
    f"{loaded.analysis_width}x{loaded.analysis_height}"
)
print(f"Smoke analysis class: {result.ore_class}")
print("Legacy talc gate: removed from final inference")
