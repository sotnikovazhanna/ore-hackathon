from __future__ import annotations

import argparse
import time
from pathlib import Path

from src.large_image import load_uploaded_image
from src.ore_pipeline import OreAnalyzer


class LocalUpload:
    def __init__(self, path: Path) -> None:
        self.path = path

    def getvalue(self) -> bytes:
        return self.path.read_bytes()


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one large ore panorama.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--max-mp", type=float, default=18.0)
    parser.add_argument("--tile-size", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=12)
    args = parser.parse_args()

    analyzer = OreAnalyzer(model_dir="models")
    started = time.time()
    loaded = load_uploaded_image(LocalUpload(args.image), args.image.name, args.max_mp)
    decode_seconds = time.time() - started

    infer_started = time.time()
    result = analyzer.analyze(
        loaded.rgb,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    inference_seconds = time.time() - infer_started
    total_seconds = time.time() - started

    print(f"Device: {analyzer.device}")
    print(f"Original: {loaded.original_width}x{loaded.original_height} ({loaded.original_megapixels:.1f} MP)")
    print(f"Analysis: {loaded.analysis_width}x{loaded.analysis_height} ({loaded.analysis_megapixels:.1f} MP)")
    print(f"Decoder: {loaded.decoder}")
    print(f"Tiles: {result.tile_count}")
    print(f"Decode: {decode_seconds:.1f} s")
    print(f"Inference: {inference_seconds:.1f} s")
    print(f"Total: {total_seconds:.1f} s")
    print(f"Class: {result.ore_class}")
    print(f"Talc area: {result.talc_share_percent:.1f}%")
    print(f"Decision: {result.decision_reason}")
    if result.warning:
        print(f"Warning: {result.warning}")


if __name__ == "__main__":
    main()
