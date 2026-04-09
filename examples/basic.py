"""Minimal end-to-end example.

Generates a CC0 tile set, picks a CC0 target, and runs mosaicraft.
Run from the repository root::

    python examples/basic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the example runnable without installing the package.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mosaicraft import MosaicGenerator, configure_logging


def ensure_demo_assets() -> tuple[Path, Path]:
    """Generate demo tiles and a demo target if they don't already exist."""
    tiles_dir = REPO_ROOT / "demo_tiles"
    target = REPO_ROOT / "demo_target.jpg"
    if not tiles_dir.exists() or not any(tiles_dir.iterdir()):
        import cv2
        from scripts.generate_demo_tiles import generate_tile

        tiles_dir.mkdir(exist_ok=True)
        for i in range(256):
            tile = generate_tile(i, 128)
            cv2.imwrite(str(tiles_dir / f"tile_{i:04d}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not target.exists():
        import cv2
        from scripts.generate_demo_target import make_target

        img = make_target(800, 1000, seed=42)
        cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return tiles_dir, target


def main() -> int:
    configure_logging(verbose=True)
    tiles_dir, target = ensure_demo_assets()
    output = REPO_ROOT / "demo_mosaic.jpg"

    gen = MosaicGenerator(tile_dir=tiles_dir, preset="ultra")
    result = gen.generate(target, output, target_tiles=400, tile_size=64)
    print(f"\nGenerated {result.n_tiles} cells -> {output}")
    print(f"Output size: {result.image.shape[1]}x{result.image.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
