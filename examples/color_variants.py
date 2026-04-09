"""Expand a small tile pool into a bigger one with Oklch hue rotation.

This example shows the v0.2.0 ``color_variants`` feature: every tile is
reused at N additional positions on the Oklab a/b plane by rotating its
hue, so a 256-tile pool becomes a 1,280-tile pool (1 original + 4 rotated
variants) without downloading a single extra photograph. The Oklab L
channel is preserved exactly, so per-tile texture and shading survive.

Run from the repository root::

    python examples/color_variants.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from examples.basic import ensure_demo_assets

from mosaicraft import MosaicGenerator, configure_logging


def main() -> int:
    configure_logging(verbose=True)
    tiles_dir, target = ensure_demo_assets()

    # Same pipeline as examples/basic.py, but with the tile pool expanded
    # 5x (1 original + 4 Oklch hue rotations at 72°/144°/216°/288°).
    gen = MosaicGenerator(
        tile_dir=tiles_dir,
        preset="ultra",
        color_variants=4,
    )
    output = REPO_ROOT / "demo_mosaic_cv4.jpg"
    result = gen.generate(target, output, target_tiles=400, tile_size=64)

    print(f"\nGenerated {result.n_tiles} cells -> {output}")
    print("Tile pool was expanded 5x by Oklch hue rotation (color_variants=4).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
