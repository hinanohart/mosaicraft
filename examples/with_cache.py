"""Pre-build a feature cache then generate multiple mosaics quickly.

For tile sets with thousands of images, the cache makes the second and
subsequent runs an order of magnitude faster.

Run from the repository root::

    python examples/with_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaicraft import MosaicGenerator, build_cache, configure_logging

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    configure_logging(verbose=True)

    tiles_dir = REPO_ROOT / "demo_tiles"
    if not tiles_dir.exists():
        print(f"Generate demo tiles first: python scripts/generate_demo_tiles.py -o {tiles_dir}")
        return 1

    target = REPO_ROOT / "demo_target.jpg"
    if not target.exists():
        print(f"Generate a demo target first: python scripts/generate_demo_target.py -o {target}")
        return 1

    cache_dir = REPO_ROOT / "demo_cache"
    print(f"Building cache at {cache_dir} (one-time)...")
    build_cache(tiles_dir, cache_dir, tile_sizes=[48, 64, 88], thumb_size=120, progress=True)

    # Generate two presets back-to-back, reusing the cache.
    for preset in ("ultra", "vivid"):
        gen = MosaicGenerator(cache_dir=cache_dir, preset=preset)
        result = gen.generate(
            target,
            REPO_ROOT / f"demo_mosaic_{preset}.jpg",
            target_tiles=400,
            tile_size=64,
        )
        print(f"  {preset}: {result.grid_cols}x{result.grid_rows} -> {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
