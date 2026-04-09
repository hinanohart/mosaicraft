"""Build a custom preset by overriding fields of an existing one.

Useful when you want most of the defaults but a specific tweak — e.g. more
vibrance with less skin protection, or a different color transfer method.

Run from the repository root::

    python examples/custom_preset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mosaicraft import MosaicGenerator, configure_logging, get_preset

REPO_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    configure_logging(verbose=True)

    # Start from "natural" and override a few fields.
    custom = get_preset("natural")
    custom.update(
        {
            "color_transfer": "mkl_hybrid",
            "vibrance": 0.5,
            "saturation_boost": 1.7,
            "skin_protection": 0.4,
        }
    )

    tiles_dir = REPO_ROOT / "demo_tiles"
    target = REPO_ROOT / "demo_target.jpg"
    if not tiles_dir.exists() or not target.exists():
        print("Run examples/basic.py first to generate demo assets.")
        return 1

    gen = MosaicGenerator(tile_dir=tiles_dir, preset=custom)
    result = gen.generate(
        target,
        REPO_ROOT / "demo_mosaic_custom.jpg",
        target_tiles=400,
        tile_size=64,
    )
    print(f"Custom preset: {result.n_tiles} cells -> {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
