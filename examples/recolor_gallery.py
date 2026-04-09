"""Recolor a finished mosaic through every named Oklch preset.

This example shows the v0.2.0 ``recolor`` feature: once you have a mosaic,
you can rotate its hue through 21 named presets (or any ``#RRGGBB``)
without regenerating a single tile. Lightness is preserved exactly, so
per-tile shading survives the rotation and no boundary artifacts appear.

Run from the repository root::

    python examples/basic.py              # produce demo_mosaic.jpg first
    python examples/recolor_gallery.py    # then fan it out into variants
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mosaicraft import configure_logging, list_recolor_presets, recolor


def main() -> int:
    configure_logging(verbose=True)

    mosaic = REPO_ROOT / "demo_mosaic.jpg"
    if not mosaic.exists():
        print(
            f"ERROR: {mosaic} does not exist.\n"
            "Run `python examples/basic.py` first to produce a mosaic."
        )
        return 1

    out_dir = REPO_ROOT / "demo_recolored"
    out_dir.mkdir(exist_ok=True)

    for name in list_recolor_presets():
        dst = out_dir / f"demo_mosaic_{name}.jpg"
        recolor(mosaic, dst, preset=name)
        print(f"  wrote {dst.relative_to(REPO_ROOT)}")

    # Any #RRGGBB also works.
    brand_hex = "#3b82f6"  # tailwind blue-500
    recolor(mosaic, out_dir / "demo_mosaic_brand.jpg", target_hex=brand_hex)
    print(f"  wrote demo_recolored/demo_mosaic_brand.jpg (custom {brand_hex})")

    print(f"\nRecolored {len(list_recolor_presets()) + 1} variants into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
