#!/usr/bin/env python3
"""Generate the selective-recolor demo gallery for mosaicraft.

The Vermeer "Girl with a Pearl Earring" turban is a textbook case of a
single, well-defined coloured object inside a richer image — exactly the
shape that ``recolor_region`` is designed to handle. This script:

1. Builds a mosaic of the painting (so the demo lives inside the mosaic
   pipeline, not just a flat image).
2. Detects the blue turban with an Oklch color-range mask.
3. Recolors only that region to seven different target colors.
4. Composites a 4x2 grid into ``docs/images/selective_recolor_turban.jpg``.
5. Also writes the binary mask to ``docs/images/selective_recolor_mask.png``
   so the README can show *what* was detected.

The painting and tile pool are CC0 / public domain (see ``docs/assets/``).

Usage::

    python scripts/generate_recolor_region_demo.py
    python scripts/generate_recolor_region_demo.py --quick   # smaller mosaic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mosaicraft import MosaicGenerator, configure_logging  # noqa: E402
from mosaicraft.recolor import recolor_region  # noqa: E402

ASSETS = REPO_ROOT / "docs" / "assets"
PAINTING = ASSETS / "paintings" / "pearl_earring.jpg"
TILES = ASSETS / "tiles"
OUT_DIR = REPO_ROOT / "docs" / "images"

# Vermeer's blue turban — Oklch hue ~250-260° in our convention.
# Wider tolerance pulls in the shaded mid-tones; chroma_min drops the
# near-black background.
TURBAN_BLUE_HEX = "#3a5d9e"
TURBAN_HUE_TOLERANCE = 28.0
TURBAN_CHROMA_MIN = 0.04
TURBAN_LIGHTNESS = (0.18, 0.78)

TARGETS: list[tuple[str, str, dict]] = [
    ("Original", "", {}),
    ("Detected mask", "", {}),
    ("Emerald",   "green",     {}),
    ("Crimson",   "red",       {}),
    ("Royal violet", "purple", {}),
    ("Solar gold",   "yellow", {"chroma": 1.05}),
    ("Mint",     "mint",       {}),
    ("Cyberpunk", "cyberpunk", {}),
]


def _label(img: np.ndarray, text: str) -> np.ndarray:
    bar_h = 36
    h, w = img.shape[:2]
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:h] = img
    out[h:] = (28, 28, 28)
    cv2.putText(
        out, text, (12, h + 25),
        cv2.FONT_HERSHEY_DUPLEX, 0.65,
        (240, 240, 240), 1, cv2.LINE_AA,
    )
    return out


def _stack_grid(panels: list[np.ndarray], cols: int) -> np.ndarray:
    rows: list[np.ndarray] = []
    for i in range(0, len(panels), cols):
        row = panels[i:i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(panels[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _build_mosaic(painting: Path, tiles: Path, *, quick: bool) -> np.ndarray:
    target_tiles = 600 if quick else 1500
    tile_size = 56 if quick else 72
    gen = MosaicGenerator(
        tile_dir=tiles,
        preset="vivid",
        augment=True,
        color_variants=4,
    )
    res = gen.generate(
        painting,
        OUT_DIR / ".tmp_pearl_mosaic.jpg",
        target_tiles=target_tiles,
        tile_size=tile_size,
        jpeg_quality=95,
    )
    return res.image


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true",
                   help="Faster, lower-resolution mosaic")
    p.add_argument(
        "--use-painting", action="store_true",
        help="Skip the mosaic pipeline and recolor the painting directly",
    )
    args = p.parse_args()

    configure_logging(verbose=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PAINTING.exists():
        print(f"Painting missing: {PAINTING}", file=sys.stderr)
        print("Run scripts/download_demo_assets.py first.", file=sys.stderr)
        return 2

    if args.use_painting or not TILES.exists() or not any(TILES.iterdir()):
        print("[demo] using raw painting (no mosaic stage)")
        base = cv2.imread(str(PAINTING))
        if base is None:
            print(f"Could not decode {PAINTING}", file=sys.stderr)
            return 2
    else:
        print("[demo] building Vermeer mosaic")
        base = _build_mosaic(PAINTING, TILES, quick=args.quick)
        # Drop the temporary mosaic file we wrote during the build.
        tmp = OUT_DIR / ".tmp_pearl_mosaic.jpg"
        if tmp.exists():
            tmp.unlink()

    # Resize for the figure: keep it README-friendly.
    fig_h = 480
    fig_w = int(base.shape[1] * fig_h / base.shape[0])
    base_small = cv2.resize(base, (fig_w, fig_h), interpolation=cv2.INTER_AREA)

    # Probe the mask once on the small figure so the gallery is consistent.
    mask = None
    panels: list[np.ndarray] = []
    for label, preset, overrides in TARGETS:
        if label == "Original":
            panels.append(_label(base_small, label))
            continue
        if label == "Detected mask":
            from mosaicraft.recolor import (
                _hex_to_bgr,  # type: ignore[attr-defined]
                _hue_from_bgr,  # type: ignore[attr-defined]
                build_oklch_region_mask,
            )

            b, g, r = _hex_to_bgr(TURBAN_BLUE_HEX)
            hue_center = _hue_from_bgr(b, g, r)
            mask = build_oklch_region_mask(
                base_small,
                hue_center_deg=hue_center,
                hue_tolerance_deg=TURBAN_HUE_TOLERANCE,
                chroma_min=TURBAN_CHROMA_MIN,
                lightness_min=TURBAN_LIGHTNESS[0],
                lightness_max=TURBAN_LIGHTNESS[1],
                morph_open=3,
                morph_close=7,
                min_area=400,
            )
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # Composite mask outline over the base for legibility.
            highlight = base_small.copy()
            highlight[mask > 0] = (0.4 * highlight[mask > 0] + 0.6 * np.array([255, 255, 255])).astype(np.uint8)
            panels.append(_label(highlight, label))
            cv2.imwrite(str(OUT_DIR / "selective_recolor_mask.png"), mask_rgb)
            continue

        kwargs: dict = {
            "source_hex": TURBAN_BLUE_HEX,
            "hue_tolerance_deg": TURBAN_HUE_TOLERANCE,
            "chroma_min": TURBAN_CHROMA_MIN,
            "lightness_min": TURBAN_LIGHTNESS[0],
            "lightness_max": TURBAN_LIGHTNESS[1],
            "morph_open": 3,
            "morph_close": 7,
            "min_area": 400,
            "feather_px": 4,
            "preset": preset,
        }
        if "chroma" in overrides:
            kwargs["chroma_scale"] = overrides["chroma"]
        out = recolor_region(base_small, **kwargs)
        panels.append(_label(out, label))

    grid = _stack_grid(panels, cols=4)
    out_path = OUT_DIR / "selective_recolor_turban.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"[demo] wrote {out_path} ({grid.shape[1]}x{grid.shape[0]})")
    print(f"[demo] wrote {OUT_DIR / 'selective_recolor_mask.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
