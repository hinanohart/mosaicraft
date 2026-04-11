"""mosaicraft: perceptual photomosaic generator.

Quick start::

    from mosaicraft import MosaicGenerator

    gen = MosaicGenerator(tile_dir="tiles/", preset="ultra")
    result = gen.generate("photo.jpg", "mosaic.jpg", target_tiles=2000)
    print(result.grid_cols, "x", result.grid_rows)

See the project README for the full API and CLI usage.
"""

from __future__ import annotations

from .color_augment import expand_color_variants, rotate_hue_oklch
from .core import MosaicGenerator, MosaicResult
from .presets import PRESETS, get_preset, list_presets
from .recolor import (
    RECOLOR_PRESETS,
    RecolorPreset,
    build_oklch_region_mask,
    get_recolor_preset,
    list_recolor_presets,
    recolor,
    recolor_region,
)
from .tiles import TileSet, build_cache, load_tiles
from .utils import calc_grid, configure_logging

__version__ = "0.3.0"

__all__ = [
    "PRESETS",
    "RECOLOR_PRESETS",
    "MosaicGenerator",
    "MosaicResult",
    "RecolorPreset",
    "TileSet",
    "__version__",
    "build_cache",
    "build_oklch_region_mask",
    "calc_grid",
    "configure_logging",
    "expand_color_variants",
    "get_preset",
    "get_recolor_preset",
    "list_presets",
    "list_recolor_presets",
    "load_tiles",
    "recolor",
    "recolor_region",
    "rotate_hue_oklch",
]
