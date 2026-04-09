"""mosaicraft: perceptual photomosaic generator.

Quick start::

    from mosaicraft import MosaicGenerator

    gen = MosaicGenerator(tile_dir="tiles/", preset="ultra")
    result = gen.generate("photo.jpg", "mosaic.jpg", target_tiles=2000)
    print(result.grid_cols, "x", result.grid_rows)

See the project README for the full API and CLI usage.
"""

from __future__ import annotations

from .core import MosaicGenerator, MosaicResult
from .presets import PRESETS, get_preset, list_presets
from .tiles import TileSet, build_cache, load_tiles
from .utils import calc_grid, configure_logging

__version__ = "0.1.0"

__all__ = [
    "PRESETS",
    "MosaicGenerator",
    "MosaicResult",
    "TileSet",
    "__version__",
    "build_cache",
    "calc_grid",
    "configure_logging",
    "get_preset",
    "list_presets",
    "load_tiles",
]
