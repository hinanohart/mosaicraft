"""mosaicraft: perceptual photomosaic generator.

Quick start::

    from mosaicraft import MosaicGenerator

    gen = MosaicGenerator(tile_dir="tiles/", preset="vivid")
    result = gen.generate("photo.jpg", "mosaic.jpg", target_tiles=2000)
    print(result.grid_cols, "x", result.grid_rows)

See the project README for the full API and CLI usage.
"""

from __future__ import annotations

from .color_augment import expand_color_variants, rotate_hue_oklch
from .core import MosaicGenerator, MosaicResult
from .presets import PRESETS, get_preset, list_presets
from .tiles import TileSet, build_cache, load_tiles
from .utils import calc_grid, configure_logging

__version__ = "0.3.2"

__all__ = [
    "PRESETS",
    "MosaicGenerator",
    "MosaicResult",
    "TileSet",
    "__version__",
    "build_cache",
    "calc_grid",
    "configure_logging",
    "expand_color_variants",
    "get_preset",
    "list_presets",
    "load_tiles",
    "rotate_hue_oklch",
]


# ---------------------------------------------------------------------------
# Withdrawn-symbol shim
# ---------------------------------------------------------------------------
_WITHDRAWN = {
    "recolor_region": "0.3.1",
    "build_oklch_region_mask": "0.3.1",
    "recolor": "0.4.0",
    "RECOLOR_PRESETS": "0.4.0",
    "RecolorPreset": "0.4.0",
    "get_recolor_preset": "0.4.0",
    "list_recolor_presets": "0.4.0",
}


def __getattr__(name: str):
    if name in _WITHDRAWN:
        version = _WITHDRAWN[name]
        raise AttributeError(
            f"mosaicraft.{name} was removed in v{version}. "
            f"Pin an older version if you depend on it, "
            f"or see https://github.com/hinanohart/mosaicraft/issues "
            f"for alternatives."
        )
    raise AttributeError(f"module 'mosaicraft' has no attribute {name!r}")
