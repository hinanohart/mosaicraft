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
from .recolor import (
    RECOLOR_PRESETS,
    RecolorPreset,
    get_recolor_preset,
    list_recolor_presets,
    recolor,
)
from .tiles import TileSet, build_cache, load_tiles
from .utils import calc_grid, configure_logging

__version__ = "0.3.2"

__all__ = [
    "PRESETS",
    "RECOLOR_PRESETS",
    "MosaicGenerator",
    "MosaicResult",
    "RecolorPreset",
    "TileSet",
    "__version__",
    "build_cache",
    "calc_grid",
    "configure_logging",
    "expand_color_variants",
    "get_preset",
    "get_recolor_preset",
    "list_presets",
    "list_recolor_presets",
    "load_tiles",
    "recolor",
    "rotate_hue_oklch",
]


# ---------------------------------------------------------------------------
# Withdrawn-symbol shim
# ---------------------------------------------------------------------------
# v0.3.0 briefly shipped `recolor_region` and `build_oklch_region_mask` as
# selective-recolor APIs. They were withdrawn in v0.3.1 because the
# colour-range mask approach could not produce the quality the README
# implied without per-image hand-tuning. Without this shim, a v0.3.0 user
# who runs `pip install -U mosaicraft` would see the bare error
#     ImportError: cannot import name 'recolor_region' from 'mosaicraft'
# with no clue about why or how to recover. The shim raises a richer
# error that points at the v0.3.0 pin and the upstream issue tracker.
_WITHDRAWN_IN_0_3_1 = {
    "recolor_region": "0.3.1",
    "build_oklch_region_mask": "0.3.1",
}


def __getattr__(name: str):
    if name in _WITHDRAWN_IN_0_3_1:
        version = _WITHDRAWN_IN_0_3_1[name]
        raise AttributeError(
            f"mosaicraft.{name} was withdrawn in v{version}. "
            f"It briefly shipped in v0.3.0 but did not produce the quality "
            f"the README implied without per-image hand tuning. "
            f"Pin `mosaicraft==0.3.0` if you depend on it, "
            f"or watch https://github.com/hinanohart/mosaicraft/issues "
            f"for an [ai] extras re-launch built on a real segmentation backbone."
        )
    raise AttributeError(f"module 'mosaicraft' has no attribute {name!r}")
