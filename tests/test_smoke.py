"""End-to-end smoke test exercising the full pipeline.

These tests use synthetic tiles and a synthetic target so they run in a
few seconds with no external assets.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from mosaicraft import MosaicGenerator, list_presets
from mosaicraft.utils import calc_grid


def test_calc_grid_basic() -> None:
    cols, rows, total = calc_grid(2000, 1920, 1080)
    assert total > 0
    assert cols * rows == total


def test_list_presets_nonempty() -> None:
    presets = list_presets()
    assert "ultra" in presets
    assert "natural" in presets
    assert len(presets) >= 5


def test_generator_end_to_end_ultra(synthetic_tile_dir: Path, synthetic_target: Path, tmp_path: Path) -> None:
    out_path = tmp_path / "mosaic.png"
    gen = MosaicGenerator(tile_dir=synthetic_tile_dir, preset="ultra")
    result = gen.generate(
        synthetic_target,
        out_path,
        target_tiles=64,
        tile_size=32,
    )
    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3
    assert result.grid_cols * result.grid_rows == result.n_tiles
    assert out_path.exists()
    saved = cv2.imread(str(out_path))
    assert saved is not None
    assert saved.shape == result.image.shape


def test_generator_end_to_end_fast(synthetic_tile_dir: Path, synthetic_target: Path, tmp_path: Path) -> None:
    gen = MosaicGenerator(tile_dir=synthetic_tile_dir, preset="fast")
    result = gen.generate(
        synthetic_target,
        tmp_path / "mosaic_fast.png",
        target_tiles=49,
        tile_size=32,
    )
    assert result.n_tiles >= 1


def test_generator_with_cache(synthetic_tile_dir: Path, synthetic_target: Path, tmp_path: Path) -> None:
    from mosaicraft.tiles import build_cache

    cache_dir = tmp_path / "cache"
    build_cache(
        synthetic_tile_dir, cache_dir, tile_sizes=[32], thumb_size=48, progress=False
    )
    gen = MosaicGenerator(cache_dir=cache_dir, preset="natural")
    result = gen.generate(
        synthetic_target,
        tmp_path / "mosaic_cached.png",
        target_tiles=49,
        tile_size=32,
    )
    assert result.n_tiles >= 1


def test_generator_requires_source() -> None:
    import pytest

    with pytest.raises(ValueError, match="tile_dir or cache_dir"):
        MosaicGenerator()
