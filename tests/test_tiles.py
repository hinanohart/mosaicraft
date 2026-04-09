"""Tests for tile loading and caching."""

from __future__ import annotations

from pathlib import Path

import pytest

from mosaicraft.features import FEATURE_DIM
from mosaicraft.tiles import (
    augment_tiles,
    build_cache,
    load_tiles,
    load_tiles_cached,
)


def test_load_tiles_basic(synthetic_tile_dir: Path) -> None:
    tileset = load_tiles(synthetic_tile_dir, tile_size=32)
    assert len(tileset) == 64
    assert tileset.features.shape == (64, FEATURE_DIM)
    assert tileset.oklab_means.shape == (64, 3)
    assert all(t.shape == (32, 32, 3) for t in tileset.tiles)


def test_augment_tiles_4x(synthetic_tile_dir: Path) -> None:
    tileset = load_tiles(synthetic_tile_dir, tile_size=32)
    augmented = augment_tiles(tileset, tile_size=32)
    assert len(augmented) == 64 * 4
    assert augmented.features.shape == (64 * 4, FEATURE_DIM)


def test_load_tiles_missing_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tiles(tmp_path / "nope", tile_size=32)


def test_build_and_load_cache(synthetic_tile_dir: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    build_cache(
        synthetic_tile_dir,
        cache_dir,
        tile_sizes=[32],
        thumb_size=48,
        progress=False,
    )
    assert (cache_dir / "features_32.npz").exists()
    assert (cache_dir / "files.json").exists()
    cached = load_tiles_cached(cache_dir, tile_size=32)
    assert len(cached) == 64 * 4
    assert cached.features.shape == (64 * 4, FEATURE_DIM)


def test_load_cached_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_tiles_cached(tmp_path, tile_size=32)
