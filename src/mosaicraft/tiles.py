"""Tile loading, augmentation, and on-disk caching.

For large tile collections, recomputing feature vectors on every run is the
biggest bottleneck. The cache stores the augmented features as ``.npz``
alongside small thumbnails so the actual tile pixels can still be reloaded
quickly at any working size.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .color import bgr_to_oklab
from .features import FEATURE_DIM, extract_features

NDArray = np.ndarray

# Image extensions to consider when scanning a tile directory.
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "TileSet",
    "augment_tiles",
    "build_cache",
    "load_tiles",
    "load_tiles_cached",
]


@dataclass
class TileSet:
    """A loaded set of tile images and their precomputed features."""

    tiles: list[NDArray]
    features: NDArray  # (n, 191) float32
    lab_stats: list[tuple[float, float, float, float, float, float]]
    grays: list[NDArray]
    oklab_means: NDArray  # (n, 3) float64

    def __len__(self) -> int:
        return len(self.tiles)


# Augmentation functions used by both ``augment_tiles`` and ``build_cache``.
# IMPORTANT: changing this list invalidates existing caches.
AUGMENT_FNS: list[tuple[str, Callable[[NDArray], NDArray]]] = [
    ("orig", lambda t: t),
    ("flip_h", lambda t: cv2.flip(t, 1)),
    (
        "bright_up",
        lambda t: np.clip(t.astype(np.float32) * 1.10, 0, 255).astype(np.uint8),
    ),
    (
        "bright_down",
        lambda t: np.clip(t.astype(np.float32) * 0.90, 0, 255).astype(np.uint8),
    ),
]


def _list_tile_files(tile_dir: Path) -> list[str]:
    return sorted(
        f
        for f in os.listdir(tile_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    )


def _stats_for(lab: NDArray) -> tuple[float, float, float, float, float, float]:
    return (
        float(lab[:, :, 0].mean()),
        float(lab[:, :, 0].std() + 1e-6),
        float(lab[:, :, 1].mean()),
        float(lab[:, :, 1].std() + 1e-6),
        float(lab[:, :, 2].mean()),
        float(lab[:, :, 2].std() + 1e-6),
    )


def load_tiles(tile_dir: str | os.PathLike[str], tile_size: int) -> TileSet:
    """Load every tile from a directory and compute features at ``tile_size``.

    Parameters
    ----------
    tile_dir : path
        Directory containing tile images. Subdirectories are not scanned.
    tile_size : int
        Tiles will be resized to ``(tile_size, tile_size)``.

    Returns
    -------
    TileSet
        A dataclass holding the tiles, features, statistics, and Oklab means.
    """
    tile_dir = Path(tile_dir)
    if not tile_dir.is_dir():
        raise FileNotFoundError(f"Tile directory not found: {tile_dir}")

    files = _list_tile_files(tile_dir)
    if not files:
        raise FileNotFoundError(
            f"No images with extensions {SUPPORTED_EXTENSIONS} in {tile_dir}"
        )

    tiles: list[NDArray] = []
    features: list[list[float]] = []
    lab_stats: list[tuple[float, float, float, float, float, float]] = []
    grays: list[NDArray] = []
    oklab_means: list[NDArray] = []

    for f in files:
        img = cv2.imread(str(tile_dir / f))
        if img is None:
            continue
        tile = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        tiles.append(tile)
        lab = cv2.cvtColor(tile, cv2.COLOR_BGR2LAB).astype(np.float32)
        features.append(extract_features(lab, tile_size))
        lab_stats.append(_stats_for(lab))
        grays.append(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY))
        oklab_means.append(bgr_to_oklab(tile).mean(axis=(0, 1)))

    if not tiles:
        raise RuntimeError(f"All images in {tile_dir} failed to decode")

    return TileSet(
        tiles=tiles,
        features=np.array(features, dtype=np.float32),
        lab_stats=lab_stats,
        grays=grays,
        oklab_means=np.array(oklab_means, dtype=np.float64),
    )


def augment_tiles(tileset: TileSet, tile_size: int) -> TileSet:
    """Apply geometric and photometric augmentations to a tile set.

    The augmentations (horizontal flip, brightness up/down) effectively grow
    the tile collection 4x, which improves matching quality especially when
    the tile pool is small.
    """
    aug_tiles: list[NDArray] = list(tileset.tiles)
    aug_feats: list[NDArray] = [tileset.features]
    aug_stats: list[tuple[float, float, float, float, float, float]] = list(
        tileset.lab_stats
    )
    aug_grays: list[NDArray] = list(tileset.grays)
    aug_oklab: list[NDArray] = list(tileset.oklab_means)

    for name, fn in AUGMENT_FNS:
        if name == "orig":
            continue
        new_tiles: list[NDArray] = []
        new_feats: list[list[float]] = []
        new_stats: list[tuple[float, float, float, float, float, float]] = []
        new_grays: list[NDArray] = []
        new_oklab: list[NDArray] = []
        for tile in tileset.tiles:
            t = fn(tile)
            new_tiles.append(t)
            lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32)
            new_feats.append(extract_features(lab, tile_size))
            new_stats.append(_stats_for(lab))
            new_grays.append(cv2.cvtColor(t, cv2.COLOR_BGR2GRAY))
            new_oklab.append(bgr_to_oklab(t).mean(axis=(0, 1)))
        aug_tiles.extend(new_tiles)
        aug_feats.append(np.array(new_feats, dtype=np.float32))
        aug_stats.extend(new_stats)
        aug_grays.extend(new_grays)
        aug_oklab.extend(new_oklab)

    return TileSet(
        tiles=aug_tiles,
        features=np.vstack(aug_feats),
        lab_stats=aug_stats,
        grays=aug_grays,
        oklab_means=np.array(aug_oklab, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# On-disk feature cache
# ---------------------------------------------------------------------------
def _cache_paths(cache_dir: Path, tile_size: int) -> tuple[Path, Path]:
    return cache_dir / f"features_{tile_size}.npz", cache_dir / "files.json"


def build_cache(
    tile_dir: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    tile_sizes: list[int],
    thumb_size: int = 120,
    progress: bool = True,
) -> None:
    """Build an on-disk feature cache for one or more tile sizes.

    The cache contains:
        * ``thumbs/`` - small JPEG thumbnails of every tile (for fast reload).
        * ``files.json`` - sorted list of tile filenames.
        * ``features_<size>.npz`` - precomputed features per requested size.
    """
    tile_dir = Path(tile_dir)
    cache_dir = Path(cache_dir)
    thumb_dir = cache_dir / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    files = _list_tile_files(tile_dir)
    if not files:
        raise FileNotFoundError(f"No tile images in {tile_dir}")

    # Step 1: build thumbnails.
    thumb_names: list[str] = []
    for i, f in enumerate(files):
        stem = Path(f).stem
        dst = thumb_dir / f"{stem}.jpg"
        thumb_names.append(dst.name)
        if dst.exists():
            continue
        img = cv2.imread(str(tile_dir / f))
        if img is None:
            continue
        thumb = cv2.resize(img, (thumb_size, thumb_size), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(dst), thumb, [cv2.IMWRITE_JPEG_QUALITY, 97])
        if progress and (i + 1) % 500 == 0:
            print(f"  thumbnails: {i + 1}/{len(files)}")

    (cache_dir / "files.json").write_text(json.dumps(thumb_names))

    # Step 2: compute features per requested tile size.
    for tile_size in tile_sizes:
        cache_npz, _ = _cache_paths(cache_dir, tile_size)
        if cache_npz.exists():
            if progress:
                print(f"  features_{tile_size}.npz already exists, skipping")
            continue

        thumb_files = sorted(p.name for p in thumb_dir.glob("*.jpg"))
        orig_tiles: list[NDArray] = []
        for tf in thumb_files:
            img = cv2.imread(str(thumb_dir / tf))
            if img is None:
                continue
            orig_tiles.append(
                cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
            )

        all_features: list[list[float]] = []
        all_stats: list[list[float]] = []
        all_oklab: list[NDArray] = []

        for _, fn in AUGMENT_FNS:
            for tile in orig_tiles:
                t = fn(tile)
                lab = cv2.cvtColor(t, cv2.COLOR_BGR2LAB).astype(np.float32)
                all_features.append(extract_features(lab, tile_size))
                all_stats.append(list(_stats_for(lab)))
                all_oklab.append(bgr_to_oklab(t).mean(axis=(0, 1)))

        np.savez_compressed(
            cache_npz,
            features=np.array(all_features, dtype=np.float32),
            oklab_means=np.array(all_oklab, dtype=np.float32),
            lab_stats=np.array(all_stats, dtype=np.float32),
            n_orig=np.array([len(orig_tiles)]),
        )
        if progress:
            mb = cache_npz.stat().st_size / 1024 / 1024
            print(f"  wrote {cache_npz} ({mb:.1f} MB)")


def load_tiles_cached(
    cache_dir: str | os.PathLike[str], tile_size: int
) -> TileSet:
    """Load a tile set from an on-disk cache built by :func:`build_cache`.

    Raises
    ------
    FileNotFoundError
        If the cache for ``tile_size`` does not exist. Build it first with
        :func:`build_cache`.
    """
    cache_dir = Path(cache_dir)
    cache_npz, _ = _cache_paths(cache_dir, tile_size)
    if not cache_npz.exists():
        raise FileNotFoundError(
            f"Feature cache missing: {cache_npz}. "
            f"Run mosaicraft.tiles.build_cache() first."
        )

    data = np.load(cache_npz)
    features: NDArray = data["features"]
    oklab_means: NDArray = data["oklab_means"].astype(np.float64)
    stats_arr: NDArray = data["lab_stats"]
    n_orig = int(data["n_orig"][0])

    if features.shape[1] != FEATURE_DIM:
        raise ValueError(
            f"Cache feature dim {features.shape[1]} != expected {FEATURE_DIM}. "
            "Rebuild the cache."
        )

    thumb_dir = cache_dir / "thumbs"
    thumb_files = sorted(p.name for p in thumb_dir.glob("*.jpg"))[:n_orig]
    orig_tiles: list[NDArray] = []
    for tf in thumb_files:
        img = cv2.imread(str(thumb_dir / tf))
        if img is None:
            continue
        orig_tiles.append(
            cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        )

    aug_tiles: list[NDArray] = []
    for _, fn in AUGMENT_FNS:
        for tile in orig_tiles:
            aug_tiles.append(fn(tile))

    grays = [cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in aug_tiles]
    lab_stats = [tuple(stats_arr[i]) for i in range(len(stats_arr))]

    return TileSet(
        tiles=aug_tiles,
        features=features,
        lab_stats=lab_stats,  # type: ignore[arg-type]
        grays=grays,
        oklab_means=oklab_means,
    )
