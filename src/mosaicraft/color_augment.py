"""Tile-pool expansion via Oklch hue rotation.

A common failure mode for photomosaics is a small tile pool: even a careful
1:1 assignment can only be as diverse as the pool itself. Adding more source
images is the obvious fix, but it is often impractical (licensing, curation
time, or a subject matter that simply has a limited number of photographs).

This module grows an existing tile pool by applying perceptually-uniform hue
rotations in Oklch. Rotating hue in Oklch:

* keeps lightness (Oklab L) exact, so every tile retains its edge structure
  and texture,
* is perceptually smoother than HSV or CIELCH rotation
  (Björn Ottosson, 2020),
* introduces no boundary artifacts because L is untouched.

The net effect is to turn an ``N``-tile pool into an ``N * (1 + k)``-tile
pool, where ``k`` is the number of rotation angles requested. Hungarian
assignment then has a larger candidate set to choose from, which
substantially raises the achievable cell diversity without the usual pixel
fidelity vs diversity tradeoff.

Example
-------
::

    from mosaicraft.tiles import load_tiles
    from mosaicraft.color_augment import expand_color_variants

    base = load_tiles("tiles/", tile_size=56)  # 1,024 tiles
    pool = expand_color_variants(base, n_variants=4, tile_size=56)
    # pool now contains 1,024 * 5 = 5,120 tiles:
    #   - the originals
    #   - 4 Oklch-hue-rotated variants per tile (72°, 144°, 216°, 288°)

Design notes
------------
* The default rotation schedule (``n_variants=4``) spreads the rotations
  evenly around the hue circle at 72° increments, which covers warm, cool,
  and complementary replacements without favoring any axis.
* ``chroma_scale`` is applied *after* rotation so that low-saturation tiles
  do not get artificially boosted into oversaturated clones of each other.
* Highlight and shadow protection are enabled by default.
* The augmentation produces fresh ``features``, ``lab_stats``, ``grays`` and
  ``oklab_means`` arrays for each variant so the returned :class:`TileSet`
  is a drop-in replacement for the input.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .color import bgr_to_oklab, oklab_to_bgr
from .features import extract_features
from .tiles import TileSet

NDArray = np.ndarray

__all__ = [
    "DEFAULT_HUE_SCHEDULE",
    "expand_color_variants",
    "rotate_hue_oklch",
]


# Default hue-rotation schedule, in degrees. Picked so each angle lands on a
# different quadrant of the Oklab a/b plane (warm, green, cool, magenta) and
# the gaps are perceptually balanced.
DEFAULT_HUE_SCHEDULE: tuple[float, ...] = (72.0, 144.0, 216.0, 288.0)


def rotate_hue_oklch(
    tile_bgr: NDArray,
    hue_shift_deg: float,
    *,
    chroma_scale: float = 1.0,
    protect_highlights: bool = True,
    protect_shadows: bool = True,
) -> NDArray:
    """Rotate a tile's hue in Oklch, preserving Oklab L exactly.

    Parameters
    ----------
    tile_bgr : np.ndarray
        BGR uint8 image of shape ``(H, W, 3)``.
    hue_shift_deg : float
        Signed rotation in degrees. Positive shifts go counter-clockwise on
        the Oklab a/b plane; 360 is a no-op.
    chroma_scale : float
        Multiplier applied to chroma after rotation. ``1.0`` keeps saturation
        unchanged. Values < 1 desaturate; > 1 boost.
    protect_highlights : bool
        If True, fade chroma to zero as L approaches 1.0 so speculars stay
        clean instead of picking up a color cast.
    protect_shadows : bool
        If True, fade chroma to zero as L approaches 0.0 so shadow depth is
        not accidentally colorized.

    Returns
    -------
    np.ndarray
        BGR uint8 image, same shape as input.
    """
    oklab = bgr_to_oklab(tile_bgr)
    lightness = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]

    chroma = np.sqrt(a * a + b * b)
    hue = np.arctan2(b, a) + np.radians(float(hue_shift_deg))

    new_chroma = chroma * float(chroma_scale)

    if protect_highlights:
        highlight = np.clip((lightness - 0.85) / 0.15, 0.0, 1.0)
        new_chroma = new_chroma * (1.0 - highlight * 0.5)
    if protect_shadows:
        shadow = np.clip((0.25 - lightness) / 0.25, 0.0, 1.0)
        new_chroma = new_chroma * (1.0 - shadow * 0.3)

    new_chroma = np.clip(new_chroma, 0.0, 0.4)

    new_a = new_chroma * np.cos(hue)
    new_b = new_chroma * np.sin(hue)

    rotated = np.stack([lightness, new_a, new_b], axis=-1)
    return oklab_to_bgr(rotated)


def _stats_for_lab(lab: NDArray) -> tuple[float, float, float, float, float, float]:
    """Mirror :func:`mosaicraft.tiles._stats_for` so features stay consistent."""
    return (
        float(lab[:, :, 0].mean()),
        float(lab[:, :, 0].std() + 1e-6),
        float(lab[:, :, 1].mean()),
        float(lab[:, :, 1].std() + 1e-6),
        float(lab[:, :, 2].mean()),
        float(lab[:, :, 2].std() + 1e-6),
    )


def expand_color_variants(
    tileset: TileSet,
    *,
    n_variants: int | None = None,
    hue_schedule: Sequence[float] | None = None,
    tile_size: int,
    chroma_scale: float = 1.0,
    protect_highlights: bool = True,
    protect_shadows: bool = True,
) -> TileSet:
    """Return a new :class:`TileSet` augmented with Oklch-rotated variants.

    Parameters
    ----------
    tileset : TileSet
        Source tile set (typically the output of
        :func:`mosaicraft.tiles.load_tiles` or ``augment_tiles``).
    n_variants : int, optional
        Number of hue-rotated copies to add *per* source tile. If
        ``hue_schedule`` is also given, ``n_variants`` is ignored. The default
        uses the evenly-spaced schedule in :data:`DEFAULT_HUE_SCHEDULE` (4
        variants → 5x pool including originals).
    hue_schedule : sequence of float, optional
        Explicit list of rotation angles in degrees. Takes precedence over
        ``n_variants``.
    tile_size : int
        Working tile size. Must match the tile size used when ``tileset`` was
        built, so feature vectors line up.
    chroma_scale : float
        Chroma multiplier applied to the rotated variants only. Use values
        slightly below 1.0 (e.g. 0.9) if you want rotated variants to read as
        "adjacent" rather than "replacement".
    protect_highlights, protect_shadows : bool
        Passed through to :func:`rotate_hue_oklch`.

    Returns
    -------
    TileSet
        A new ``TileSet`` containing the original tiles first, then all
        hue-rotated variants in stable order. The ``features``, ``lab_stats``,
        ``grays`` and ``oklab_means`` arrays are recomputed from the rotated
        pixel data so downstream placement stays numerically consistent.
    """
    if hue_schedule is None:
        if n_variants is None or n_variants <= 0:
            return tileset
        n = int(n_variants)
        if n == len(DEFAULT_HUE_SCHEDULE):
            schedule: tuple[float, ...] = DEFAULT_HUE_SCHEDULE
        else:
            # Evenly spaced on the unit circle, skipping 0 so the variants are
            # always distinct from the originals.
            step = 360.0 / (n + 1)
            schedule = tuple(step * (i + 1) for i in range(n))
    else:
        schedule = tuple(float(a) for a in hue_schedule)
        if not schedule:
            return tileset

    new_tiles: list[NDArray] = list(tileset.tiles)
    feat_chunks: list[NDArray] = [tileset.features]
    new_stats = list(tileset.lab_stats)
    new_grays: list[NDArray] = list(tileset.grays)
    oklab_chunks: list[NDArray] = [tileset.oklab_means]

    for angle in schedule:
        rotated_tiles: list[NDArray] = []
        rotated_feats: list[list[float]] = []
        rotated_stats: list[tuple[float, float, float, float, float, float]] = []
        rotated_grays: list[NDArray] = []
        rotated_oklab: list[NDArray] = []

        for tile in tileset.tiles:
            rt = rotate_hue_oklch(
                tile,
                hue_shift_deg=angle,
                chroma_scale=chroma_scale,
                protect_highlights=protect_highlights,
                protect_shadows=protect_shadows,
            )
            rotated_tiles.append(rt)
            lab = cv2.cvtColor(rt, cv2.COLOR_BGR2LAB).astype(np.float32)
            rotated_feats.append(extract_features(lab, tile_size))
            rotated_stats.append(_stats_for_lab(lab))
            rotated_grays.append(cv2.cvtColor(rt, cv2.COLOR_BGR2GRAY))
            rotated_oklab.append(bgr_to_oklab(rt).mean(axis=(0, 1)))

        new_tiles.extend(rotated_tiles)
        feat_chunks.append(np.array(rotated_feats, dtype=np.float32))
        new_stats.extend(rotated_stats)
        new_grays.extend(rotated_grays)
        oklab_chunks.append(np.array(rotated_oklab, dtype=np.float64))

    return TileSet(
        tiles=new_tiles,
        features=np.vstack(feat_chunks),
        lab_stats=new_stats,
        grays=new_grays,
        oklab_means=np.vstack(oklab_chunks),
    )
