"""Perceptual color science: Oklab/Oklch conversions and color transfer.

This module implements:
    * Oklab / Oklch color space conversions (Björn Ottosson, 2020).
    * Reinhard statistical color transfer.
    * Monge-Kantorovich Linear (MKL) optimal transport color transfer.
    * Histogram matching color transfer.
    * Hybrid color transfer methods.

Oklab is a perceptual color space (Björn Ottosson, 2020) designed to be
noticeably more uniform than CIELAB on saturated colours, which is the
regime a photomosaic spends most of its compute budget in.

References:
    Björn Ottosson, "A perceptual color space for image processing":
        https://bottosson.github.io/posts/oklab/
    Pitié, F. et al., "N-dimensional probability density function transfer
        and its application to color transfer", ICCV 2005.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from skimage.exposure import match_histograms

NDArray = np.ndarray
EPS = 1e-10

__all__ = [
    "apply_color_transfer",
    "bgr_to_oklab",
    "histogram_transfer",
    "hybrid_transfer",
    "mkl_hybrid_transfer",
    "mkl_transfer",
    "oklab_to_bgr",
    "reinhard_transfer",
    "vibrance_oklch",
]


# ---------------------------------------------------------------------------
# sRGB <-> Linear RGB
# ---------------------------------------------------------------------------
def _srgb_to_linear(c: NDArray) -> NDArray:
    """Convert sRGB (0-1) to linear RGB using the standard gamma curve."""
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c: NDArray) -> NDArray:
    """Convert linear RGB to sRGB (0-1)."""
    return np.where(
        c <= 0.0031308,
        c * 12.92,
        1.055 * np.power(np.maximum(c, 0), 1.0 / 2.4) - 0.055,
    )


# ---------------------------------------------------------------------------
# Oklab conversions (Björn Ottosson)
# ---------------------------------------------------------------------------
def bgr_to_oklab(bgr_uint8: NDArray) -> NDArray:
    """Convert OpenCV BGR uint8 image to Oklab float64.

    Parameters
    ----------
    bgr_uint8 : np.ndarray
        BGR image with dtype uint8 and shape (H, W, 3).

    Returns
    -------
    np.ndarray
        Oklab image with dtype float64 and shape (H, W, 3).
        L is in [0, 1], a/b are roughly in [-0.4, 0.4].
    """
    rgb = bgr_uint8[..., ::-1].astype(np.float64) / 255.0
    linear = _srgb_to_linear(rgb)
    long = (
        0.4122214708 * linear[..., 0]
        + 0.5363325363 * linear[..., 1]
        + 0.0514459929 * linear[..., 2]
    )
    medium = (
        0.2119034982 * linear[..., 0]
        + 0.6806995451 * linear[..., 1]
        + 0.1073969566 * linear[..., 2]
    )
    short = (
        0.0883024619 * linear[..., 0]
        + 0.2817188376 * linear[..., 1]
        + 0.6299787005 * linear[..., 2]
    )
    long_ = np.cbrt(np.maximum(long, 0))
    medium_ = np.cbrt(np.maximum(medium, 0))
    short_ = np.cbrt(np.maximum(short, 0))
    lightness = 0.2104542553 * long_ + 0.7936177850 * medium_ - 0.0040720468 * short_
    a = 1.9779984951 * long_ - 2.4285922050 * medium_ + 0.4505937099 * short_
    b = 0.0259040371 * long_ + 0.7827717662 * medium_ - 0.8086757660 * short_
    return np.stack([lightness, a, b], axis=-1)


def oklab_to_bgr(oklab: NDArray) -> NDArray:
    """Convert Oklab float64 to OpenCV BGR uint8.

    Parameters
    ----------
    oklab : np.ndarray
        Oklab image with shape (H, W, 3), float64.

    Returns
    -------
    np.ndarray
        BGR uint8 image clipped to [0, 255].
    """
    lightness = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    long_ = lightness + 0.3963377774 * a + 0.2158037573 * b
    medium_ = lightness - 0.1055613458 * a - 0.0638541728 * b
    short_ = lightness - 0.0894841775 * a - 1.2914855480 * b
    long = long_ * long_ * long_
    medium = medium_ * medium_ * medium_
    short = short_ * short_ * short_
    r = +4.0767416621 * long - 3.3077115913 * medium + 0.2309699292 * short
    g = -1.2684380046 * long + 2.6097574011 * medium - 0.3413193965 * short
    b_ch = -0.0041960863 * long - 0.7034186147 * medium + 1.7076147010 * short
    rgb_linear = np.stack([r, g, b_ch], axis=-1)
    rgb = _linear_to_srgb(np.clip(rgb_linear, 0, 1))
    return np.clip(rgb[..., ::-1] * 255, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Color transfer
# ---------------------------------------------------------------------------
def reinhard_transfer(
    tile_bgr: NDArray,
    target_lab: NDArray,
    strength_l: float = 0.6,
    strength_ab: float = 0.42,
) -> NDArray:
    """Adaptive Reinhard color transfer in CIELAB.

    Reinhard et al. (2001), with adaptive strength scaling based on the
    color distance between source and target. The smaller the distance,
    the stronger the transfer (avoids over-correcting wildly different
    palettes).

    Parameters
    ----------
    tile_bgr : np.ndarray
        Source tile in BGR uint8.
    target_lab : np.ndarray
        Target region statistics, in CIELAB float (already converted).
    strength_l : float
        Lightness channel transfer strength (0-1).
    strength_ab : float
        Chromaticity (a, b) channel transfer strength (0-1).
    """
    tile_lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    for ch in range(3):
        src_mean = tile_lab[:, :, ch].mean()
        src_std = tile_lab[:, :, ch].std() + EPS
        tgt_mean = target_lab[:, :, ch].mean()
        tgt_std = target_lab[:, :, ch].std() + EPS
        color_diff = abs(src_mean - tgt_mean) / 128.0
        adaptive_factor = max(0.3, 1.0 - color_diff * 0.3)
        strength = (strength_l if ch == 0 else strength_ab) * adaptive_factor
        std_ratio = float(np.clip(tgt_std / src_std, 0.5, 2.0))
        transferred = (tile_lab[:, :, ch] - src_mean) * std_ratio + tgt_mean
        tile_lab[:, :, ch] = tile_lab[:, :, ch] * (1 - strength) + transferred * strength
    tile_lab = np.clip(tile_lab, 0, 255)
    return cv2.cvtColor(tile_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _mkl_transform(a_cov: NDArray, b_cov: NDArray) -> NDArray:
    """Compute the Monge-Kantorovich linear transport matrix.

    See Pitié et al., "The linear Monge-Kantorovitch linear colour mapping
    for example-based colour transfer", IET-CVMP 2007.
    """
    da_sq, ua = np.linalg.eigh(a_cov)
    da_sq = np.maximum(da_sq, EPS)
    da = np.sqrt(da_sq)
    da_inv = np.diag(1.0 / da)
    c = np.diag(da) @ ua.T @ b_cov @ ua @ np.diag(da)
    dc_sq, uc = np.linalg.eigh(c)
    dc_sq = np.maximum(dc_sq, EPS)
    dc = np.sqrt(dc_sq)
    return ua @ da_inv @ uc @ np.diag(dc) @ uc.T @ da_inv @ ua.T


def mkl_transfer(
    tile_bgr: NDArray,
    target_bgr: NDArray,
    strength: float = 0.55,
    adaptive: bool = True,
    adaptive_scale: float = 30.0,
) -> NDArray:
    """MKL optimal transport color transfer in CIELAB.

    Preserves the *shape* of the source color distribution while shifting
    its first and second moments to match the target. This produces more
    vivid and natural results than pure Reinhard for chromatically distant
    images.

    When ``adaptive`` is True (default), the blend strength is smoothly
    raised toward 1.0 for **flat, chromatically-distant** target cells —
    the regime where a halfway transfer leaves mid-gray tiles in regions
    the eye expects to be a solid colour (e.g. a white background or a
    saturated-hue illustration patch). Textured cells (faces, fabric,
    foliage) stay near ``strength`` so tile identity is preserved.

    The boost factor is ``mean_distance * (1 - normalized_target_std)``,
    passed through ``exp(-d/adaptive_scale)``. Set ``adaptive=False`` for
    the pre-v0.4 behaviour.
    """
    tile_lab = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    target_lab = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    src = tile_lab.reshape(-1, 3)
    tgt = target_lab.reshape(-1, 3)
    a_cov = np.cov(src, rowvar=False) + np.eye(3) * 1e-6
    b_cov = np.cov(tgt, rowvar=False) + np.eye(3) * 1e-6
    t = _mkl_transform(a_cov, b_cov)
    src_mean = src.mean(axis=0)
    tgt_mean = tgt.mean(axis=0)
    if adaptive and strength < 1.0:
        d = float(np.linalg.norm(tgt_mean - src_mean))
        # Target-cell flatness: 0 for textured cells, 1 for perfectly flat.
        # Normalize L/a/b std by a rough 20-unit reference.
        tgt_std = float(np.linalg.norm(tgt.std(axis=0)))
        flatness = float(np.clip(1.0 - tgt_std / 20.0, 0.0, 1.0))
        boost = 1.0 - float(np.exp(-d * flatness / max(adaptive_scale, EPS)))
        strength = strength + (1.0 - strength) * boost
    transferred = (src - src_mean) @ np.real(t) + tgt_mean
    result = src * (1 - strength) + transferred * strength
    result_lab = np.clip(result, 0, 255).astype(np.uint8).reshape(tile_lab.shape)
    return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)


def histogram_transfer(
    tile_bgr: NDArray, target_bgr: NDArray, blend_alpha: float = 0.6
) -> NDArray:
    """Histogram matching color transfer (per-channel CDF mapping)."""
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    target_rgb = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2RGB)
    matched = match_histograms(tile_rgb, target_rgb, channel_axis=-1)
    matched_bgr = cv2.cvtColor(matched.astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(tile_bgr, 1 - blend_alpha, matched_bgr, blend_alpha, 0)


def hybrid_transfer(
    tile_bgr: NDArray,
    target_bgr: NDArray,
    target_lab: NDArray,
    strength_l: float,
    strength_ab: float,
) -> NDArray:
    """Reinhard followed by histogram matching."""
    reinhard = reinhard_transfer(tile_bgr, target_lab, strength_l, strength_ab)
    return histogram_transfer(reinhard, target_bgr, blend_alpha=0.35)


def mkl_hybrid_transfer(
    tile_bgr: NDArray, target_bgr: NDArray, strength: float
) -> NDArray:
    """MKL optimal transport followed by histogram matching."""
    mkl_result = mkl_transfer(tile_bgr, target_bgr, strength)
    return histogram_transfer(mkl_result, target_bgr, blend_alpha=0.30)


def apply_color_transfer(
    tile_bgr: NDArray,
    target_bgr: NDArray,
    target_lab: NDArray,
    method: str,
    profile: dict[str, Any],
) -> NDArray:
    """Dispatch color transfer by method name.

    Supported methods:
        * ``"none"`` - no transfer
        * ``"adaptive_reinhard"`` - Reinhard with adaptive strength
        * ``"histogram"`` - histogram matching
        * ``"hybrid"`` - Reinhard + histogram
        * ``"mkl"`` - MKL optimal transport
        * ``"mkl_hybrid"`` - MKL + histogram
    """
    if method == "none":
        return tile_bgr
    if method == "adaptive_reinhard":
        return reinhard_transfer(
            tile_bgr, target_lab, profile["reinhard_l"], profile["reinhard_ab"]
        )
    if method == "histogram":
        return histogram_transfer(tile_bgr, target_bgr, blend_alpha=0.55)
    if method == "hybrid":
        return hybrid_transfer(
            tile_bgr,
            target_bgr,
            target_lab,
            profile["reinhard_l"],
            profile["reinhard_ab"],
        )
    if method == "mkl":
        return mkl_transfer(tile_bgr, target_bgr, profile["reinhard_l"])
    if method == "mkl_hybrid":
        return mkl_hybrid_transfer(tile_bgr, target_bgr, profile["reinhard_l"])
    raise ValueError(f"Unknown color transfer method: {method!r}")


# ---------------------------------------------------------------------------
# Vibrance in Oklch
# ---------------------------------------------------------------------------
def vibrance_oklch(
    img_bgr: NDArray,
    amount: float = 0.4,
    skin_mask: NDArray | None = None,
    skin_protection: float = 0.0,
) -> NDArray:
    """Non-linear vibrance in Oklch (perceptual chroma).

    Boosts low-saturation pixels more strongly than already-saturated ones,
    which avoids the over-saturated, neon look of naive HSV vibrance. A soft
    knee gamut mapping prevents hard clipping.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR uint8 image.
    amount : float
        Vibrance amount (0-1, typical 0.3-0.6).
    skin_mask : np.ndarray, optional
        Soft mask (0-1) marking skin regions to protect.
    skin_protection : float
        How strongly to protect masked skin (0 = no protection, 1 = full).
    """
    oklab = bgr_to_oklab(img_bgr)
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    chroma = np.sqrt(a**2 + b**2)
    hue = np.arctan2(b, a)

    chroma_safe = chroma[chroma > 0.001]
    chroma_max = (
        np.percentile(chroma_safe, 99) + EPS if len(chroma_safe) > 0 else 0.3
    )
    normalized = np.clip(chroma / chroma_max, 0, 1)

    boost = 1.0 + amount * (1.0 - normalized) ** 2

    if skin_mask is not None and skin_protection > 0:
        boost = boost * (1.0 - skin_mask * skin_protection) + 1.0 * (
            skin_mask * skin_protection
        )

    chroma_new = chroma * boost

    knee = 0.85
    threshold = chroma_max * knee
    over = np.maximum(chroma_new - threshold, 0)
    headroom = chroma_max * 0.5
    chroma_new = np.where(
        chroma_new > threshold,
        threshold + over / (1.0 + over / headroom),
        chroma_new,
    )

    oklab_out = np.stack(
        [L, chroma_new * np.cos(hue), chroma_new * np.sin(hue)], axis=-1
    )
    return oklab_to_bgr(oklab_out)
