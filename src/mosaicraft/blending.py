"""Mosaic assembly with seamless cell blending.

Two strategies are supported:

* :func:`assemble_laplacian` - Laplacian pyramid blend at the cell borders.
  Highest quality, smoothes out grid lines without losing detail.
* :func:`assemble_feather`   - per-cell feather mask blend. Faster, also
  produces clean joins for most use cases.

Both functions also apply per-tile color transfer and a saliency-driven
blend with the underlying target image.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .color import apply_color_transfer

NDArray = np.ndarray

__all__ = ["assemble_feather", "assemble_laplacian"]


# ---------------------------------------------------------------------------
# Laplacian pyramid helpers
# ---------------------------------------------------------------------------
def _build_laplacian_pyramid(img: NDArray, levels: int) -> list[NDArray]:
    gaussian = [img.astype(np.float64)]
    for _ in range(levels):
        gaussian.append(cv2.pyrDown(gaussian[-1]))
    laplacian: list[NDArray] = []
    for i in range(levels):
        up = cv2.pyrUp(
            gaussian[i + 1], dstsize=(gaussian[i].shape[1], gaussian[i].shape[0])
        )
        laplacian.append(gaussian[i] - up)
    laplacian.append(gaussian[-1])
    return laplacian


def _reconstruct_from_laplacian(pyramid: list[NDArray]) -> NDArray:
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        up = cv2.pyrUp(img, dstsize=(pyramid[i].shape[1], pyramid[i].shape[0]))
        img = up + pyramid[i]
    return img


def _blend_alpha(saliency: float, blend_min: float, blend_max: float) -> float:
    sal_clamped = float(np.clip(saliency, 0.5, 2.0))
    return blend_min + (blend_max - blend_min) * (sal_clamped - 0.5) / 1.5


# ---------------------------------------------------------------------------
# Laplacian assembly
# ---------------------------------------------------------------------------
def assemble_laplacian(
    grid: NDArray,
    tiles: list[NDArray],
    target_resized: NDArray,
    grid_cols: int,
    grid_rows: int,
    tile_size: int,
    profile: dict[str, Any],
    saliency_weights: NDArray,
    levels: int = 4,
) -> NDArray:
    """Assemble a mosaic using Laplacian pyramid blending at cell borders."""
    blend_min = profile["base_blend_min"]
    blend_max = profile["base_blend_max"]
    method = profile["color_transfer"]

    target_h = grid_rows * tile_size
    target_w = grid_cols * tile_size
    target_lab_full = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB).astype(np.float64)

    mosaic = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            y0, y1 = gy * tile_size, (gy + 1) * tile_size
            x0, x1 = gx * tile_size, (gx + 1) * tile_size
            tile_bgr = tiles[grid[gy, gx]]
            target_bgr = target_resized[y0:y1, x0:x1]
            target_lab = target_lab_full[y0:y1, x0:x1]
            corrected = apply_color_transfer(
                tile_bgr, target_bgr, target_lab, method, profile
            )
            blend = _blend_alpha(saliency_weights[gy, gx], blend_min, blend_max)
            corrected = cv2.addWeighted(corrected, 1.0 - blend, target_bgr, blend, 0)
            mosaic[y0:y1, x0:x1] = corrected

    mosaic_f = mosaic.astype(np.float64)

    # Build a per-cell border mask (1 at borders, 0 at center).
    border_w = max(2, tile_size // 15)
    cell_center = np.ones((tile_size, tile_size), dtype=np.float64)
    for i in range(border_w):
        alpha = (i + 1) / (border_w + 1)
        cell_center[i, :] = alpha
        cell_center[-(i + 1), :] = np.minimum(cell_center[-(i + 1), :], alpha)
        cell_center[:, i] = np.minimum(cell_center[:, i], alpha)
        cell_center[:, -(i + 1)] = np.minimum(cell_center[:, -(i + 1)], alpha)

    center_mask = np.zeros((target_h, target_w), dtype=np.float64)
    for gy in range(grid_rows):
        for gx in range(grid_cols):
            y0, y1 = gy * tile_size, (gy + 1) * tile_size
            x0, x1 = gx * tile_size, (gx + 1) * tile_size
            center_mask[y0:y1, x0:x1] = cell_center
    border_mask = 1.0 - center_mask

    pyr = _build_laplacian_pyramid(mosaic_f, levels)
    border3 = np.stack([border_mask] * 3, axis=-1)
    mask_pyr = [border3]
    for _ in range(levels):
        mask_pyr.append(cv2.pyrDown(mask_pyr[-1]))
    for i in range(min(2, levels)):
        attenuation = 0.4 if i == 0 else 0.6
        pyr[i] = pyr[i] * (1.0 - mask_pyr[i] * (1.0 - attenuation))
    result = _reconstruct_from_laplacian(pyr)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feather assembly (faster)
# ---------------------------------------------------------------------------
def assemble_feather(
    grid: NDArray,
    tiles: list[NDArray],
    target_resized: NDArray,
    grid_cols: int,
    grid_rows: int,
    tile_size: int,
    profile: dict[str, Any],
    saliency_weights: NDArray,
) -> NDArray:
    """Assemble a mosaic with per-cell feather masks at the borders."""
    fw = profile["feather_width"]
    blend_min = profile["base_blend_min"]
    blend_max = profile["base_blend_max"]
    method = profile["color_transfer"]

    target_h = grid_rows * tile_size
    target_w = grid_cols * tile_size
    target_lab_full = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB).astype(np.float64)

    feather_mask: NDArray | None = None
    if fw > 0:
        layers = [np.ones((tile_size, tile_size), dtype=np.float32) for _ in range(4)]
        for i in range(fw):
            alpha = (i + 1) / (fw + 1)
            layers[0][i, :] = alpha
            layers[1][-(i + 1), :] = alpha
            layers[2][:, i] = alpha
            layers[3][:, -(i + 1)] = alpha
        feather_mask = np.minimum(
            np.minimum(layers[0], layers[1]), np.minimum(layers[2], layers[3])
        )

    if feather_mask is not None:
        color_sum = np.zeros((target_h, target_w, 3), dtype=np.float64)
        weight_sum = np.zeros((target_h, target_w, 1), dtype=np.float64)
        mask3 = np.stack([feather_mask] * 3, axis=-1).astype(np.float64)
        mask1 = feather_mask[:, :, np.newaxis].astype(np.float64)
    else:
        mosaic = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            y0, y1 = gy * tile_size, (gy + 1) * tile_size
            x0, x1 = gx * tile_size, (gx + 1) * tile_size
            tile_bgr = tiles[grid[gy, gx]]
            target_bgr = target_resized[y0:y1, x0:x1]
            target_lab = target_lab_full[y0:y1, x0:x1]
            corrected = apply_color_transfer(
                tile_bgr, target_bgr, target_lab, method, profile
            )
            blend = _blend_alpha(saliency_weights[gy, gx], blend_min, blend_max)
            corrected = cv2.addWeighted(corrected, 1.0 - blend, target_bgr, blend, 0)
            if feather_mask is not None:
                color_sum[y0:y1, x0:x1] += corrected.astype(np.float64) * mask3
                weight_sum[y0:y1, x0:x1] += mask1
            else:
                mosaic[y0:y1, x0:x1] = corrected

    if feather_mask is not None:
        return (color_sum / (weight_sum + 1e-10)).astype(np.uint8)
    return mosaic
