"""Tile feature extraction for similarity matching.

Each tile is described by a 191-dimensional feature vector composed of:

    * Quadrant means in CIELAB        : 5x5 quadrants x 3 channels = 75 dim
    * Per-channel histograms           : 12 bins x 3 channels       = 36 dim
    * Gradient orientation histograms  : 8 bins x 4 blocks + 8 dim  = 40 dim
    * Local Binary Pattern histograms  : 10 bins x 4 blocks         = 40 dim

Total: 75 + 36 + 40 + 40 = 191.

The vector captures both color statistics and texture, which gives a much
better matching signal than mean color alone.
"""

from __future__ import annotations

import cv2
import numpy as np

NDArray = np.ndarray

QUADRANTS = 5
HIST_BINS = 12
HIST_SCALE = 50.0
GRAD_BINS = 8
GRAD_SCALE = 30.0
LBP_BINS = 10
LBP_SCALE = 20.0
FEATURE_DIM = (QUADRANTS * QUADRANTS * 3) + (HIST_BINS * 3) + (GRAD_BINS * 5) + (LBP_BINS * 4)
assert FEATURE_DIM == 191, f"Expected 191-dim features, got {FEATURE_DIM}"

__all__ = ["FEATURE_DIM", "compute_lbp", "extract_features"]


def compute_lbp(gray_img: NDArray) -> NDArray:
    """Compute the 8-neighbor Local Binary Pattern.

    Parameters
    ----------
    gray_img : np.ndarray
        Grayscale image, uint8.

    Returns
    -------
    np.ndarray
        LBP image, shape (H-2, W-2), uint8.
    """
    h, w = gray_img.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    center = gray_img[1:-1, 1:-1]
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for i, (dy, dx) in enumerate(neighbors):
        neighbor = gray_img[1 + dy : h - 1 + dy, 1 + dx : w - 1 + dx]
        lbp |= (neighbor >= center).astype(np.uint8) << i
    return lbp


def extract_features(lab_img: NDArray, tile_size: int) -> list[float]:
    """Extract a 191-dimensional feature vector from a CIELAB tile.

    Parameters
    ----------
    lab_img : np.ndarray
        Tile in CIELAB float32, shape (tile_size, tile_size, 3).
    tile_size : int
        Side length of the (square) tile in pixels.

    Returns
    -------
    list[float]
        Concatenated feature vector of length :data:`FEATURE_DIM`.
    """
    # 1) Quadrant means (5x5 grid of LAB means).
    qh = qw = tile_size // QUADRANTS
    feat_quad: list[float] = []
    for gy in range(QUADRANTS):
        for gx in range(QUADRANTS):
            region = lab_img[gy * qh : (gy + 1) * qh, gx * qw : (gx + 1) * qw]
            feat_quad.extend(region.mean(axis=(0, 1)).tolist())

    # 2) Per-channel histograms.
    feat_hist: list[float] = []
    for ch in range(3):
        hist, _ = np.histogram(lab_img[:, :, ch], bins=HIST_BINS, range=(0, 255))
        hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-6)
        feat_hist.extend((hist_norm * HIST_SCALE).tolist())

    # 3) Gradient orientation histograms (per-block + global).
    l_ch = lab_img[:, :, 0]
    gx_s = cv2.Sobel(l_ch, cv2.CV_32F, 1, 0, ksize=3)
    gy_s = cv2.Sobel(l_ch, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx_s**2 + gy_s**2)
    angle = np.arctan2(gy_s, gx_s)
    bh = bw = tile_size // 2
    feat_grad: list[float] = []
    for by, bx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        block_mag = mag[by * bh : (by + 1) * bh, bx * bw : (bx + 1) * bw].flatten()
        block_angle = angle[by * bh : (by + 1) * bh, bx * bw : (bx + 1) * bw].flatten()
        hist, _ = np.histogram(
            block_angle, bins=GRAD_BINS, range=(-np.pi, np.pi), weights=block_mag
        )
        hist_norm = hist / (hist.sum() + 1e-6)
        feat_grad.extend((hist_norm * GRAD_SCALE).tolist())
    glob_hist, _ = np.histogram(
        angle.flatten(), bins=GRAD_BINS, range=(-np.pi, np.pi), weights=mag.flatten()
    )
    glob_norm = glob_hist / (glob_hist.sum() + 1e-6)
    feat_grad.extend((glob_norm * GRAD_SCALE).tolist())

    # 4) LBP histograms (per-block).
    l_u8 = np.clip(l_ch, 0, 255).astype(np.uint8)
    feat_lbp: list[float] = []
    if l_u8.shape[0] >= 4 and l_u8.shape[1] >= 4:
        lbp = compute_lbp(l_u8)
        lbp_h, lbp_w = lbp.shape
        bh2, bw2 = lbp_h // 2, lbp_w // 2
        for by, bx in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            block = lbp[by * bh2 : (by + 1) * bh2, bx * bw2 : (bx + 1) * bw2].flatten()
            hist, _ = np.histogram(block, bins=LBP_BINS, range=(0, 256))
            hist_norm = hist.astype(np.float32) / (hist.sum() + 1e-6)
            feat_lbp.extend((hist_norm * LBP_SCALE).tolist())
    else:
        feat_lbp = [0.0] * (LBP_BINS * 4)

    return feat_quad + feat_hist + feat_grad + feat_lbp
