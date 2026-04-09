"""Per-cell saliency weights for cost-matrix biasing.

Important regions of the target image (faces, edges, saturated colors) get
higher weights so the cost matrix is forced to spend its best matches there.
"""

from __future__ import annotations

import cv2
import numpy as np

NDArray = np.ndarray

__all__ = ["compute_saliency_weights"]


def compute_saliency_weights(
    main_gray: NDArray,
    main_bgr: NDArray,
    grid_cols: int,
    grid_rows: int,
    tile_size: int,
) -> NDArray:
    """Compute per-cell saliency weights.

    The score for each cell is a linear combination of:
        * Edge density       (Canny edges within the cell)
        * Laplacian energy   (high-frequency content)
        * HSV saturation     (chromatic richness)
        * Center bias        (people usually compose subjects centrally)

    Parameters
    ----------
    main_gray : np.ndarray
        Grayscale target image.
    main_bgr : np.ndarray
        BGR target image at the working resolution.
    grid_cols, grid_rows : int
        Mosaic grid dimensions.
    tile_size : int
        Pixel size of each cell.

    Returns
    -------
    np.ndarray
        Saliency weights, shape ``(grid_rows, grid_cols)``, normalized so that
        the mean is 1.0.
    """
    edges = cv2.Canny(main_gray, 50, 150)
    blurred = cv2.GaussianBlur(main_gray, (5, 5), 1.5)
    log = np.abs(cv2.Laplacian(blurred, cv2.CV_64F))
    log_norm = log / (log.max() + 1e-6)
    hsv = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2HSV)
    sat_map = hsv[:, :, 1].astype(np.float64) / 255.0

    h, w = grid_rows * tile_size, grid_cols * tile_size
    edges_grid = edges[:h, :w].reshape(grid_rows, tile_size, grid_cols, tile_size)
    log_grid = log_norm[:h, :w].reshape(grid_rows, tile_size, grid_cols, tile_size)
    sat_grid = sat_map[:h, :w].reshape(grid_rows, tile_size, grid_cols, tile_size)

    edge_density = (edges_grid > 0).sum(axis=(1, 3)).astype(np.float64) / (
        tile_size * tile_size
    )
    log_density = log_grid.mean(axis=(1, 3))
    sat_density = sat_grid.mean(axis=(1, 3))

    gy_arr = np.arange(grid_rows)[:, None]
    gx_arr = np.arange(grid_cols)[None, :]
    cy, cx = grid_rows / 2.0, grid_cols / 2.0
    dist = np.sqrt((gy_arr - cy) ** 2 + (gx_arr - cx) ** 2)
    max_dist = float(np.sqrt(cy**2 + cx**2))
    center_bias = 1.0 - 0.3 * (dist / max_dist)

    weights = (
        0.2 + edge_density * 1.5 + log_density * 1.0 + sat_density * 0.5
    ) * center_bias
    weights /= weights.mean() + 1e-6
    return weights
