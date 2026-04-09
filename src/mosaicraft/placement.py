"""Tile-to-cell assignment algorithms.

This module provides three placement strategies:

1. ``hungarian``           - exact assignment via the Jonker-Volgenant
                             algorithm (scipy ``linear_sum_assignment``).
                             Best quality, requires that ``n_tiles >=
                             n_cells``. Memory cost grows as O(N*M).
2. ``faiss_diffusion``     - greedy nearest-neighbor with Floyd-Steinberg
                             error diffusion. Used as a fallback when the
                             cost matrix is too large for Hungarian.
3. ``neighbor_swap``       - local refinement applied after either of the
                             above to escape suboptimal pairings.

A 2-stage NCC + SSIM rerank is also provided to improve perceptual quality
after assignment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity as ssim

if TYPE_CHECKING:
    pass

NDArray = np.ndarray

__all__ = [
    "FAISS_AVAILABLE",
    "compute_cost_matrix",
    "neighbor_swap_refinement",
    "place_faiss_diffusion",
    "place_hungarian",
    "ssim_rerank",
]

try:
    import faiss  # type: ignore

    FAISS_AVAILABLE = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore
    FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Cost matrix
# ---------------------------------------------------------------------------
def compute_cost_matrix(
    grid_features: NDArray,
    tile_features: NDArray,
    grid_oklab_means: NDArray,
    tile_oklab_means: NDArray,
    saliency_weights: NDArray,
    *,
    oklab_weight: float = 0.20,
    k_top: int = 100,
) -> tuple[NDArray, NDArray]:
    """Build a cost matrix combining feature L2 distance and Oklab distance.

    Returns
    -------
    cost_matrix : np.ndarray, shape (n_cells, n_tiles), float64
    top_indices : np.ndarray, shape (n_cells, k_top), int
        Per-cell candidate tile indices sorted by feature L2 distance.
        Used by :func:`ssim_rerank` to limit the rerank search.
    """
    n_cells = len(grid_features)
    n_tiles = len(tile_features)

    # Pure L2 in feature space, vectorized.
    grid_sq = np.sum(grid_features**2, axis=1, keepdims=True)
    tile_sq = np.sum(tile_features**2, axis=1, keepdims=True)
    dot = grid_features @ tile_features.T
    full_l2 = grid_sq + tile_sq.T - 2 * dot
    np.maximum(full_l2, 0, out=full_l2)

    # Top-K candidates per cell. Faiss is faster but optional.
    k_top = min(k_top, n_tiles)
    if FAISS_AVAILABLE:
        index = faiss.IndexFlatL2(grid_features.shape[1])
        index.add(tile_features)
        _, top_indices = index.search(grid_features, k_top)
    else:
        top_indices = np.argpartition(full_l2, k_top - 1, axis=1)[:, :k_top]
        # Sort within each row.
        row_idx = np.arange(n_cells)[:, None]
        ordering = np.argsort(full_l2[row_idx, top_indices], axis=1)
        top_indices = top_indices[row_idx, ordering]

    global_max = full_l2.max() + 1e-6
    cost_matrix = (full_l2 / global_max).astype(np.float64)

    # Add Oklab perceptual distance only on top-K candidates.
    grid_ok_exp = grid_oklab_means[:, np.newaxis, :]
    cand_ok = tile_oklab_means[top_indices]
    diff = grid_ok_exp - cand_ok
    oklab_dist = np.sqrt(np.sum(diff**2, axis=2))
    cost_matrix[np.arange(n_cells)[:, None], top_indices] += oklab_dist * oklab_weight

    # Apply per-cell saliency weighting.
    sal_flat = saliency_weights.flatten()
    cost_matrix *= sal_flat[:, np.newaxis]

    return cost_matrix, top_indices


# ---------------------------------------------------------------------------
# Hungarian assignment
# ---------------------------------------------------------------------------
def place_hungarian(
    cost_matrix: NDArray, grid_cols: int, grid_rows: int
) -> NDArray:
    """Solve the rectangular assignment problem with scipy's solver.

    Returns
    -------
    np.ndarray
        Tile indices laid out as ``(grid_rows, grid_cols)``.
    """
    n_cells = grid_rows * grid_cols
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assignment = np.zeros(n_cells, dtype=np.int32)
    for r, c in zip(row_ind, col_ind):
        assignment[r] = c
    return assignment.reshape(grid_rows, grid_cols)


def neighbor_swap_refinement(
    grid: NDArray, cost_matrix: NDArray, rounds: int, grid_cols: int, grid_rows: int
) -> NDArray:
    """Greedy 2-opt swap of neighboring cells until no improvement is found."""
    if rounds <= 0:
        return grid
    assignment = grid.flatten().copy()
    for _ in range(rounds):
        improved_this_round = 0
        for gy in range(grid_rows):
            for gx in range(grid_cols):
                idx_a = gy * grid_cols + gx
                ta = assignment[idx_a]
                for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1)]:
                    ny, nx = gy + dy, gx + dx
                    if not (0 <= ny < grid_rows and 0 <= nx < grid_cols):
                        continue
                    idx_b = ny * grid_cols + nx
                    tb = assignment[idx_b]
                    delta = (
                        cost_matrix[idx_a, tb]
                        + cost_matrix[idx_b, ta]
                        - cost_matrix[idx_a, ta]
                        - cost_matrix[idx_b, tb]
                    )
                    if delta < -1e-10:
                        assignment[idx_a], assignment[idx_b] = tb, ta
                        ta = tb
                        improved_this_round += 1
        if improved_this_round == 0:
            break
    return assignment.reshape(grid_rows, grid_cols)


# ---------------------------------------------------------------------------
# Two-stage rerank: NCC then SSIM
# ---------------------------------------------------------------------------
def _ncc(img1: NDArray, img2: NDArray) -> float:
    """Normalized cross-correlation."""
    i1 = img1.astype(np.float64).ravel()
    i2 = img2.astype(np.float64).ravel()
    m1, m2 = i1.mean(), i2.mean()
    s1 = i1.std() + 1e-8
    s2 = i2.std() + 1e-8
    return float(np.dot(i1 - m1, i2 - m2) / (len(i1) * s1 * s2))


def ssim_rerank(
    grid: NDArray,
    tiles_gray: list[NDArray],
    main_gray: NDArray,
    grid_cols: int,
    grid_rows: int,
    tile_size: int,
    top_indices: NDArray,
    n_rerank: int = 25,
) -> NDArray:
    """Improve a placement by reranking with NCC then SSIM.

    Stage 1: rank ``n_rerank`` candidates from the cost matrix by NCC, keep
        the top 5.
    Stage 2: pick the highest SSIM among those 5, replacing the current tile
        if it scores better.

    The two-stage approach is roughly 5x faster than running SSIM on the full
    candidate set with no measurable quality loss.
    """
    assignment = grid.flatten().copy()
    used_tiles = set(assignment.tolist())

    for gy in range(grid_rows):
        for gx in range(grid_cols):
            idx = gy * grid_cols + gx
            y0, y1 = gy * tile_size, (gy + 1) * tile_size
            x0, x1 = gx * tile_size, (gx + 1) * tile_size
            cell_gray = main_gray[y0:y1, x0:x1]
            current_tile = int(assignment[idx])

            candidates = top_indices[idx][:n_rerank]
            ncc_scores: list[tuple[float, int]] = [
                (_ncc(cell_gray, tiles_gray[current_tile]), current_tile)
            ]
            for cand in candidates:
                cand = int(cand)
                if cand == current_tile or cand in used_tiles:
                    continue
                ncc_scores.append((_ncc(cell_gray, tiles_gray[cand]), cand))
            ncc_scores.sort(key=lambda pair: -pair[0])
            top5 = [c for _, c in ncc_scores[:5]]

            best_ssim = ssim(cell_gray, tiles_gray[current_tile], data_range=255)
            best_tile = current_tile
            for cand in top5:
                if cand == current_tile:
                    continue
                s = ssim(cell_gray, tiles_gray[cand], data_range=255)
                if s > best_ssim:
                    best_ssim = s
                    best_tile = cand

            if best_tile != current_tile:
                used_tiles.discard(current_tile)
                used_tiles.add(best_tile)
                assignment[idx] = best_tile

    return assignment.reshape(grid_rows, grid_cols)


# ---------------------------------------------------------------------------
# FAISS + Floyd-Steinberg error diffusion (fallback for huge problems)
# ---------------------------------------------------------------------------
def place_faiss_diffusion(
    grid_features: NDArray,
    tile_features: NDArray,
    grid_cols: int,
    grid_rows: int,
    dedup_radius: int,
    strength: float = 0.85,
) -> NDArray:
    """Greedy placement with Floyd-Steinberg error diffusion.

    This algorithm is used when the Hungarian cost matrix is too large to
    fit in memory. ``dedup_radius`` blocks reuse of the same tile within a
    box neighborhood to keep duplicates from clumping.

    Falls back to a numpy-only K-NN if faiss is not installed.
    """
    n_tiles = len(tile_features)
    k = min(150, n_tiles)
    dim = tile_features.shape[1]

    if FAISS_AVAILABLE:
        index = faiss.IndexFlatL2(dim)
        index.add(tile_features)

        def search(q: NDArray) -> NDArray:
            _, candidates = index.search(q, k)
            return candidates[0]

    else:  # pragma: no cover - exercised in faiss-less environments
        tile_sq = np.sum(tile_features.astype(np.float64) ** 2, axis=1)

        def search(q: NDArray) -> NDArray:
            q64 = q[0].astype(np.float64)
            d = tile_sq - 2 * (tile_features @ q64) + np.dot(q64, q64)
            return np.argpartition(d, k - 1)[:k]

    gf_2d = grid_features.reshape(grid_rows, grid_cols, -1).copy().astype(np.float64)
    grid = np.full((grid_rows, grid_cols), -1, dtype=np.int32)
    for gy in range(grid_rows):
        col_range = range(grid_cols) if gy % 2 == 0 else range(grid_cols - 1, -1, -1)
        for gx in col_range:
            query = gf_2d[gy, gx : gx + 1].astype(np.float32)
            candidates = search(query)
            blocked: set[int] = set()
            for dy in range(-dedup_radius, dedup_radius + 1):
                for dx in range(-dedup_radius, dedup_radius + 1):
                    ny, nx = gy + dy, gx + dx
                    if 0 <= ny < grid_rows and 0 <= nx < grid_cols and grid[ny, nx] >= 0:
                        blocked.add(int(grid[ny, nx]))
            chosen = int(candidates[0])
            for c in candidates:
                if int(c) not in blocked:
                    chosen = int(c)
                    break
            grid[gy, gx] = chosen
            chosen_feat = tile_features[chosen].astype(np.float64)
            error = (gf_2d[gy, gx] - chosen_feat) * strength
            if gy % 2 == 0:
                if gx + 1 < grid_cols:
                    gf_2d[gy, gx + 1] += error * 7 / 16
                if gy + 1 < grid_rows:
                    if gx - 1 >= 0:
                        gf_2d[gy + 1, gx - 1] += error * 3 / 16
                    gf_2d[gy + 1, gx] += error * 5 / 16
                    if gx + 1 < grid_cols:
                        gf_2d[gy + 1, gx + 1] += error * 1 / 16
            else:
                if gx - 1 >= 0:
                    gf_2d[gy, gx - 1] += error * 7 / 16
                if gy + 1 < grid_rows:
                    if gx + 1 < grid_cols:
                        gf_2d[gy + 1, gx + 1] += error * 3 / 16
                    gf_2d[gy + 1, gx] += error * 5 / 16
                    if gx - 1 >= 0:
                        gf_2d[gy + 1, gx - 1] += error * 1 / 16
    return grid
