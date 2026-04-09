"""Tests for placement algorithms."""

from __future__ import annotations

import numpy as np

from mosaicraft.placement import (
    compute_cost_matrix,
    neighbor_swap_refinement,
    place_faiss_diffusion,
    place_hungarian,
)


def _random_features(n: int, dim: int = 191, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n, dim)).astype(np.float32)


def _random_oklab(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        [
            rng.uniform(0, 1, n),
            rng.uniform(-0.4, 0.4, n),
            rng.uniform(-0.4, 0.4, n),
        ]
    )


def test_compute_cost_matrix_shape() -> None:
    grid = _random_features(20, seed=0)
    tiles = _random_features(50, seed=1)
    cost, top = compute_cost_matrix(
        grid,
        tiles,
        _random_oklab(20),
        _random_oklab(50),
        np.ones((4, 5)),
    )
    assert cost.shape == (20, 50)
    assert top.shape[0] == 20
    assert top.shape[1] <= 50


def test_place_hungarian_returns_grid_shape() -> None:
    cols, rows = 5, 4
    n_cells = cols * rows
    n_tiles = 30
    cost = np.random.default_rng(0).random((n_cells, n_tiles))
    grid = place_hungarian(cost, cols, rows)
    assert grid.shape == (rows, cols)
    # Each cell should hold a valid tile index.
    assert grid.min() >= 0
    assert grid.max() < n_tiles
    # All chosen indices unique (assignment is a permutation onto cells).
    assert len(np.unique(grid)) == n_cells


def test_neighbor_swap_does_not_increase_cost() -> None:
    cols, rows = 4, 4
    n_cells = cols * rows
    rng = np.random.default_rng(2)
    cost = rng.random((n_cells, 32))
    grid = place_hungarian(cost, cols, rows)
    before = sum(cost[i, grid.flat[i]] for i in range(n_cells))
    refined = neighbor_swap_refinement(grid, cost, rounds=3, grid_cols=cols, grid_rows=rows)
    after = sum(cost[i, refined.flat[i]] for i in range(n_cells))
    assert after <= before + 1e-9


def test_faiss_diffusion_fills_grid() -> None:
    cols, rows = 6, 5
    grid_features = _random_features(cols * rows, seed=10)
    tile_features = _random_features(40, seed=11)
    grid = place_faiss_diffusion(grid_features, tile_features, cols, rows, dedup_radius=1)
    assert grid.shape == (rows, cols)
    assert grid.min() >= 0
