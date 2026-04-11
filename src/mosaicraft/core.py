"""High-level orchestration: the :class:`MosaicGenerator` class.

Typical use::

    from mosaicraft import MosaicGenerator

    gen = MosaicGenerator(
        tile_dir="path/to/tiles",
        preset="vivid",
    )
    gen.generate("input.jpg", "output.jpg", target_tiles=2000)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .blending import assemble_feather, assemble_laplacian
from .color_augment import expand_color_variants
from .features import extract_features
from .placement import (
    compute_cost_matrix,
    neighbor_swap_refinement,
    place_faiss_diffusion,
    place_hungarian,
    ssim_rerank,
)
from .postprocess import postprocess
from .presets import get_preset
from .saliency import compute_saliency_weights
from .tiles import TileSet, augment_tiles, load_tiles, load_tiles_cached
from .utils import calc_grid, logger, stage

NDArray = np.ndarray

__all__ = ["MosaicGenerator", "MosaicResult"]

# Hungarian cost-matrix memory cap. Above this, fall back to FAISS placement.
DEFAULT_HUNGARIAN_MEM_LIMIT_MB = 3000


@dataclass
class MosaicResult:
    """Container returned by :meth:`MosaicGenerator.generate`."""

    image: NDArray
    grid_cols: int
    grid_rows: int
    tile_size: int
    output_path: Path | None = None

    @property
    def n_tiles(self) -> int:
        return self.grid_cols * self.grid_rows


class MosaicGenerator:
    """Build mosaics from a fixed tile set with configurable presets.

    Parameters
    ----------
    tile_dir : path
        Directory containing the tile images. Required if ``cache_dir`` is
        not given.
    cache_dir : path, optional
        Directory holding a precomputed feature cache. If both ``tile_dir``
        and ``cache_dir`` are given, the cache is preferred.
    preset : str or dict
        Preset name (see :data:`mosaicraft.presets.PRESETS`) or a fully
        specified profile dict.
    augment : bool
        Whether to apply geometric/photometric augmentations when loading
        tiles. Ignored when reading from a cache (the cache always contains
        augmented features).
    color_variants : int
        Number of Oklch hue-rotated copies to add per tile after the base
        augmentation, growing the effective pool ``(1 + color_variants)x``.
        ``0`` (the default) matches pre-0.2 behavior. Rotations are applied
        to the already-augmented tile pool, so ``color_variants=4`` on a
        1,024-tile pool with ``augment=True`` yields ``1024 * 4 * 5 = 20,480``
        candidates — a 5x diversity ceiling for Hungarian placement.
    hungarian_mem_limit_mb : float
        Cost matrix size cap; above this, FAISS placement is used.
    """

    def __init__(
        self,
        tile_dir: str | os.PathLike[str] | None = None,
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        preset: str | dict[str, Any] = "vivid",
        augment: bool = True,
        color_variants: int = 0,
        hungarian_mem_limit_mb: float = DEFAULT_HUNGARIAN_MEM_LIMIT_MB,
    ) -> None:
        if tile_dir is None and cache_dir is None:
            raise ValueError("Either tile_dir or cache_dir must be provided")
        self.tile_dir = Path(tile_dir) if tile_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.profile: dict[str, Any] = (
            get_preset(preset) if isinstance(preset, str) else dict(preset)
        )
        self.augment = augment
        self.color_variants = int(color_variants)
        self.hungarian_mem_limit_mb = hungarian_mem_limit_mb
        self._tile_cache: dict[int, TileSet] = {}

    # ------------------------------------------------------------------
    # Tile loading (cached per tile_size)
    # ------------------------------------------------------------------
    def _get_tiles(self, tile_size: int) -> TileSet:
        if tile_size in self._tile_cache:
            return self._tile_cache[tile_size]

        if self.cache_dir is not None and (
            self.cache_dir / f"features_{tile_size}.npz"
        ).exists():
            with stage(f"Loading tiles from cache (tile_size={tile_size})"):
                tileset = load_tiles_cached(self.cache_dir, tile_size)
        else:
            if self.tile_dir is None:
                raise FileNotFoundError(
                    f"No cache for tile_size={tile_size} and no tile_dir given"
                )
            with stage(f"Loading tiles from {self.tile_dir} (tile_size={tile_size})"):
                tileset = load_tiles(self.tile_dir, tile_size)
                if self.augment:
                    tileset = augment_tiles(tileset, tile_size)

        if self.color_variants > 0:
            with stage(
                f"Expanding color variants (x{self.color_variants} hue rotations)"
            ):
                tileset = expand_color_variants(
                    tileset,
                    n_variants=self.color_variants,
                    tile_size=tile_size,
                )

        logger.info("Loaded %d tiles", len(tileset))
        self._tile_cache[tile_size] = tileset
        return tileset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str] | None = None,
        *,
        target_tiles: int = 2000,
        tile_size: int = 88,
        dedup_radius: int = 4,
        jpeg_quality: int = 95,
    ) -> MosaicResult:
        """Generate a mosaic from ``input_path``.

        Parameters
        ----------
        input_path : path
            Target image to reproduce as a mosaic.
        output_path : path, optional
            Where to write the mosaic. If ``None``, the result is not saved.
        target_tiles : int
            Approximate number of cells; the actual count is chosen by
            :func:`mosaicraft.utils.calc_grid`.
        tile_size : int
            Side length of each cell in pixels.
        dedup_radius : int
            Used by FAISS placement to suppress duplicate tiles within a
            box neighborhood. Ignored by Hungarian placement.
        jpeg_quality : int
            JPEG quality if writing to a ``.jpg`` / ``.jpeg`` file.

        Returns
        -------
        MosaicResult
            Bundled output (image array, grid dims, output path if saved).
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input image not found: {input_path}")

        target = cv2.imread(str(input_path))
        if target is None:
            raise OSError(f"Could not decode {input_path}")

        cols, rows, n_cells = calc_grid(target_tiles, target.shape[1], target.shape[0])
        logger.info(
            "Mosaic grid: %dx%d = %d cells, tile_size=%dpx, output=%dx%d",
            cols,
            rows,
            n_cells,
            tile_size,
            cols * tile_size,
            rows * tile_size,
        )

        tileset = self._get_tiles(tile_size)
        target_w = cols * tile_size
        target_h = rows * tile_size
        target_resized = cv2.resize(
            target, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4
        )
        target_lab = cv2.cvtColor(target_resized, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_gray = cv2.cvtColor(target_resized, cv2.COLOR_BGR2GRAY)

        with stage("Computing saliency weights"):
            saliency_weights = compute_saliency_weights(
                target_gray, target_resized, cols, rows, tile_size
            )

        with stage("Extracting per-cell features"):
            grid_features, grid_oklab_means = self._extract_grid_features(
                target_lab, target_resized, cols, rows, tile_size
            )

        del target_lab

        grid = self._place(
            grid_features,
            grid_oklab_means,
            tileset,
            saliency_weights,
            cols,
            rows,
            tile_size,
            target_gray,
            dedup_radius,
            n_cells,
        )

        with stage("Assembling mosaic"):
            if self.profile.get("laplacian_blend", False):
                mosaic = assemble_laplacian(
                    grid,
                    tileset.tiles,
                    target_resized,
                    cols,
                    rows,
                    tile_size,
                    self.profile,
                    saliency_weights,
                    self.profile.get("laplacian_levels", 4),
                )
            else:
                mosaic = assemble_feather(
                    grid,
                    tileset.tiles,
                    target_resized,
                    cols,
                    rows,
                    tile_size,
                    self.profile,
                    saliency_weights,
                )

        del target_resized

        with stage("Postprocessing"):
            mosaic = postprocess(mosaic, self.profile)

        out_path: Path | None = None
        if output_path is not None:
            out_path = self._save(mosaic, Path(output_path), jpeg_quality)

        return MosaicResult(
            image=mosaic,
            grid_cols=cols,
            grid_rows=rows,
            tile_size=tile_size,
            output_path=out_path,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_grid_features(
        target_lab: NDArray,
        target_resized: NDArray,
        cols: int,
        rows: int,
        tile_size: int,
    ) -> tuple[NDArray, NDArray]:
        from .color import bgr_to_oklab

        grid_features: list[list[float]] = []
        grid_oklab_means: list[NDArray] = []
        for gy in range(rows):
            for gx in range(cols):
                cell_lab = target_lab[
                    gy * tile_size : (gy + 1) * tile_size,
                    gx * tile_size : (gx + 1) * tile_size,
                ]
                grid_features.append(extract_features(cell_lab, tile_size))
                cell_bgr = target_resized[
                    gy * tile_size : (gy + 1) * tile_size,
                    gx * tile_size : (gx + 1) * tile_size,
                ]
                grid_oklab_means.append(bgr_to_oklab(cell_bgr).mean(axis=(0, 1)))
        return (
            np.array(grid_features, dtype=np.float32),
            np.array(grid_oklab_means, dtype=np.float64),
        )

    def _place(
        self,
        grid_features: NDArray,
        grid_oklab_means: NDArray,
        tileset: TileSet,
        saliency_weights: NDArray,
        cols: int,
        rows: int,
        tile_size: int,
        target_gray: NDArray,
        dedup_radius: int,
        n_cells: int,
    ) -> NDArray:
        n_tiles = len(tileset)
        cost_matrix_mb = (n_cells * n_tiles * 8) / (1024**2)
        wants_hungarian = self.profile["placement"] == "hungarian"

        # Hungarian assignment requires n_tiles >= n_cells (the assignment
        # is 1:1 — every cell needs a distinct tile). scipy's
        # `linear_sum_assignment` raises a generic
        # ValueError("cost matrix is infeasible") that gives the user no
        # actionable hint, so we raise a richer error up-front.
        if wants_hungarian and n_tiles < n_cells:
            raise ValueError(
                f"tile pool too small for Hungarian placement: "
                f"{n_tiles} tiles but mosaic needs {n_cells} cells. "
                f"Reduce --target-tiles to {n_tiles} or below, "
                f"add more tiles to the pool, "
                f"or use --preset fast (FAISS placement) which allows tile reuse."
            )

        use_hungarian = wants_hungarian and cost_matrix_mb < self.hungarian_mem_limit_mb

        if wants_hungarian and not use_hungarian:
            logger.warning(
                "Cost matrix would be %.0f MB (limit %.0f MB), "
                "falling back to FAISS placement",
                cost_matrix_mb,
                self.hungarian_mem_limit_mb,
            )

        if use_hungarian:
            with stage(f"Computing cost matrix ({n_cells} x {n_tiles})"):
                cost_matrix, top_indices = compute_cost_matrix(
                    grid_features,
                    tileset.features,
                    grid_oklab_means,
                    tileset.oklab_means,
                    saliency_weights,
                )
            with stage("Hungarian assignment"):
                grid = place_hungarian(cost_matrix, cols, rows)
            with stage(
                f"Neighbor swap refinement ({self.profile['neighbor_swap_rounds']} rounds)"
            ):
                grid = neighbor_swap_refinement(
                    grid,
                    cost_matrix,
                    self.profile["neighbor_swap_rounds"],
                    cols,
                    rows,
                )
            with stage("Two-stage rerank (NCC + SSIM)"):
                grid = ssim_rerank(
                    grid,
                    tileset.grays,
                    target_gray,
                    cols,
                    rows,
                    tile_size,
                    top_indices,
                )
            del cost_matrix
        else:
            with stage("FAISS placement with error diffusion"):
                grid = place_faiss_diffusion(
                    grid_features,
                    tileset.features,
                    cols,
                    rows,
                    dedup_radius,
                    strength=self.profile.get("error_diffusion_strength", 0.85),
                )
        return grid

    @staticmethod
    def _save(mosaic: NDArray, out_path: Path, jpeg_quality: int) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        params: list[int] = []
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 1, 100))]
        ok = cv2.imwrite(str(out_path), mosaic, params)
        if not ok:
            raise OSError(f"Failed to write {out_path}")
        size_mb = out_path.stat().st_size / 1024 / 1024
        logger.info("Wrote %s (%.1f MB)", out_path, size_mb)
        return out_path
