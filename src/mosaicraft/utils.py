"""Small shared utilities."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter

import numpy as np

logger = logging.getLogger("mosaicraft")

__all__ = ["calc_grid", "configure_logging", "logger", "stage"]


def configure_logging(verbose: bool = False) -> None:
    """Configure the package logger.

    Idempotent: re-calling will not duplicate handlers.
    """
    if logger.handlers:
        logger.setLevel(logging.DEBUG if verbose else logging.INFO)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False


def calc_grid(
    target_tiles: int, aspect_w: int, aspect_h: int
) -> tuple[int, int, int]:
    """Choose grid dimensions that approximate ``target_tiles`` for an aspect.

    Parameters
    ----------
    target_tiles : int
        Approximate desired number of cells in the mosaic.
    aspect_w, aspect_h : int
        Width and height of the target image (used for aspect ratio only).

    Returns
    -------
    cols, rows, total : tuple[int, int, int]
    """
    if target_tiles < 1:
        raise ValueError(f"target_tiles must be >= 1, got {target_tiles}")
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError("aspect_w and aspect_h must be positive")
    rows = max(1, int(np.sqrt(target_tiles * aspect_h / aspect_w)))
    cols = max(1, int(rows * aspect_w / aspect_h))
    return cols, rows, cols * rows


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Log the elapsed wall time of a named pipeline stage."""
    t0 = perf_counter()
    logger.info("%s ...", name)
    try:
        yield
    finally:
        logger.info("%s done in %.1fs", name, perf_counter() - t0)
