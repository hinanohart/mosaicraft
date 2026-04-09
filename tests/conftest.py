"""Shared pytest fixtures.

These fixtures synthesize tile and target images on the fly so the test
suite never needs binary assets in version control.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

NDArray = np.ndarray


def _make_tile(seed: int, size: int = 64) -> NDArray:
    """Generate a synthetic tile with a unique color/texture signature."""
    rng = np.random.default_rng(seed)
    base = np.full((size, size, 3), rng.integers(0, 255, size=3), dtype=np.uint8)
    # Add a textured noise overlay so feature extraction has structure.
    noise = rng.integers(-30, 30, size=(size, size, 3))
    base = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Draw a couple of shapes for gradient/LBP variety.
    cv2.circle(base, (size // 2, size // 2), size // 4, (255, 255, 255), 2)
    cv2.line(base, (0, 0), (size - 1, size - 1), (0, 0, 0), 2)
    return base


@pytest.fixture
def synthetic_tile_dir(tmp_path: Path) -> Path:
    """Create a directory of 64 synthetic tiles."""
    tile_dir = tmp_path / "tiles"
    tile_dir.mkdir()
    for i in range(64):
        tile = _make_tile(i)
        cv2.imwrite(str(tile_dir / f"tile_{i:03d}.png"), tile)
    return tile_dir


@pytest.fixture
def synthetic_target(tmp_path: Path) -> Path:
    """Create a 256x256 gradient target image."""
    h = w = 256
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img = np.stack(
        [
            np.broadcast_to(x, (h, w)),
            np.broadcast_to(y, (h, w)),
            np.broadcast_to((x + y) // 2, (h, w)),
        ],
        axis=-1,
    ).astype(np.uint8)
    out = tmp_path / "target.png"
    cv2.imwrite(str(out), img)
    return out
