#!/usr/bin/env python3
"""Generate a procedural CC0 demo tile set for trying mosaicraft.

The tiles are synthesized from scratch — no copyrighted material — so the
output is free to redistribute. Each tile is a small abstract composition
with a unique color signature, which gives the matcher a useful pool to
choose from while keeping the repository light.

Usage::

    python scripts/generate_demo_tiles.py --output demo_tiles --count 256
    python scripts/generate_demo_tiles.py -o demo_tiles -n 512 --size 96
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

NDArray = np.ndarray


def _hsv_color(rng: np.random.Generator) -> tuple[int, int, int]:
    """Sample a random HSV color biased toward saturated mid-tones."""
    h = int(rng.integers(0, 180))
    s = int(rng.integers(80, 255))
    v = int(rng.integers(80, 255))
    return h, s, v


def _add_noise(img: NDArray, rng: np.random.Generator, amplitude: int = 25) -> NDArray:
    noise = rng.integers(-amplitude, amplitude + 1, size=img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _hsv_to_bgr(hsv: tuple[int, int, int]) -> tuple[int, int, int]:
    arr = np.uint8([[list(hsv)]])
    bgr = cv2.cvtColor(arr, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def _gradient_tile(rng: np.random.Generator, size: int) -> NDArray:
    color_a = _hsv_to_bgr(_hsv_color(rng))
    color_b = _hsv_to_bgr(_hsv_color(rng))
    direction = rng.integers(0, 4)
    if direction == 0:  # horizontal
        t = np.linspace(0, 1, size)[None, :, None]
    elif direction == 1:  # vertical
        t = np.linspace(0, 1, size)[:, None, None]
    elif direction == 2:  # diagonal
        x = np.linspace(0, 1, size)
        t = ((x[None, :] + x[:, None]) / 2.0)[..., None]
    else:  # radial
        cy, cx = size / 2, size / 2
        yy, xx = np.indices((size, size), dtype=np.float32)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        t = (d / d.max())[..., None]
    a = np.array(color_a, dtype=np.float32)
    b = np.array(color_b, dtype=np.float32)
    out = a * (1 - t) + b * t
    return out.astype(np.uint8)


def _shape_tile(rng: np.random.Generator, size: int) -> NDArray:
    bg = _hsv_to_bgr(_hsv_color(rng))
    fg = _hsv_to_bgr(_hsv_color(rng))
    img = np.full((size, size, 3), bg, dtype=np.uint8)
    shape = rng.integers(0, 3)
    if shape == 0:  # circle
        center = (int(rng.integers(size // 4, 3 * size // 4)),
                  int(rng.integers(size // 4, 3 * size // 4)))
        radius = int(rng.integers(size // 6, size // 3))
        cv2.circle(img, center, radius, fg, -1)
    elif shape == 1:  # rectangle
        x0 = int(rng.integers(0, size // 2))
        y0 = int(rng.integers(0, size // 2))
        x1 = int(rng.integers(size // 2, size))
        y1 = int(rng.integers(size // 2, size))
        cv2.rectangle(img, (x0, y0), (x1, y1), fg, -1)
    else:  # triangle
        pts = np.array(
            [
                [rng.integers(0, size), rng.integers(0, size // 2)],
                [rng.integers(0, size // 2), rng.integers(size // 2, size)],
                [rng.integers(size // 2, size), rng.integers(size // 2, size)],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(img, [pts], fg)
    return img


def _stripe_tile(rng: np.random.Generator, size: int) -> NDArray:
    color_a = _hsv_to_bgr(_hsv_color(rng))
    color_b = _hsv_to_bgr(_hsv_color(rng))
    img = np.zeros((size, size, 3), dtype=np.uint8)
    stripe_width = int(rng.integers(4, 12))
    horizontal = bool(rng.integers(0, 2))
    for i in range(size):
        coord = i // stripe_width
        color = color_a if coord % 2 == 0 else color_b
        if horizontal:
            img[i, :] = color
        else:
            img[:, i] = color
    return img


def _checker_tile(rng: np.random.Generator, size: int) -> NDArray:
    color_a = _hsv_to_bgr(_hsv_color(rng))
    color_b = _hsv_to_bgr(_hsv_color(rng))
    cells = int(rng.integers(2, 6))
    cell = max(1, size // cells)
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for i in range(cells + 1):
        for j in range(cells + 1):
            color = color_a if (i + j) % 2 == 0 else color_b
            y0, y1 = i * cell, min(size, (i + 1) * cell)
            x0, x1 = j * cell, min(size, (j + 1) * cell)
            img[y0:y1, x0:x1] = color
    return img


GENERATORS = [_gradient_tile, _shape_tile, _stripe_tile, _checker_tile]


def generate_tile(seed: int, size: int) -> NDArray:
    """Generate a single demo tile from a seed."""
    rng = np.random.default_rng(seed)
    fn = GENERATORS[seed % len(GENERATORS)]
    tile = fn(rng, size)
    return _add_noise(tile, rng, amplitude=20)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("demo_tiles"),
        help="Output directory (default: demo_tiles)",
    )
    parser.add_argument(
        "-n",
        "--count",
        type=int,
        default=256,
        help="Number of tiles to generate (default: 256)",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=128,
        help="Tile side length in pixels (default: 128)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed (default: 0)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        tile = generate_tile(args.seed + i, args.size)
        path = args.output / f"tile_{i:04d}.jpg"
        cv2.imwrite(str(path), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if (i + 1) % 64 == 0:
            print(f"  {i + 1}/{args.count}")
    print(f"Wrote {args.count} CC0 demo tiles to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
