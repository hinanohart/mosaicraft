#!/usr/bin/env python3
"""Generate a CC0 demo target image — abstract artwork, no copyright.

Useful for trying mosaicraft without having to find a free-to-use photo.

Usage::

    python scripts/generate_demo_target.py --output demo_target.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def make_target(width: int = 800, height: int = 1000, seed: int = 0) -> np.ndarray:
    """Create an abstract gradient target image."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.float32)
    cy, cx = height / 2, width / 2

    # Three radial gradients of saturated colors blended together.
    out = np.zeros((height, width, 3), dtype=np.float32)
    for _ in range(3):
        cx2 = cx + rng.uniform(-cx / 2, cx / 2)
        cy2 = cy + rng.uniform(-cy / 2, cy / 2)
        d = np.sqrt((xx - cx2) ** 2 + (yy - cy2) ** 2)
        norm = 1.0 - np.clip(d / max(width, height), 0, 1)
        intensity = norm**2
        color = rng.uniform(80, 255, size=3).astype(np.float32)
        out += intensity[..., None] * color[None, None, :]

    # Add concentric rings for visual interest.
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rings = (np.sin(rr / 18) + 1) * 30
    out += rings[..., None]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("demo_target.jpg"),
        help="Output image path (default: demo_target.jpg)",
    )
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    img = make_target(args.width, args.height, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Wrote demo target {args.width}x{args.height} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
