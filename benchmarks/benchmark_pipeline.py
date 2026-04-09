"""Benchmark the end-to-end mosaicraft pipeline.

Generates synthetic CC0 demo assets, then times each preset across a few
target-tile counts. Output is a small Markdown table you can paste into a
PR description or copy into the README.

Run from the repository root::

    python benchmarks/benchmark_pipeline.py
    python benchmarks/benchmark_pipeline.py --presets ultra fast --tiles 200 800
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mosaicraft import MosaicGenerator, configure_logging  # noqa: E402


def ensure_assets() -> tuple[Path, Path]:
    """Reuse examples/basic.py asset bootstrap so the bench is self-contained."""
    tiles_dir = REPO_ROOT / "demo_tiles"
    target = REPO_ROOT / "demo_target.jpg"
    if not tiles_dir.exists() or not any(tiles_dir.iterdir()):
        import cv2
        from scripts.generate_demo_tiles import generate_tile

        tiles_dir.mkdir(exist_ok=True)
        for i in range(256):
            tile = generate_tile(i, 128)
            cv2.imwrite(str(tiles_dir / f"tile_{i:04d}.jpg"), tile, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not target.exists():
        import cv2
        from scripts.generate_demo_target import make_target

        img = make_target(800, 1000, seed=42)
        cv2.imwrite(str(target), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return tiles_dir, target


def run_one(tiles_dir: Path, target: Path, preset: str, n_tiles: int) -> float:
    out = REPO_ROOT / f"bench_{preset}_{n_tiles}.jpg"
    gen = MosaicGenerator(tile_dir=tiles_dir, preset=preset)
    t0 = time.perf_counter()
    gen.generate(target, out, target_tiles=n_tiles, tile_size=64)
    elapsed = time.perf_counter() - t0
    out.unlink(missing_ok=True)
    return elapsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--presets",
        nargs="+",
        default=["fast", "natural", "ultra", "vivid"],
        help="Presets to benchmark",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        type=int,
        default=[200, 500, 1000],
        help="target_tiles values to sweep",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(verbose=False)
    tiles_dir, target = ensure_assets()

    print(f"\n## mosaicraft benchmark — tiles={args.tiles}, presets={args.presets}\n")
    header = "| preset | " + " | ".join(f"{n} cells" for n in args.tiles) + " |"
    sep = "|" + "|".join(["---"] * (len(args.tiles) + 1)) + "|"
    print(header)
    print(sep)

    for preset in args.presets:
        row = [preset]
        for n in args.tiles:
            elapsed = run_one(tiles_dir, target, preset, n)
            row.append(f"{elapsed:.2f}s")
        print("| " + " | ".join(row) + " |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
