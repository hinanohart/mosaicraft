"""Benchmark the end-to-end mosaicraft pipeline.

Measures wall time and peak RSS across a sweep of target-cell counts and
presets. Two scales are supported:

* ``--scale small``  — 200 / 500 / 1000 cells on the synthetic demo pool
  (256 tiles). Fast, no asset bootstrap. Used by the README table.
* ``--scale large``  — 5,000 / 10,000 / 20,000 / 30,000 cells on the CC0
  demo pool (1,024 tiles from picsum.photos). Requires
  ``scripts/download_demo_assets.py`` to be run once. Used for the
  "large-regime" table in the README.

Run from the repository root::

    # small (default): ~1 min on a laptop
    python benchmarks/benchmark_pipeline.py

    # 30k cells regime: 5-15 min depending on CPU
    python benchmarks/benchmark_pipeline.py --scale large

    # Fully custom
    python benchmarks/benchmark_pipeline.py --presets ultra --tiles 5000 10000
"""

from __future__ import annotations

import argparse
import platform
import resource
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from mosaicraft import MosaicGenerator, __version__, configure_logging


def ensure_small_assets() -> tuple[Path, Path]:
    """Synthetic tiles + target. Fast, no network."""
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


def ensure_large_assets() -> tuple[Path, Path]:
    """CC0 pool (1024 tiles) + a public-domain painting as target.

    Requires ``python scripts/download_demo_assets.py`` to have been run.
    """
    tiles_dir = REPO_ROOT / "docs" / "assets" / "tiles"
    paintings_dir = REPO_ROOT / "docs" / "assets" / "paintings"
    target = paintings_dir / "pearl_earring.jpg"
    if not tiles_dir.exists() or not any(tiles_dir.glob("*.jpg")):
        raise SystemExit(
            "Large-scale benchmark needs the CC0 demo pool. "
            "Run `python scripts/download_demo_assets.py` first."
        )
    if not target.exists():
        raise SystemExit(
            f"Target painting {target} not found. "
            "Run `python scripts/download_demo_assets.py` first."
        )
    return tiles_dir, target


def _peak_rss_mb() -> float:
    """Peak resident set size in MB (ru_maxrss is KB on Linux, B on macOS)."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / (1024 * 1024)
    return maxrss / 1024


def run_one(
    tiles_dir: Path,
    target: Path,
    preset: str,
    n_tiles: int,
    *,
    tile_size: int = 64,
) -> tuple[float, float, int]:
    """Return (wall_time_s, delta_peak_rss_mb, n_cells_actual)."""
    out = REPO_ROOT / f"bench_{preset}_{n_tiles}.jpg"
    gen = MosaicGenerator(tile_dir=tiles_dir, preset=preset)
    rss_before = _peak_rss_mb()
    t0 = time.perf_counter()
    result = gen.generate(target, out, target_tiles=n_tiles, tile_size=tile_size)
    elapsed = time.perf_counter() - t0
    rss_after = _peak_rss_mb()
    out.unlink(missing_ok=True)
    return elapsed, max(0.0, rss_after - rss_before), result.n_tiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scale",
        choices=("small", "large"),
        default="small",
        help="small = 256-tile synthetic pool (fast); large = 1024 CC0 tiles (30k regime)",
    )
    parser.add_argument(
        "--presets",
        nargs="+",
        default=None,
        help="Presets to benchmark (default: fast/natural/ultra/vivid for small, fast/ultra for large)",
    )
    parser.add_argument(
        "--tiles",
        nargs="+",
        type=int,
        default=None,
        help="Target-cell counts to sweep (default depends on --scale)",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Cell size in pixels (default 64 small, 56 large)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(verbose=False)

    if args.scale == "small":
        tiles_dir, target = ensure_small_assets()
        presets = args.presets or ["fast", "natural", "ultra", "vivid"]
        cell_counts = args.tiles or [200, 500, 1000]
        tile_size = args.tile_size or 64
    else:
        tiles_dir, target = ensure_large_assets()
        presets = args.presets or ["fast", "ultra"]
        cell_counts = args.tiles or [5000, 10000, 20000, 30000]
        tile_size = args.tile_size or 56

    # Environment banner — helps anyone comparing numbers from different machines.
    uname = platform.uname()
    print(f"\n## mosaicraft benchmark — {args.scale} scale")
    print()
    print(
        f"- mosaicraft `{__version__}` · Python {platform.python_version()} · "
        f"{uname.system} {uname.release}"
    )
    print(f"- CPU: `{uname.machine}` · target: `{target.name}` · tiles: `{tiles_dir.relative_to(REPO_ROOT)}`")
    print(f"- cell size: {tile_size}px · {len(list(tiles_dir.glob('*.jpg')))} tiles in pool")
    print()

    # Table header.
    col_headers = [f"{n:,} cells" for n in cell_counts]
    print("| preset | metric | " + " | ".join(col_headers) + " |")
    print("|" + "|".join(["---"] * (len(cell_counts) + 2)) + "|")

    for preset in presets:
        times: list[str] = []
        mems: list[str] = []
        for n in cell_counts:
            elapsed, delta_rss, _actual = run_one(
                tiles_dir, target, preset, n, tile_size=tile_size
            )
            times.append(f"{elapsed:.1f}s")
            mems.append(f"{delta_rss:,.0f} MB")
        print(f"| {preset} | wall time | " + " | ".join(times) + " |")
        print(f"| {preset} | Δ peak RSS | " + " | ".join(mems) + " |")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
