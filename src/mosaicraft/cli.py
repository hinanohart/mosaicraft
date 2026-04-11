"""Command-line interface for mosaicraft.

Examples::

    # Build a mosaic from a directory of tile images.
    mosaicraft generate photo.jpg --tiles tiles/ --output mosaic.jpg

    # Use a specific preset and target tile count.
    mosaicraft generate photo.jpg -t tiles/ -o out.jpg --preset vivid -n 5000

    # Pre-build the feature cache for fast iteration.
    mosaicraft cache --tiles tiles/ --cache-dir .cache --sizes 56 88 120

    # List available presets.
    mosaicraft presets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .core import MosaicGenerator
from .presets import get_preset, list_presets
from .recolor import get_recolor_preset, list_recolor_presets, recolor
from .tiles import build_cache
from .utils import configure_logging, logger

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mosaicraft",
        description=(
            "Perceptual photomosaic generator with Oklab color space, "
            "Hungarian 1:1 placement, MKL optimal-transport color matching, "
            "Laplacian blending, and Oklch tile-pool expansion + recoloring."
        ),
    )
    parser.add_argument(
        "--version", action="version", version=f"mosaicraft {__version__}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # generate
    g = sub.add_parser("generate", help="Generate a mosaic from an image")
    g.add_argument("input", type=Path, help="Path to the target image")
    g.add_argument(
        "-t",
        "--tiles",
        type=Path,
        help="Directory containing tile images",
    )
    g.add_argument(
        "-c",
        "--cache-dir",
        type=Path,
        help="Directory containing a precomputed feature cache",
    )
    g.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output image path (.jpg, .png, ...)",
    )
    g.add_argument(
        "-p",
        "--preset",
        default="vivid",
        choices=list_presets(),
        help="Profile preset (default: vivid)",
    )
    g.add_argument(
        "-n",
        "--target-tiles",
        type=int,
        default=2000,
        help="Approximate number of cells (default: 2000)",
    )
    g.add_argument(
        "-s",
        "--tile-size",
        type=int,
        default=88,
        help="Cell size in pixels (default: 88)",
    )
    g.add_argument(
        "--dedup-radius",
        type=int,
        default=4,
        help="Dedup radius for FAISS placement (default: 4)",
    )
    g.add_argument(
        "--no-augment",
        action="store_true",
        help="Disable tile augmentation when loading",
    )
    g.add_argument(
        "--color-variants",
        type=int,
        default=0,
        help=(
            "Expand the tile pool by N Oklch hue-rotated variants per tile. "
            "For a pool of 1,000 tiles and --color-variants 4 you get 5,000 "
            "candidates (diversity ceiling ~5x higher). Default: 0 (off)."
        ),
    )
    g.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )
    g.add_argument(
        "--mem-limit-mb",
        type=float,
        default=3000,
        help="Cost matrix memory cap in MB (default: 3000)",
    )

    # cache
    c = sub.add_parser("cache", help="Build a feature cache for fast iteration")
    c.add_argument(
        "-t", "--tiles", type=Path, required=True, help="Tile directory"
    )
    c.add_argument(
        "-c",
        "--cache-dir",
        type=Path,
        required=True,
        help="Output cache directory",
    )
    c.add_argument(
        "-s",
        "--sizes",
        type=int,
        nargs="+",
        required=True,
        help="One or more tile sizes to precompute (e.g. 56 88 120)",
    )
    c.add_argument(
        "--thumb-size",
        type=int,
        default=120,
        help="Thumbnail size to store in the cache (default: 120)",
    )

    # recolor
    r = sub.add_parser(
        "recolor",
        help="Recolor an image perceptually in Oklch (hue rotation + chroma)",
    )
    r.add_argument("input", type=Path, help="Input image path")
    r.add_argument("-o", "--output", type=Path, required=True, help="Output image path")
    group = r.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-p",
        "--preset",
        choices=list_recolor_presets(),
        help="Named recolor preset (e.g. blue, sepia, cyberpunk)",
    )
    group.add_argument(
        "--hex",
        dest="target_hex",
        help="Target color as #RRGGBB (any valid sRGB color)",
    )
    group.add_argument(
        "--hue",
        dest="hue_shift_deg",
        type=float,
        help="Relative hue rotation in degrees (e.g. 60, -30)",
    )
    r.add_argument(
        "--chroma",
        dest="chroma_scale",
        type=float,
        default=None,
        help="Override chroma scale (0.0 = grayscale, 1.0 = keep, >1 = boost)",
    )
    r.add_argument(
        "--lightness-gamma",
        type=float,
        default=None,
        help="Override lightness gamma (<1 lifts shadows, >1 deepens midtones)",
    )
    r.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Blend factor 0-1 (default: 1.0 = full recolor)",
    )
    r.add_argument(
        "--no-protect-highlights",
        dest="protect_highlights",
        action="store_false",
        help="Do not fade chroma in highlights",
    )
    r.add_argument(
        "--no-protect-shadows",
        dest="protect_shadows",
        action="store_false",
        help="Do not fade chroma in shadows",
    )
    r.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality 1-100 (default: 95)",
    )

    # presets
    sub.add_parser("presets", help="List available mosaic presets")

    # recolor-presets
    sub.add_parser("recolor-presets", help="List available recolor presets")

    return parser


def _cmd_generate(args: argparse.Namespace) -> int:
    if args.tiles is None and args.cache_dir is None:
        logger.error("Either --tiles or --cache-dir must be specified")
        return 2
    gen = MosaicGenerator(
        tile_dir=args.tiles,
        cache_dir=args.cache_dir,
        preset=args.preset,
        augment=not args.no_augment,
        color_variants=args.color_variants,
        hungarian_mem_limit_mb=args.mem_limit_mb,
    )
    result = gen.generate(
        args.input,
        args.output,
        target_tiles=args.target_tiles,
        tile_size=args.tile_size,
        dedup_radius=args.dedup_radius,
        jpeg_quality=args.jpeg_quality,
    )
    logger.info(
        "Result: %dx%d cells (%d), %dx%d px",
        result.grid_cols,
        result.grid_rows,
        result.n_tiles,
        result.image.shape[1],
        result.image.shape[0],
    )
    return 0


def _cmd_cache(args: argparse.Namespace) -> int:
    build_cache(
        tile_dir=args.tiles,
        cache_dir=args.cache_dir,
        tile_sizes=list(args.sizes),
        thumb_size=args.thumb_size,
    )
    logger.info("Cache build complete: %s", args.cache_dir)
    return 0


def _cmd_presets(_args: argparse.Namespace) -> int:
    print("Available presets:\n")
    for name in list_presets():
        info = get_preset(name)
        desc = info.get("description", "(no description)")
        print(f"  {name:14s} - {desc}")
    return 0


def _cmd_recolor_presets(_args: argparse.Namespace) -> int:
    print("Available recolor presets:\n")
    for name in list_recolor_presets():
        p = get_recolor_preset(name)
        print(f"  {name:12s} - {p.description}")
    return 0


def _cmd_recolor(args: argparse.Namespace) -> int:
    recolor(
        args.input,
        args.output,
        preset=args.preset,
        target_hex=args.target_hex,
        hue_shift_deg=args.hue_shift_deg,
        chroma_scale=args.chroma_scale,
        lightness_gamma=args.lightness_gamma,
        strength=args.strength,
        protect_highlights=args.protect_highlights,
        protect_shadows=args.protect_shadows,
        jpeg_quality=args.jpeg_quality,
    )
    logger.info("Wrote %s", args.output)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(verbose=getattr(args, "verbose", False))
    try:
        if args.command == "generate":
            return _cmd_generate(args)
        if args.command == "cache":
            return _cmd_cache(args)
        if args.command == "presets":
            return _cmd_presets(args)
        if args.command == "recolor":
            return _cmd_recolor(args)
        if args.command == "recolor-presets":
            return _cmd_recolor_presets(args)
        parser.error(f"Unknown command: {args.command}")
    except KeyboardInterrupt:
        logger.error("Interrupted")
        return 130
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("%s: %s", type(exc).__name__, exc)
        if getattr(args, "verbose", False):
            raise
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
