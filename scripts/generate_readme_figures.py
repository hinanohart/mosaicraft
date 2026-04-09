#!/usr/bin/env python3
"""Generate README figures for mosaicraft.

Builds a CC0 procedural landscape target, a procedural tile pool, then
renders mosaics with several presets and composites comparison figures
into ``docs/images/``. Every asset is synthesized from scratch so the
output is free to redistribute - no external photography involved.

Usage::

    python scripts/generate_readme_figures.py
    python scripts/generate_readme_figures.py --quick     # smaller, faster
    python scripts/generate_readme_figures.py --seed 7
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mosaicraft import MosaicGenerator, configure_logging
from scripts.generate_demo_tiles import generate_tile


# --------------------------------------------------------------------------- #
# Procedural landscape target (CC0).
# --------------------------------------------------------------------------- #

def _interp_channels(t: np.ndarray, stops_t: np.ndarray, stops_c: np.ndarray) -> np.ndarray:
    out = np.empty((*t.shape, 3), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(t, stops_t, stops_c[:, ch])
    return out


def make_landscape_target(
    width: int = 1600,
    height: int = 1200,
    seed: int = 42,
) -> np.ndarray:
    """Return a BGR uint8 image of a procedural mountain-sunset landscape."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((height, width), dtype=np.float32)
    y_norm = yy / height
    x_norm = xx / width

    horizon = 0.58
    horizon_y = height * horizon

    # ---- Sky gradient: deep violet → magenta → orange → gold → cream ----
    stops_t = np.array([0.00, 0.30, 0.55, 0.78, 1.00], dtype=np.float32)
    stops_c = np.array(
        [
            [130,  50, 105],  # deep violet (BGR)
            [125,  85, 185],  # magenta
            [ 85, 130, 235],  # warm orange
            [110, 200, 250],  # gold
            [180, 230, 250],  # pale cream
        ],
        dtype=np.float32,
    )
    sky_t = np.clip(yy / horizon_y, 0.0, 1.0)
    sky_color = _interp_channels(sky_t, stops_t, stops_c)

    # ---- Sun with glow ----
    sun_cx = width * 0.68
    sun_cy = horizon_y * 0.66
    sun_r = min(width, height) * 0.045
    d_sun = np.sqrt((xx - sun_cx) ** 2 + (yy - sun_cy) ** 2)
    glow = np.exp(-((d_sun / (sun_r * 3.2)) ** 2))
    sky_color += glow[..., None] * np.array(
        [190, 235, 255], dtype=np.float32
    )[None, None, :] * 0.45
    core = np.clip(1.0 - d_sun / sun_r, 0.0, 1.0) ** 0.6
    sky_color = sky_color * (1.0 - core[..., None]) + core[..., None] * np.array(
        [235, 250, 255], dtype=np.float32
    )[None, None, :]

    # ---- Cloud bands (low-frequency multi-octave sine noise) ----
    cloud = np.zeros_like(yy)
    for octave in range(4):
        f = 2 ** octave
        phase_x = rng.uniform(0.0, 2.0 * np.pi)
        phase_y = rng.uniform(0.0, 2.0 * np.pi)
        cloud += (
            np.sin(x_norm * np.pi * 3.0 * f + phase_x)
            * np.sin(y_norm * np.pi * 2.0 * f + phase_y)
            / f
        )
    # Concentrate clouds in upper sky
    cloud = np.clip(cloud * 0.5 + 0.5, 0.0, 1.0)
    cloud *= np.exp(-(((y_norm - 0.22) * 4.0) ** 2))
    sky_mask = (yy < horizon_y).astype(np.float32)
    sky_color += (cloud * sky_mask)[..., None] * np.array(
        [50, 70, 90], dtype=np.float32
    )[None, None, :]

    # ---- Mountain silhouette layers (far → near) ----
    layers = [
        (0.48, 30.0, np.array([100,  80,  95], dtype=np.float32)),
        (0.53, 55.0, np.array([ 70,  55,  80], dtype=np.float32)),
        (0.58, 85.0, np.array([ 35,  28,  50], dtype=np.float32)),
    ]
    for elev, rough, color in layers:
        ridge = np.zeros(width, dtype=np.float32)
        for k in range(6):
            freq = rng.integers(2, 9)
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp = rough / (k + 1.0)
            ridge += np.sin(np.arange(width) / width * freq * 2.0 * np.pi + phase) * amp
        ridge_y = elev * height + ridge
        ridge_map = np.broadcast_to(ridge_y[None, :], (height, width))
        mask = (yy >= ridge_map) & (yy < horizon_y)
        sky_color[mask] = color

    # ---- Water: reflect the sky, darken, add ripples ----
    water_y = yy - horizon_y
    reflect_y = np.clip(horizon_y - water_y * 0.55, 0.0, horizon_y - 1.0).astype(np.int32)
    x_i = xx.astype(np.int32)
    reflected = sky_color[reflect_y, x_i]
    depth_t = np.clip(water_y / max(1.0, height - horizon_y), 0.0, 1.0)
    water = reflected * (0.45 + 0.25 * (1.0 - depth_t))[..., None]

    # Ripples (visible in reflection)
    ripple = (
        np.sin(yy * 0.35 + xx * 0.015) * 6.0
        + np.sin(yy * 0.18 - xx * 0.010) * 4.0
        + np.sin(yy * 0.08 + xx * 0.004) * 2.5
    )
    water = water + ripple[..., None]

    water_mask = (yy >= horizon_y).astype(np.float32)[..., None]
    out = sky_color * (1.0 - water_mask) + water * water_mask

    # Subtle grain
    grain = rng.integers(-4, 5, size=out.shape, dtype=np.int16)
    out = np.clip(out + grain, 0, 255).astype(np.uint8)
    return out


# --------------------------------------------------------------------------- #
# Tiles.
# --------------------------------------------------------------------------- #

def build_tile_pool(tiles_dir: Path, count: int, tile_size: int, seed: int) -> None:
    tiles_dir.mkdir(parents=True, exist_ok=True)
    # Clear stale pool so stray images don't influence matching.
    for p in tiles_dir.glob("*.jpg"):
        p.unlink()
    for i in range(count):
        tile = generate_tile(seed + i, tile_size)
        cv2.imwrite(
            str(tiles_dir / f"tile_{i:05d}.jpg"),
            tile,
            [cv2.IMWRITE_JPEG_QUALITY, 92],
        )


# --------------------------------------------------------------------------- #
# Figure composition helpers.
# --------------------------------------------------------------------------- #

def _label_bar(
    width: int,
    text: str,
    *,
    height: int = 48,
    bg: tuple[int, int, int] = (22, 22, 24),
    fg: tuple[int, int, int] = (240, 240, 240),
    font_scale: float = 0.9,
) -> np.ndarray:
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, font_scale, 1)
    x = (width - tw) // 2
    y = (height + th) // 2 - 2
    cv2.putText(
        bar,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        font_scale,
        fg,
        1,
        cv2.LINE_AA,
    )
    return bar


def _fit_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h:
        return img
    scale = target_h / h
    new_w = max(1, round(w * scale))
    return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)


def _hstack_with_labels(
    images: list[np.ndarray],
    labels: list[str],
    *,
    target_h: int = 900,
    gap: int = 12,
    bg: tuple[int, int, int] = (22, 22, 24),
) -> np.ndarray:
    resized = [_fit_height(im, target_h) for im in images]
    widths = [im.shape[1] for im in resized]
    total_w = sum(widths) + gap * (len(images) + 1)

    label_bars = [
        _label_bar(w, t, height=56, bg=bg, fg=(240, 240, 240), font_scale=1.0)
        for w, t in zip(widths, labels)
    ]

    top_label_h = 56
    canvas_h = top_label_h + gap + target_h + gap
    canvas = np.full((canvas_h, total_w, 3), bg, dtype=np.uint8)
    x = gap
    for im, bar in zip(resized, label_bars):
        w = im.shape[1]
        canvas[0:top_label_h, x : x + w] = bar
        y = top_label_h + gap
        canvas[y : y + target_h, x : x + w] = im
        x += w + gap
    return canvas


def _center_crop(img: np.ndarray, box: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w, box)
    y0 = (h - s) // 2
    x0 = (w - s) // 2
    return img[y0 : y0 + s, x0 : x0 + s]


_JPEG_Q = 86  # balance file size vs visible blockiness on GitHub


def make_hero(target: np.ndarray, mosaic: np.ndarray, out_path: Path) -> None:
    """Side-by-side target vs mosaic for the top of the README."""
    fig = _hstack_with_labels(
        [target, mosaic],
        ["Target (CC0 procedural)", "mosaicraft output - preset: ultra"],
        target_h=720,
        gap=14,
    )
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_before_after(target: np.ndarray, mosaic: np.ndarray, out_path: Path) -> None:
    fig = _hstack_with_labels(
        [target, mosaic],
        ["Before", "After - preset: ultra"],
        target_h=640,
        gap=12,
    )
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_presets_comparison(
    mosaics: dict[str, np.ndarray], out_path: Path
) -> None:
    order = ["natural", "ultra", "vivid_max"]
    imgs = [mosaics[p] for p in order]
    labels = [
        "natural (restrained saturation)",
        "ultra (Hungarian + Laplacian)",
        "vivid_max (MKL optimal transport)",
    ]
    fig = _hstack_with_labels(imgs, labels, target_h=620, gap=12)
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_zoom_detail(mosaic: np.ndarray, out_path: Path) -> None:
    """Two crops of the mosaic at different zoom levels to show tile granularity."""
    h, w = mosaic.shape[:2]
    # Wide crop: center 50% of the image
    wide = mosaic[h // 4 : h * 3 // 4, w // 4 : w * 3 // 4]
    # Close crop: center 18% of the image, upscaled 2x (nearest → crisp tile edges)
    box = int(min(h, w) * 0.18)
    cy, cx = h // 2, w // 2
    close = mosaic[cy - box // 2 : cy + box // 2, cx - box // 2 : cx + box // 2]
    close = cv2.resize(
        close, (close.shape[1] * 2, close.shape[0] * 2), interpolation=cv2.INTER_NEAREST
    )
    target_h = 640
    fig = _hstack_with_labels(
        [wide, close],
        ["Whole mosaic (center 50%)", "Zoom 2x - individual tiles visible"],
        target_h=target_h,
        gap=12,
    )
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_tiles_sample(tiles_dir: Path, out_path: Path, *, rows: int = 4, cols: int = 12) -> None:
    files = sorted(tiles_dir.glob("*.jpg"))[: rows * cols * 3 : 3]  # stride to vary
    if not files:
        return
    sample = [cv2.imread(str(p)) for p in files[: rows * cols]]
    if not sample or any(im is None for im in sample):
        return
    tile_h, tile_w = sample[0].shape[:2]
    canvas = np.full((rows * tile_h + (rows + 1) * 4, cols * tile_w + (cols + 1) * 4, 3), 22, dtype=np.uint8)
    for i, im in enumerate(sample):
        r, c = divmod(i, cols)
        y = 4 + r * (tile_h + 4)
        x = 4 + c * (tile_w + 4)
        canvas[y : y + tile_h, x : x + tile_w] = im
    # Top label
    label = _label_bar(canvas.shape[1], "Procedural tile pool (CC0) - gradients, shapes, stripes, checkers", height=44, font_scale=0.7)
    fig = np.vstack([label, canvas])
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, 88])


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "images")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / ".readme_figures_cache",
        help="Scratch directory for target/tiles/mosaics",
    )
    parser.add_argument("--quick", action="store_true", help="Smaller assets, faster to iterate")
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep the scratch directory (useful for debugging)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(verbose=False)

    if args.quick:
        tgt_w, tgt_h = 1000, 750
        n_tiles = 768
        tile_px = 80
        n_cells = 1200
        cell_px = 56
    else:
        tgt_w, tgt_h = 1600, 1200
        n_tiles = 2048
        tile_px = 96
        n_cells = 3200
        cell_px = 72

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tiles_dir = args.work_dir / "tiles"
    target_path = args.work_dir / "target.jpg"

    # 1. Target
    t0 = time.perf_counter()
    print(f"[1/4] Building landscape target {tgt_w}x{tgt_h} (seed={args.seed}) ...")
    target = make_landscape_target(tgt_w, tgt_h, seed=args.seed)
    cv2.imwrite(str(target_path), target, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"     target ready in {time.perf_counter() - t0:.1f}s")

    # 2. Tiles
    t0 = time.perf_counter()
    print(f"[2/4] Generating {n_tiles} procedural tiles ({tile_px}px) ...")
    build_tile_pool(tiles_dir, count=n_tiles, tile_size=tile_px, seed=args.seed)
    print(f"     tile pool ready in {time.perf_counter() - t0:.1f}s")

    # 3. Mosaics
    mosaics: dict[str, np.ndarray] = {}
    for preset in ["natural", "ultra", "vivid_max"]:
        t0 = time.perf_counter()
        print(f"[3/4] Rendering mosaic (preset={preset}, cells={n_cells}) ...")
        gen = MosaicGenerator(tile_dir=tiles_dir, preset=preset)
        out_path = args.work_dir / f"mosaic_{preset}.jpg"
        result = gen.generate(
            target_path,
            out_path,
            target_tiles=n_cells,
            tile_size=cell_px,
            jpeg_quality=94,
        )
        mosaics[preset] = result.image
        print(
            f"     {preset} ready in {time.perf_counter() - t0:.1f}s"
            f" ({result.grid_cols}x{result.grid_rows} = {result.n_tiles} cells)"
        )

    # 4. Composites
    t0 = time.perf_counter()
    print("[4/4] Composing comparison figures ...")
    make_hero(target, mosaics["ultra"], args.output_dir / "hero.jpg")
    make_before_after(target, mosaics["ultra"], args.output_dir / "before_after.jpg")
    make_presets_comparison(mosaics, args.output_dir / "presets_comparison.jpg")
    make_zoom_detail(mosaics["ultra"], args.output_dir / "zoom_detail.jpg")
    make_tiles_sample(tiles_dir, args.output_dir / "tiles_sample.jpg")
    print(f"     figures ready in {time.perf_counter() - t0:.1f}s")

    if not args.keep_work:
        shutil.rmtree(args.work_dir, ignore_errors=True)

    print()
    print("Wrote:")
    for p in sorted(args.output_dir.glob("*.jpg")):
        size_kb = p.stat().st_size / 1024
        print(f"  {p.relative_to(REPO_ROOT)}   ({size_kb:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
