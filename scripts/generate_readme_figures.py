#!/usr/bin/env python3
"""Generate README figures for mosaicraft.

Loads public-domain paintings from Wikimedia Commons and a CC0 tile pool
from picsum.photos (both bootstrapped by
``scripts/download_demo_assets.py``), then renders mosaics with several
presets and composites the README comparison figures into
``docs/images/``.

Every source image is freely redistributable:

* Paintings — public domain (pre-1929, Wikimedia Commons)
* Tiles    — CC0 / Unsplash License via picsum.photos

Usage::

    # one-time bootstrap (downloads ~8 MB into docs/assets/)
    python scripts/download_demo_assets.py

    # then render figures
    python scripts/generate_readme_figures.py
    python scripts/generate_readme_figures.py --target pearl_earring
    python scripts/generate_readme_figures.py --quick  # fewer cells, faster iteration
"""

from __future__ import annotations

import argparse
import json
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

ASSETS_DIR = REPO_ROOT / "docs" / "assets"
PAINTINGS_DIR = ASSETS_DIR / "paintings"
TILES_DIR = ASSETS_DIR / "tiles"
MANIFEST_PATH = ASSETS_DIR / "MANIFEST.json"


# --------------------------------------------------------------------------- #
# Asset loading.
# --------------------------------------------------------------------------- #

# Hero caption text is pulled from MANIFEST.json so it stays in sync with the
# licensing metadata. Keys here must match the ``name`` field in
# ``scripts/download_demo_assets.py``'s ``PAINTINGS`` list.
TARGET_CHOICES = {
    "pearl_earring": "pearl_earring.jpg",
}


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            "docs/assets/MANIFEST.json not found. "
            "Run `python scripts/download_demo_assets.py` first."
        )
    return json.loads(MANIFEST_PATH.read_text())


def painting_caption(manifest: dict, filename: str) -> str:
    """Return a short attribution string like 'Vermeer - Girl with a Pearl Earring (PD)'."""
    for entry in manifest.get("paintings", []):
        if entry["path"].endswith(filename):
            artist_last = entry["artist"].split()[-1]
            return f"{artist_last} - {entry['title']} (PD)"
    return filename


def load_target(filename: str, max_side: int = 1600) -> np.ndarray:
    """Load a painting and resize so its longest side equals *max_side*."""
    src = PAINTINGS_DIR / filename
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Run `python scripts/download_demo_assets.py` first."
        )
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"failed to decode {src}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        new_w = max(1, round(w * scale))
        new_h = max(1, round(h * scale))
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


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
    min_pad: int = 12,
) -> np.ndarray:
    """Render a label bar that auto-shrinks the font when the text overflows.

    Bug history: the previous version computed ``x = (width - tw) // 2`` and
    silently let cv2.putText clip the left edge when ``tw > width`` —
    every "Target: Vermeer..." caption was rendering as "t: Vermeer...".
    The fix shrinks ``font_scale`` until the text fits with at least
    ``min_pad`` pixels on each side, falling back to a hard floor at
    ``font_scale = 0.35`` so labels stay legible.
    """
    bar = np.full((height, width, 3), bg, dtype=np.uint8)
    available = max(1, width - 2 * min_pad)
    fs = float(font_scale)
    while True:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
        if tw <= available or fs <= 0.35:
            break
        fs = max(0.35, fs * 0.9)
    x = max(min_pad, (width - tw) // 2)
    y = (height + th) // 2 - 2
    cv2.putText(
        bar,
        text,
        (x, y),
        cv2.FONT_HERSHEY_DUPLEX,
        fs,
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


def make_hero(
    target: np.ndarray, mosaic: np.ndarray, out_path: Path, *, target_caption: str
) -> None:
    """Side-by-side target vs mosaic for the top of the README."""
    fig = _hstack_with_labels(
        [target, mosaic],
        [f"Target: {target_caption}", "mosaicraft output - preset: vivid"],
        target_h=720,
        gap=14,
    )
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_before_after(
    target: np.ndarray, mosaic: np.ndarray, out_path: Path
) -> None:
    fig = _hstack_with_labels(
        [target, mosaic],
        ["Before (original painting)", "After - preset: vivid"],
        target_h=640,
        gap=12,
    )
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_presets_comparison(
    mosaics: dict[str, np.ndarray], out_path: Path
) -> None:
    """5-up preset comparison covering every preset shipped in `presets.py`."""
    order = ["vivid", "ultra", "natural", "tile", "fast"]
    imgs: list[np.ndarray] = []
    labels: list[str] = []
    label_map = {
        "vivid":   "vivid (MKL OT, recommended)",
        "ultra":   "ultra (Hungarian + Laplacian)",
        "natural": "natural (restrained saturation)",
        "tile":    "tile (mosaic look)",
        "fast":    "fast (FAISS + error diffusion)",
    }
    for key in order:
        if key in mosaics:
            imgs.append(mosaics[key])
            labels.append(label_map[key])
    if not imgs:
        return
    fig = _hstack_with_labels(imgs, labels, target_h=540, gap=10)
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


def make_tiles_sample(
    tiles_dir: Path, out_path: Path, *, rows: int = 4, cols: int = 12
) -> None:
    files = sorted(tiles_dir.glob("*.jpg"))
    if not files:
        return
    # Stride through the pool so the sample shows visual variety.
    stride = max(1, len(files) // (rows * cols))
    picked = files[::stride][: rows * cols]
    sample = [cv2.imread(str(p)) for p in picked]
    if not sample or any(im is None for im in sample):
        return
    tile_h, tile_w = sample[0].shape[:2]
    canvas = np.full(
        (rows * tile_h + (rows + 1) * 4, cols * tile_w + (cols + 1) * 4, 3),
        22,
        dtype=np.uint8,
    )
    for i, im in enumerate(sample):
        r, c = divmod(i, cols)
        y = 4 + r * (tile_h + 4)
        x = 4 + c * (tile_w + 4)
        canvas[y : y + tile_h, x : x + tile_w] = im
    label = _label_bar(
        canvas.shape[1],
        "CC0 tile pool (Unsplash License via picsum.photos) - 1024 photographs",
        height=44,
        font_scale=0.7,
    )
    fig = np.vstack([label, canvas])
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, 88])


def make_target_comparison(
    targets: dict[str, np.ndarray],
    mosaics: dict[str, np.ndarray],
    out_path: Path,
) -> None:
    """Before/after grid for each target image."""
    panels: list[np.ndarray] = []
    labels: list[str] = []
    for key in TARGET_CHOICES:
        if key not in targets or key not in mosaics:
            continue
        panels.append(targets[key])
        panels.append(mosaics[key])
        pretty = key.replace("_", " ").title()
        labels.append(f"{pretty} - Original")
        labels.append(f"{pretty} - mosaicraft vivid")
    if not panels:
        return

    target_h = 360
    resized = [_fit_height(im, target_h) for im in panels]
    bar_h = 44
    gap = 8
    rows = []
    for i in range(0, len(resized), 2):
        pair = resized[i : i + 2]
        labs = labels[i : i + 2]
        widths = [im.shape[1] for im in pair]
        row_w = sum(widths) + gap * 3
        row = np.full((bar_h + target_h + gap, row_w, 3), 22, dtype=np.uint8)
        x = gap
        for im, lab in zip(pair, labs):
            w = im.shape[1]
            row[0:bar_h, x : x + w] = _label_bar(w, lab, height=bar_h, font_scale=0.7)
            row[bar_h : bar_h + target_h, x : x + w] = im
            x += w + gap
        rows.append(row)
    # Pad rows to same width.
    max_w = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 22, dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)
    fig = np.vstack(padded)
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


_DIVERSITY_BAR_COLORS = {
    "codebox":              (180, 180, 180),
    "photomosaic_py":       (200, 200, 200),
    "mosaicraft_fast":      (140, 200, 120),
    "mosaicraft_vivid":     (120, 180, 240),
    "mosaicraft_vivid_cv4": (80, 120, 250),
}

_DIVERSITY_BAR_LABELS = {
    "codebox":              "codebox/mosaic",
    "photomosaic_py":       "photomosaic 0.3.1",
    "mosaicraft_fast":      "mosaicraft fast",
    "mosaicraft_vivid":     "mosaicraft vivid",
    "mosaicraft_vivid_cv4": "mosaicraft vivid + cv4",
}


def _load_diversity_bars(metrics_path: Path) -> tuple[list[tuple[str, float, tuple[int, int, int]]], str]:
    """Read metrics.json and return bar tuples + the target label.

    Returns ``([], "?")`` when the file is missing or unreadable, so the
    caller can decide whether to skip the chart instead of crashing the
    whole figure pipeline.
    """
    if not metrics_path.exists():
        return [], "?"
    try:
        payload = json.loads(metrics_path.read_text())
    except (OSError, ValueError):
        return [], "?"
    bars: list[tuple[str, float, tuple[int, int, int]]] = []
    for entry in payload.get("results", []):
        tool_id = entry.get("tool_id", "")
        diversity = float(entry.get("metrics", {}).get("cell_diversity", 0.0))
        label = _DIVERSITY_BAR_LABELS.get(tool_id, tool_id)
        color = _DIVERSITY_BAR_COLORS.get(tool_id, (160, 160, 160))
        bars.append((label, diversity, color))
    return bars, str(payload.get("target", "?"))


def make_diversity_chart(out_path: Path) -> bool:
    """Cell-diversity bar chart, rendered from the live benchmark JSON.

    Returns ``True`` if the chart was rendered, ``False`` if the metrics
    file is missing (in which case the caller should log a warning and
    continue with the rest of the figure pipeline). The previous version
    raised ``SystemExit`` from inside the figure pipeline, which killed
    every other figure if the bench had not been run.
    """
    metrics_path = REPO_ROOT / "docs" / "assets" / "bench_outputs" / "metrics.json"
    bars, target_label = _load_diversity_bars(metrics_path)
    if not bars:
        return False

    width = 1180
    height = 480
    pad_left = 320
    pad_right = 80
    pad_top = 80
    pad_bot = 70
    bar_area_w = width - pad_left - pad_right
    bar_area_h = height - pad_top - pad_bot
    canvas = np.full((height, width, 3), 22, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"Cell diversity on {target_label} (higher is better)",
        (32, 40),
        cv2.FONT_HERSHEY_DUPLEX, 0.7, (240, 240, 240), 1, cv2.LINE_AA,
    )

    # Auto-scale the x axis to the next 0.1 above the largest bar, with
    # at least 15% headroom so the value label never collides with the
    # right edge. Clamped to [0.10, 1.0].
    max_val = max(v for _, v, _ in bars)
    x_max = (np.ceil(max_val * 10 + 1) / 10).item()
    x_max = float(min(1.0, max(0.10, x_max, max_val * 1.15)))

    n = len(bars)
    bar_h = max(20, bar_area_h // n - 16)
    for i, (name, val, color) in enumerate(bars):
        y = pad_top + i * (bar_h + 16)
        bar_w = int(val * bar_area_w / x_max)
        cv2.line(
            canvas,
            (pad_left, y + bar_h + 8),
            (width - pad_right, y + bar_h + 8),
            (60, 60, 60), 1,
        )
        cv2.rectangle(
            canvas,
            (pad_left, y),
            (pad_left + bar_w, y + bar_h),
            color, -1,
        )
        cv2.putText(
            canvas, name, (16, y + bar_h - 9),
            cv2.FONT_HERSHEY_DUPLEX, 0.55, (240, 240, 240), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, f"{val * 100:.1f}%",
            (pad_left + bar_w + 10, y + bar_h - 7),
            cv2.FONT_HERSHEY_DUPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        "source: docs/assets/bench_outputs/metrics.json (compare_tools.py)",
        (16, height - 18),
        cv2.FONT_HERSHEY_DUPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA,
    )
    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return True


# --------------------------------------------------------------------------- #
# Driver.
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_CHOICES.keys()),
        default="pearl_earring",
        help="Which public-domain painting to feature as the hero target",
    )
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "docs" / "images")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / ".readme_figures_cache",
        help="Scratch directory for the resized target / intermediate mosaics",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer cells, faster to iterate (for layout tweaks)",
    )
    parser.add_argument(
        "--keep-work",
        action="store_true",
        help="Keep the scratch directory (useful for debugging)",
    )
    parser.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip the target comparison grid (faster iteration)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(verbose=False)
    manifest = load_manifest()

    if args.quick:
        max_side = 1000
        n_cells = 1200
        cell_px = 56
    else:
        max_side = 1600
        n_cells = 3200
        cell_px = 72

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_file = TARGET_CHOICES[args.target]
    target_path = args.work_dir / f"target_{args.target}.jpg"

    # 1. Target
    t0 = time.perf_counter()
    print(f"[1/4] Loading target painting: {target_file} ...")
    target = load_target(target_file, max_side=max_side)
    cv2.imwrite(str(target_path), target, [cv2.IMWRITE_JPEG_QUALITY, 94])
    caption = painting_caption(manifest, target_file)
    print(f"     target {target.shape[1]}x{target.shape[0]} ready in {time.perf_counter() - t0:.1f}s")
    print(f"     caption: {caption}")

    # 2. Tiles — pulled straight from the committed CC0 pool.
    n_tiles = sum(1 for _ in TILES_DIR.glob("*.jpg"))
    if n_tiles == 0:
        raise SystemExit(
            f"no tiles in {TILES_DIR}. "
            "Run `python scripts/download_demo_assets.py` first."
        )
    print(f"[2/4] Using {n_tiles} CC0 tiles from {TILES_DIR.relative_to(REPO_ROOT)}")

    # 3. Mosaics — render every preset shipped in `presets.py` so the
    #    presets_comparison figure isn't a partial subset of the README table.
    mosaics: dict[str, np.ndarray] = {}
    for preset in ["vivid", "ultra", "natural", "tile", "fast"]:
        t0 = time.perf_counter()
        print(f"[3/4] Rendering mosaic (preset={preset}, cells={n_cells}) ...")
        gen = MosaicGenerator(tile_dir=TILES_DIR, preset=preset)
        out_path = args.work_dir / f"mosaic_{args.target}_{preset}.jpg"
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

    # 3b. Optional target comparison batch.
    multi_mosaics: dict[str, np.ndarray] = {args.target: mosaics["vivid"]}
    multi_targets: dict[str, np.ndarray] = {args.target: target}
    if not args.skip_grid:
        small_cells = max(800, n_cells // 4)
        small_px = max(48, cell_px - 16)
        for key, fname in TARGET_CHOICES.items():
            if key == args.target:
                continue
            t0g = time.perf_counter()
            print(f"[3b] Rendering {key} (vivid, {small_cells} cells) ...")
            t_img = load_target(fname, max_side=1000)
            tmp_target = args.work_dir / f"target_{key}.jpg"
            cv2.imwrite(str(tmp_target), t_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            gen = MosaicGenerator(tile_dir=TILES_DIR, preset="vivid")
            res = gen.generate(
                tmp_target,
                args.work_dir / f"mosaic_{key}_vivid.jpg",
                target_tiles=small_cells,
                tile_size=small_px,
                jpeg_quality=92,
            )
            multi_targets[key] = t_img
            multi_mosaics[key] = res.image
            print(f"     {key} ready in {time.perf_counter() - t0g:.1f}s")

    # 4. Composites
    t0 = time.perf_counter()
    print("[4/4] Composing figures ...")
    make_hero(
        target,
        mosaics["vivid"],
        args.output_dir / "hero.jpg",
        target_caption=caption,
    )
    make_before_after(target, mosaics["vivid"], args.output_dir / "before_after.jpg")
    make_presets_comparison(mosaics, args.output_dir / "presets_comparison.jpg")
    make_zoom_detail(mosaics["vivid"], args.output_dir / "zoom_detail.jpg")
    make_tiles_sample(TILES_DIR, args.output_dir / "tiles_sample.jpg")
    if not args.skip_grid and len(multi_mosaics) >= 2:
        make_target_comparison(
            multi_targets, multi_mosaics,
            args.output_dir / "target_comparison.jpg",
        )
    if not make_diversity_chart(args.output_dir / "diversity_chart.jpg"):
        print(
            "  WARNING: docs/assets/bench_outputs/metrics.json not found; "
            "diversity_chart.jpg skipped. "
            "Run `python benchmarks/compare_tools.py --target pearl_earring.jpg "
            "--grid 40` to generate it."
        )
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
