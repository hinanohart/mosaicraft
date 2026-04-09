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
    "starry_night": "starry_night.jpg",
    "great_wave": "great_wave.jpg",
    "red_fuji": "red_fuji.jpg",
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


def make_hero(
    target: np.ndarray, mosaic: np.ndarray, out_path: Path, *, target_caption: str
) -> None:
    """Side-by-side target vs mosaic for the top of the README."""
    fig = _hstack_with_labels(
        [target, mosaic],
        [f"Target: {target_caption}", "mosaicraft output - preset: ultra"],
        target_h=720,
        gap=14,
    )
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


def make_before_after(
    target: np.ndarray, mosaic: np.ndarray, out_path: Path
) -> None:
    fig = _hstack_with_labels(
        [target, mosaic],
        ["Before (original painting)", "After - preset: ultra"],
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


def make_paintings_gallery(
    manifest: dict, out_path: Path, *, target_h: int = 520
) -> None:
    """4-up gallery of all public-domain paintings with artist labels."""
    entries = manifest.get("paintings", [])
    if not entries:
        return
    imgs = []
    labels = []
    for entry in entries:
        src = ASSETS_DIR / entry["path"]
        im = cv2.imread(str(src))
        if im is None:
            continue
        imgs.append(im)
        artist_last = entry["artist"].split()[-1]
        labels.append(f"{artist_last} - {entry['title']}")
    if not imgs:
        return
    fig = _hstack_with_labels(imgs, labels, target_h=target_h, gap=10)
    cv2.imwrite(str(out_path), fig, [cv2.IMWRITE_JPEG_QUALITY, _JPEG_Q])


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

    # 3. Mosaics
    mosaics: dict[str, np.ndarray] = {}
    for preset in ["natural", "ultra", "vivid_max"]:
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

    # 4. Composites
    t0 = time.perf_counter()
    print("[4/4] Composing figures ...")
    make_hero(
        target,
        mosaics["ultra"],
        args.output_dir / "hero.jpg",
        target_caption=caption,
    )
    make_before_after(target, mosaics["ultra"], args.output_dir / "before_after.jpg")
    make_presets_comparison(mosaics, args.output_dir / "presets_comparison.jpg")
    make_zoom_detail(mosaics["ultra"], args.output_dir / "zoom_detail.jpg")
    make_tiles_sample(TILES_DIR, args.output_dir / "tiles_sample.jpg")
    make_paintings_gallery(manifest, args.output_dir / "paintings_gallery.jpg")
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
