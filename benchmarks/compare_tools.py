"""Compare mosaicraft against other open-source photomosaic generators.

Runs four tools on the same (target, tile-pool) pair and reports wall time
plus a mix of pixel-level and perceptual quality metrics.

Pixel-level (included for completeness; penalise photomosaics that
intentionally inject tile-level detail):

  * **SSIM** — Structural similarity vs the target (higher is better).
  * **ΔE2000** — CIEDE2000 mean colour error in CIELAB (lower is better).
  * **Edge-corr** — Pearson correlation between target and mosaic Sobel edge
    maps (higher is better).

Perceptual (correlate with human judgement and with how photomosaics are
actually viewed — step back and look):

  * **SSIM_blur** — SSIM after Gaussian blur of both images with σ ≈ 2% of
    the image diagonal. Approximates "view from reading distance"; a
    well-made photomosaic should score high here even if it loses pixel SSIM.
  * **LPIPS (AlexNet)** — Learned Perceptual Image Patch Similarity
    (Zhang et al., CVPR 2018). Deep-feature distance; lower is better.
    LPIPS correlates with human two-alternative forced-choice judgements far
    better than SSIM/PSNR, and is the standard reference metric for
    generative image quality.

Photomosaic-specific:

  * **Cell diversity** — fraction of grid cells whose mean colour is
    distinct after 5-bit quantisation (higher = more of the tile pool is
    actually being used). Photomosaics are meant to *use* the pool.

Tools under test (all invoked from a clean subprocess / fresh Python context):

  1. **mosaicraft — fast preset**   (ours, Oklab + histograms, optimised for speed)
  2. **mosaicraft — vivid preset**  (ours, Oklab + MKL OT + Hungarian + SSIM rerank)
  2b. **mosaicraft — vivid + cv4**  (ours, vivid preset on a 5×-expanded Oklch pool)
  3. **photomosaic 0.3.1** (danielballan, PyPI) — CIELAB + cKDTree
  4. **codebox/mosaic** (git @ codebox/mosaic) — naive RGB mean matching

Run from the repository root after ``scripts/download_demo_assets.py``::

    python benchmarks/compare_tools.py
    python benchmarks/compare_tools.py --target red_fuji.jpg --grid 40
    python benchmarks/compare_tools.py --skip codebox  # if not cloned

Outputs:

  * ``docs/assets/bench_outputs/<tool>.jpg`` — each tool's raw output
  * ``docs/assets/bench_outputs/metrics.json`` — raw numbers
    (consumed by ``scripts/generate_readme_figures.py`` to render
    ``docs/images/diversity_chart.jpg``)
  * stdout — Markdown table for the README comparison section
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity as ssim_func

from mosaicraft import MosaicGenerator, configure_logging

ASSETS_DIR = REPO_ROOT / "docs" / "assets"
PAINTINGS_DIR = ASSETS_DIR / "paintings"
TILES_DIR = ASSETS_DIR / "tiles"
BENCH_DIR = ASSETS_DIR / "bench_outputs"
METRICS_JSON = BENCH_DIR / "metrics.json"

# Where we cloned / expect to find codebox/mosaic.
CODEBOX_REPO = Path("/tmp/oss_compare/codebox-mosaic")

# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _to_uint8_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def ssim_rgb(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    # skimage SSIM with channel_axis is the canonical implementation.
    return float(
        ssim_func(
            a_rgb,
            b_rgb,
            channel_axis=-1,
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )


def delta_e_mean(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    # CIEDE2000 over downsampled arrays so it runs in under a second.
    scale = 4
    small_a = cv2.resize(a_rgb, (a_rgb.shape[1] // scale, a_rgb.shape[0] // scale))
    small_b = cv2.resize(b_rgb, (b_rgb.shape[1] // scale, b_rgb.shape[0] // scale))
    lab_a = rgb2lab(small_a.astype(np.float32) / 255.0)
    lab_b = rgb2lab(small_b.astype(np.float32) / 255.0)
    return float(deltaE_ciede2000(lab_a, lab_b).mean())


def edge_correlation(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    gray_a = cv2.cvtColor(a_rgb, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(b_rgb, cv2.COLOR_RGB2GRAY)
    sobel_a = cv2.magnitude(
        cv2.Sobel(gray_a, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_a, cv2.CV_32F, 0, 1, ksize=3),
    )
    sobel_b = cv2.magnitude(
        cv2.Sobel(gray_b, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(gray_b, cv2.CV_32F, 0, 1, ksize=3),
    )
    # Pearson correlation between the two edge maps.
    fa = sobel_a.ravel() - sobel_a.mean()
    fb = sobel_b.ravel() - sobel_b.mean()
    denom = float(np.linalg.norm(fa) * np.linalg.norm(fb))
    if denom == 0:
        return 0.0
    return float(np.dot(fa, fb) / denom)


def cell_diversity(mosaic_bgr: np.ndarray, n_grid: int) -> float:
    """Fraction of grid cells whose mean BGR is distinct (5-bit quantized).

    Higher values indicate the mosaic uses a wider variety of tiles per cell,
    instead of repeating a small color palette. This is the metric where a
    photomosaic-specific tool (1:1 tile assignment, no reuse) dramatically
    outperforms naive colour-matchers that pick the same "best" tile over and
    over when the target is dominated by one colour.
    """
    h, w = mosaic_bgr.shape[:2]
    cell_h, cell_w = h // n_grid, w // n_grid
    if cell_h < 2 or cell_w < 2:
        return 0.0
    means: set[tuple[int, int, int]] = set()
    total = 0
    for i in range(n_grid):
        for j in range(n_grid):
            cell = mosaic_bgr[i * cell_h : (i + 1) * cell_h, j * cell_w : (j + 1) * cell_w]
            b, g, r = (int(v) for v in cell.mean(axis=(0, 1)))
            # 5-bit quantization → 32^3 = 32768 buckets, small perceptual delta.
            means.add((b >> 3, g >> 3, r >> 3))
            total += 1
    return len(means) / total if total else 0.0


def blurred_ssim_rgb(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """SSIM computed after Gaussian blurring both target and mosaic.

    Photomosaics are designed to be recognisable at reading distance, not
    pixel-identical at 100% zoom. We blur with σ ≈ 2% of the longest side —
    the same rule graphic designers use when simulating "step back and look"
    — and compute SSIM on the blurred pair. This is the evaluation mode that
    best matches how a photomosaic is actually consumed.
    """
    sigma = max(1.0, 0.02 * float(max(a_rgb.shape[:2])))
    ksize = int(2 * round(3 * sigma) + 1)
    a_blur = cv2.GaussianBlur(a_rgb, (ksize, ksize), sigma)
    b_blur = cv2.GaussianBlur(b_rgb, (ksize, ksize), sigma)
    return ssim_rgb(a_blur, b_blur)


# Lazy singleton — torch and lpips are heavy; only initialise if the metric
# is actually requested.
_LPIPS_MODEL: Any = None


def _get_lpips_model() -> Any:
    global _LPIPS_MODEL
    if _LPIPS_MODEL is None:
        import lpips  # type: ignore[import-untyped]
        import torch  # type: ignore[import-untyped]

        # net='alex' is the fast/accurate combination recommended by the
        # original paper (Zhang et al. 2018) for image quality benchmarking.
        model = lpips.LPIPS(net="alex", verbose=False)
        model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        _LPIPS_MODEL = model
    return _LPIPS_MODEL


def lpips_distance(a_rgb: np.ndarray, b_rgb: np.ndarray) -> float:
    """LPIPS (AlexNet) distance between two RGB uint8 images.

    Returns a float in ``[0, ~1]`` — 0 means identical, higher is worse.
    LPIPS correlates with human two-alternative forced-choice perceptual
    judgements far better than SSIM or PSNR (Zhang et al., CVPR 2018:
    https://arxiv.org/abs/1801.03924).

    Operates on 256-px downsampled copies for speed; the conv features are
    translation-invariant so this is a negligible accuracy loss.
    """
    try:
        import torch  # type: ignore[import-untyped]
    except ImportError:
        return float("nan")

    model = _get_lpips_model()
    # Resize to a fixed 256 so the metric is comparable across image sizes.
    target_side = 256
    small_a = cv2.resize(a_rgb, (target_side, target_side), interpolation=cv2.INTER_AREA)
    small_b = cv2.resize(b_rgb, (target_side, target_side), interpolation=cv2.INTER_AREA)

    def _prep(x: np.ndarray) -> Any:
        t = torch.from_numpy(x).float().permute(2, 0, 1)[None] / 127.5 - 1.0
        return t

    with torch.no_grad():
        dist = model(_prep(small_a), _prep(small_b)).item()
    return float(dist)


def compute_metrics(
    target_bgr: np.ndarray, mosaic_bgr: np.ndarray, n_grid: int
) -> dict[str, float]:
    # Resize mosaic to target dims for fair pixel-wise comparison.
    h, w = target_bgr.shape[:2]
    resized = cv2.resize(mosaic_bgr, (w, h), interpolation=cv2.INTER_AREA)
    a = _to_uint8_rgb(target_bgr)
    b = _to_uint8_rgb(resized)
    return {
        "ssim": ssim_rgb(a, b),
        "ssim_blurred": blurred_ssim_rgb(a, b),
        "delta_e2000_mean": delta_e_mean(a, b),
        "edge_corr": edge_correlation(a, b),
        "cell_diversity": cell_diversity(mosaic_bgr, n_grid),
        "lpips": lpips_distance(a, b),
    }


# --------------------------------------------------------------------------- #
# Tool runners — each returns (elapsed_seconds, output_path)
# --------------------------------------------------------------------------- #


@dataclass
class ToolResult:
    tool_id: str
    tool_label: str
    output_path: Path | None = None
    elapsed_sec: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None


def run_mosaicraft(
    preset: str,
    target_path: Path,
    tiles_dir: Path,
    grid_tiles: int,
    out_path: Path,
    *,
    color_variants: int = 0,
) -> tuple[float, Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gen = MosaicGenerator(
        tile_dir=tiles_dir,
        preset=preset,
        color_variants=color_variants,
    )
    t0 = time.perf_counter()
    gen.generate(target_path, out_path, target_tiles=grid_tiles, tile_size=64)
    return time.perf_counter() - t0, out_path


def _patch_photomosaic_for_modern_stack() -> None:
    """Monkey-patch photomosaic 0.3.1 for numpy 2.x / scikit-image > 0.19.

    Two upstream bugs bite when you use photomosaic 0.3.1 (2018) with a
    modern Python stack:

    1. ``crop_to_fit`` passes np.float64 values to skimage.util.crop, which
       rejects them as slice indices.
    2. ``np.product`` was removed in numpy 2.0 (use ``np.prod``); photomosaic
       still calls ``np.product`` in several helpers.

    We patch both so the comparison can actually run.
    """
    import photomosaic.photomosaic as pmm
    from skimage.transform import resize

    if getattr(pmm, "_mosaicraft_patched", False):
        return

    # 1) restore np.product as alias for np.prod.
    if not hasattr(np, "product"):
        np.product = np.prod  # type: ignore[attr-defined]

    # 2) re-implement crop_to_fit with explicit int casts.
    def _patched_crop_to_fit(image: np.ndarray, shape: Any) -> np.ndarray:
        h, w = image.shape[:2]
        target_h, target_w = int(shape[0]), int(shape[1])
        if h >= target_h and w >= target_w:
            top = (h - target_h) // 2
            left = (w - target_w) // 2
            return image[top : top + target_h, left : left + target_w]
        scale = max(target_h / h, target_w / w)
        new_h = int(np.ceil(h * scale))
        new_w = int(np.ceil(w * scale))
        new_shape = (new_h, new_w, *tuple(image.shape[2:]))
        resized = resize(image, new_shape, mode="constant", anti_aliasing=False)
        return _patched_crop_to_fit(resized, shape)

    pmm.crop_to_fit = _patched_crop_to_fit
    pmm._mosaicraft_patched = True


def run_photomosaic_pypi(
    target_path: Path,
    tiles_dir: Path,
    grid_dims: tuple[int, int],
    out_path: Path,
) -> tuple[float, Path]:
    """Use danielballan/photomosaic (PyPI 0.3.1) via its Python API.

    We pre-resize the target to a size exactly commensurate with
    ``grid_dims`` (i.e. ``grid × k`` for some integer ``k``) so that
    ``rescale_commensurate`` is a no-op and our monkey-patched crop_to_fit
    is not exercised on marginal float/int crops. Empirically this is what
    the difference between "readable face" and "random chessboard" output
    hinges on.
    """
    import photomosaic as pm

    _patch_photomosaic_for_modern_stack()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-resize target: choose k so the shorter side of the pre-sized image
    # is just above the original while remaining divisible by grid.
    raw = cv2.imread(str(target_path))
    h, w = raw.shape[:2]
    gh, gw = grid_dims
    cell_h = max(1, int(np.ceil(h / gh)))
    cell_w = max(1, int(np.ceil(w / gw)))
    new_h, new_w = cell_h * gh, cell_w * gw
    sized = cv2.resize(raw, (new_w, new_h), interpolation=cv2.INTER_AREA)
    work_target = out_path.with_suffix(".target.jpg")
    cv2.imwrite(str(work_target), sized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    t0 = time.perf_counter()
    pool = pm.make_pool(str(tiles_dir / "*.jpg"))
    target_img = pm.imread(str(work_target))
    if target_img.dtype != np.uint8:
        target_img = (np.clip(target_img, 0, 1) * 255).astype(np.uint8)
    mosaic = pm.basic_mosaic(target_img, pool, grid_dims=grid_dims, depth=0)
    elapsed = time.perf_counter() - t0

    if mosaic.dtype != np.uint8:
        mosaic_u8 = (np.clip(mosaic, 0, 1) * 255).astype(np.uint8)
    else:
        mosaic_u8 = mosaic
    if mosaic_u8.ndim == 3 and mosaic_u8.shape[2] == 3:
        cv2.imwrite(str(out_path), cv2.cvtColor(mosaic_u8, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(out_path), mosaic_u8)
    work_target.unlink(missing_ok=True)
    return elapsed, out_path


def run_codebox_mosaic(
    target_path: Path,
    tiles_dir: Path,
    out_path: Path,
    grid: int,
) -> tuple[float, Path]:
    """codebox/mosaic has constants hardcoded at the top of mosaic.py; we patch
    them in a temporary copy so (a) the output matches the requested *grid*
    side length, and (b) the output isn't 8× the target size (the upstream
    default, which is huge for a 2048px painting).

    We achieve a ``grid × grid`` cell layout by pre-resizing the target to
    ``(grid * 32) × (grid * 32)`` pixels and running codebox with
    ``TILE_SIZE = 32`` and ``ENLARGEMENT = 1``.
    """
    if not CODEBOX_REPO.exists():
        raise FileNotFoundError(
            f"codebox/mosaic not cloned at {CODEBOX_REPO} — run: "
            f"git clone --depth 1 https://github.com/codebox/mosaic.git {CODEBOX_REPO}"
        )
    work = BENCH_DIR / "_codebox_work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True, exist_ok=True)

    tile_size = 32
    side = grid * tile_size
    target_img = cv2.imread(str(target_path))
    if target_img is None:
        raise RuntimeError(f"codebox: could not read target {target_path}")
    resized = cv2.resize(target_img, (side, side), interpolation=cv2.INTER_AREA)
    codebox_target = work / "target.jpg"
    cv2.imwrite(str(codebox_target), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

    src = (CODEBOX_REPO / "mosaic.py").read_text()
    patched = src.replace(
        "TILE_SIZE      = 50", f"TILE_SIZE      = {tile_size}"
    ).replace("ENLARGEMENT    = 8", "ENLARGEMENT    = 1")
    (work / "mosaic.py").write_text(patched)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "mosaic.py", str(codebox_target.resolve()), str(tiles_dir.resolve())],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"codebox/mosaic failed ({proc.returncode}): {proc.stderr[-500:] or proc.stdout[-500:]}"
        )
    produced = work / "mosaic.jpeg"
    if not produced.exists():
        raise FileNotFoundError(
            f"codebox/mosaic did not produce {produced}; stdout tail: {proc.stdout[-300:]}"
        )
    shutil.move(str(produced), out_path)
    return elapsed, out_path


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


TOOLS: list[dict[str, Any]] = [
    {
        "id": "codebox",
        "label": "codebox/mosaic\n(naive RGB mean)",
        "runner": "codebox",
    },
    {
        "id": "photomosaic_py",
        "label": "photomosaic 0.3.1\n(CIELAB + kd-tree)",
        "runner": "photomosaic_py",
    },
    {
        "id": "mosaicraft_fast",
        "label": "mosaicraft fast\n(Oklab + histograms)",
        "runner": "mosaicraft_fast",
    },
    {
        "id": "mosaicraft_vivid",
        "label": "mosaicraft vivid\n(MKL optimal transport)",
        "runner": "mosaicraft_vivid",
    },
    {
        "id": "mosaicraft_vivid_cv4",
        "label": "mosaicraft vivid + cv4\n(MKL + Oklch pool x5)",
        "runner": "mosaicraft_vivid_cv4",
    },
]


def make_runner(
    runner_id: str,
    target_path: Path,
    grid: int,
    out_dir: Path,
) -> tuple[str, Callable[[], tuple[float, Path]]]:
    out_path = out_dir / f"{runner_id}.jpg"
    if runner_id == "codebox":
        return "codebox", lambda: run_codebox_mosaic(target_path, TILES_DIR, out_path, grid)
    if runner_id == "photomosaic_py":
        return "photomosaic_py", lambda: run_photomosaic_pypi(
            target_path, TILES_DIR, (grid, grid), out_path
        )
    if runner_id == "mosaicraft_fast":
        return "mosaicraft_fast", lambda: run_mosaicraft(
            "fast", target_path, TILES_DIR, grid * grid, out_path
        )
    if runner_id == "mosaicraft_vivid":
        return "mosaicraft_vivid", lambda: run_mosaicraft(
            "vivid", target_path, TILES_DIR, grid * grid, out_path
        )
    if runner_id == "mosaicraft_vivid_cv4":
        return "mosaicraft_vivid_cv4", lambda: run_mosaicraft(
            "vivid", target_path, TILES_DIR, grid * grid, out_path, color_variants=4
        )
    raise ValueError(f"unknown runner {runner_id}")


# --------------------------------------------------------------------------- #
# Markdown report
# --------------------------------------------------------------------------- #


def format_markdown_table(target_name: str, grid: int, results: list[ToolResult]) -> str:
    lines = [
        f"### Photomosaic tool comparison — target = {target_name}, grid = {grid}×{grid}",
        "",
        "| tool | time | LPIPS ↓ | SSIM_blur ↑ | SSIM ↑ | ΔE2000 ↓ | edge ↑ | diversity ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        if r.error:
            lines.append(
                f"| {r.tool_id} | — | — | — | — | — | — | — _(error: {r.error[:60]})_ |"
            )
            continue
        lines.append(
            "| {tool} | {t:.1f}s | {l:.3f} | {sb:.3f} | {s:.3f} | {d:.2f} | {e:.3f} | {v:.3f} |".format(
                tool=r.tool_id.replace("_", " "),
                t=r.elapsed_sec,
                l=r.metrics.get("lpips", float("nan")),
                sb=r.metrics.get("ssim_blurred", float("nan")),
                s=r.metrics.get("ssim", float("nan")),
                d=r.metrics.get("delta_e2000_mean", float("nan")),
                e=r.metrics.get("edge_corr", float("nan")),
                v=r.metrics.get("cell_diversity", float("nan")),
            )
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def prepare_target(target_name: str, side_px: int = 1024) -> tuple[Path, np.ndarray]:
    """Center-crop the painting to a square and resize to *side_px*.

    A square target means every tool gets the same effective grid (``grid × grid``)
    since mosaicraft picks a grid shape from the target aspect ratio. Without
    squaring, mosaicraft ends up with fewer cells than photomosaic / codebox
    for a landscape painting, which is an apples-to-oranges comparison.
    """
    src = PAINTINGS_DIR / target_name
    if not src.exists():
        raise SystemExit(f"target {src} missing — run scripts/download_demo_assets.py first")
    img = cv2.imread(str(src))
    if img is None:
        raise SystemExit(f"could not read {src}")
    h, w = img.shape[:2]
    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    cropped = img[top : top + side, left : left + side]
    img = cv2.resize(cropped, (side_px, side_px), interpolation=cv2.INTER_AREA)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    dst = BENCH_DIR / f"target_{Path(target_name).stem}.jpg"
    cv2.imwrite(str(dst), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return dst, img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--target",
        default="red_fuji.jpg",
        choices=["starry_night.jpg", "great_wave.jpg", "red_fuji.jpg", "pearl_earring.jpg"],
        help="painting to use as the comparison target",
    )
    p.add_argument("--grid", type=int, default=40, help="grid side length (cells)")
    p.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="tool ids to skip, e.g. --skip codebox photomosaic_py",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(verbose=False)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    print(f"preparing target: {args.target}")
    target_path, target_bgr = prepare_target(args.target)
    print(f"  → {target_path.relative_to(REPO_ROOT)}  ({target_bgr.shape[1]}x{target_bgr.shape[0]})")

    print(f"tile pool:        {TILES_DIR.relative_to(REPO_ROOT)}")
    n_tiles = len(list(TILES_DIR.glob("*.jpg")))
    print(f"  → {n_tiles} tiles")

    grid = args.grid
    results: list[ToolResult] = []
    for tool in TOOLS:
        tid = tool["id"]
        result = ToolResult(tool_id=tid, tool_label=tool["label"])
        if tid in args.skip:
            result.error = "skipped"
            print(f"\n[{tid}] skipped")
            results.append(result)
            continue

        print(f"\n[{tid}] running ...")
        _runner_id, runner = make_runner(tid, target_path, grid, BENCH_DIR)
        try:
            elapsed, out_path = runner()
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc(limit=3)
            result.error = str(e)[:200]
            results.append(result)
            continue

        result.elapsed_sec = elapsed
        result.output_path = out_path
        mosaic_bgr = cv2.imread(str(out_path))
        if mosaic_bgr is None:
            result.error = f"cv2.imread failed on {out_path}"
            results.append(result)
            continue
        result.metrics = compute_metrics(target_bgr, mosaic_bgr, grid)
        print(
            "  done: {t:.1f}s  LPIPS={l:.3f}  SSIM_blur={sb:.3f}  SSIM={s:.3f}  ΔE={d:.2f}  div={v:.3f}".format(
                t=elapsed,
                l=result.metrics["lpips"],
                sb=result.metrics["ssim_blurred"],
                s=result.metrics["ssim"],
                d=result.metrics["delta_e2000_mean"],
                v=result.metrics["cell_diversity"],
            )
        )
        results.append(result)

    # Save metrics JSON. (Composite figure rendering moved out of this
    # benchmark script — `scripts/generate_readme_figures.py` now reads
    # METRICS_JSON and renders `docs/images/diversity_chart.jpg` from it,
    # so the bench script doesn't have to know anything about README
    # layout.)
    METRICS_JSON.write_text(
        json.dumps(
            {
                "target": args.target,
                "grid": grid,
                "tile_pool_size": n_tiles,
                "results": [asdict(r) | {"output_path": str(r.output_path) if r.output_path else None}
                            for r in results],
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    print(f"  wrote {METRICS_JSON.relative_to(REPO_ROOT)}")

    # Print Markdown table.
    print()
    print(format_markdown_table(args.target, grid, results))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
