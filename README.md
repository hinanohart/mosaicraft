# mosaicraft

**A Python photomosaic generator built on the Oklab perceptual color space, MKL optimal transport, Laplacian pyramid blending, and Oklch tile-pool expansion.**

[![PyPI version](https://img.shields.io/pypi/v/mosaicraft.svg)](https://pypi.org/project/mosaicraft/)
[![Python](https://img.shields.io/pypi/pyversions/mosaicraft.svg)](https://pypi.org/project/mosaicraft/)
[![CI](https://github.com/hinanohart/mosaicraft/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/mosaicraft/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hinanohart/mosaicraft/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

![Target vs mosaicraft output](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/hero.jpg)

---

`mosaicraft` rebuilds a target image as a grid of smaller tile photographs. Most photomosaic libraries use mean-color matching in RGB or HSV. `mosaicraft` does something different: every step of the pipeline runs in a perceptual color space, and every cell of the output is a distinct photograph.

What's inside:

- **Oklab perceptual color space** — Björn Ottosson's 2020 colour space, noticeably more uniform than CIELAB on the saturated colours photomosaics spend most of their compute budget matching, at the same compute cost.
- **MKL optimal transport color transfer** — matches the full covariance of each tile's color distribution to the target, preserving the shape of the original tile instead of flattening it.
- **Hungarian 1:1 placement** — globally optimal assignment of tiles to cells via the Jonker–Volgenant algorithm. Falls back to FAISS + Floyd–Steinberg error diffusion when the cost matrix exceeds memory.
- **Laplacian pyramid blending** — removes grid lines without blurring detail.
- **Oklch tile-pool expansion** — generates N hue-rotated variants of every tile in the pool, multiplying the effective catalog size by (N+1) with zero extra photographs.

The hero image above is reproducible from this repository. `python scripts/download_demo_assets.py` fetches ~8 MB of demo assets and CC0 tiles; `python scripts/generate_readme_figures.py` then writes every image in this README.

## Installation

```bash
pip install mosaicraft                # PyPI
pip install "mosaicraft[faiss]"       # with FAISS for huge tile pools
```

Requires Python 3.9+, NumPy ≥ 1.23, OpenCV ≥ 4.6, SciPy ≥ 1.10, scikit-image ≥ 0.20. No GPU required; FAISS is optional.

## Quick start

### CLI

```bash
# Basic: target image + tile directory.
mosaicraft generate photo.jpg --tiles ./tiles --output mosaic.jpg

# Pick a preset and target cell count.
mosaicraft generate photo.jpg -t ./tiles -o vivid.jpg --preset vivid -n 5000

# Expand a 1,024-tile pool into 5,120 candidates with Oklch hue rotation.
mosaicraft generate photo.jpg -t ./tiles -o big.jpg --color-variants 4

# Pre-build a feature cache so subsequent runs load in under a second.
mosaicraft cache --tiles ./tiles --cache-dir ./cache --sizes 56 88 120

# List all presets.
mosaicraft presets
```

![Before and after](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/before_after.jpg)

*Target: Vermeer, Girl with a Pearl Earring (1,366 × 1,600 px). 1,024-image CC0 tile pool × 4 augmentations = 4,096 candidates. 52 × 61 = 3,172 cells. Preset `vivid`.*

### Python API

```python
from mosaicraft import MosaicGenerator

gen = MosaicGenerator(
    tile_dir="./tiles",
    preset="vivid",
    color_variants=4,              # 1,024 tiles -> 5,120 candidates
)
result = gen.generate("photo.jpg", "mosaic.jpg", target_tiles=5000)
```

## Pipeline

```
                  ┌─────────────────────┐
                  │  Tile collection    │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐    ┌────────────────────┐
                  │  Feature extraction │───▶│ 4x flip/brightness │
                  │   (191 dimensions)  │    │ + Oklch variants   │
                  └──────────┬──────────┘    └─────────┬──────────┘
                             │                         │
                             └────────────┬────────────┘
                                          │
   ┌────────────────────┐       ┌─────────▼───────────┐
   │  Target image      │──────▶│  Per-cell features  │
   └────────────────────┘       │  + Oklab means      │
                                └─────────┬───────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Saliency-weighted cost matrix      │
                       │  (191-D L2 + Oklab ΔE)              │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Hungarian 1:1 assignment           │
                       │  (or FAISS + Floyd–Steinberg)       │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Neighbor-swap refinement (2-opt)   │
                       │  then NCC + SSIM rerank             │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Per-tile MKL optimal transport     │
                       │  Laplacian pyramid blend            │
                       │  Oklch vibrance / skin protection   │
                       └──────────────────┬──────────────────┘
                                          ▼
                                       output
```

**Why Oklab?** CIELAB was calibrated on small colour differences; it underestimates perceptual distance for the large jumps a photomosaic routinely makes. Oklab ([Björn Ottosson, 2020](https://bottosson.github.io/posts/oklab/)) was rebuilt on modern colour-difference data and is noticeably more uniform on the saturated regions a photomosaic spends most of its compute budget matching. Dropping it into the cost function is free and visibly improves matches on saturated photos.

**Why MKL optimal transport?** Reinhard color transfer matches the first and second moments of the LAB distribution. MKL ([Pitié et al., 2007](https://www.researchgate.net/publication/220056262)) matches the full covariance, so the *shape* of the tile's color distribution is preserved as its statistics slide toward the target cell. Details survive; averages don't win.

![Zoom detail](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/zoom_detail.jpg)

*Left: the center of the mosaic — at reading distance, the painting is recognizable. Right: a 2× nearest-neighbor zoom — every cell is a distinct CC0 photograph.*

## Oklch tile-pool expansion

![Tile pool sample](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/tiles_sample.jpg)

One of the hardest problems in photomosaic generation is having enough tiles. A 1,000-image pool gives ~1,000 mean colors, so a 5,000-cell mosaic is forced to repeat. `color_variants=N` rotates every tile through N evenly-spaced hue shifts in Oklch (the default schedule is 72° / 144° / 216° / 288°), reusing the same photograph at four new positions on the a/b plane:

```python
gen = MosaicGenerator(tile_dir="./tiles", preset="vivid", color_variants=4)
```

Lightness is preserved exactly, so texture and shading are untouched — only hue and chroma move. For a 1,024-tile pool this turns into **5,120 candidates after Oklch expansion** (1,024 × 5 = original + 4 hue rotations), or **20,480 once the default flip + brightness augmentation is layered on top**. The Hungarian assignment then has an order of magnitude more material to work with.

## Presets

| Preset    | Best for                                                   |
| --------- | ---------------------------------------------------------- |
| `vivid`   | **Recommended.** MKL optimal transport with skin protection. |
| `ultra`   | Hungarian + Laplacian blend. Highest pixel fidelity.       |
| `natural` | Photo-realistic look, restrained saturation.               |
| `tile`    | Emphasizes individual tiles. Strongest mosaic look.        |
| `fast`    | FAISS + error diffusion only. No rerank, no Hungarian.     |

Pass a dict to `MosaicGenerator(preset={...})` to override individual keys. See [`src/mosaicraft/presets.py`](https://github.com/hinanohart/mosaicraft/blob/main/src/mosaicraft/presets.py) for the full schema.

![Preset comparison](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/presets_comparison.jpg)

## Benchmarks

### Small-pool wall time (256-tile pool, cold start)

Produced by `python benchmarks/benchmark_pipeline.py` — a single `MosaicGenerator` pass, tiles loaded from disk every time, no feature cache, no GPU, no FAISS.

| preset  | 200 cells | 500 cells | 1,000 cells |
| ------- | --------: | --------: | ----------: |
| fast    | 3.00 s    | 4.42 s    | 6.87 s      |
| natural | 2.79 s    | 4.38 s    | 7.49 s      |
| ultra   | 2.86 s    | 4.64 s    | 7.61 s      |
| vivid   | 2.92 s    | 4.69 s    | 7.85 s      |

*AMD Ryzen 7 7735HS, WSL2 / Ubuntu 24.04, Python 3.12, NumPy + OpenCV wheels.*

### Large-pool regime (1,024-tile pool, up to 30,000 cells)

Run `python benchmarks/benchmark_pipeline.py --scale large` to reproduce. Every cell is one tile selected from the 1,024 CC0 photograph pool × 4 augmentations (1 horizontal flip + 3 brightness shifts) = 4,096 candidates. Every case is run cold — tiles loaded from disk on every invocation.

| preset | metric     | 5,000 cells | 10,000 cells | 20,000 cells | 30,000 cells |
| ------ | ---------- | ----------: | -----------: | -----------: | -----------: |
| fast   | wall time  |      28.3 s |       51.1 s |       95.0 s |      190.2 s |
| fast   | peak RSS   |    4,691 MB |     4,840 MB |     9,373 MB |     7,264 MB |
| ultra  | wall time  |      73.9 s |       99.8 s |      110.7 s |      181.7 s |

The 30,000-cell output is **8,904 × 10,472 px ≈ 93 megapixels** and the finished JPEG is ~47 MB. (`ultra` runs faster than `fast` at the 20k / 30k end because the Hungarian assignment saturates before the FAISS + error-diffusion code path stops benefiting from more cells; your mileage will vary with the tile pool / cell size ratio.)

**50,000-cell estimate** (CPU only, no GPU):

| preset | est. time | est. peak RAM |
| ------ | --------: | ------------: |
| `fast` | ~5–7 min  |     8–12 GB   |
| `vivid`| ~4–6 min  |    12–16 GB   |
| `vivid --color-variants 4` | ~10–15 min | 20–25 GB |

Output: ~14,000 × 14,000 px ≈ 200 megapixels. The dominant memory cost is the dense Hungarian cost matrix (`n_cells × n_candidates × 8 bytes`); `fast` avoids it via FAISS.

## Compared against other photomosaic OSS

![Cell diversity vs other tools](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/diversity_chart.jpg)

Cell diversity is what separates a photomosaic from a four-colour halftone — at 8% the grid is essentially the same dozen tiles repeated all over, at 40%+ each cell is its own photo. The chart above shows the metric on Vermeer's *Girl with a Pearl Earring* against a shared 1,024-image CC0 pool, regenerated by `benchmarks/compare_tools.py` and saved to `docs/assets/bench_outputs/metrics.json`.

![Target comparison](https://raw.githubusercontent.com/hinanohart/mosaicraft/main/docs/images/target_comparison.jpg)

The same `vivid` preset on two very different source styles — a 17th-century oil painting and a modern illustration — using one shared 1,024-image CC0 tile pool. No tile pool was tuned per target.

mosaicraft is not the only library that uses [linear-sum assignment](https://en.wikipedia.org/wiki/Assignment_problem) for tile placement: [`phomo`](https://github.com/loiccoyle/phomo), [`phomo-rs`](https://github.com/loiccoyle/phomo-rs), and [`image-collage-maker`](https://github.com/hanzhi713/image-collage-maker) all do 1:1 placement too. The combination that makes mosaicraft different is **Oklab perceptual colour matching + per-tile MKL optimal-transport colour transfer + Oklch hue-rotation pool expansion** in the same pipeline. To my knowledge no other OSS photomosaic library ships all three.

<details>
<summary>Pixel metrics (click to expand)</summary>

Target: Vermeer, *Girl with a Pearl Earring*. Grid: 40×40 = 1,600 cells. 1,024 CC0 tiles. Numbers below are read directly from `docs/assets/bench_outputs/metrics.json`; rerun `python benchmarks/compare_tools.py --target pearl_earring.jpg --grid 40` to refresh.

| Tool                              |   Wall | SSIM ↑ | LPIPS ↓ | ΔE2000 ↓ | Diversity ↑ |
| --------------------------------- | -----: | -----: | ------: | -------: | ----------: |
| codebox/mosaic (RGB mean)         |  1.3 s |  0.250 |   0.544 |    10.32 |       0.079 |
| photomosaic 0.3.1 (CIELAB)        |  2.1 s |  0.065 |   0.776 |    37.18 |       0.111 |
| mosaicraft `fast`                 | 17.2 s |  0.216 |   0.630 |    10.85 |       0.341 |
| mosaicraft `vivid`                | 22.2 s |  0.148 |   0.627 |    15.12 |   **0.424** |
| mosaicraft `vivid --cv 4`         | 77.6 s |  0.224 |   0.559 |    11.06 |       0.384 |

SSIM and ΔE2000 reward pixel fidelity, which structurally favours mean-matching tools that reuse the same tiles. LPIPS (Zhang et al., CVPR 2018) correlates better with human judgement. Cell diversity counts the number of visually distinct cells (5-bit-quantised mean colour) — at 0.42 the same Vermeer that looks like a flat halftone in `codebox` (0.08) is built from ~670 distinct photos out of 1,600 cells.

`vivid --cv 4` trades a small amount of bucket diversity (0.424 → 0.384) for an LPIPS gain (0.627 → 0.559) and a ΔE2000 improvement (15.12 → 11.06): the 4× hue-rotated pool gives the cost-matrix more colour matches, so cells get colour-closer tiles instead of more *distinct* tiles. Pick `vivid` if cell diversity matters most, `vivid --cv 4` if perceptual fidelity does.

</details>

```bash
python benchmarks/compare_tools.py --target pearl_earring.jpg --grid 40
```

## Python API

```python
from mosaicraft import MosaicGenerator, rotate_hue_oklch

# Generator
gen = MosaicGenerator(
    tile_dir="./tiles",          # or cache_dir="./cache"
    preset="vivid",              # preset name or dict
    augment=True,                # 4x flip + brightness augmentation
    color_variants=0,            # set to >0 to expand pool via Oklch rotation
)
result = gen.generate("photo.jpg", "mosaic.jpg", target_tiles=2000, tile_size=88)

# Rotate a single tile or patch in Oklch (preserves L exactly)
rotated_bgr = rotate_hue_oklch(tile_bgr, hue_shift_deg=90)
```

`MosaicResult` exposes `image` (numpy BGR), `grid_cols`, `grid_rows`, `tile_size`, `output_path`, `n_tiles`.

Helpers:

- `mosaicraft.list_presets()` — mosaic preset names.
- `mosaicraft.build_cache(tile_dir, cache_dir, tile_sizes, thumb_size=120)` — precompute features.
- `mosaicraft.calc_grid(target_tiles, aspect_w, aspect_h)` — pick a grid for a desired cell count.

Lower-level building blocks live in `mosaicraft.color`, `mosaicraft.features`, `mosaicraft.placement`, `mosaicraft.blending`, `mosaicraft.postprocess`, `mosaicraft.saliency`, `mosaicraft.color_augment`, `mosaicraft.tiles`, and `mosaicraft.utils`.

## Reproducible figures

Every image in this README — hero, before/after, preset comparison, zoom detail, tile sample, and comparison table — is produced by two self-contained scripts:

```bash
# 1. Bootstrap public-domain demo assets (~8 MB, one time).
python scripts/download_demo_assets.py
python scripts/download_demo_assets.py --verify-only   # SHA256 integrity check

# 2. Render figures.
python scripts/generate_readme_figures.py
python scripts/generate_readme_figures.py --quick                 # faster iteration

# 3. Run the OSS comparison benchmark.
python benchmarks/compare_tools.py --target pearl_earring.jpg --grid 40
```

SHA256 and license metadata for every bootstrapped file live in [`docs/assets/MANIFEST.json`](https://github.com/hinanohart/mosaicraft/blob/main/docs/assets/MANIFEST.json). Downloaded images (paintings from Wikimedia, tiles from picsum) are not committed; only the manifest and the Zundamon target (79 KB, committed under the [Tohoku Zunko Guidelines](https://zunko.jp/guideline.html)) ship with the repository.

## Testing

```bash
pip install -e ".[dev]"
pytest                        # unit + pipeline + CLI tests
ruff check src tests          # lint
bandit -r src -ll             # security scan
```

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](https://github.com/hinanohart/mosaicraft/blob/main/CONTRIBUTING.md) for the development workflow. Security issues: see [SECURITY.md](https://github.com/hinanohart/mosaicraft/blob/main/SECURITY.md).

## License and image credits

MIT License for human use. See [LICENSE](https://github.com/hinanohart/mosaicraft/blob/main/LICENSE).

**AI/ML training opt-out**: this repository is opted out of AI/ML training, fine-tuning, and embedding generation — see [ai.txt](./ai.txt). Training use requires separately negotiated written permission.

Image credits:

- **Target painting** — Johannes Vermeer, *Girl with a Pearl Earring* (c. 1665), public domain, via [Wikimedia Commons](https://commons.wikimedia.org/).
- **Zundamon** — character by SSS LLC / Tohoku Zunko Project. Used under the [Tohoku Zunko Guidelines](https://zunko.jp/guideline.html).
- **Tile pool** — 1,024 photographs from [picsum.photos](https://picsum.photos) (Unsplash-sourced, [Unsplash License](https://unsplash.com/license) — effectively CC0).

## References

mosaicraft stands on the following classic and modern work:

- Björn Ottosson, *A perceptual color space for image processing* (2020, blog). Oklab.
- Pitié, F. et al., *The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer* (IET-CVMP 2007). MKL.
- Reinhard, E. et al., *Color transfer between images* (IEEE CGA 2001).
- Zhang, R. et al., *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric* (CVPR 2018). LPIPS.
- Wang, Z. et al., *Image quality assessment: from error visibility to structural similarity* (IEEE TIP 2004). SSIM.
- Tesfaldet, M. et al., *Convolutional Photomosaic Generation via Multi-Scale Perceptual Losses* (ECCVW 2018). Multi-scale perceptual loss for photomosaic quality assessment.
- Burt, P. & Adelson, E., *A multiresolution spline with application to image mosaics* (ACM ToG 1983). Laplacian pyramid blending.
- Kuhn, H. W., *The Hungarian method for the assignment problem* (Naval Research Logistics 1955).
