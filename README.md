<div align="center">

# mosaicraft

**Perceptual photomosaic generator with Oklab color science, MKL optimal transport, and Laplacian pyramid blending.**

[![CI](https://github.com/hinanohart/mosaicraft/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/mosaicraft/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mosaicraft.svg)](https://pypi.org/project/mosaicraft/)
[![Python](https://img.shields.io/pypi/pyversions/mosaicraft.svg)](https://pypi.org/project/mosaicraft/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-47%20passing-brightgreen.svg)](#testing)

[Features](#-features) · [Quick Start](#-quick-start) · [Algorithm](#-algorithm) · [Comparison](#-comparison-with-other-photomosaic-tools) · [Presets](#-presets) · [Benchmarks](#-benchmarks) · [API](#-python-api) · [日本語](#-日本語)

![Target vs mosaicraft output](docs/images/hero.jpg)

</div>

---

`mosaicraft` reproduces a target image as a grid of smaller tile images. Most photomosaic libraries use mean-color matching in RGB or HSV; mosaicraft works in **Oklab** perceptual color space with **MKL optimal transport** color transfer, **Hungarian** placement, and **Laplacian pyramid** boundary blending. The result is a mosaic that looks closer to the target *and* preserves the look of the individual tiles.

> *The hero image is reproducible end-to-end from this repository. Bootstrap the demo assets once, then render:*
> ```bash
> python scripts/download_demo_assets.py       # ~8 MB of public-domain paintings + CC0 tiles
> python scripts/generate_readme_figures.py    # writes docs/images/*.jpg
> ```
> *Target: Johannes Vermeer, **Girl with a Pearl Earring** (c. 1665, public domain, Wikimedia Commons). Tile pool: 1,024 CC0 photographs via picsum.photos (Unsplash License). See [Image credits](#-image-credits).*

## ✨ Features

- **Oklab perceptual color matching** &nbsp;— roughly 8.5x more perceptually uniform than CIELAB for chroma differences. Tile selection is dramatically more accurate, especially for vivid colors.
- **191-dimensional tile features** &nbsp;— quadrant means + per-channel histograms + gradient orientation histograms + Local Binary Pattern. Captures color *and* texture, not just average color.
- **Hungarian assignment** &nbsp;— globally optimal one-to-one tile placement via the Jonker-Volgenant algorithm. Falls back to FAISS + Floyd-Steinberg error diffusion for huge problems where the cost matrix would not fit in RAM.
- **MKL optimal transport color transfer** &nbsp;— preserves the *shape* of each tile's color distribution while shifting its statistics to match the target cell. Far better than naive Reinhard for chromatically distant images.
- **Two-stage NCC + SSIM rerank** &nbsp;— after Hungarian, every cell is reranked using normalized cross-correlation then structural similarity. Roughly 5x faster than running SSIM on the full candidate set with no measurable quality loss.
- **Laplacian pyramid blending** &nbsp;— removes the visible grid lines without losing detail.
- **Saliency-aware cost biasing** &nbsp;— faces, edges, and saturated regions get the best matches.
- **Skin protection** &nbsp;— optional HSV ∩ YCrCb skin mask preserves natural luminance during high-saturation postprocessing, so people don't end up looking like Oompa Loompas.
- **Persistent feature cache** &nbsp;— compute features once, generate mosaics in seconds.
- **First-class CLI and Python API.**
- **Pure Python + NumPy/OpenCV/SciPy** &nbsp;— no GPU required. FAISS is optional.

## 📦 Installation

From PyPI:

```bash
pip install mosaicraft
```

With FAISS for faster nearest-neighbor search on huge tile sets:

```bash
pip install "mosaicraft[faiss]"
```

From source:

```bash
git clone https://github.com/hinanohart/mosaicraft.git
cd mosaicraft
pip install -e ".[dev]"
```

**Requirements**: Python 3.9+, NumPy ≥ 1.23, OpenCV ≥ 4.6, SciPy ≥ 1.10, scikit-image ≥ 0.20.

## 🚀 Quick Start

### CLI

```bash
# 1. Point at any directory of tile images and a target photo.
mosaicraft generate photo.jpg --tiles ./tiles --output mosaic.jpg

# 2. Pick a preset and a target tile count.
mosaicraft generate photo.jpg -t ./tiles -o vivid.jpg --preset vivid -n 5000

# 3. Pre-build a feature cache for faster iteration.
mosaicraft cache --tiles ./tiles --cache-dir ./cache --sizes 56 88 120

# 4. Then generate from the cache.
mosaicraft generate photo.jpg --cache-dir ./cache -o out.jpg --tile-size 88

# 5. List all presets.
mosaicraft presets
```

![Before and after](docs/images/before_after.jpg)

*Target: Vermeer, *Girl with a Pearl Earring* (1,366×1,600 px). 1,024-image CC0 tile pool × 4 augmentations. 52×61 = 3,172 cells. Preset `ultra`.*

### Python API

```python
from mosaicraft import MosaicGenerator

gen = MosaicGenerator(
    tile_dir="./tiles",   # or cache_dir="./cache"
    preset="ultra",
)

result = gen.generate(
    "photo.jpg",
    "mosaic.jpg",
    target_tiles=2000,
    tile_size=88,
)

print(f"{result.grid_cols}x{result.grid_rows} = {result.n_tiles} cells")
print(f"Image: {result.image.shape}")  # numpy uint8 BGR array
```

## 🧠 Algorithm

The pipeline is built from independent stages you can mix and match:

```
                  ┌─────────────────────┐
                  │  Tile collection    │
                  │  (any directory)    │
                  └──────────┬──────────┘
                             │
                  ┌──────────▼──────────┐    ┌────────────────────┐
                  │  Feature extraction │    │  Augmentation x4   │
                  │   (191 dimensions)  │───▶│ (flip + bright±)   │
                  └──────────┬──────────┘    └─────────┬──────────┘
                             │                         │
                             └────────────┬────────────┘
                                          │
                                          ▼
   ┌────────────────────┐       ┌─────────────────────┐
   │  Target image      │──────▶│  Per-cell features  │
   └────────────────────┘       │  + Oklab means      │
                                └─────────┬───────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Cost matrix (191-D L2 + Oklab)     │
                       │  weighted by saliency               │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Hungarian assignment               │
                       │  (or FAISS + Floyd-Steinberg)       │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Neighbor swap refinement (2-opt)   │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Two-stage NCC + SSIM rerank        │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Per-tile color transfer            │
                       │  (Reinhard / MKL / Histogram)       │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Laplacian pyramid blend            │
                       └──────────────────┬──────────────────┘
                                          │
                       ┌──────────────────▼──────────────────┐
                       │  Postprocess: gamma → CLAHE →       │
                       │  Oklch vibrance → HSV saturation →  │
                       │  contrast → sharpness               │
                       └──────────────────┬──────────────────┘
                                          ▼
                                       output
```

### Why Oklab?

CIELAB was designed for small color differences. For the *large* color jumps you encounter in photomosaic matching (where a tile may be far from any cell), CIELAB underestimates perceptual distance, especially in vivid yellows and blues. Oklab ([Björn Ottosson, 2020](https://bottosson.github.io/posts/oklab/)) was redesigned with modern data and is **roughly 8.5x more perceptually uniform** for chroma. Switching from CIELAB to Oklab in the cost function noticeably improves matching quality on saturated images at zero compute cost.

### Why MKL optimal transport?

Reinhard color transfer matches the *first and second moments* of the LAB distributions. MKL ([Pitié et al., 2007](https://www.researchgate.net/publication/220056262)) matches the *full covariance*, preserving the shape of the source distribution as it shifts toward the target. The result keeps the texture of the original tile while making it blend with the surrounding cells.

![Zoom detail](docs/images/zoom_detail.jpg)

*Left: the center 50% of the mosaic — at reading distance the painting is recognizable. Right: a 2× nearest-neighbor zoom into the same region — every cell is a distinct CC0 photograph from the tile pool.*

## 🆚 Comparison with other photomosaic tools

`benchmarks/compare_tools.py` runs mosaicraft side-by-side with two reference OSS photomosaic tools against the same Wikimedia public-domain target (Vermeer, *Girl with a Pearl Earring*), the same 1,024-image CC0 tile pool (Unsplash License via picsum.photos), and an identical 40×40 grid. The output figure and the raw metrics file are committed under `docs/assets/bench_outputs/`.

![Side-by-side comparison](docs/images/comparison.jpg)

| Tool                                                                           | Wall time | SSIM ↑ | ΔE2000 ↓ | Edge corr ↑ | Cell diversity ↑ |
| ------------------------------------------------------------------------------ | --------: | -----: | -------: | ----------: | ---------------: |
| [codebox/mosaic](https://github.com/codebox/mosaic) (naive RGB mean)           |   1.57 s  |  0.250 |    10.32 |       0.209 |            0.079 |
| [photomosaic 0.3.1](https://pypi.org/project/photomosaic/) (CIELAB + kd-tree)  |   2.09 s  |  0.068 |    37.18 |      −0.079 |            0.110 |
| **mosaicraft — `fast`** (Oklab + 191-D features + FAISS)                       |  16.3  s  |  0.217 |    10.80 |       0.165 |        **0.339** |
| **mosaicraft — `ultra`** (Oklab + MKL OT + Hungarian + NCC/SSIM + Laplacian)   |  20.1  s  |  0.166 |    13.84 |       0.106 |        **0.367** |

<sub>SSIM and ΔE2000 (CIEDE2000) are computed against the original painting. Edge correlation is the Pearson correlation of Sobel-gradient magnitudes between target and mosaic. Cell diversity is the fraction of unique 5-bit-quantized cell means across the mosaic — higher means the mosaic uses more of the tile pool's variety. Rerun on your own machine with `python benchmarks/compare_tools.py --target pearl_earring --grid 40`.</sub>

### How to read these numbers

Photomosaic tools aren't all optimizing for the same thing, and the metrics make the tradeoff explicit:

- **codebox** wins on raw pixel fidelity (SSIM, ΔE2000) because it uses low-pass mean-color matching with no constraint on tile reuse. That produces the smoothest pixel reproduction but reuses a tiny fraction of the pool — only **7.9%** of its cells are visually distinct.
- **photomosaic 0.3.1** (2018-era CIELAB + kd-tree) loses on every metric here because its default pipeline expects much larger tile pools than this 1,024-image benchmark allows, and its skimage-era codepaths needed a compatibility shim to even run on modern NumPy. It is included as a historical baseline.
- **mosaicraft** enforces strict one-to-one Hungarian assignment between cells and augmented tiles, then post-processes with MKL optimal transport, Oklch vibrance, and saturation. That shifts the output *away* from the target pixel values on purpose, in exchange for **4–5× higher cell diversity**: every cell of a mosaicraft output is a meaningfully different photograph, which is the entire point of a photomosaic. At reading distance the target is clearly recognizable; at arm's length every tile is its own image.

**tl;dr**: if you want the closest possible pixel match, use a low-pass tool. If you want a photomosaic that *looks like a photomosaic* — where the tiles are recognizable under close inspection — mosaicraft is doing something different on purpose.

## 🎨 Presets

| Preset         | Best for                                              | Speed |
| -------------- | ----------------------------------------------------- | ----- |
| `ultra`        | Highest quality. Hungarian + Laplacian blend.        | ⭐⭐⭐  |
| `natural`      | Photo-realistic look, restrained saturation.          | ⭐⭐⭐⭐ |
| `vivid`        | Vivid output via MKL optimal transport.               | ⭐⭐⭐  |
| `vivid_strong` | Strong saturation with skin protection.               | ⭐⭐⭐  |
| `vivid_max`    | Maximum saturation, full skin protection.             | ⭐⭐⭐  |
| `tile`         | Emphasize individual tiles, max mosaic look.          | ⭐⭐⭐⭐ |
| `fast`         | FAISS + error diffusion only. No rerank, no Hungarian.| ⭐⭐⭐⭐⭐ |

You can also pass a custom dict to `MosaicGenerator(preset={...})`. See [`presets.py`](src/mosaicraft/presets.py) for the full key list.

![Preset comparison](docs/images/presets_comparison.jpg)

*Same target, same tile pool, three presets — saturation and color-transfer behavior visibly diverge.*

## 📊 Benchmarks

### End-to-end wall time (cold start, 256-tile pool)

Produced by `python benchmarks/benchmark_pipeline.py` — a single `MosaicGenerator` pass, tiles loaded from disk every time, no feature cache. Each run ends with the mosaic written to JPEG.

| preset  | 200 cells | 500 cells | 1,000 cells |
| ------- | --------: | --------: | ----------: |
| fast    | 3.00 s    | 4.42 s    | 6.87 s      |
| natural | 2.79 s    | 4.38 s    | 7.49 s      |
| ultra   | 2.86 s    | 4.64 s    | 7.61 s      |
| vivid   | 2.92 s    | 4.69 s    | 7.85 s      |

<sub>AMD Ryzen 7 7735HS, WSL2 / Ubuntu 24.04, Python 3.12, NumPy + OpenCV wheels — no GPU, no FAISS. Rerun the script on your own machine to verify.</sub>

### Per-stage breakdown (64-cell synthetic)

| Stage                              | Wall time (synthetic) |
| ---------------------------------- | --------------------: |
| Tile feature extraction (×4 aug)   | ~0.05 s               |
| Cost matrix (256 cells × 256 tiles)| ~0.01 s               |
| Hungarian assignment               | ~0.005 s              |
| Neighbor swap (5 rounds)           | ~0.01 s               |
| NCC + SSIM rerank                  | ~0.04 s               |
| Laplacian assembly                 | ~0.05 s               |
| Postprocess                        | ~0.08 s               |
| **Total (preset=ultra, 64 cells)** | **~0.3 s**            |

A 5,000-cell mosaic from a 4,000-tile pool typically completes in 30–90 seconds on a modern laptop CPU. Pre-building the feature cache with `mosaicraft cache` reduces tile load time from minutes to under one second on subsequent runs — the cold-start numbers above are a pessimistic floor.

### Reproducible demo figures

Every figure in this README — hero, before/after, preset comparison, zoom detail, paintings gallery, and the side-by-side comparison against other tools — is produced by two self-contained scripts that bootstrap ~8 MB of public-domain demo assets on first run:

```bash
# 1. Download public-domain demo assets (~8 MB, one time).
#    Writes docs/assets/MANIFEST.json with per-file SHA256 for integrity.
python scripts/download_demo_assets.py

# 2. Render README figures from the bootstrapped assets.
python scripts/generate_readme_figures.py                       # full resolution
python scripts/generate_readme_figures.py --quick               # faster iteration
python scripts/generate_readme_figures.py --target starry_night # swap the hero painting

# 3. Run the side-by-side benchmark against other OSS photomosaic tools.
python benchmarks/compare_tools.py --target pearl_earring --grid 40
```

Asset integrity is verified against `docs/assets/MANIFEST.json` via SHA256 — rerun `python scripts/download_demo_assets.py --verify-only` any time to check.

![Tile pool sample](docs/images/tiles_sample.jpg)

*A stride-sampled thumbnail of the 1,024-image CC0 tile pool (Unsplash License via picsum.photos).*

![Public-domain paintings gallery](docs/images/paintings_gallery.jpg)

*All four public-domain targets the scripts can feature — swap with `--target {pearl_earring,starry_night,great_wave,red_fuji}`.*

## 📚 Python API

### `MosaicGenerator`

```python
MosaicGenerator(
    tile_dir: str | None = None,        # tile directory (or use cache_dir)
    cache_dir: str | None = None,       # precomputed feature cache
    preset: str | dict = "ultra",       # preset name or custom dict
    augment: bool = True,               # 4x tile augmentation
    hungarian_mem_limit_mb: float = 3000,
)
```

Methods:

- `generate(input_path, output_path=None, *, target_tiles=2000, tile_size=88, dedup_radius=4, jpeg_quality=95) -> MosaicResult`

### `MosaicResult`

Attributes: `image` (numpy BGR array), `grid_cols`, `grid_rows`, `tile_size`, `output_path`, `n_tiles`.

### Helpers

- `mosaicraft.list_presets()` &nbsp;— list preset names.
- `mosaicraft.get_preset(name)` &nbsp;— deep copy of a preset dict.
- `mosaicraft.build_cache(tile_dir, cache_dir, tile_sizes, thumb_size=120)` &nbsp;— precompute features.
- `mosaicraft.calc_grid(target_tiles, aspect_w, aspect_h)` &nbsp;— pick a grid for a desired cell count.
- `mosaicraft.configure_logging(verbose=False)` &nbsp;— enable info/debug logging.

Lower-level building blocks live in `mosaicraft.color`, `mosaicraft.features`, `mosaicraft.placement`, `mosaicraft.blending`, `mosaicraft.postprocess`, and `mosaicraft.tiles` and are documented in their docstrings.

## 🧪 Testing

```bash
pip install -e ".[dev]"
pytest
```

47 tests covering color science, feature extraction, placement, postprocessing, end-to-end pipeline, and the CLI surface. The suite uses synthetically generated images so it has zero binary fixtures.

```bash
ruff check src tests          # lint
bandit -r src -ll             # security scan
```

## 🤝 Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and code style. Security issues: please follow [SECURITY.md](SECURITY.md).

## 📄 License

MIT License — see [LICENSE](LICENSE).

## 🖼 Image credits

Every demo figure in this README is reproducible from public-domain / CC0 sources:

**Target paintings** — public domain (pre-1929), via [Wikimedia Commons](https://commons.wikimedia.org/):

- Johannes Vermeer, *Girl with a Pearl Earring* (c. 1665)
- Vincent van Gogh, *The Starry Night* (1889)
- Katsushika Hokusai, *The Great Wave off Kanagawa* (c. 1831)
- Katsushika Hokusai, *Fine Wind, Clear Morning (Red Fuji)* (c. 1831)

**Tile pool** — 1,024 photographs from [picsum.photos](https://picsum.photos) (Unsplash-sourced, [Unsplash License](https://unsplash.com/license) — free for any use, attribution appreciated but not required).

Per-file SHA256 and license metadata are pinned in [`docs/assets/MANIFEST.json`](docs/assets/MANIFEST.json), which is committed to the repository; the raw image files are not. Run `python scripts/download_demo_assets.py` to fetch them and `--verify-only` to check integrity.

## 🙏 Acknowledgments

mosaicraft builds on classic and modern color science:

- Björn Ottosson, *A perceptual color space for image processing* (2020).
- Pitié, F. et al., *The linear Monge-Kantorovitch linear colour mapping for example-based colour transfer* (IET-CVMP 2007).
- Reinhard, E. et al., *Color transfer between images* (IEEE CGA 2001).
- Burt, P. & Adelson, E., *A multiresolution spline with application to image mosaics* (ACM ToG 1983).
- Kuhn, H. W., *The Hungarian method for the assignment problem* (Naval Research Logistics 1955).

---

## 🌐 日本語

![ターゲット画像と mosaicraft 出力の比較](docs/images/before_after.jpg)

`mosaicraft` は、画像をタイル写真の集合として再構成する**フォトモザイク**ジェネレータです。多くの既存ライブラリが RGB/HSV の平均色マッチングを使うのに対し、mosaicraft は **Oklab 知覚色空間** + **MKL 最適輸送色転写** + **ハンガリアン法による配置** + **ラプラシアンピラミッドブレンディング** を統合し、より精度の高いマッチングと自然な見た目を両立します。

> 上の比較画像はリポジトリ内で完全に再現可能です:
> ```bash
> python scripts/download_demo_assets.py       # 公開画像アセットを一度DL（約8MB）
> python scripts/generate_readme_figures.py    # docs/images/*.jpg を生成
> ```
> ターゲット画像は **フェルメール「真珠の耳飾りの少女」**（c. 1665、パブリックドメイン、Wikimedia Commons）、タイルプールは **picsum.photos** の CC0 写真 1,024 枚（Unsplash License）。詳細は [Image credits](#-image-credits) を参照。

### 特徴

- **Oklab 知覚色マッチング** — CIELAB より約 8.5 倍知覚均一性が高く、特に鮮やかな色で精度が向上。
- **191 次元タイル特徴** — クアドラント平均 + ヒストグラム + 勾配方向 + Local Binary Pattern。色とテクスチャの両方を捉えます。
- **ハンガリアン配置** — 大域最適な 1 対 1 割当。コスト行列が大きすぎる場合は FAISS + Floyd-Steinberg にフォールバック。
- **MKL 最適輸送色転写** — タイル本来の色分布の形状を保ちながら統計量を目標セルに合わせます。Reinhard より自然。
- **NCC + SSIM 二段階リランク** — Hungarian 後のリランクで品質が大きく向上、しかも高速。
- **ラプラシアンピラミッドブレンディング** — グリッド線を消しつつディテールを保持。
- **サリエンシー重み付け** — 顔・エッジ・彩度の高い領域に最良のタイルを優先割当。
- **肌保護** — 高彩度処理時に肌の明度を元に近づけ、人物の不自然な変色を防ぎます。
- **特徴キャッシュ** — 1 度計算した特徴を保存し、以降の生成を数秒で実行可能。
- **CLI と Python API の両対応**。
- **GPU 不要**（FAISS はオプション）。

### インストール

```bash
pip install mosaicraft           # PyPI
pip install "mosaicraft[faiss]"  # FAISS 込み
```

### 使い方

```bash
# 基本
mosaicraft generate 写真.jpg --tiles ./タイル --output mosaic.jpg

# プリセット指定 + タイル数指定
mosaicraft generate 写真.jpg -t ./タイル -o vivid.jpg --preset vivid -n 5000

# 特徴キャッシュ事前構築
mosaicraft cache --tiles ./タイル --cache-dir ./cache --sizes 56 88 120

# プリセット一覧
mosaicraft presets
```

```python
from mosaicraft import MosaicGenerator

gen = MosaicGenerator(tile_dir="./tiles", preset="ultra")
result = gen.generate("photo.jpg", "mosaic.jpg", target_tiles=2000)
```

### プリセット

| Preset         | 用途                                                  |
| -------------- | ----------------------------------------------------- |
| `ultra`        | 最高品質。Hungarian + ラプラシアンブレンド             |
| `natural`      | 自然なフォトリアリスティック仕上がり                   |
| `vivid`        | MKL 最適輸送によるビビッドな色                         |
| `vivid_strong` | 強めの彩度 + 肌保護                                    |
| `vivid_max`    | 最大彩度 + 肌保護フル                                  |
| `tile`         | タイル感を最大化                                       |
| `fast`         | FAISS のみ。最速、品質は ultra より控えめ              |

### 既存 OSS ツールとの比較

`benchmarks/compare_tools.py` で Wikimedia の公開画像（フェルメール「真珠の耳飾りの少女」）をターゲットに、mosaicraft と 2 本の参考 OSS フォトモザイクツールを、同じ 1,024 枚 CC0 タイルプール（picsum.photos 経由の Unsplash License）・同じ 40×40 グリッドで走らせた結果が下表です。生成された比較画像と生メトリクスは `docs/assets/bench_outputs/` にコミットされています。

![比較図](docs/images/comparison.jpg)

| ツール                                                                       | 所要時間 | SSIM ↑ | ΔE2000 ↓ | Edge corr ↑ | セル多様性 ↑ |
| ---------------------------------------------------------------------------- | -------: | -----: | -------: | ----------: | -----------: |
| [codebox/mosaic](https://github.com/codebox/mosaic)（単純 RGB 平均）         |  1.57 s  |  0.250 |    10.32 |       0.209 |        0.079 |
| [photomosaic 0.3.1](https://pypi.org/project/photomosaic/)（CIELAB + kd-tree）|  2.09 s  |  0.068 |    37.18 |      −0.079 |        0.110 |
| **mosaicraft — `fast`**（Oklab + 191 次元特徴 + FAISS）                      | 16.3  s  |  0.217 |    10.80 |       0.165 |    **0.339** |
| **mosaicraft — `ultra`**（Oklab + MKL OT + ハンガリアン + NCC/SSIM + Laplacian）| 20.1  s |  0.166 |    13.84 |       0.106 |    **0.367** |

**読み方**: ピクセル単位の忠実度（SSIM・ΔE2000）では codebox の単純平均が最良ですが、これは色の低域通過でマッチしているだけで、タイルの多様性は 7.9% しかありません（モザイク全体で似たような平均色のタイルが大量に重複使用されている状態）。mosaicraft は厳密な 1 対 1 のハンガリアン割当 + MKL 最適輸送 + 彩度ブーストを通すので、あえてターゲットのピクセル値から少し離れる代わりに、**セル多様性が約 4〜5 倍（37%）**になります。近距離で見たとき、モザイクを構成する各セルが別々の写真として認識できる──「フォトモザイクらしいフォトモザイク」になる、という設計思想です。純粋な色再現度を最優先するなら低域通過型、近くで見たときにタイルの意味を残したいなら mosaicraft、と目的で使い分けるのが実態に即しています。

比較は `python benchmarks/compare_tools.py --target pearl_earring --grid 40` でローカル再現可能です。

### ベンチマーク（実測値）

AMD Ryzen 7 7735HS / WSL2 Ubuntu 24.04 / Python 3.12。256 タイルプールからのコールドスタート実測（特徴キャッシュ未使用）。

| preset  | 200 セル | 500 セル | 1,000 セル |
| ------- | -------: | -------: | ---------: |
| fast    | 3.00 s   | 4.42 s   | 6.87 s     |
| natural | 2.79 s   | 4.38 s   | 7.49 s     |
| ultra   | 2.86 s   | 4.64 s   | 7.61 s     |
| vivid   | 2.92 s   | 4.69 s   | 7.85 s     |

`python benchmarks/benchmark_pipeline.py` でご自身の環境でも再実行可能です。`mosaicraft cache` で特徴キャッシュを事前構築すると、2 回目以降のタイル読み込みが数秒→1 秒未満に短縮されます。

### 画像クレジット

README で使用する全ての図版はパブリックドメイン / CC0 ソースから再現可能です。

- **ターゲット絵画**: フェルメール「真珠の耳飾りの少女」（c. 1665）、ゴッホ「星月夜」（1889）、北斎「神奈川沖浪裏」「凱風快晴（赤富士）」（c. 1831）。いずれも Wikimedia Commons のパブリックドメイン版。
- **タイルプール**: [picsum.photos](https://picsum.photos) の 1,024 枚（Unsplash-sourced、[Unsplash License](https://unsplash.com/license) = 事実上 CC0）。

各ファイルの SHA256 とライセンスメタデータは [`docs/assets/MANIFEST.json`](docs/assets/MANIFEST.json) にコミット済み（画像本体は gitignore）。`python scripts/download_demo_assets.py` で取得、`--verify-only` で整合性チェック。

### ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照。
