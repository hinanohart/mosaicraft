<div align="center">

# mosaicraft

**Perceptual photomosaic generator with Oklab color science, MKL optimal transport, and Laplacian pyramid blending.**

[![CI](https://github.com/hinanohart/mosaicraft/actions/workflows/ci.yml/badge.svg)](https://github.com/hinanohart/mosaicraft/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/mosaicraft.svg)](https://pypi.org/project/mosaicraft/)
[![Python](https://img.shields.io/pypi/pyversions/mosaicraft.svg)](https://pypi.org/project/mosaicraft/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/tests-47%20passing-brightgreen.svg)](#testing)

[Features](#-features) · [Quick Start](#-quick-start) · [Algorithm](#-algorithm) · [Presets](#-presets) · [Benchmarks](#-benchmarks) · [API](#-python-api) · [日本語](#-日本語)

![Target vs mosaicraft output](docs/images/hero.jpg)

</div>

---

`mosaicraft` reproduces a target image as a grid of smaller tile images. Most photomosaic libraries use mean-color matching in RGB or HSV; mosaicraft works in **Oklab** perceptual color space with **MKL optimal transport** color transfer, **Hungarian** placement, and **Laplacian pyramid** boundary blending. The result is a mosaic that looks closer to the target *and* preserves the look of the individual tiles.

> *The hero image above is reproducible end-to-end from this repository with no external assets:*
> ```bash
> python scripts/generate_readme_figures.py
> ```
> *The target, the tile pool, and every mosaic are synthesized procedurally and released under CC0.*

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

*1,600×1,200 target, 2,048-tile procedural pool, 3,072 cells, preset `ultra`.*

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

*Left: the center 50% of the mosaic — at reading distance the landscape is recognizable. Right: a 2x nearest-neighbor zoom into the same region — every cell is a distinct tile from the procedural pool.*

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

The figures in this README — the hero image, the before/after, the preset comparison, and the zoom detail — are produced by a single self-contained script using only procedural assets. Running it locally is the fastest way to sanity-check the pipeline on your own hardware:

```bash
python scripts/generate_readme_figures.py          # full resolution (~2 minutes)
python scripts/generate_readme_figures.py --quick  # smaller, faster (~40 seconds)
```

The `docs/images/tiles_sample.jpg` thumbnail below shows a subset of the procedural tile pool — gradients, shape primitives, stripes, and checkerboards with perturbed HSV, all synthesized on the fly:

![Procedural tile pool](docs/images/tiles_sample.jpg)

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

> 上の比較画像は `python scripts/generate_readme_figures.py` を実行するとリポジトリ内で完全に再現できます。ターゲット・タイル・モザイクすべて手続き的に生成する CC0 アセットで、外部の写真は一切使用していません。

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

### ベンチマーク（実測値）

AMD Ryzen 7 7735HS / WSL2 Ubuntu 24.04 / Python 3.12。256 タイルプールからのコールドスタート実測（特徴キャッシュ未使用）。

| preset  | 200 セル | 500 セル | 1,000 セル |
| ------- | -------: | -------: | ---------: |
| fast    | 3.00 s   | 4.42 s   | 6.87 s     |
| natural | 2.79 s   | 4.38 s   | 7.49 s     |
| ultra   | 2.86 s   | 4.64 s   | 7.61 s     |
| vivid   | 2.92 s   | 4.69 s   | 7.85 s     |

`python benchmarks/benchmark_pipeline.py` でご自身の環境でも再実行可能です。`mosaicraft cache` で特徴キャッシュを事前構築すると、2 回目以降のタイル読み込みが数秒→1 秒未満に短縮されます。

### ライセンス

MIT License。詳細は [LICENSE](LICENSE) を参照。
