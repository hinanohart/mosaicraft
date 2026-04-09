# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

* `scripts/download_demo_assets.py` — bootstraps ~8 MB of public-domain demo
  assets: four Wikimedia Commons paintings (Vermeer's *Girl with a Pearl
  Earring*, Van Gogh's *Starry Night*, Hokusai's *Great Wave off Kanagawa*
  and *Red Fuji*) and a 1,024-image CC0 tile pool from `picsum.photos`
  (Unsplash License). Writes `docs/assets/MANIFEST.json` with per-file
  SHA256 so the download is bit-exact reproducible; supports `--verify-only`,
  `--offline`, and `--force`.
* `benchmarks/compare_tools.py` — side-by-side benchmark runner that
  compares mosaicraft (`fast` and `ultra` presets) against two reference
  OSS photomosaic tools ([codebox/mosaic](https://github.com/codebox/mosaic)
  and [photomosaic 0.3.1](https://pypi.org/project/photomosaic/)) on an
  identical target, tile pool, and grid. Emits a 5-panel comparison figure
  to `docs/images/comparison.jpg` plus `docs/assets/bench_outputs/metrics.json`
  with SSIM, ΔE2000 (CIEDE2000), edge correlation, cell diversity, and wall
  time. Includes a monkey-patch shim so `photomosaic 0.3.1` (skimage-era)
  runs on modern NumPy / scikit-image.
* `scripts/generate_readme_figures.py` — rewritten to render hero,
  before/after, preset comparison, zoom detail, tile-pool sample, and a
  4-painting gallery figure from the real bootstrapped assets. Adds a
  `--target {pearl_earring,starry_night,great_wave,red_fuji}` flag.
* README: new "Comparison with other photomosaic tools" section (plus
  Japanese mirror) with a metrics table and a plain-language explanation of
  the fundamental tradeoff between pixel-fidelity and tile-diversity
  optimization. Image Credits section lists every public-domain / CC0
  source. All existing captions updated to reference the real Wikimedia
  paintings.
* `docs/assets/MANIFEST.json` committed; the raw image files under
  `docs/assets/{paintings,tiles,bench_outputs}/` are gitignored and
  reproducible via `python scripts/download_demo_assets.py`.

### Changed

* Dropped the procedural CC0 landscape / procedural tile pool used in the
  previous README figures. The recognisable public-domain paintings and the
  real photograph tile pool are a better showcase of mosaicraft's Hungarian
  + MKL + Oklab pipeline.

## [0.1.0] - 2026-04-09

### Added

* Initial public release.
* `MosaicGenerator` high-level orchestrator with seven built-in presets
  (`ultra`, `natural`, `vivid`, `vivid_strong`, `vivid_max`, `tile`, `fast`).
* Oklab / Oklch color space conversions and Oklch non-linear vibrance.
* Color transfer methods: adaptive Reinhard, MKL optimal transport,
  histogram matching, and hybrid combinations.
* 191-dimensional tile feature extraction (quadrant means + histograms +
  gradient orientations + Local Binary Pattern).
* Hungarian assignment via `scipy.optimize.linear_sum_assignment`, with a
  FAISS + Floyd-Steinberg fallback for problems too large for the cost
  matrix.
* Two-stage NCC + SSIM rerank.
* Saliency-aware cost biasing (edges, Laplacian energy, saturation, center).
* Laplacian pyramid blending and feather blending.
* Skin protection via HSV ∩ YCrCb mask, with optional luminance restoration.
* Persistent on-disk feature cache (`build_cache` / `load_tiles_cached`).
* CLI with `generate`, `cache`, and `presets` subcommands.
* 47 unit and end-to-end tests using synthetically generated fixtures.
* MIT license, contributor docs, security policy, and code of conduct.

[Unreleased]: https://github.com/hinanohart/mosaicraft/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hinanohart/mosaicraft/releases/tag/v0.1.0
