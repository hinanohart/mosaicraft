# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.0] - 2026-04-11

### Added

* **Selective recoloring** (`mosaicraft.recolor.recolor_region`, new public
  function and `mosaicraft recolor-region` CLI subcommand). Where `recolor()`
  shifts the hue of every pixel, `recolor_region()` is the surgical version:
  it isolates a single coloured object — a blue turban, a red ribbon, a
  yellow lantern — and rotates only its hue in Oklch, leaving the rest of
  the image byte-for-byte identical. Lightness is still preserved exactly,
  so the recoloured region carries no boundary artifacts.

  Region specification is one of, in priority order:

  1. An explicit binary mask (`mask=` PNG path or `ndarray`).
  2. A rectangular `bbox=(y1, x1, y2, x2)` window.
  3. A perceptual Oklch colour-range mask built from `source_hex=`
     (default behaviour) or `source_hue_deg=`, with `hue_tolerance_deg`,
     `chroma_min/max`, and `lightness_min/max` gates.

  Target colour is specified the same way as `recolor()`
  (`preset=`, `target_hex=`, or `hue_shift_deg=`). The mask is cleaned
  with morphology + connected-component area filtering and Gaussian
  feathering for soft edges. Returns the recoloured BGR image, or
  `(image, mask)` when `return_mask=True`.

  ```python
  from mosaicraft import recolor_region

  recolor_region(
      "girl.jpg", "green_turban.jpg",
      source_hex="#3a5d9e",     # detect the blue turban
      preset="green",            # rotate to Oklch green
      hue_tolerance_deg=28,
  )
  ```

  ```bash
  mosaicraft recolor-region girl.jpg -o green.jpg \
      --source-hex "#3a5d9e" --preset green --hue-tolerance 28
  ```

* `build_oklch_region_mask()` — public helper that returns the binary
  Oklch colour-range mask used by `recolor_region`. Useful for previewing
  the detected region before committing to a recolour.
* `scripts/generate_recolor_region_demo.py` — standalone demo script that
  builds the Vermeer turban gallery (eight panels: original, detected
  mask, and six target colours) without touching the rest of the README
  figure pipeline.
* New README figures committed under `docs/images/`:
  * `selective_recolor_turban.jpg` — 4×2 panel gallery of the Vermeer
    turban recoloured to six different hues, plus the detected mask.
  * `selective_recolor_mask.png` — the raw binary mask, so the README
    can show the reader exactly which pixels were detected.
  * `diversity_chart.jpg` — pure-cv2 horizontal bar chart visualising
    cell diversity vs. `codebox` and `worldveil/photomosaic` (8% / 11% /
    30% / 42% / 57%). Renders without a `matplotlib` dependency.
  * `comparison_four_targets.jpg` — 4-painting before/after grid
    (Vermeer, Van Gogh, Hokusai ×2) showing the same `vivid` preset
    against four very different source styles in one figure.
* `scripts/generate_readme_figures.py` learned `--skip-grid`, the
  `make_four_target_comparison`, `make_diversity_chart`, and
  `make_selective_recolor_figure` figure makers, plus a 3b. multi-target
  mosaic stage that renders the other three paintings at a smaller cell
  budget when the headline grid is requested.
* `tests/test_recolor.py` grew a `TestRegionMask` and `TestRecolorRegion`
  class (10 new tests covering colour-range masks, explicit masks,
  bounding boxes, bbox clipping, empty masks, file IO, feathering, and
  the `return_mask` tuple form). Total recolor test count is now 34.

### Changed

* `mosaicraft.__version__` is now `"0.3.0"`. The top-level package
  re-exports `recolor_region` and `build_oklch_region_mask`.

## [0.2.0] - 2026-04-10

### Added

* **Oklch tile-pool expansion** (`mosaicraft.color_augment`, new module).
  `MosaicGenerator(color_variants=N)` and the CLI flag `--color-variants N`
  rotate every tile through N evenly-spaced Oklch hue shifts (default
  schedule 72° / 144° / 216° / 288°) to expand the candidate pool by (N+1)×
  with zero new photographs. Lightness is preserved exactly, so per-tile
  texture and shading survive the rotation. Exposes `rotate_hue_oklch` and
  `expand_color_variants` from the top-level package.
* **Oklch whole-image recoloring** (`mosaicraft.recolor`, new module). The
  `recolor()` function and `mosaicraft recolor` CLI subcommand rotate a
  finished mosaic through 21 named presets (`blue`, `cyan`, `teal`,
  `purple`, `pink`, `sepia`, `cyberpunk`, ...) or any `#RRGGBB` target,
  preserving the Oklab L channel exactly so the result has no boundary
  artifacts. Supports strength blending, highlight/shadow chroma protection,
  relative hue shifts, and lightness gamma overrides.
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
  identical target, tile pool, and grid. Emits a comparison figure to
  `docs/images/comparison.jpg` plus `docs/assets/bench_outputs/metrics.json`
  with SSIM, blurred SSIM, ΔE2000 (CIEDE2000), edge correlation, cell
  diversity, LPIPS, and wall time. Includes a monkey-patch shim so
  `photomosaic 0.3.1` (skimage-era) runs on modern NumPy / scikit-image.
* `benchmarks/benchmark_pipeline.py --scale large` — 5k / 10k / 20k / 30k
  cell regime against the 1,024-image CC0 pool, up to 8,904 × 10,472 px
  (~93 megapixels).
* `scripts/generate_readme_figures.py` — renders hero, before/after, preset
  comparison, zoom detail, tile-pool sample, 4-painting gallery, and the
  9-preset Oklch recolor gallery from the real bootstrapped assets. Adds a
  `--target {pearl_earring,starry_night,great_wave,red_fuji}` flag.
* README: new sections for `--color-variants`, Oklch recoloring, the large
  benchmark regime, and a rewritten Comparison section with blurred SSIM +
  LPIPS added to the metric table. English is now the primary README and
  Japanese has moved to `README.ja.md`.
* `docs/assets/MANIFEST.json` committed; the raw image files under
  `docs/assets/{paintings,tiles,bench_outputs}/` are gitignored and
  reproducible via `python scripts/download_demo_assets.py`.

### Changed

* Presets consolidated from seven to five (`ultra`, `natural`, `vivid`,
  `tile`, `fast`). The old `vivid_strong` and `vivid_max` were collapsed
  into `vivid` with skin protection always on.
* LPIPS and blurred SSIM added to the `compare_tools.py` metric set. The
  README comparison section is now honest about the pixel-fidelity trade-off
  rather than leading with a metric mosaicraft wins by construction.
* Dropped the procedural CC0 landscape / procedural tile pool used in the
  previous README figures in favor of Wikimedia Commons paintings and the
  1,024-image CC0 photograph pool.

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

[Unreleased]: https://github.com/hinanohart/mosaicraft/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/hinanohart/mosaicraft/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hinanohart/mosaicraft/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hinanohart/mosaicraft/releases/tag/v0.1.0
