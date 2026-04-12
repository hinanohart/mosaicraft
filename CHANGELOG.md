# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

* README demo targets reduced from four public-domain paintings to
  Vermeer's *Girl with a Pearl Earring* + a Zundamon illustration,
  showing the pipeline works on both fine art and modern illustrations.

### Removed

* **Oklch whole-image recoloring** (`mosaicraft.recolor` module, `recolor()`
  function, `mosaicraft recolor` / `mosaicraft recolor-presets` CLI
  subcommands, 21 named presets, and `examples/recolor_gallery.py`).
  The feature was an optional post-processing step unrelated to the core
  mosaic pipeline and added maintenance weight without clear user demand.
  Pin `mosaicraft<=0.3.2` if you depend on it.
* Starry Night, Great Wave, and Red Fuji demo targets removed from
  `scripts/download_demo_assets.py` and README figures.

## [0.3.2] - 2026-04-11

This is a follow-up to v0.3.1 that fixes the post-publish audit findings —
all of these are blockers a v0.3.1 user would have hit immediately.

### Security

* **`opencv-python` floor raised from `>=4.6` to `>=4.8.1.78`** to close
  CVE-2023-4863 (the libwebp heap overflow / Chrome 0day). The bundled
  opencv-python in v0.3.1 already resolved to 4.13 in practice, but the
  loose floor admitted vulnerable versions for downstream users with
  reproducible-build lockfiles. `numpy`, `scipy`, and `scikit-image`
  floors tightened in the same commit.

### Fixed

* **PyPI README rendered with broken images.** All nine
  `![...](docs/images/X.jpg)` references and every relative repo link
  (`LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`, `src/...`,
  `docs/assets/...`, `benchmarks/...`) have been rewritten to absolute
  `https://raw.githubusercontent.com/hinanohart/mosaicraft/main/...`
  URLs so PyPI's `pypi.org/project/mosaicraft/` page actually shows the
  hero, the diversity chart, and the gallery. v0.3.1's PyPI page was a
  broken-image graveyard.
* **`from mosaicraft import recolor_region` after upgrading from
  v0.3.0** previously raised the bare
  `ImportError: cannot import name 'recolor_region' from 'mosaicraft'`.
  `mosaicraft/__init__.py` now defines a `__getattr__` shim that
  raises a richer error pointing at the v0.3.0 pin and the upstream
  issue tracker, so users who skip the CHANGELOG still find their way
  out.
* **`mosaicraft generate` with a tile pool smaller than the requested
  cell count** previously crashed inside `scipy.optimize.linear_sum_assignment`
  with a generic `ValueError: cost matrix is infeasible`.
  `MosaicGenerator._compose_grid` now validates `n_tiles >= n_cells`
  up-front for Hungarian placement and raises a clear error suggesting
  `--target-tiles N` or `--preset fast`.
* **`mosaicraft recolor-region`** (the withdrawn v0.3.0 subcommand) now
  exits with a custom message pointing at the v0.3.0 pin instead of
  argparse's generic `invalid choice` error.
* **`make_diversity_chart` x-axis math** rounded to a meaningless step
  (the `np.ceil(max_val * 12) / 10` line claimed to round to "the
  nearest 0.083" — it didn't), and could squeeze the largest bar
  against the right margin when the auto-scaled `x_max` was clamped to
  1.0. Replaced with a clean "round up to the next 0.1" rule.
* **`make_diversity_chart` SystemExit on missing metrics.json** killed
  every other figure in the pipeline if the user ran
  `scripts/generate_readme_figures.py` on a fresh clone without first
  running the benchmark. The function now returns `False` and the
  caller logs a soft warning.
* CHANGELOG link table no longer orphans the `[0.3.0]` compare URL —
  v0.3.0 is marked `[YANKED-LIKE]` (the release stays on PyPI for
  pinning compatibility, but every public symbol it added has been
  withdrawn in v0.3.1).

### Added

* README "Upgrading from v0.3.0?" notice at the very top so the
  migration path is the first thing every reader sees, not buried in
  the CHANGELOG.

## [0.3.1] - 2026-04-11

### Removed (deliberate withdrawal of v0.3.0 features)

* **`recolor_region()`, `build_oklch_region_mask()`, and the
  `mosaicraft recolor-region` CLI subcommand** that shipped in v0.3.0
  are gone. The colour-range-mask approach to selective recoloring
  could not produce the quality the README screenshots implied on
  realistic targets — masks were sensitive to source-colour pick,
  morphology kernel size, and lightness gating, and the failure mode
  (a recoloured halo over the wrong region) was hard to detect
  programmatically. Rather than ship a feature that needs hand-tuning
  per image, the API has been withdrawn. A future release may bring it
  back built on a real segmentation backbone (SegFormer / SAM2) under
  an `[ai]` extras group, but that work has not started.
* `scripts/generate_recolor_region_demo.py`,
  `docs/images/selective_recolor_turban.jpg`, and
  `docs/images/selective_recolor_mask.png` are removed for the same
  reason.

  **Migration**: pin `mosaicraft==0.3.0` if you depended on
  `recolor_region`. The 0.3.0 release on PyPI is unchanged.

### Changed

* Default preset is now `vivid` (was `ultra`) — both in
  `MosaicGenerator(...)` and on the CLI's `mosaicraft generate`
  subcommand. The README's "Recommended" label now matches the runtime
  default.
* `make_diversity_chart` reads its values from
  `docs/assets/bench_outputs/metrics.json` instead of a hard-coded list,
  so the published chart can no longer drift away from
  `benchmarks/compare_tools.py` output.
* `make_paintings_gallery` now renders an original-vs-mosaic spread for
  every public-domain painting (was: originals only).
* `make_presets_comparison` covers all five built-in presets (was:
  `vivid`/`ultra`/`natural` only) so the figure matches the README table.
* `_label_bar` (the README figure caption renderer) auto-shrinks the
  font and clamps `x` to a positive padding so labels no longer get
  silently clipped on the left edge.
* `mosaicraft --help` description now mentions Hungarian placement and
  Oklch recoloring, matching the README tagline.
* CLI default preset reflected in core.py / cli.py / module docstring;
  README "Recommended" label is no longer a documentation-only claim.

### Removed

* `docs/images/comparison.jpg`, `comparison_zoom.jpg`,
  `comparison_zoom_pearl.jpg`, and `comparison_zoom_starry.jpg` — these
  were stale screenshots from earlier benchmark runs; the same
  comparison is now told by the live `diversity_chart.jpg` plus the
  4-painting `comparison_four_targets.jpg`.

### Fixed

* README diversity numbers were inconsistent with
  `docs/assets/bench_outputs/metrics.json`. Replaced
  "38–57% diversity" with the actual measured range
  (`fast` 0.341, `vivid` 0.424, `vivid + cv4` 0.384) and removed the
  "every cell is a distinct photograph" overclaim.
* "8.5× more perceptually uniform than CIELAB" was an unsourced number
  that did not appear in Ottosson 2020. Softened to "noticeably more
  uniform" in README, `recolor.py`, `color.py`, and `color_augment.py`.
* `color_augment.py` example previously claimed the default rotation
  schedule was "(90°, 180°, 270°, mid)"; the actual default is
  `(72°, 144°, 216°, 288°)`. Docstring corrected.
* README pipeline diagram and Python API example previously labelled
  the augmentation step "4× geometric"; only one of the four
  augmentations is geometric (horizontal flip), the other three are
  photometric brightness shifts. Re-labelled "4× flip + brightness".
* README "Compared against other photomosaic OSS" section no longer
  implies Hungarian 1:1 placement is unique to mosaicraft (`phomo`,
  `phomo-rs`, and `image-collage-maker` also implement it). The
  differentiator is now stated as the *combination* of Oklab + MKL
  optimal-transport color matching + Oklch hue-rotation pool expansion.

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

[Unreleased]: https://github.com/hinanohart/mosaicraft/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/hinanohart/mosaicraft/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/hinanohart/mosaicraft/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/hinanohart/mosaicraft/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/hinanohart/mosaicraft/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hinanohart/mosaicraft/releases/tag/v0.1.0
