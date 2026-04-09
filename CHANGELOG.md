# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
