"""Tests for postprocessing filters."""

from __future__ import annotations

import numpy as np

from mosaicraft.postprocess import (
    apply_color_harmony,
    apply_contrast,
    apply_frequency_enhance,
    apply_gamma,
    apply_local_contrast,
    apply_shadow_lift,
    apply_unsharp_mask,
    boost_saturation_hsv,
    detect_skin_mask,
    postprocess,
    protect_skin_luminance,
)
from mosaicraft.presets import get_preset


def _img(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (64, 64, 3), dtype=np.uint8)


def test_gamma_identity() -> None:
    img = _img(0)
    np.testing.assert_array_equal(apply_gamma(img, 1.0), img)


def test_gamma_changes() -> None:
    img = _img(1)
    out = apply_gamma(img, 0.7)
    assert out.dtype == np.uint8
    assert out.shape == img.shape


def test_shadow_lift() -> None:
    img = _img(2)
    out = apply_shadow_lift(img, lift=15)
    assert out.mean() >= img.mean()


def test_detect_skin_mask_range() -> None:
    img = _img(3)
    mask = detect_skin_mask(img)
    assert mask.shape == img.shape[:2]
    assert mask.min() >= 0
    assert mask.max() <= 1


def test_boost_saturation_hsv() -> None:
    img = _img(4)
    out = boost_saturation_hsv(img, 1.5, 1.5)
    assert out.shape == img.shape


def test_protect_skin_luminance() -> None:
    img = _img(5)
    altered = apply_gamma(img, 0.5)
    out = protect_skin_luminance(img, altered, strength=0.5)
    assert out.shape == img.shape


def test_apply_contrast() -> None:
    img = _img(6)
    out = apply_contrast(img, 1.3)
    assert out.shape == img.shape


def test_unsharp_mask() -> None:
    img = _img(7)
    assert apply_unsharp_mask(img, 0.0).shape == img.shape
    assert apply_unsharp_mask(img, 0.5).shape == img.shape


def test_local_contrast_clahe() -> None:
    img = _img(8)
    out = apply_local_contrast(img, 2.0, 8)
    assert out.shape == img.shape


def test_color_harmony() -> None:
    img = _img(9)
    np.testing.assert_array_equal(apply_color_harmony(img, 0), img)
    out = apply_color_harmony(img, 0.2)
    assert out.shape == img.shape


def test_frequency_enhance() -> None:
    out = apply_frequency_enhance(_img(10))
    assert out.shape == (64, 64, 3)


def test_postprocess_full_pipeline_runs() -> None:
    profile = get_preset("ultra")
    out = postprocess(_img(11), profile)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8
