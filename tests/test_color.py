"""Tests for the color module: Oklab conversions and color transfer."""

from __future__ import annotations

import numpy as np
import pytest

from mosaicraft.color import (
    apply_color_transfer,
    bgr_to_oklab,
    histogram_transfer,
    mkl_transfer,
    oklab_to_bgr,
    reinhard_transfer,
    vibrance_oklch,
)


def _random_image(seed: int = 0, h: int = 32, w: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


class TestOklabRoundTrip:
    def test_pure_colors_round_trip(self) -> None:
        colors = np.array(
            [
                [[0, 0, 0]],
                [[255, 255, 255]],
                [[255, 0, 0]],
                [[0, 255, 0]],
                [[0, 0, 255]],
            ],
            dtype=np.uint8,
        )
        recovered = oklab_to_bgr(bgr_to_oklab(colors))
        assert np.max(np.abs(colors.astype(int) - recovered.astype(int))) <= 2

    def test_random_round_trip(self) -> None:
        img = _random_image(seed=42, h=64, w=64)
        recovered = oklab_to_bgr(bgr_to_oklab(img))
        # sRGB <-> linear has small numerical drift; allow a tolerance.
        diff = np.abs(img.astype(int) - recovered.astype(int))
        assert diff.mean() < 1.5
        assert diff.max() <= 4

    def test_oklab_l_in_unit_range(self) -> None:
        img = _random_image(seed=7)
        oklab = bgr_to_oklab(img)
        assert oklab.shape == img.shape
        assert (oklab[..., 0] >= 0).all()
        assert (oklab[..., 0] <= 1.001).all()


class TestColorTransfer:
    def test_reinhard_returns_uint8(self) -> None:
        import cv2

        tile = _random_image(seed=1)
        target = _random_image(seed=2)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float64)
        out = reinhard_transfer(tile, target_lab, 0.6, 0.42)
        assert out.dtype == np.uint8
        assert out.shape == tile.shape

    def test_mkl_transfers_distribution(self) -> None:
        rng = np.random.default_rng(0)
        # Tile = mostly red, target = mostly blue.
        tile = np.zeros((32, 32, 3), dtype=np.uint8)
        tile[..., 2] = 200 + rng.integers(0, 30, (32, 32))
        target = np.zeros((32, 32, 3), dtype=np.uint8)
        target[..., 0] = 200 + rng.integers(0, 30, (32, 32))
        out = mkl_transfer(tile, target, strength=1.0)
        # The output should be more blue than the original tile.
        assert out[..., 0].mean() > tile[..., 0].mean()

    def test_histogram_transfer_preserves_shape(self) -> None:
        out = histogram_transfer(_random_image(0), _random_image(1), 0.6)
        assert out.shape == (32, 32, 3)

    def test_apply_color_transfer_dispatch(self) -> None:
        import cv2

        tile = _random_image(0)
        target = _random_image(1)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float64)
        profile = {"reinhard_l": 0.5, "reinhard_ab": 0.4}
        for method in ("none", "adaptive_reinhard", "histogram", "hybrid", "mkl", "mkl_hybrid"):
            out = apply_color_transfer(tile, target, target_lab, method, profile)
            assert out.shape == tile.shape
            assert out.dtype == np.uint8

    def test_apply_color_transfer_unknown(self) -> None:
        import cv2

        tile = _random_image(0)
        target = _random_image(1)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float64)
        with pytest.raises(ValueError, match="Unknown color transfer method"):
            apply_color_transfer(tile, target, target_lab, "bogus", {})


class TestVibrance:
    def test_vibrance_increases_chroma(self) -> None:
        img = _random_image(seed=10)
        boosted = vibrance_oklch(img, amount=0.6)
        assert boosted.shape == img.shape

        # Mean Oklab chroma should be higher after boost.
        before = bgr_to_oklab(img)
        after = bgr_to_oklab(boosted)
        c_before = np.sqrt(before[..., 1] ** 2 + before[..., 2] ** 2).mean()
        c_after = np.sqrt(after[..., 1] ** 2 + after[..., 2] ** 2).mean()
        assert c_after >= c_before

    def test_vibrance_amount_zero_is_noop(self) -> None:
        img = _random_image(seed=11)
        out = vibrance_oklch(img, amount=0.0)
        # Mean intensity should not move materially.
        assert abs(int(out.mean()) - int(img.mean())) <= 3
