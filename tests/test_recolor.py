"""Tests for perceptual Oklch recoloring."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mosaicraft import (
    RECOLOR_PRESETS,
    RecolorPreset,
    get_recolor_preset,
    list_recolor_presets,
    recolor,
)
from mosaicraft.color import bgr_to_oklab


def _gradient_image(h: int = 48, w: int = 48) -> np.ndarray:
    """A colorful gradient so recoloring has something to shift."""
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return np.stack(
        [
            np.broadcast_to(x, (h, w)),
            np.broadcast_to(y, (h, w)),
            np.broadcast_to((x + y) // 2, (h, w)),
        ],
        axis=-1,
    ).astype(np.uint8)


class TestPresetTable:
    def test_all_presets_listed(self) -> None:
        names = list_recolor_presets()
        assert "blue" in names
        assert "monochrome" in names
        assert "cyberpunk" in names
        assert names == sorted(names)

    def test_get_preset_returns_frozen_dataclass(self) -> None:
        p = get_recolor_preset("blue")
        assert isinstance(p, RecolorPreset)
        assert p.name == "blue"
        assert p.target_hue_deg is not None

    def test_unknown_preset_raises(self) -> None:
        with pytest.raises(KeyError, match="Unknown recolor preset"):
            get_recolor_preset("not-a-real-preset")

    def test_monochrome_has_zero_chroma(self) -> None:
        p = get_recolor_preset("monochrome")
        assert p.chroma_scale == 0.0
        assert p.target_hue_deg is None

    def test_preset_count(self) -> None:
        # Sanity: we claim "more than a handful" in the README.
        assert len(RECOLOR_PRESETS) >= 20


class TestRecolorCore:
    def test_requires_a_target(self) -> None:
        img = _gradient_image()
        with pytest.raises(ValueError, match="Specify one of"):
            recolor(img)

    def test_preset_returns_same_shape(self) -> None:
        img = _gradient_image()
        out = recolor(img, preset="blue")
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_hue_shift_returns_same_shape(self) -> None:
        img = _gradient_image()
        out = recolor(img, hue_shift_deg=90)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_hex_target_returns_same_shape(self) -> None:
        img = _gradient_image()
        out = recolor(img, target_hex="#3b82f6")
        assert out.shape == img.shape

    def test_bad_hex_raises(self) -> None:
        img = _gradient_image()
        with pytest.raises(ValueError, match="Invalid hex color"):
            recolor(img, target_hex="nope")

    def test_monochrome_has_near_zero_chroma(self) -> None:
        img = _gradient_image()
        out = recolor(img, preset="monochrome")
        oklab = bgr_to_oklab(out)
        chroma = np.sqrt(oklab[..., 1] ** 2 + oklab[..., 2] ** 2).mean()
        assert chroma < 0.01

    def test_lightness_is_preserved(self) -> None:
        """L stays within 3% after a hue rotation — the core promise."""
        img = _gradient_image()
        before = bgr_to_oklab(img)[..., 0]
        out = recolor(img, preset="blue")
        after = bgr_to_oklab(out)[..., 0]
        # Mean L should be essentially unchanged.
        assert abs(float(after.mean()) - float(before.mean())) < 0.03

    def test_blue_preset_actually_turns_blue(self) -> None:
        img = _gradient_image()
        out = recolor(img, preset="blue")
        # Mean BGR should now be blue-dominant.
        mean_bgr = out.reshape(-1, 3).mean(axis=0)
        assert mean_bgr[0] > mean_bgr[2]  # B > R

    def test_red_preset_actually_turns_red(self) -> None:
        img = _gradient_image()
        out = recolor(img, preset="red")
        mean_bgr = out.reshape(-1, 3).mean(axis=0)
        assert mean_bgr[2] > mean_bgr[0]  # R > B

    def test_strength_zero_is_noop(self) -> None:
        img = _gradient_image()
        out = recolor(img, preset="blue", strength=0.0)
        # With strength=0 the result should match the input very closely.
        diff = np.abs(img.astype(int) - out.astype(int))
        assert diff.mean() < 3.0

    def test_explicit_chroma_scale_override(self) -> None:
        img = _gradient_image()
        gray = recolor(img, preset="blue", chroma_scale=0.0)
        oklab = bgr_to_oklab(gray)
        chroma = np.sqrt(oklab[..., 1] ** 2 + oklab[..., 2] ** 2).mean()
        assert chroma < 0.01


class TestRecolorFile:
    def test_writes_file(self, tmp_path: Path) -> None:
        img = _gradient_image()
        in_path = tmp_path / "in.png"
        out_path = tmp_path / "out.png"
        cv2.imwrite(str(in_path), img)
        recolor(in_path, out_path, preset="sepia")
        assert out_path.exists()
        loaded = cv2.imread(str(out_path))
        assert loaded is not None
        assert loaded.shape == img.shape

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            recolor(tmp_path / "does_not_exist.jpg", preset="blue")

    def test_jpeg_output(self, tmp_path: Path) -> None:
        img = _gradient_image()
        in_path = tmp_path / "in.png"
        out_path = tmp_path / "out.jpg"
        cv2.imwrite(str(in_path), img)
        recolor(in_path, out_path, preset="cyberpunk", jpeg_quality=80)
        assert out_path.exists()


class TestProtection:
    def test_highlights_dimmed(self) -> None:
        """Bright pixels should not blow out after a hue shift."""
        bright = np.full((32, 32, 3), 245, dtype=np.uint8)
        out = recolor(bright, preset="red")
        # A near-white pixel should stay very bright after recolor.
        assert out.mean() > 200

    def test_shadows_dimmed(self) -> None:
        """Dark pixels should retain their depth."""
        dark = np.full((32, 32, 3), 10, dtype=np.uint8)
        out = recolor(dark, preset="red")
        assert out.mean() < 50
