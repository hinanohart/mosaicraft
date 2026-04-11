"""Tests for perceptual Oklch recoloring."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from mosaicraft import (
    RECOLOR_PRESETS,
    RecolorPreset,
    build_oklch_region_mask,
    get_recolor_preset,
    list_recolor_presets,
    recolor,
    recolor_region,
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


def _two_color_image(h: int = 60, w: int = 60) -> np.ndarray:
    """Image with a saturated blue background and a saturated green patch."""
    img = np.full((h, w, 3), [200, 50, 30], dtype=np.uint8)  # blue (BGR)
    img[15:35, 20:50] = [50, 200, 50]  # green patch
    return img


class TestRegionMask:
    def test_isolates_green_patch(self) -> None:
        img = _two_color_image()
        # Green BGR (50,200,50) → derive its Oklch hue and use as center.
        from mosaicraft.recolor import _hue_from_bgr  # type: ignore[attr-defined]

        hue = _hue_from_bgr(50, 200, 50)
        mask = build_oklch_region_mask(
            img, hue_center_deg=hue, hue_tolerance_deg=25,
            chroma_min=0.02, min_area=10,
        )
        # Mask should cover the 20*30 = 600 patch pixels (allowing morphology slop).
        assert int(mask.sum() / 255) > 400
        # Background must be excluded.
        assert mask[0, 0] == 0

    def test_min_area_filters_noise(self) -> None:
        img = _two_color_image()
        # Add a single tiny green dot far away.
        img[55, 55] = [50, 200, 50]
        from mosaicraft.recolor import _hue_from_bgr  # type: ignore[attr-defined]

        hue = _hue_from_bgr(50, 200, 50)
        mask = build_oklch_region_mask(
            img, hue_center_deg=hue, hue_tolerance_deg=25,
            chroma_min=0.02, min_area=200, morph_open=1, morph_close=1,
        )
        assert mask[55, 55] == 0  # tiny island dropped

    def test_chroma_gate_drops_neutrals(self) -> None:
        img = np.full((30, 30, 3), 128, dtype=np.uint8)  # neutral grey
        mask = build_oklch_region_mask(
            img, hue_center_deg=0.0, hue_tolerance_deg=180,
            chroma_min=0.05,
        )
        assert int(mask.sum()) == 0


class TestRecolorRegion:
    def test_unmask_area_preserved_color(self) -> None:
        img = _two_color_image()
        out = recolor_region(
            img,
            source_hex="#32C832",  # green target
            preset="red",
            hue_tolerance_deg=25,
            chroma_min=0.02,
            min_area=10,
            feather_px=0,
        )
        # Top-left corner is blue background — must stay byte-identical.
        assert tuple(out[0, 0]) == tuple(img[0, 0])

    def test_region_actually_recolored(self) -> None:
        img = _two_color_image()
        out = recolor_region(
            img,
            source_hex="#32C832",
            preset="red",
            hue_tolerance_deg=25,
            chroma_min=0.02,
            min_area=10,
            feather_px=0,
        )
        # Patch center should now be reddish (R > B in BGR).
        center = out[25, 35]
        assert center[2] > center[0]

    def test_explicit_mask_path(self, tmp_path: Path) -> None:
        img = _two_color_image()
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask[15:35, 20:50] = 255
        mask_path = tmp_path / "m.png"
        cv2.imwrite(str(mask_path), mask)
        out = recolor_region(img, mask=mask_path, preset="red", feather_px=0)
        assert tuple(out[0, 0]) == tuple(img[0, 0])
        # Inside the mask the channels must change.
        assert not np.array_equal(out[25, 35], img[25, 35])

    def test_bbox_region(self) -> None:
        img = _two_color_image()
        out = recolor_region(
            img,
            bbox=(15, 20, 35, 50),
            preset="red",
            feather_px=0,
        )
        assert tuple(out[0, 0]) == tuple(img[0, 0])
        assert not np.array_equal(out[25, 35], img[25, 35])

    def test_bbox_clipping(self) -> None:
        img = _two_color_image()
        # Bbox that extends past image bounds should clip silently.
        out = recolor_region(
            img,
            bbox=(-5, -5, 1000, 1000),
            preset="blue",
            feather_px=0,
        )
        assert out.shape == img.shape

    def test_empty_mask_returns_input(self) -> None:
        img = _two_color_image()
        # Pick a hue with no pixels in the image.
        out = recolor_region(
            img,
            source_hue_deg=0.0,
            hue_tolerance_deg=1.0,
            chroma_min=0.99,
            preset="red",
            feather_px=0,
        )
        assert np.array_equal(out, img)

    def test_no_region_specified_raises(self) -> None:
        img = _two_color_image()
        with pytest.raises(ValueError, match="Specify region"):
            recolor_region(img, preset="red")

    def test_return_mask_tuple(self) -> None:
        img = _two_color_image()
        out, mask = recolor_region(
            img,
            source_hex="#32C832",
            preset="red",
            hue_tolerance_deg=25,
            chroma_min=0.02,
            min_area=10,
            return_mask=True,
        )
        assert mask.shape == img.shape[:2]
        assert mask.dtype == np.uint8
        assert out.shape == img.shape

    def test_writes_file(self, tmp_path: Path) -> None:
        img = _two_color_image()
        in_path = tmp_path / "in.png"
        out_path = tmp_path / "out.png"
        cv2.imwrite(str(in_path), img)
        recolor_region(
            in_path, out_path,
            source_hex="#32C832",
            preset="red",
            hue_tolerance_deg=25,
            chroma_min=0.02,
            min_area=10,
        )
        assert out_path.exists()

    def test_feather_blends_edges(self) -> None:
        img = _two_color_image()
        # With feather > 0, edge pixels should be a soft blend, not a hard cut.
        out = recolor_region(
            img,
            source_hex="#32C832",
            preset="red",
            hue_tolerance_deg=25,
            chroma_min=0.02,
            min_area=10,
            feather_px=4,
        )
        # An edge-adjacent pixel should differ slightly from raw input.
        edge = out[14, 35]
        raw = img[14, 35]
        assert not np.array_equal(edge, raw)
