"""Tests for Oklch tile-pool expansion."""

from __future__ import annotations

import numpy as np
import pytest

from mosaicraft import expand_color_variants, rotate_hue_oklch
from mosaicraft.color import bgr_to_oklab
from mosaicraft.tiles import TileSet, load_tiles


def _mk_tileset(tmp_path, n: int = 8, size: int = 24) -> tuple[TileSet, int]:
    """Write ``n`` synthetic tiles to ``tmp_path`` and load them as a TileSet."""
    import cv2

    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    rng = np.random.default_rng(0)
    for i in range(n):
        # Each tile is a pair of flat colored bands so features are non-trivial
        # but cheap to compute.
        img = np.zeros((size, size, 3), dtype=np.uint8)
        base = rng.integers(80, 200, size=3, dtype=np.uint8)
        img[: size // 2, :, :] = base
        img[size // 2 :, :, :] = 255 - base
        cv2.imwrite(str(tiles_dir / f"tile_{i:03d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return load_tiles(tiles_dir, tile_size=size), size


class TestRotateHueOklch:
    def test_shape_and_dtype_preserved(self) -> None:
        rng = np.random.default_rng(1)
        tile = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        out = rotate_hue_oklch(tile, hue_shift_deg=90)
        assert out.shape == tile.shape
        assert out.dtype == np.uint8

    def test_zero_rotation_is_approximately_identity(self) -> None:
        rng = np.random.default_rng(2)
        tile = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        out = rotate_hue_oklch(tile, hue_shift_deg=0.0)
        # sRGB round-trip through Oklab is not bit-identical but should be
        # within a few gray levels on average.
        assert np.abs(tile.astype(int) - out.astype(int)).mean() < 4.0

    def test_lightness_is_preserved(self) -> None:
        rng = np.random.default_rng(3)
        tile = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        before = bgr_to_oklab(tile)[..., 0]
        out = rotate_hue_oklch(tile, hue_shift_deg=180)
        after = bgr_to_oklab(out)[..., 0]
        # Mean lightness stays within ~3% — the whole point of Oklch rotation.
        assert abs(float(before.mean()) - float(after.mean())) < 0.03

    def test_rotation_actually_changes_hue(self) -> None:
        # A saturated red patch should not stay red after a 120° rotation.
        patch = np.full((16, 16, 3), (40, 40, 220), dtype=np.uint8)  # BGR red
        out = rotate_hue_oklch(patch, hue_shift_deg=120)
        mean_bgr = out.reshape(-1, 3).mean(axis=0)
        # After 120° the dominant channel should no longer be red.
        assert mean_bgr.argmax() != 2

    def test_chroma_scale_zero_is_gray(self) -> None:
        patch = np.full((16, 16, 3), (40, 40, 220), dtype=np.uint8)
        out = rotate_hue_oklch(patch, hue_shift_deg=60, chroma_scale=0.0)
        oklab = bgr_to_oklab(out)
        chroma = np.sqrt(oklab[..., 1] ** 2 + oklab[..., 2] ** 2).mean()
        assert chroma < 0.01


class TestExpandColorVariants:
    def test_n_variants_zero_is_identity(self, tmp_path) -> None:
        ts, size = _mk_tileset(tmp_path, n=6, size=24)
        out = expand_color_variants(ts, n_variants=0, tile_size=size)
        assert len(out.tiles) == len(ts.tiles)
        assert out.features.shape == ts.features.shape

    def test_default_schedule_quadruples_pool(self, tmp_path) -> None:
        ts, size = _mk_tileset(tmp_path, n=6, size=24)
        out = expand_color_variants(ts, n_variants=4, tile_size=size)
        # 1 original + 4 variants = 5x
        assert len(out.tiles) == len(ts.tiles) * 5
        assert out.features.shape[0] == ts.features.shape[0] * 5
        assert out.features.shape[1] == ts.features.shape[1]
        assert len(out.lab_stats) == len(ts.lab_stats) * 5
        assert len(out.grays) == len(ts.grays) * 5
        assert out.oklab_means.shape[0] == ts.oklab_means.shape[0] * 5

    def test_custom_schedule(self, tmp_path) -> None:
        ts, size = _mk_tileset(tmp_path, n=4, size=24)
        out = expand_color_variants(
            ts,
            hue_schedule=(90.0, 180.0, 270.0),
            tile_size=size,
        )
        assert len(out.tiles) == len(ts.tiles) * 4  # 1 + 3

    def test_empty_schedule_is_identity(self, tmp_path) -> None:
        ts, size = _mk_tileset(tmp_path, n=3, size=24)
        out = expand_color_variants(ts, hue_schedule=(), tile_size=size)
        assert len(out.tiles) == len(ts.tiles)

    def test_variants_actually_differ_from_originals(self, tmp_path) -> None:
        """Saturated originals → rotated block should land on different hue.

        Uses saturated synthetic tiles so chroma is high enough that a hue
        rotation produces a visible shift on the Oklab a/b plane. Averaging
        over opposite-color bands — as in ``_mk_tileset`` — cancels most of
        the per-tile mean chroma and masks the rotation; here we build
        single-color saturated patches instead.
        """
        import cv2

        tiles_dir = tmp_path / "sat_tiles"
        tiles_dir.mkdir()
        # Saturated reds / greens / blues / yellows in BGR.
        colors = [(40, 40, 220), (40, 200, 40), (220, 60, 40), (40, 220, 220)]
        for i, (b, g, r) in enumerate(colors):
            img = np.full((24, 24, 3), (b, g, r), dtype=np.uint8)
            cv2.imwrite(str(tiles_dir / f"t_{i}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        ts = load_tiles(tiles_dir, tile_size=24)

        out = expand_color_variants(ts, n_variants=4, tile_size=24)
        n = len(ts.tiles)
        orig_oklab = out.oklab_means[:n]
        rotated_oklab = out.oklab_means[n : 2 * n]
        ab_shift = np.linalg.norm(orig_oklab[:, 1:] - rotated_oklab[:, 1:], axis=1)
        assert ab_shift.mean() > 0.05


class TestPipelineIntegration:
    def test_generator_accepts_color_variants(self, tmp_path) -> None:
        """End-to-end smoke test: the CLI flag reaches core without errors."""
        import cv2

        from mosaicraft import MosaicGenerator

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()
        rng = np.random.default_rng(4)
        for i in range(16):
            img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(tiles_dir / f"t_{i:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        target = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        target_path = tmp_path / "target.jpg"
        cv2.imwrite(str(target_path), target, [cv2.IMWRITE_JPEG_QUALITY, 95])

        gen = MosaicGenerator(
            tile_dir=tiles_dir,
            preset="fast",
            color_variants=2,
        )
        result = gen.generate(target_path, None, target_tiles=16, tile_size=16)
        assert result.image.shape[2] == 3
        assert result.grid_cols > 0
        # 16 tiles * 4 base augs * (1 + 2 color variants) = 192 candidates
        # after the pipeline finishes; the cache should reflect that.
        ts = gen._tile_cache[16]
        assert len(ts.tiles) == 16 * 4 * 3


@pytest.mark.parametrize("angle", [0.0, 45.0, 90.0, 180.0, 360.0])
def test_rotate_any_angle_is_legal_uint8(angle: float) -> None:
    rng = np.random.default_rng(5)
    tile = rng.integers(0, 256, size=(24, 24, 3), dtype=np.uint8)
    out = rotate_hue_oklch(tile, hue_shift_deg=angle)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
