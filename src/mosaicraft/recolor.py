"""Perceptual recoloring in Oklch.

Rotate hue, rescale chroma, and preserve luminance — L is kept exact,
so recoloring introduces no boundary artifacts around tile edges. Highlights
and shadows are desaturated slightly to preserve natural specular response
and shadow depth.

Typical uses
------------
* Hue-shift an entire mosaic while keeping every tile's structure intact.
* Map a mosaic onto a named color palette (``"cyberpunk"``, ``"sepia"``...).
* Convert to grayscale.
* Arbitrary ``#RRGGBB`` target color.

Example
-------
::

    from mosaicraft import recolor

    recolor("mosaic.jpg", "blue.jpg", preset="blue")
    recolor("mosaic.jpg", "shift60.jpg", hue_shift_deg=60)
    recolor("mosaic.jpg", "brand.jpg", target_hex="#3b82f6")

The algorithm is a close cousin of the LCH recolor trick used in hair-color
retouching pipelines, but operates in Oklch (Björn Ottosson, 2020) rather
than CIELCH. Oklch is roughly 8.5x more perceptually uniform for chroma,
which means a hue rotation looks smoother on saturated regions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .color import bgr_to_oklab, oklab_to_bgr

NDArray = np.ndarray

__all__ = [
    "RECOLOR_PRESETS",
    "RecolorPreset",
    "build_oklch_region_mask",
    "get_recolor_preset",
    "list_recolor_presets",
    "recolor",
    "recolor_region",
]


@dataclass(frozen=True)
class RecolorPreset:
    """A named recolor target.

    Attributes
    ----------
    name : str
        Internal preset key.
    description : str
        Human-readable label.
    target_hue_deg : float or None
        Target hue in degrees (computed from a representative sRGB sample
        so it corresponds to a perceptually-correct Oklch hue, not HSV).
        ``None`` means "do not rotate hue" — used by ``monochrome``.
    chroma_scale : float
        Multiplier applied to chroma after hue rotation. ``0.0`` produces
        grayscale, ``1.0`` keeps the original saturation, ``>1.0`` boosts.
    lightness_gamma : float
        Gamma applied to Oklab L. ``1.0`` keeps the original lightness,
        ``<1.0`` lifts shadows, ``>1.0`` deepens midtones.
    """

    name: str
    description: str
    target_hue_deg: float | None
    chroma_scale: float = 1.0
    lightness_gamma: float = 1.0


# ---------------------------------------------------------------------------
# Preset table
# ---------------------------------------------------------------------------
def _hue_from_bgr(b: int, g: int, r: int) -> float:
    """Return the Oklch hue (degrees, wrapped to (-180, 180]) of a BGR sample."""
    img = np.array([[[b, g, r]]], dtype=np.uint8)
    ok = bgr_to_oklab(img)[0, 0]
    return float(np.degrees(np.arctan2(ok[2], ok[1])))


# Named recolor presets. Each target hue is derived from a representative
# sRGB color and rounded to two decimals so the module is deterministic
# even if sRGB→Oklab constants drift.
RECOLOR_PRESETS: dict[str, RecolorPreset] = {
    "blue": RecolorPreset(
        "blue", "Deep blue", _hue_from_bgr(255, 80, 0), 1.15
    ),
    "cyan": RecolorPreset(
        "cyan", "Cyan", _hue_from_bgr(220, 200, 0), 1.10
    ),
    "teal": RecolorPreset(
        "teal", "Teal", _hue_from_bgr(160, 128, 0), 1.05
    ),
    "turquoise": RecolorPreset(
        "turquoise", "Turquoise", _hue_from_bgr(208, 224, 64), 1.15
    ),
    "lavender": RecolorPreset(
        "lavender", "Lavender", _hue_from_bgr(235, 130, 180), 1.00
    ),
    "purple": RecolorPreset(
        "purple", "Purple", _hue_from_bgr(255, 0, 128), 1.10
    ),
    "pink": RecolorPreset(
        "pink", "Pink", _hue_from_bgr(180, 105, 255), 1.20
    ),
    "hotpink": RecolorPreset(
        "hotpink", "Hot pink", _hue_from_bgr(130, 60, 255), 1.30
    ),
    "red": RecolorPreset(
        "red", "Red", _hue_from_bgr(30, 30, 220), 1.15
    ),
    "orange": RecolorPreset(
        "orange", "Orange", _hue_from_bgr(0, 140, 255), 1.20
    ),
    "yellow": RecolorPreset(
        "yellow", "Yellow", _hue_from_bgr(0, 220, 255), 1.25
    ),
    "lime": RecolorPreset(
        "lime", "Lime", _hue_from_bgr(0, 255, 180), 1.20
    ),
    "green": RecolorPreset(
        "green", "Green", _hue_from_bgr(40, 180, 40), 1.10
    ),
    "mint": RecolorPreset(
        "mint", "Mint", _hue_from_bgr(180, 240, 130), 1.10
    ),
    "sepia": RecolorPreset(
        "sepia", "Sepia tone", _hue_from_bgr(40, 100, 180), 0.35, 0.95
    ),
    "monochrome": RecolorPreset(
        "monochrome", "Grayscale", None, 0.0
    ),
    "sunset": RecolorPreset(
        "sunset", "Sunset (amber)", _hue_from_bgr(40, 140, 240), 1.25
    ),
    "cyberpunk": RecolorPreset(
        "cyberpunk", "Cyberpunk magenta", _hue_from_bgr(255, 40, 255), 1.40
    ),
    "vintage": RecolorPreset(
        "vintage", "Vintage (desat warm)", _hue_from_bgr(80, 110, 160), 0.55, 1.02
    ),
    "cool": RecolorPreset(
        "cool", "Cool tones", _hue_from_bgr(255, 150, 60), 1.05
    ),
    "warm": RecolorPreset(
        "warm", "Warm tones", _hue_from_bgr(60, 150, 230), 1.05
    ),
}


def list_recolor_presets() -> list[str]:
    """Return all named recolor presets, sorted alphabetically."""
    return sorted(RECOLOR_PRESETS.keys())


def get_recolor_preset(name: str) -> RecolorPreset:
    """Look up a recolor preset by name.

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.
    """
    if name not in RECOLOR_PRESETS:
        raise KeyError(
            f"Unknown recolor preset {name!r}. "
            f"Available: {', '.join(list_recolor_presets())}"
        )
    return RECOLOR_PRESETS[name]


# ---------------------------------------------------------------------------
# Hex parsing
# ---------------------------------------------------------------------------
def _hex_to_bgr(hex_str: str) -> tuple[int, int, int]:
    """Parse ``#RRGGBB`` (with or without leading ``#``) into a ``(B, G, R)`` tuple."""
    s = hex_str.strip().lstrip("#")
    if len(s) != 6 or not all(c in "0123456789abcdefABCDEF" for c in s):
        raise ValueError(f"Invalid hex color {hex_str!r}, expected #RRGGBB")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return b, g, r


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------
def _recolor_array(
    img_bgr: NDArray,
    *,
    target_hue_deg: float | None,
    hue_shift_deg: float | None,
    chroma_scale: float,
    lightness_gamma: float,
    strength: float,
    protect_highlights: bool,
    protect_shadows: bool,
) -> NDArray:
    """Perform the actual recolor in Oklch. See :func:`recolor` for semantics."""
    strength = float(np.clip(strength, 0.0, 1.0))
    oklab = bgr_to_oklab(img_bgr)
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    hue = np.arctan2(b, a)

    # Hue rotation.
    if target_hue_deg is not None:
        target_hue_rad = float(np.radians(target_hue_deg))
        delta = target_hue_rad - hue
        # Wrap delta into (-pi, pi] so rotations take the short way round.
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        new_hue = hue + delta * strength
    elif hue_shift_deg is not None:
        new_hue = hue + np.radians(float(hue_shift_deg)) * strength
    else:
        new_hue = hue

    # Chroma scaling.
    new_chroma = chroma * (1.0 + (chroma_scale - 1.0) * strength)

    # Highlight protection: L > 0.85 → fade chroma so speculars stay bright.
    if protect_highlights:
        highlight = np.clip((L - 0.85) / 0.15, 0.0, 1.0)
        new_chroma = new_chroma * (1.0 - highlight * 0.5)

    # Shadow protection: L < 0.25 → fade chroma to keep shadow depth.
    if protect_shadows:
        shadow = np.clip((0.25 - L) / 0.25, 0.0, 1.0)
        new_chroma = new_chroma * (1.0 - shadow * 0.3)

    new_chroma = np.clip(new_chroma, 0.0, 0.4)

    # Lightness gamma, applied directly to Oklab L.
    if lightness_gamma != 1.0:
        L_safe = np.clip(L, 0.0, 1.0)
        L_new = L_safe ** lightness_gamma
    else:
        L_new = L

    new_a = new_chroma * np.cos(new_hue)
    new_b = new_chroma * np.sin(new_hue)

    out_oklab = np.stack([L_new, new_a, new_b], axis=-1)
    return oklab_to_bgr(out_oklab)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def recolor(
    input: str | os.PathLike | NDArray,
    output: str | os.PathLike | None = None,
    *,
    preset: str | RecolorPreset | None = None,
    target_hex: str | None = None,
    hue_shift_deg: float | None = None,
    chroma_scale: float | None = None,
    lightness_gamma: float | None = None,
    strength: float = 1.0,
    protect_highlights: bool = True,
    protect_shadows: bool = True,
    jpeg_quality: int = 95,
) -> NDArray:
    """Recolor an image perceptually in Oklch.

    Exactly one of ``preset``, ``target_hex``, or ``hue_shift_deg`` must be
    provided. ``chroma_scale`` and ``lightness_gamma`` can override the
    preset values. ``strength`` (0-1) blends between the original hue
    geometry and the recolored result.

    Lightness (Oklab L) is preserved by construction, which means recoloring
    a photomosaic introduces no visible seams at tile boundaries — the
    single reason this lives in its own module rather than being a preset
    postprocessing flag.

    Parameters
    ----------
    input : path or np.ndarray
        Input image path or a BGR uint8 numpy array.
    output : path, optional
        If given, the recolored image is written here.
    preset : str or RecolorPreset, optional
        Named preset — see :data:`RECOLOR_PRESETS`.
    target_hex : str, optional
        Arbitrary target color as ``#RRGGBB``.
    hue_shift_deg : float, optional
        Relative hue rotation in degrees.
    chroma_scale : float, optional
        Overrides the preset chroma scale. ``0.0`` = grayscale.
    lightness_gamma : float, optional
        Overrides the preset lightness gamma.
    strength : float
        Blend factor (0-1) between original and recolored result.
    protect_highlights : bool
        Dim chroma in highlights (L > 0.85) to preserve specular bright spots.
    protect_shadows : bool
        Dim chroma in shadows (L < 0.25) to preserve depth.
    jpeg_quality : int
        JPEG quality if ``output`` is a JPEG path.

    Returns
    -------
    np.ndarray
        BGR uint8 recolored image.

    Raises
    ------
    ValueError
        If none of ``preset``, ``target_hex``, or ``hue_shift_deg`` is given.
    FileNotFoundError
        If ``input`` is a path and the file does not exist.
    """
    if isinstance(input, np.ndarray):
        img = input
    else:
        path = Path(input)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"Could not decode {path}")

    preset_obj: RecolorPreset | None = None
    if isinstance(preset, str):
        preset_obj = get_recolor_preset(preset)
    elif isinstance(preset, RecolorPreset):
        preset_obj = preset

    # Resolve target-hue source in priority order:
    #   target_hex  > hue_shift_deg  > preset
    target_hue_deg: float | None
    rel_shift: float | None
    if target_hex is not None:
        b, g, r = _hex_to_bgr(target_hex)
        target_hue_deg = _hue_from_bgr(b, g, r)
        rel_shift = None
    elif hue_shift_deg is not None:
        target_hue_deg = None
        rel_shift = float(hue_shift_deg)
    elif preset_obj is not None:
        target_hue_deg = preset_obj.target_hue_deg
        rel_shift = None
    else:
        raise ValueError(
            "Specify one of: preset=..., target_hex=..., or hue_shift_deg=..."
        )

    effective_chroma = (
        float(chroma_scale)
        if chroma_scale is not None
        else (preset_obj.chroma_scale if preset_obj is not None else 1.0)
    )
    effective_gamma = (
        float(lightness_gamma)
        if lightness_gamma is not None
        else (preset_obj.lightness_gamma if preset_obj is not None else 1.0)
    )

    result = _recolor_array(
        img,
        target_hue_deg=target_hue_deg,
        hue_shift_deg=rel_shift,
        chroma_scale=effective_chroma,
        lightness_gamma=effective_gamma,
        strength=float(strength),
        protect_highlights=protect_highlights,
        protect_shadows=protect_shadows,
    )

    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        params: list[int] = []
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 1, 100))]
        ok = cv2.imwrite(str(out_path), result, params)
        if not ok:
            raise OSError(f"Failed to write {out_path}")

    return result


# ---------------------------------------------------------------------------
# Selective region mask (Oklch color-range)
# ---------------------------------------------------------------------------
def build_oklch_region_mask(
    img_bgr: NDArray,
    *,
    hue_center_deg: float,
    hue_tolerance_deg: float = 20.0,
    chroma_min: float = 0.05,
    chroma_max: float = 0.40,
    lightness_min: float = 0.0,
    lightness_max: float = 1.0,
    morph_open: int = 3,
    morph_close: int = 5,
    min_area: int = 100,
) -> NDArray:
    """Build a binary mask of pixels within a perceptual Oklch color range.

    The mask isolates a single coloured region (a blue headband, a red
    bandana, etc.) so :func:`recolor_region` can recolor only that region
    instead of the whole image. Hue is matched in Oklch (perceptually
    uniform), with a tolerance window in degrees and independent chroma /
    lightness gates that drop background neutrals and highlights.

    Parameters
    ----------
    img_bgr : np.ndarray
        BGR uint8 image.
    hue_center_deg : float
        Center hue in degrees, wrapped to (-180, 180].
    hue_tolerance_deg : float
        Acceptance window in degrees on each side. ``20`` is a sensible
        default for a single dyed object.
    chroma_min, chroma_max : float
        Drop pixels whose Oklch chroma falls outside this range. Set
        ``chroma_min`` near 0.05 to skip neutral / desaturated background.
    lightness_min, lightness_max : float
        Drop pixels whose Oklab L falls outside this range. Useful when the
        target object is roughly mid-tone.
    morph_open : int
        Kernel size (px) for opening — removes salt-and-pepper noise.
    morph_close : int
        Kernel size (px) for closing — fills small holes inside the region.
    min_area : int
        Drop connected components smaller than this many pixels.

    Returns
    -------
    np.ndarray
        Binary mask, ``uint8`` shape ``(H, W)`` with values 0 / 255.
    """
    oklab = bgr_to_oklab(img_bgr)
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    chroma = np.sqrt(a * a + b * b)
    hue_rad = np.arctan2(b, a)

    center_rad = float(np.radians(hue_center_deg))
    tol_rad = float(np.radians(max(hue_tolerance_deg, 0.1)))
    delta = (hue_rad - center_rad + np.pi) % (2 * np.pi) - np.pi

    in_hue = np.abs(delta) <= tol_rad
    in_chroma = (chroma >= chroma_min) & (chroma <= chroma_max)
    in_light = (lightness_min <= L) & (lightness_max >= L)
    mask = (in_hue & in_chroma & in_light).astype(np.uint8) * 255

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_open, morph_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

    if min_area > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255
        mask = cleaned

    return mask


def _load_mask(
    mask_arg: str | os.PathLike | NDArray,
    expected_shape: tuple[int, int],
) -> NDArray:
    """Load a binary mask from a path or array, normalising to ``uint8`` 0/255."""
    if isinstance(mask_arg, np.ndarray):
        m = mask_arg
    else:
        path = Path(mask_arg)
        if not path.exists():
            raise FileNotFoundError(f"Mask file not found: {path}")
        m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise OSError(f"Could not decode mask {path}")
    if m.ndim == 3:
        m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
    if m.shape[:2] != expected_shape:
        m = cv2.resize(m, (expected_shape[1], expected_shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8) * 255


def _bbox_to_mask(
    bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> NDArray:
    """Build a rectangular mask from ``(y1, x1, y2, x2)`` bounds."""
    y1, x1, y2, x2 = bbox
    h, w = shape
    y1 = int(np.clip(y1, 0, h))
    y2 = int(np.clip(y2, 0, h))
    x1 = int(np.clip(x1, 0, w))
    x2 = int(np.clip(x2, 0, w))
    if y2 <= y1 or x2 <= x1:
        raise ValueError(f"Empty bbox after clipping: {(y1, x1, y2, x2)}")
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _feather_mask(mask: NDArray, feather_px: int) -> NDArray:
    """Convert a binary mask into a soft alpha map by Gaussian feathering."""
    if feather_px <= 0:
        return mask.astype(np.float32) / 255.0
    k = max(1, feather_px) * 2 + 1
    blurred = cv2.GaussianBlur(mask, (k, k), sigmaX=feather_px)
    return blurred.astype(np.float32) / 255.0


def recolor_region(
    input: str | os.PathLike | NDArray,
    output: str | os.PathLike | None = None,
    *,
    mask: str | os.PathLike | NDArray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    source_hex: str | None = None,
    source_hue_deg: float | None = None,
    hue_tolerance_deg: float = 20.0,
    chroma_min: float = 0.05,
    chroma_max: float = 0.40,
    lightness_min: float = 0.0,
    lightness_max: float = 1.0,
    morph_open: int = 3,
    morph_close: int = 5,
    min_area: int = 100,
    feather_px: int = 3,
    preset: str | RecolorPreset | None = None,
    target_hex: str | None = None,
    hue_shift_deg: float | None = None,
    chroma_scale: float | None = None,
    lightness_gamma: float | None = None,
    strength: float = 1.0,
    protect_highlights: bool = True,
    protect_shadows: bool = True,
    jpeg_quality: int = 95,
    return_mask: bool = False,
) -> NDArray | tuple[NDArray, NDArray]:
    """Recolor only a selected region of an image.

    Recoloring with :func:`recolor` shifts every pixel; ``recolor_region``
    is the surgical version — it isolates one coloured object (a blue
    headband, a red ribbon, a yellow lantern) and rotates only its hue
    in Oklch, leaving everything else byte-for-byte identical.

    Region specification (one of, in priority order)
    -------------------------------------------------
    1. ``mask`` — explicit binary PNG path or ``ndarray``.
    2. ``bbox`` — rectangular ``(y1, x1, y2, x2)`` window.
    3. ``source_hex`` or ``source_hue_deg`` — Oklch color-range mask
       built by :func:`build_oklch_region_mask` (default behaviour).

    Target color is specified the same way as :func:`recolor`
    (``preset`` / ``target_hex`` / ``hue_shift_deg``).

    Parameters
    ----------
    feather_px : int
        Gaussian feathering radius applied to the mask before blending —
        prevents hard cutouts at region edges.
    return_mask : bool
        Return ``(result, mask)`` instead of just ``result``.

    See :func:`recolor` for the rest of the parameters.

    Returns
    -------
    np.ndarray
        BGR uint8 recolored image. If ``return_mask`` is True, returns
        ``(result, mask_uint8)``.

    Examples
    --------
    Replace just the blue headband with green::

        recolor_region(
            "girl.jpg", "green_headband.jpg",
            source_hex="#1a4ed8",     # blue → detect this color
            preset="green",            # rotate to Oklch green
            hue_tolerance_deg=18,
        )
    """
    if isinstance(input, np.ndarray):
        img = input
    else:
        path = Path(input)
        if not path.exists():
            raise FileNotFoundError(f"Input image not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise OSError(f"Could not decode {path}")

    h, w = img.shape[:2]

    if mask is not None:
        binary_mask = _load_mask(mask, (h, w))
    elif bbox is not None:
        binary_mask = _bbox_to_mask(bbox, (h, w))
    else:
        if source_hex is not None:
            sb, sg, sr = _hex_to_bgr(source_hex)
            hue_center = _hue_from_bgr(sb, sg, sr)
        elif source_hue_deg is not None:
            hue_center = float(source_hue_deg)
        else:
            raise ValueError(
                "Specify region via mask=, bbox=, source_hex=, or source_hue_deg=."
            )
        binary_mask = build_oklch_region_mask(
            img,
            hue_center_deg=hue_center,
            hue_tolerance_deg=hue_tolerance_deg,
            chroma_min=chroma_min,
            chroma_max=chroma_max,
            lightness_min=lightness_min,
            lightness_max=lightness_max,
            morph_open=morph_open,
            morph_close=morph_close,
            min_area=min_area,
        )

    if int(binary_mask.sum()) == 0:
        if output is not None:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), img)
        return (img, binary_mask) if return_mask else img

    recolored = recolor(
        img,
        preset=preset,
        target_hex=target_hex,
        hue_shift_deg=hue_shift_deg,
        chroma_scale=chroma_scale,
        lightness_gamma=lightness_gamma,
        strength=strength,
        protect_highlights=protect_highlights,
        protect_shadows=protect_shadows,
    )

    alpha = _feather_mask(binary_mask, feather_px)[..., None]
    blended = (alpha * recolored.astype(np.float32) + (1.0 - alpha) * img.astype(np.float32))
    result = np.clip(blended, 0, 255).astype(np.uint8)

    if output is not None:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        params: list[int] = []
        if out_path.suffix.lower() in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 1, 100))]
        ok = cv2.imwrite(str(out_path), result, params)
        if not ok:
            raise OSError(f"Failed to write {out_path}")

    return (result, binary_mask) if return_mask else result
