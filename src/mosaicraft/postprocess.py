"""Postprocessing filters applied after mosaic assembly.

Each filter is a small, self-contained step. The default
:func:`postprocess` applies them in the order proven empirically to give the
best perceptual results: tonal -> spatial -> color -> sharpness.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .color import vibrance_oklch

NDArray = np.ndarray

__all__ = [
    "apply_color_harmony",
    "apply_contrast",
    "apply_frequency_enhance",
    "apply_gamma",
    "apply_local_contrast",
    "apply_shadow_lift",
    "apply_unsharp_mask",
    "boost_saturation_hsv",
    "detect_skin_mask",
    "postprocess",
    "protect_skin_luminance",
]


def apply_gamma(img: NDArray, gamma: float) -> NDArray:
    """Apply a gamma curve via 256-entry LUT."""
    if gamma == 1.0:
        return img
    lut = np.array(
        [((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(img, lut)


def apply_shadow_lift(img: NDArray, lift: int) -> NDArray:
    """Brighten dark pixels while leaving highlights mostly intact."""
    if lift <= 0:
        return img
    f = img.astype(np.float32)
    weight = 1.0 - f / 255.0
    return np.clip(f + lift * weight, 0, 255).astype(np.uint8)


def detect_skin_mask(img_bgr: NDArray) -> NDArray:
    """Soft skin probability mask in [0, 1] using HSV ∩ YCrCb gates."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    hue_mask = ((h >= 0) & (h <= 28)) | (h >= 170)
    sat_mask = (s >= 25) & (s <= 180)
    val_mask = v >= 60
    skin = (hue_mask & sat_mask & val_mask).astype(np.float32)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    cr, cb = ycrcb[:, :, 1], ycrcb[:, :, 2]
    ycrcb_skin = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(
        np.float32
    )
    skin = skin * ycrcb_skin
    return cv2.GaussianBlur(skin, (15, 15), 4)


def boost_saturation_hsv(
    img_bgr: NDArray, boost_cool: float, boost_warm: float
) -> NDArray:
    """Direct HSV saturation boost with separate cool/warm coefficients."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = hsv[:, :, 0]  # OpenCV H is 0-180
    factor_map = np.full_like(h, boost_cool, dtype=np.float32)
    warm_mask = (h <= 25) | (h >= 170)
    factor_map[warm_mask] = boost_warm
    transition = (h > 25) & (h < 40)
    t = (h[transition] - 25.0) / 15.0
    factor_map[transition] = boost_warm + (boost_cool - boost_warm) * t
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor_map, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def protect_skin_luminance(
    img_before: NDArray, img_after: NDArray, strength: float = 0.7
) -> NDArray:
    """Restore the original luminance over detected skin regions."""
    skin_mask = detect_skin_mask(img_before)
    if skin_mask.mean() < 0.01:
        return img_after
    lab_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_restored = lab_after[:, :, 0] * (1.0 - skin_mask * strength) + lab_before[
        :, :, 0
    ] * skin_mask * strength
    lab_after[:, :, 0] = np.clip(l_restored, 0, 255)
    return cv2.cvtColor(lab_after.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_contrast(
    img: NDArray,
    factor: float,
    skin_mask: NDArray | None = None,
    skin_protection: float = 0.0,
) -> NDArray:
    """Contrast around mid-grey, with optional skin protection."""
    f = img.astype(np.float32)
    contrasted = 128 + (f - 128) * factor
    if skin_mask is not None and skin_protection > 0:
        sm3 = np.stack([skin_mask] * 3, axis=-1)
        result = contrasted * (1.0 - sm3 * skin_protection) + f * sm3 * skin_protection
        return np.clip(result, 0, 255).astype(np.uint8)
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def apply_unsharp_mask(
    img: NDArray, strength: float = 0.3, kernel_size: int = 5
) -> NDArray:
    """Unsharp mask sharpening."""
    if strength <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_local_contrast(
    img: NDArray, clip_limit: float = 2.0, grid_size: int = 8
) -> NDArray:
    """CLAHE local contrast on the L channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_color_harmony(img: NDArray, strength: float = 0.1) -> NDArray:
    """Pull a/b channels toward their image mean for color harmony."""
    if strength <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_a = lab[:, :, 1].mean()
    mean_b = lab[:, :, 2].mean()
    lab[:, :, 1] = lab[:, :, 1] * (1 - strength) + mean_a * strength
    lab[:, :, 2] = lab[:, :, 2] * (1 - strength) + mean_b * strength
    lab = np.clip(lab, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def apply_frequency_enhance(img: NDArray) -> NDArray:
    """Mild high-pass enhancement: boost high-frequency texture."""
    f = img.astype(np.float64)
    low = cv2.GaussianBlur(f, (15, 15), 5)
    high = f - low
    low = 128 + (low - 128) * 1.03
    high *= 1.08
    result = low + high
    return np.clip(result, 0, 255).astype(np.uint8)


def postprocess(mosaic: NDArray, profile: dict[str, Any]) -> NDArray:
    """Apply the full postprocessing chain dictated by ``profile``.

    Order: gamma -> shadow lift -> CLAHE -> frequency enhance -> vibrance ->
    HSV saturation -> contrast -> color harmony -> sharpness -> skin restore.
    """
    p = profile
    skin_prot = p.get("skin_protection", 0.0)
    boost_cool = p.get("boost_cool", p["saturation_boost"])
    boost_warm = p.get("boost_warm", p["saturation_boost"])
    skin_lum = p.get("skin_lum_protection", 0.0)

    result = apply_gamma(mosaic, p["gamma"])
    result = apply_shadow_lift(result, p["shadow_lift"])
    result = apply_local_contrast(
        result,
        clip_limit=p["local_contrast_clip"],
        grid_size=p["local_contrast_grid"],
    )
    result = apply_frequency_enhance(result)

    pre_color = result.copy() if skin_lum > 0 else None

    skin_mask: NDArray | None = None
    if skin_prot > 0:
        skin_mask = detect_skin_mask(result)

    result = vibrance_oklch(result, p["vibrance"], skin_mask, skin_prot)
    result = boost_saturation_hsv(result, boost_cool, boost_warm)
    result = apply_contrast(result, p["contrast_boost"], skin_mask, skin_prot)
    result = apply_color_harmony(result, p.get("color_harmony", 0.1))
    result = apply_unsharp_mask(result, p["sharpness"])

    if pre_color is not None and skin_lum > 0:
        result = protect_skin_luminance(pre_color, result, skin_lum)

    return result
