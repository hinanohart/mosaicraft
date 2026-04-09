"""Tunable preset profiles for the mosaic generator.

Each preset is a fully-specified set of parameters covering placement,
color transfer, blending, and postprocessing. The keys are stable across
versions; renaming will be a breaking change.

Pick a preset that matches the *style* you want and pass it to
:class:`mosaicraft.MosaicGenerator`. You can also build a custom dict.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

__all__ = ["PRESETS", "get_preset", "list_presets"]

PRESETS: dict[str, dict[str, Any]] = {
    "ultra": {
        "description": "Highest quality. Hungarian + Laplacian blend.",
        "placement": "hungarian",
        "neighbor_swap_rounds": 8,
        "color_transfer": "hybrid",
        "reinhard_l": 0.60,
        "reinhard_ab": 0.42,
        "feather_width": 0,
        "laplacian_blend": True,
        "laplacian_levels": 4,
        "base_blend_min": 0.05,
        "base_blend_max": 0.16,
        "gamma": 0.95,
        "shadow_lift": 8,
        "saturation_boost": 1.45,
        "vibrance": 0.35,
        "contrast_boost": 1.04,
        "sharpness": 0.25,
        "local_contrast_clip": 1.5,
        "local_contrast_grid": 16,
        "color_harmony": 0.15,
        "skin_protection": 0.65,
    },
    "natural": {
        "description": "Natural, photo-realistic look.",
        "placement": "hungarian",
        "neighbor_swap_rounds": 5,
        "color_transfer": "adaptive_reinhard",
        "reinhard_l": 0.65,
        "reinhard_ab": 0.48,
        "feather_width": 2,
        "laplacian_blend": False,
        "laplacian_levels": 3,
        "base_blend_min": 0.07,
        "base_blend_max": 0.18,
        "gamma": 0.95,
        "shadow_lift": 8,
        "saturation_boost": 1.40,
        "vibrance": 0.32,
        "contrast_boost": 1.04,
        "sharpness": 0.23,
        "local_contrast_clip": 1.5,
        "local_contrast_grid": 16,
        "color_harmony": 0.12,
        "skin_protection": 0.70,
    },
    "vivid": {
        "description": "Vivid MKL optimal transport with skin protection.",
        "placement": "hungarian",
        "neighbor_swap_rounds": 5,
        "color_transfer": "mkl_hybrid",
        "reinhard_l": 0.50,
        "reinhard_ab": 0.35,
        "feather_width": 1,
        "laplacian_blend": False,
        "laplacian_levels": 3,
        "base_blend_min": 0.05,
        "base_blend_max": 0.14,
        "gamma": 0.94,
        "shadow_lift": 9,
        "saturation_boost": 1.75,
        "vibrance": 0.48,
        "contrast_boost": 1.06,
        "sharpness": 0.28,
        "local_contrast_clip": 1.8,
        "local_contrast_grid": 14,
        "color_harmony": 0.07,
        "skin_protection": 0.75,
        "boost_cool": 1.75,
        "boost_warm": 1.45,
        "skin_lum_protection": 0.65,
    },
    "tile": {
        "description": "Emphasizes individual tiles. Strongest mosaic look.",
        "placement": "hungarian",
        "neighbor_swap_rounds": 5,
        "color_transfer": "adaptive_reinhard",
        "reinhard_l": 0.40,
        "reinhard_ab": 0.28,
        "feather_width": 0,
        "laplacian_blend": False,
        "laplacian_levels": 3,
        "base_blend_min": 0.03,
        "base_blend_max": 0.10,
        "gamma": 0.96,
        "shadow_lift": 9,
        "saturation_boost": 1.48,
        "vibrance": 0.38,
        "contrast_boost": 1.03,
        "sharpness": 0.25,
        "local_contrast_clip": 1.5,
        "local_contrast_grid": 16,
        "color_harmony": 0.05,
        "skin_protection": 0.65,
    },
    "fast": {
        "description": "Faster preset (no Hungarian, no rerank).",
        "placement": "faiss_diffusion",
        "error_diffusion_strength": 0.85,
        "neighbor_swap_rounds": 0,
        "color_transfer": "adaptive_reinhard",
        "reinhard_l": 0.55,
        "reinhard_ab": 0.40,
        "feather_width": 1,
        "laplacian_blend": False,
        "laplacian_levels": 3,
        "base_blend_min": 0.06,
        "base_blend_max": 0.15,
        "gamma": 0.96,
        "shadow_lift": 8,
        "saturation_boost": 1.40,
        "vibrance": 0.30,
        "contrast_boost": 1.04,
        "sharpness": 0.22,
        "local_contrast_clip": 1.5,
        "local_contrast_grid": 16,
        "color_harmony": 0.10,
        "skin_protection": 0.50,
    },
}


def list_presets() -> list[str]:
    """Return all available preset names."""
    return sorted(PRESETS.keys())


def get_preset(name: str) -> dict[str, Any]:
    """Return a deep copy of the named preset.

    Raises
    ------
    KeyError
        If ``name`` is not a known preset.
    """
    if name not in PRESETS:
        raise KeyError(
            f"Unknown preset {name!r}. Available: {', '.join(list_presets())}"
        )
    return deepcopy(PRESETS[name])
