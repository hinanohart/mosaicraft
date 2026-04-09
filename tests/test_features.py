"""Tests for feature extraction."""

from __future__ import annotations

import cv2
import numpy as np

from mosaicraft.features import FEATURE_DIM, compute_lbp, extract_features


def test_feature_dim_constant() -> None:
    assert FEATURE_DIM == 191


def test_lbp_shape() -> None:
    img = np.random.default_rng(0).integers(0, 256, (16, 16), dtype=np.uint8)
    lbp = compute_lbp(img)
    assert lbp.shape == (14, 14)
    assert lbp.dtype == np.uint8


def test_extract_features_length() -> None:
    rng = np.random.default_rng(1)
    bgr = rng.integers(0, 256, (60, 60, 3), dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    feats = extract_features(lab, 60)
    assert len(feats) == FEATURE_DIM
    assert all(np.isfinite(feats))


def test_extract_features_deterministic() -> None:
    rng = np.random.default_rng(2)
    bgr = rng.integers(0, 256, (40, 40, 3), dtype=np.uint8)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = extract_features(lab, 40)
    b = extract_features(lab, 40)
    assert a == b
