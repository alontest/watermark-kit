"""blend 模块单元测试。"""

from __future__ import annotations

import numpy as np
import pytest

from watermark import blend


def test_apply_then_remove_recovers_exact():
    rng = np.random.default_rng(7)
    img = rng.random((10, 10, 3)).astype(np.float32) * 255.0
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[2:5, 3:8] = 1.0
    color = np.asarray([90, 90, 90], dtype=np.float32)
    opacity = 0.3
    out, _ = blend.apply_layer(img, mask, color, opacity)
    rec, unrecoverable = blend.remove_layer(out, mask, color, opacity)
    assert unrecoverable == 0.0
    np.testing.assert_allclose(rec, img, atol=1e-3)


def test_unrecoverable_ratio_counted():
    img = np.full((10, 10, 3), 128.0, dtype=np.float32)
    mask = np.zeros((10, 10), dtype=np.float32)
    mask[0:2, :] = 1.0  # opacity 0.95 -> strength 0.95 < 1-eps，可逆
    color = np.zeros(3, dtype=np.float32)
    out, _ = blend.apply_layer(img, mask, color, 0.95)
    _, unrecoverable = blend.remove_layer(out, mask, color, 0.95)
    assert unrecoverable == 0.0


def test_psnr_identical_is_inf():
    img = np.full((4, 4, 3), 100.0)
    assert blend.psnr(img, img) == float("inf")


def test_psnr_known_value():
    ref = np.full((10, 10), 128.0)
    tgt = ref + 1.0
    expected = 10 * np.log10(255**2 / 1.0)
    assert blend.psnr(ref, tgt) == pytest.approx(expected)


def test_psnr_shape_mismatch_raises():
    with pytest.raises(ValueError):
        blend.psnr(np.zeros((2, 2)), np.zeros((3, 3)))


def test_diff_stats_values():
    ref = np.full((5, 5), 10.0)
    tgt = np.full((5, 5), 13.0)
    stats = blend.diff_stats(ref, tgt)
    assert stats["max"] == 3.0
    assert stats["mean"] == 3.0
    assert stats["rmse"] == pytest.approx(3.0)


def test_apply_convex_never_exceeds_bounds():
    rng = np.random.default_rng(3)
    img = rng.random((8, 8, 3)).astype(np.float32) * 255.0
    mask = np.ones((8, 8), dtype=np.float32)
    color = np.asarray([255, 0, 128], dtype=np.float32)
    out, _ = blend.apply_layer(img, mask, color, 0.95)
    assert out.min() >= 0.0 and out.max() <= 255.0
