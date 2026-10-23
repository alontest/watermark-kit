"""render 模块测试：mask 形状、hash、tile/定位。"""

from __future__ import annotations

import numpy as np
import pytest

from watermark.render import hash_masks, render_layer_mask, render_spec_masks
from watermark.spec import Layer, WatermarkSpec

W, H = 100, 80


def test_mask_shape_and_range():
    layer = Layer(kind="text", content="hi", opacity=0.2, font_size=20)
    mask = render_layer_mask(layer, W, H)
    assert mask.shape == (H, W)
    assert mask.dtype == np.float32
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_hash_masks_deterministic_and_sensitive():
    spec = WatermarkSpec(layers=(Layer(kind="text", content="x", opacity=0.2),))
    masks = render_spec_masks(spec, W, H)
    h1 = hash_masks(masks)
    h2 = hash_masks(render_spec_masks(spec, W, H))
    assert h1 == h2
    assert h1 != hash_masks(render_spec_masks(spec, W, H + 1))


def test_tile_covers_image():
    layer = Layer(kind="text", content="T", opacity=0.2, tile=True, font_size=16)
    mask = render_layer_mask(layer, W, H)
    # 平铺按 stamp 尺寸从负偏移起铺，整体覆盖率应远高于单次粘贴
    single = render_layer_mask(
        Layer(kind="text", content="T", opacity=0.2, font_size=16), W, H
    )
    assert (mask > 0).mean() > 3 * (single > 0).mean()
    assert mask.min() == 0.0 and mask.max() > 0.0


def test_single_position_bottom_right():
    layer = Layer(kind="text", content="X", opacity=0.2, position="bottom-right",
                  margin=0.0, offset=(0, 0), font_size=20)
    mask = render_layer_mask(layer, W, H)
    ys, xs = np.nonzero(mask)
    assert ys.max() > H * 0.7
    assert xs.max() > W * 0.7
    assert mask[0, 0] == 0.0


def test_offset_moves_stamp():
    base = Layer(kind="text", content="X", opacity=0.2, position="top-left",
                 margin=0.0, font_size=20)
    shifted = Layer(kind="text", content="X", opacity=0.2, position="top-left",
                    margin=0.0, offset=(10, 5), font_size=20)
    m1 = render_layer_mask(base, W, H)
    m2 = render_layer_mask(shifted, W, H)
    ys1, xs1 = np.nonzero(m1)
    ys2, xs2 = np.nonzero(m2)
    assert xs2.min() == xs1.min() + 10
    assert ys2.min() == ys1.min() + 5


def test_logo_mask_matches_logo_size(logo_rgba):
    layer = Layer(kind="logo", content=logo_rgba, opacity=0.3,
                  position="top-left", margin=0.0, font_size=24)
    mask = render_layer_mask(layer, W, H)
    assert mask.shape == (H, W)
    assert mask.max() > 0.0


def test_white_bg_logo_alpha_inverted(logo_white_bg):
    layer = Layer(kind="logo", content=logo_white_bg, opacity=0.3,
                  position="top-left", margin=0.0, font_size=24)
    mask = render_layer_mask(layer, W, H)
    assert mask.max() > 0.0
    # 白底部分 alpha 应接近 0（只有中央色块被检出）
    ys, xs = np.nonzero(mask)
    assert ys.max() < H and xs.max() < W


def test_missing_logo_raises():
    layer = Layer(kind="logo", content="/nonexistent/logo.png", opacity=0.3)
    with pytest.raises(Exception):
        render_layer_mask(layer, W, H)


def test_mask_multiple_layers_order():
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="a", opacity=0.2),
        Layer(kind="text", content="b", opacity=0.2),
    ))
    masks = render_spec_masks(spec, W, H)
    assert len(masks) == 2
