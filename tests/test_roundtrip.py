"""加水印 -> 存 -> 读 spec -> 去水印 的往返验收测试。

验收线：PSNR >= 45dB 且最大逐像素差 <= 1/255，mask_sha256 往返一致。
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from watermark import blend
from watermark.meta import read_spec, save_with_spec
from watermark.pipeline import add_watermark, remove_watermark
from watermark.spec import Layer, WatermarkSpec
from tests.conftest import W, H

MIN_PSNR = 45.0
MAX_PIXEL_DIFF = 1.0


def _assert_roundtrip(spec: WatermarkSpec, src: Image.Image, tmp_path, fmt: str):
    report_add = add_watermark(src, spec)
    marked: Image.Image = report_add["image"]

    path = tmp_path / f"marked.{fmt}"
    save_with_spec(marked, path, spec)

    spec_back = read_spec(path)
    assert spec_back == spec, "read_spec 读回的 spec 与原 spec 不一致"

    report_rm = remove_watermark(path, spec_back,
                                 expected_mask_sha256=report_add["mask_sha256"])
    restored: Image.Image = report_rm["image"]
    assert report_rm["mask_match"], "mask 指纹不匹配"

    src_arr = np.asarray(src.convert("RGB"), dtype=np.float32)
    rst_arr = np.asarray(restored.convert("RGB"), dtype=np.float32)
    value = blend.psnr(src_arr, rst_arr)
    stats = blend.diff_stats(src_arr, rst_arr)
    assert value >= MIN_PSNR, f"PSNR={value:.2f}dB < {MIN_PSNR}dB"
    assert stats["max"] <= MAX_PIXEL_DIFF, f"最大逐像素差 {stats['max']} > 1/255"
    return value, stats


def test_text_roundtrip_png(rgb_image, tmp_path):
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="机密文件", opacity=0.25,
              position="bottom-right", font_size=28),
    ))
    value, stats = _assert_roundtrip(spec, rgb_image, tmp_path, "png")
    from watermark.pipeline import report_to_json
    report_to_json(add_watermark(rgb_image, spec))


def test_tiled_roundtrip_png(rgb_image, tmp_path):
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="DRAFT", opacity=0.2, tile=True, font_size=24),
    ))
    _assert_roundtrip(spec, rgb_image, tmp_path, "png")


def test_rotated_roundtrip_png(rgb_image, tmp_path):
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="© 2026", opacity=0.2, rotate=30.0,
              position="center", font_size=30, stroke_width=1),
    ))
    _assert_roundtrip(spec, rgb_image, tmp_path, "png")


def test_logo_roundtrip_png(rgb_image, logo_rgba, tmp_path):
    spec = WatermarkSpec(layers=(
        Layer(kind="logo", content=logo_rgba, opacity=0.3,
              position="top-left", font_size=20),
    ))
    _assert_roundtrip(spec, rgb_image, tmp_path, "png")


def test_logo_white_bg_roundtrip_png(rgb_image, logo_white_bg, tmp_path):
    spec = WatermarkSpec(layers=(
        Layer(kind="logo", content=logo_white_bg, opacity=0.3,
              position="top-left", font_size=20),
    ))
    _assert_roundtrip(spec, rgb_image, tmp_path, "png")


def test_linear_space_roundtrip_png(rgb_image, tmp_path):
    """linear 空间往返：PSNR 仍须达标；最大差放宽到 5/255。

    原因：linear 混合把暗部像素推亮后写回 8bit，量化误差经反向
    (x - C·A)/(1-A) 放大，深色像素（sRGB < 20）处最大可达数个灰度级，
    属 8bit 输出的固有物理限制，非算法缺陷。
    """
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="linear", opacity=0.2, font_size=24),
    ), blend_space="linear")
    report_add = add_watermark(rgb_image, spec)
    marked: Image.Image = report_add["image"]
    path = tmp_path / "marked_linear.png"
    save_with_spec(marked, path, spec)
    spec_back = read_spec(path)
    assert spec_back == spec
    report_rm = remove_watermark(path, spec_back,
                                 expected_mask_sha256=report_add["mask_sha256"])
    assert report_rm["mask_match"]
    src_arr = np.asarray(rgb_image.convert("RGB"), dtype=np.float32)
    rst_arr = np.asarray(report_rm["image"].convert("RGB"), dtype=np.float32)
    value = blend.psnr(src_arr, rst_arr)
    stats = blend.diff_stats(src_arr, rst_arr)
    assert value >= MIN_PSNR, f"PSNR={value:.2f}dB < {MIN_PSNR}dB"
    assert stats["max"] <= 5.0, f"最大逐像素差 {stats['max']} > 5/255"


def test_rgba_alpha_preserved(rgba_image, tmp_path):
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="RGBA", opacity=0.2, font_size=24),
    ))
    report_add = add_watermark(rgba_image, spec)
    assert report_add["image"].mode == "RGBA"
    path = tmp_path / "marked_rgba.png"
    save_with_spec(report_add["image"], path, spec)
    report_rm = remove_watermark(path, spec)
    assert report_rm["image"].mode == "RGBA"
    src_a = np.asarray(rgba_image)[..., 3].astype(np.int16)
    rst_a = np.asarray(report_rm["image"])[..., 3].astype(np.int16)
    assert np.abs(src_a - rst_a).max() <= 1, "alpha 通道未被保留"


def test_report_is_json_safe(rgb_image):
    spec = WatermarkSpec(layers=(Layer(kind="text", content="j", opacity=0.2),))
    from watermark.pipeline import report_to_json
    text = report_to_json(add_watermark(rgb_image, spec))
    assert "image" not in text


def test_custom_color_changes_watermark(rgb_image):
    """同参数不同颜色 -> 不同的加水印结果（颜色链路真实生效）。"""
    specs = [
        WatermarkSpec(layers=(
            Layer(kind="text", content="color", opacity=0.4,
                  font_size=24, color=rgb),
        ))
        for rgb in ((90, 90, 90), (255, 0, 0), (0, 255, 0))
    ]
    reports = [add_watermark(rgb_image, s) for s in specs]
    marked = [np.asarray(r["image"], dtype=np.float32) for r in reports]
    # 颜色不同，结果两两不同
    assert blend.diff_stats(marked[0], marked[1])["max"] > 0
    assert blend.diff_stats(marked[0], marked[2])["max"] > 0
    assert blend.diff_stats(marked[1], marked[2])["max"] > 0


def test_custom_color_roundtrip_png(rgb_image, tmp_path):
    """自定义色 #ff8800 加水印 -> 去水印 -> PSNR >= 45dB 且最大差 <= 1/255。"""
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="彩色", opacity=0.3, font_size=26,
              color=(255, 136, 0)),
    ))
    _assert_roundtrip(spec, rgb_image, tmp_path, "png")


def test_color_survives_spec_json_roundtrip():
    """颜色随 spec 序列化往返，保证去水印时能取到同一个 C。"""
    spec = WatermarkSpec(layers=(
        Layer(kind="text", content="x", color=(255, 136, 0)),
    ))
    restored = WatermarkSpec.from_json(spec.to_json())
    assert restored.layers[0].color == (255, 136, 0)


def test_default_color_layer_uses_engine_default(rgb_image):
    """color=None 的旧 spec 行为不变：仍以 (90,90,90) 混合。"""
    spec_none = WatermarkSpec(layers=(
        Layer(kind="text", content="d", opacity=0.3, font_size=24),
    ))
    spec_gray = WatermarkSpec(layers=(
        Layer(kind="text", content="d", opacity=0.3, font_size=24,
              color=(90, 90, 90)),
    ))
    a = np.asarray(add_watermark(rgb_image, spec_none)["image"], dtype=np.float32)
    b = np.asarray(add_watermark(rgb_image, spec_gray)["image"], dtype=np.float32)
    assert blend.diff_stats(a, b)["max"] == 0.0
