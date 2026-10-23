"""pipeline 模块测试：report 结构、inspect、方向归正。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageOps

from watermark.meta import read_spec
from watermark.pipeline import (
    add_watermark,
    inspect_spec,
    remove_watermark,
    save,
)
from watermark.spec import Layer, WatermarkSpec


def _spec() -> WatermarkSpec:
    return WatermarkSpec(layers=(
        Layer(kind="text", content="P", opacity=0.2, font_size=20),
    ))


def _image(w: int = 90, h: int = 70) -> Image.Image:
    rng = np.random.default_rng(21)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def test_add_report_structure():
    report = add_watermark(_image(), _spec())
    for key in ("mode", "width", "height", "blend_space", "layers",
                "mask_sha256", "unrecoverable_ratio", "predicted_psnr", "warnings"):
        assert key in report
    assert report["width"] == 90 and report["height"] == 70
    assert report["layers"] == 1
    assert report["unrecoverable_ratio"] == 0.0
    assert report["image"].mode == "RGB"


def test_add_accepts_path(rgb_image, tmp_path):
    path = tmp_path / "src.png"
    rgb_image.save(path)
    report = add_watermark(path, _spec())
    assert report["image"].size == rgb_image.size


def test_remove_report_mask_match():
    src = _image()
    spec = _spec()
    report_add = add_watermark(src, spec)
    report_rm = remove_watermark(report_add["image"], spec,
                                 expected_mask_sha256=report_add["mask_sha256"])
    assert report_rm["mask_match"] is True
    assert report_rm["warnings"] == []


def test_remove_warns_on_mask_mismatch():
    src = _image()
    spec = _spec()
    report_add = add_watermark(src, spec)
    report_rm = remove_watermark(report_add["image"], spec,
                                 expected_mask_sha256="0" * 64)
    assert report_rm["mask_match"] is False
    assert any("指纹不匹配" in w for w in report_rm["warnings"])


def test_inspect_spec():
    summary = inspect_spec(_spec(), 64, 48)
    assert summary["width"] == 64 and summary["height"] == 48
    assert len(summary["mask_sha256"]) == 64
    assert len(summary["masks"]) == 1
    assert summary["masks"][0].shape == (48, 64)


def test_save_and_readback(tmp_path):
    src = _image()
    spec = _spec()
    report = add_watermark(src, spec)
    path = tmp_path / "out.png"
    save(report["image"], path, spec)
    assert read_spec(path) == spec


def test_report_json_serializable():
    from watermark.pipeline import report_to_json
    report = add_watermark(_image(), _spec())
    data = json.loads(report_to_json(report))
    assert data["layers"] == 1


def test_exif_orientation_normalized(tmp_path):
    """带 EXIF 方向的图加水印后尺寸按归正后的方向处理。"""
    src = _image(120, 60)  # 横图
    src = ImageOps.exif_transpose(src)
    rotated = src.transpose(Image.ROTATE_90)  # 60x120
    exif = rotated.getexif()
    exif[274] = 6  # Orientation: RightTop
    rotated.save(tmp_path / "oriented.jpg", exif=exif, quality=95)
    report = add_watermark(tmp_path / "oriented.jpg", _spec())
    # orientation=6 归正后回到原始横图方向 120x60
    assert report["width"] == 120 and report["height"] == 60
