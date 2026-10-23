"""meta 模块测试：PNG tEXt / JPEG EXIF / sidecar 读写。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from watermark.meta import (
    SPEC_TEXT_KEY,
    dump_sidecar,
    read_spec,
    save_with_spec,
    sidecar_path_for,
)
from watermark.spec import SpecError, WatermarkSpec, Layer


def _spec() -> WatermarkSpec:
    return WatermarkSpec(layers=(
        Layer(kind="text", content="meta 测试", opacity=0.2),
    ))


def _image() -> Image.Image:
    rng = np.random.default_rng(11)
    return Image.fromarray(rng.integers(0, 256, (32, 48, 3), dtype=np.uint8), "RGB")


def test_png_text_roundtrip(tmp_path):
    spec = _spec()
    path = tmp_path / "a.png"
    save_with_spec(_image(), path, spec)
    with Image.open(path) as img:
        assert SPEC_TEXT_KEY in img.text
    assert read_spec(path) == spec


def test_jpeg_exif_roundtrip(tmp_path):
    spec = _spec()
    path = tmp_path / "a.jpg"
    save_with_spec(_image(), path, spec, quality=95)
    assert read_spec(path) == spec


def test_read_spec_missing_returns_none(tmp_path):
    path = tmp_path / "plain.png"
    _image().save(path)
    assert read_spec(path) is None


def test_sidecar_roundtrip(tmp_path):
    spec = _spec()
    path = tmp_path / "a.webp"
    save_with_spec(_image(), path, spec)
    sidecar = sidecar_path_for(path)
    assert sidecar.name == "a.webp.wm.json"
    assert sidecar.is_file()
    assert read_spec(path) == spec


def test_dump_sidecar_explicit(tmp_path):
    spec = _spec()
    path = tmp_path / "b.png"
    _image().save(path)
    out = dump_sidecar(spec, path)
    assert out.is_file()
    assert read_spec(path) == spec


def test_corrupt_spec_raises(tmp_path):
    path = tmp_path / "bad.png"
    _image().save(path)
    # 直接塞一个坏 JSON
    from PIL.PngImagePlugin import PngInfo
    info = PngInfo()
    info.add_text(SPEC_TEXT_KEY, "{not json")
    _image().save(path, pnginfo=info)
    with pytest.raises(SpecError):
        read_spec(path)


def test_saved_png_pixels_unchanged(tmp_path):
    src = _image()
    path = tmp_path / "c.png"
    save_with_spec(src, path, _spec())
    with Image.open(path) as img:
        np.testing.assert_array_equal(np.asarray(img), np.asarray(src))
