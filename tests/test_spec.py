"""spec 校验与 JSON 往返测试。"""

from __future__ import annotations

import json

import pytest

from watermark.spec import Layer, SpecError, WatermarkSpec


def test_layer_json_roundtrip():
    layer = Layer(kind="text", content="水印", opacity=0.3, position="top-left",
                  offset=(3, 4), font_size=28, stroke_width=2, rotate=15.0)
    restored = Layer.from_dict(json.loads(layer.to_json()))
    assert restored == layer


def test_spec_json_roundtrip():
    spec = WatermarkSpec(
        layers=(
            Layer(kind="text", content="a", opacity=0.2),
            Layer(kind="logo", content="/tmp/x.png", opacity=0.5, position="center"),
        ),
        blend_space="linear",
    )
    restored = WatermarkSpec.from_json(spec.to_json())
    assert restored == spec


def test_spec_to_json_ensure_ascii_false():
    spec = WatermarkSpec(layers=(Layer(kind="text", content="水印"),))
    assert "水印" in spec.to_json()


def test_opacity_out_of_range_rejected():
    for bad in (1.0, 0.0, -0.1, 1.5, 0.99):
        with pytest.raises(SpecError):
            Layer(kind="text", content="x", opacity=bad)


def test_opacity_upper_bound_095_allowed():
    layer = Layer(kind="text", content="x", opacity=0.95)
    assert layer.opacity == 0.95


def test_invalid_position_rejected():
    with pytest.raises(SpecError):
        Layer(kind="text", content="x", position="middle-middle")


def test_invalid_blend_space_rejected():
    with pytest.raises(SpecError):
        WatermarkSpec(layers=(Layer(kind="text", content="x"),), blend_space="hsv")


def test_empty_layers_rejected():
    with pytest.raises(SpecError):
        WatermarkSpec(layers=())


def test_bad_version_rejected():
    with pytest.raises(SpecError):
        WatermarkSpec.from_dict({"version": 2, "layers": [{"kind": "text", "content": "x"}]})


def test_font_size_must_be_positive():
    with pytest.raises(SpecError):
        Layer(kind="text", content="x", font_size=0)


def test_logo_tile_rejected():
    with pytest.raises(SpecError):
        Layer(kind="logo", content="x.png", tile=True)


def test_negative_offset_rejected():
    with pytest.raises(SpecError):
        Layer(kind="text", content="x", offset=(-1, 0))


def test_from_dict_unknown_field_rejected():
    with pytest.raises(SpecError):
        Layer.from_dict({"kind": "text", "content": "x", "bogus": 1})
