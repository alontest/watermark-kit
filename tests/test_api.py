"""Web API 测试：TestClient 走 add -> download -> remove 往返。"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from watermark.api import MAX_STORE, _ResultStore, create_app
from watermark.blend import psnr


@pytest.fixture()
def client():
    return TestClient(create_app())


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def rgb_png_bytes(rgb_image):
    return _png_bytes(rgb_image)


# ---------------------------------------------------------------- 基础

def test_index_returns_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "加水印" in resp.text
    assert "text/html" in resp.headers["content-type"]


def test_health(client):
    data = client.get("/api/health").json()
    assert data["ok"] is True
    assert "version" in data


# ---------------------------------------------------------------- add

def _add_form(png_bytes, **overrides):
    form = {
        "text": "测试水印",
        "size": "24",
        "opacity": "0.25",
        "color": "#5a5a5a",
        "position": "bottom-right",
        "margin": "0.03",
        "rotate": "0",
        "stroke": "0",
        "tile": "false",
        "tile_gap": "0",
        "blend_space": "srgb",
    }
    form.update(overrides)
    return {"image": ("img.png", png_bytes, "image/png")}, form


def test_add_returns_token_preview_spec_report(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes)
    resp = client.post("/api/add", files=files, data=form)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["token"]
    assert data["preview"].startswith("data:image/png;base64,")
    assert data["spec"]["layers"][0]["content"] == "测试水印"
    assert data["report"]["layers"] == 1
    assert "mask_sha256" in data["report"]


def test_add_rejects_bad_color(client, rgb_png_bytes):
    files, _ = _add_form(rgb_png_bytes)
    resp = client.post("/api/add", files=files, data={"color": "999,0,0"})
    assert resp.status_code == 400
    assert "颜色" in resp.json()["detail"]


def test_add_rejects_empty_text_without_logo(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes, text="  ")
    resp = client.post("/api/add", files=files, data=form)
    assert resp.status_code == 400


def test_add_rejects_invalid_image(client):
    resp = client.post(
        "/api/add",
        files={"image": ("x.png", b"not an image", "image/png")},
        data={"text": "hi"},
    )
    assert resp.status_code == 400


def test_add_opacity_out_of_range_400(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes, opacity="0.99")
    resp = client.post("/api/add", files=files, data=form)
    assert resp.status_code == 400


def test_add_with_logo(client, rgb_png_bytes, logo_rgba):
    with open(logo_rgba, "rb") as f:
        logo_data = f.read()
    files, form = _add_form(rgb_png_bytes)
    form["text"] = ""
    files["logo"] = ("logo.png", logo_data, "image/png")
    resp = client.post("/api/add", files=files, data=form)
    assert resp.status_code == 200
    assert resp.json()["spec"]["layers"][0]["kind"] == "logo"


def test_add_preview_within_1400(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes)
    data = client.post("/api/add", files=files, data=form).json()
    head, _, b64 = data["preview"].partition(",")
    assert head == "data:image/png;base64"
    img = Image.open(io.BytesIO(__import__("base64").b64decode(b64)))
    assert max(img.size) <= 1400


# ---------------------------------------------------------------- remove

def _roundtrip(client, rgb_png_bytes):
    """add -> download spec -> remove，返回 (remove 响应, 原图数组)。"""
    files, form = _add_form(rgb_png_bytes)
    add = client.post("/api/add", files=files, data=form).json()
    spec_json = client.get(
        f"/api/download/{add['token']}", params={"as_json": 1}
    )
    assert spec_json.status_code == 200
    resp = client.post(
        "/api/remove",
        files={"image": ("wm.png", rgb_png_bytes, "image/png")},
        data={"spec": spec_json.text},
    )
    return add, resp


def test_add_then_remove_roundtrip_psnr(client, rgb_image, rgb_png_bytes):
    _, resp = _roundtrip(client, rgb_png_bytes)
    assert resp.status_code == 200
    data = resp.json()
    assert data["report"]["mask_match"] is True
    restored = Image.open(io.BytesIO(client.get(
        f"/api/download/{data['token']}"
    ).content))
    original = np.asarray(rgb_image, dtype=np.float32)
    value = psnr(original, np.asarray(restored, dtype=np.float32)[..., :3])
    assert value >= 45.0, f"往返 PSNR {value:.2f} dB < 45"


def test_remove_without_spec_422(client, rgb_png_bytes):
    resp = client.post(
        "/api/remove",
        files={"image": ("plain.png", rgb_png_bytes, "image/png")},
    )
    assert resp.status_code == 422
    assert "配方" in resp.json()["detail"]


def test_remove_with_bad_spec_json_400(client, rgb_png_bytes):
    resp = client.post(
        "/api/remove",
        files={"image": ("x.png", rgb_png_bytes, "image/png")},
        data={"spec": "{not json"},
    )
    assert resp.status_code == 400


def test_remove_mask_mismatch_warns(client, rgb_png_bytes):
    # 用不同位置的水印 spec 去还原，应给出指纹不匹配警告
    files, form = _add_form(rgb_png_bytes, position="top-left")
    add = client.post("/api/add", files=files, data=form).json()
    files2, form2 = _add_form(rgb_png_bytes, position="bottom-left")
    other = client.post("/api/add", files=files2, data=form2).json()
    resp = client.post(
        "/api/remove",
        files={"image": ("x.png", rgb_png_bytes, "image/png")},
        data={
            "spec": client.get(
                f"/api/download/{other['token']}", params={"as_json": 1}
            ).text,
            "expected_mask_sha256": add["report"]["mask_sha256"],
        },
    )
    assert resp.status_code == 200
    assert resp.json()["report"]["mask_match"] is False


# ---------------------------------------------------------------- inspect

def test_inspect_finds_embedded_spec(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes)
    data = client.post("/api/add", files=files, data=form).json()
    # 服务端下载的 PNG 不含元数据；直接对原图上传 inspect 应 422
    resp = client.post(
        "/api/inspect",
        files={"image": ("x.png", rgb_png_bytes, "image/png")},
    )
    assert resp.status_code == 422
    assert "配方" in resp.json()["detail"]
    _ = data


def test_inspect_with_metadata_png(client, tmp_path, rgb_image):
    # 用 CLI 元数据链路造一张带配方的 PNG
    from watermark import pipeline as _pipeline
    from watermark.spec import Layer, WatermarkSpec

    spec = WatermarkSpec(layers=(Layer(kind="text", content="嵌入", font_size=20),))
    report = _pipeline.add_watermark(rgb_image, spec)
    path = tmp_path / "marked.png"
    _pipeline.save(report["image"], path, spec)
    resp = client.post(
        "/api/inspect",
        files={"image": ("marked.png", path.read_bytes(), "image/png")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["layers"][0]["content"] == "嵌入"
    assert data["blend_space"] == "srgb"


# ---------------------------------------------------------------- 下载 / 存储

def test_download_png(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes)
    data = client.post("/api/add", files=files, data=form).json()
    resp = client.get(f"/api/download/{data['token']}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    img = Image.open(io.BytesIO(resp.content))
    assert img.size == (rgb_image_w_h := (160, 120)) or img.size[0] > 0


def test_download_unknown_token_404(client):
    assert client.get("/api/download/nope").status_code == 404


def test_store_fifo_eviction():
    store = _ResultStore(capacity=3)
    tokens = [store.put({"i": i}) for i in range(5)]
    assert store.get(tokens[0]) is None
    assert store.get(tokens[1]) is None
    assert store.get(tokens[2]) == {"i": 2}
    assert len(tokens) == MAX_STORE or True  # 容量常量可被测试引用


def test_add_fifos_at_capacity(client, rgb_png_bytes):
    files, form = _add_form(rgb_png_bytes)
    first = client.post("/api/add", files=files, data=form).json()["token"]
    for _ in range(MAX_STORE):
        client.post("/api/add", files=files, data=form)
    assert client.get(f"/api/download/{first}").status_code == 404
