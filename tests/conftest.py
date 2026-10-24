"""测试公共夹具：随机 RGB / RGBA 图、logo 图、已登录的 Web 客户端。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

W, H = 160, 120
SEED = 2026

# Web 测试统一使用的口令与同源请求头（TestClient 的 base_url 为 http://testserver）
AUTH_PASSWORD = "fixture-pw-2026"
TEST_ORIGIN = "http://testserver"


def _random_rgb(seed: int, w: int = W, h: int = H) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


@pytest.fixture()
def rgb_image() -> Image.Image:
    return _random_rgb(SEED)


@pytest.fixture()
def rgba_image() -> Image.Image:
    rng = np.random.default_rng(SEED)
    rgb = rng.integers(0, 256, (H, W, 3), dtype=np.uint8)
    alpha = rng.integers(0, 256, (H, W, 1), dtype=np.uint8)
    return Image.fromarray(
        np.concatenate([rgb, alpha], axis=-1), "RGBA"
    )


@pytest.fixture()
def logo_rgba(tmp_path) -> str:
    """一个带 alpha 通道的 24x12 logo 文件，返回路径。"""
    rng = np.random.default_rng(SEED + 1)
    rgb = rng.integers(0, 256, (12, 24, 3), dtype=np.uint8)
    alpha = np.full((12, 24, 1), 200, dtype=np.uint8)
    logo = Image.fromarray(np.concatenate([rgb, alpha], axis=-1), "RGBA")
    path = tmp_path / "logo.png"
    logo.save(path)
    return str(path)


@pytest.fixture()
def logo_white_bg(tmp_path) -> str:
    """白底无 alpha 的 logo 文件（用于四角检测反相）。"""
    rng = np.random.default_rng(SEED + 2)
    rgb = np.full((12, 24, 3), 255, dtype=np.uint8)
    rgb[3:9, 5:19] = rng.integers(0, 256, (6, 14, 3), dtype=np.uint8)
    logo = Image.fromarray(rgb, "RGB")
    path = tmp_path / "logo_white.png"
    logo.save(path)
    return str(path)


# ---------------------------------------------------------------- Web 鉴权

@pytest.fixture(autouse=True)
def _isolated_config_dir(tmp_path, monkeypatch):
    """隔离真实凭据文件，并确保 WATERMARK_PASSWORD 不干扰测试。"""
    monkeypatch.delenv("WATERMARK_PASSWORD", raising=False)
    monkeypatch.setenv("WATERMARK_CONFIG_DIR", str(tmp_path / "config"))


@pytest.fixture()
def auth_service(tmp_path):
    """空凭据目录上的认证服务（未配置口令）。"""
    from watermark.auth import AuthService

    return AuthService(path=tmp_path / "config" / "auth.json")


@pytest.fixture()
def auth_app(auth_service):
    """未登录的应用实例（用于测试登录/首次设置链路）。"""
    from watermark.api import create_app

    return create_app(auth_service)


def _login(client, password: str = AUTH_PASSWORD, *, setup: bool = True):
    """调用 /api/login 并返回响应；setup=True 时附带确认口令。"""
    form = {"password": password}
    if setup:
        form["confirm"] = password
    return client.post("/api/login", data=form, headers={"Origin": TEST_ORIGIN})


@pytest.fixture()
def auth_client(auth_service):
    """已完成首次设置并登录的 TestClient。

    默认带上同源 Origin，模拟浏览器对写请求的常规行为；
    CSRF 拒绝逻辑由 test_auth.py 用不带 Origin 的客户端单独验证。
    """
    from fastapi.testclient import TestClient

    from watermark.api import create_app

    auth_service.set_password(AUTH_PASSWORD)
    client = TestClient(create_app(auth_service), headers={"Origin": TEST_ORIGIN})
    assert _login(client, setup=False).status_code == 200
    return client
