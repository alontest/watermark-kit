"""测试公共夹具：随机 RGB / RGBA 图、logo 图。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

W, H = 160, 120
SEED = 2026


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
