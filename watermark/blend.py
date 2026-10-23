"""混合与还原：apply_layer / remove_layer / psnr / diff_stats。

混合模型（逐通道，float32）：
  正向  I_out = I_in * (1 - A) + C * A      # 凸组合，永不越界
  反向  I_in  = (I_out - C * A) / (1 - A)   # A -> 1 时除零，spec 已限制 opacity <= 0.95

不可逆条件：A >= 1 - eps（eps=1e-4）的像素按原样保留并在报告中计占比。
"""

from __future__ import annotations

import numpy as np

# 1 - A 低于该值视为不可逆（避免除零）
REVERSIBILITY_EPS = 1e-4


def _broadcast_color(
    image: np.ndarray, color: np.ndarray
) -> np.ndarray:
    """把 (C,) 色值广播到与 (H, W, C) 图像一致的形状；灰度图取均值。"""
    channels = image.shape[2]
    if channels == 1:
        fill = float(np.mean(color))
    else:
        if color.size == channels:
            fill = None  # 逐通道直接用
        else:
            fill = float(np.mean(color))
    if fill is None:
        return color.astype(np.float32)
    return np.full((channels,), fill, dtype=np.float32)


def apply_layer(
    image: np.ndarray, mask: np.ndarray, color: np.ndarray, opacity: float
) -> tuple[np.ndarray, float]:
    """正向混合一层。返回 (新图 float32, 不可逆像素占比)。"""
    img = image.astype(np.float32)
    alpha = mask.astype(np.float32) * float(opacity)
    strength = alpha[..., np.newaxis]
    c = _broadcast_color(img, color)
    out = img * (1.0 - strength) + c * strength
    return np.clip(out, 0.0, 255.0), 0.0  # 凸组合理论上不越界，clip 兜底


def remove_layer(
    image: np.ndarray, mask: np.ndarray, color: np.ndarray, opacity: float
) -> tuple[np.ndarray, float]:
    """反向还原一层。返回 (还原图 float32, 不可逆像素占比)。

    不可逆像素（strength >= 1 - eps）保留原样，不做除法。
    """
    img = image.astype(np.float32)
    alpha = mask.astype(np.float32) * float(opacity)
    strength = alpha[..., np.newaxis]
    keep = strength >= (1.0 - REVERSIBILITY_EPS)
    denom = np.where(keep, 1.0, 1.0 - strength)
    c = _broadcast_color(img, color)
    restored = (img - c * strength) / denom
    out = np.where(keep, img, restored)
    unrecoverable = float(keep[..., 0].mean()) if keep.size else 0.0
    return out, unrecoverable


def psnr(reference: np.ndarray, target: np.ndarray) -> float:
    """两幅图之间的 PSNR（dB），输入任意同形状数组，值域建议 [0, 255]。"""
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError(f"psnr 形状不一致: {ref.shape} vs {tgt.shape}")
    mse = float(np.mean((ref - tgt) ** 2))
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(255.0**2 / mse))


def diff_stats(
    reference: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """逐像素差统计：max / mean / rmse。"""
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if ref.shape != tgt.shape:
        raise ValueError(f"diff_stats 形状不一致: {ref.shape} vs {tgt.shape}")
    diff = np.abs(ref - tgt)
    return {
        "max": float(diff.max()),
        "mean": float(diff.mean()),
        "rmse": float(np.sqrt(np.mean((ref - tgt) ** 2))),
    }
