"""渲染：spec -> 与原图同尺寸的 float32 有效 alpha 图 (H, W)。

每个 Layer 渲染为独立 mask，再由 blend 模块逐层混合。文字用
textbbox + 描边 + rotate(expand=True)，logo 用其 alpha 通道（无 alpha
时按四角颜色检测白底并反相）。九宫格定位支持 margin(比例) + offset(px)，
tile 平铺用 np.maximum 叠加、从负偏移起铺。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from watermark.fonts import load_font, resolve_font
from watermark.spec import Layer, SpecError, WatermarkSpec

# hash_masks 的量化位数
MASK_QUANT_BITS = 8

_ANCHOR_MAP = {
    "top-left": ("left", "top"),
    "top-center": ("center", "top"),
    "top-right": ("right", "top"),
    "middle-left": ("left", "center"),
    "center": ("center", "center"),
    "middle-right": ("right", "center"),
    "bottom-left": ("left", "bottom"),
    "bottom-center": ("center", "bottom"),
    "bottom-right": ("right", "bottom"),
}


def _render_text_layer(
    layer: Layer, width: int, height: int
) -> Image.Image:
    """渲染文字为 L 模式的 RGBA-ish (这里只画 mask) 图。"""
    font_path = resolve_font(layer.font, size=layer.font_size)
    font = load_font(font_path, layer.font_size)
    # 先在足够大的画布上画字，再旋转 expand
    text = layer.content
    bbox = ImageDraw.Draw(Image.new("L", (1, 1))).textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = max(layer.stroke_width, 4) * 2
    canvas_w = text_w + 2 * pad + layer.stroke_width * 2
    canvas_h = text_h + 2 * pad + layer.stroke_width * 2
    stamp = Image.new("L", (max(canvas_w, 1), max(canvas_h, 1)), 0)
    draw = ImageDraw.Draw(stamp)
    # 描边画两遍：粗的作为描边，细的作为主体
    origin = (pad - bbox[0], pad - bbox[1])
    if layer.stroke_width > 0:
        draw.text(
            origin, text, font=font, fill=255,
            stroke_width=layer.stroke_width, stroke_fill=255,
        )
    else:
        draw.text(origin, text, font=font, fill=255)
    if layer.rotate % 360 != 0:
        stamp = stamp.rotate(layer.rotate, expand=True, resample=Image.BICUBIC)
    return stamp


def _load_logo_rgba(path: str | Path) -> Image.Image:
    """读 logo，无 alpha 通道时按四角颜色检测纯色底并反相生成 alpha。"""
    logo_path = Path(path)
    if not logo_path.is_file():
        raise SpecError(f"logo 文件不存在: {logo_path}")
    logo = Image.open(logo_path)
    if logo.mode in ("RGBA", "LA") or (
        logo.mode == "P" and "transparency" in logo.info
    ):
        return logo.convert("RGBA")
    rgb = np.asarray(logo.convert("RGB"), dtype=np.float32)
    corners = np.stack([
        rgb[0, 0], rgb[0, -1], rgb[-1, 0], rgb[-1, -1]
    ])
    corner_mean = corners.mean(axis=0)
    # 白底：与四角均值的最大差 <= 2 视为背景
    bg_dist = np.abs(rgb - corner_mean).max(axis=-1)
    is_bg = bg_dist <= 2.0
    alpha = np.where(is_bg, 0.0, 255.0).astype(np.uint8)
    rgba = np.dstack([
        rgb.astype(np.uint8), alpha,
    ])
    return Image.fromarray(rgba, mode="RGBA")


def _render_logo_layer(layer: Layer, width: int, height: int) -> Image.Image:
    """渲染 logo 为 L 模式的 alpha 图（必要时缩放到 font_size 当最大边）。"""
    logo = _load_logo_rgba(layer.content)
    if layer.font_size > 0:
        scale = layer.font_size / max(logo.size)
        if scale < 1.0:
            new_size = (
                max(1, round(logo.width * scale)),
                max(1, round(logo.height * scale)),
            )
            logo = logo.resize(new_size, Image.LANCZOS)
    alpha = np.asarray(logo.getchannel("A"), dtype=np.float32) / 255.0
    return Image.fromarray((alpha * 255).astype(np.uint8), mode="L")


def _stamp_size(stamp: Image.Image) -> tuple[int, int]:
    return stamp.width, stamp.height


def _place_single(
    stamp: Image.Image,
    layer: Layer,
    width: int,
    height: int,
) -> tuple[int, int]:
    """九宫格定位 + margin(比例) + offset(px)，返回粘贴左上角。"""
    h_ax, v_ax = _ANCHOR_MAP[layer.position]
    margin_x = int(round(layer.margin * width))
    margin_y = int(round(layer.margin * height))
    if h_ax == "left":
        x = margin_x
    elif h_ax == "center":
        x = (width - stamp.width) // 2
    else:
        x = width - stamp.width - margin_x
    if v_ax == "top":
        y = margin_y
    elif v_ax == "center":
        y = (height - stamp.height) // 2
    else:
        y = height - stamp.height - margin_y
    return x + int(round(layer.offset[0])), y + int(round(layer.offset[1]))


def render_layer_mask(layer: Layer, width: int, height: int) -> np.ndarray:
    """渲染单个 Layer -> (H, W) float32 有效 alpha，取值 [0, 1]。"""
    if layer.kind == "text":
        stamp = _render_text_layer(layer, width, height)
    else:
        stamp = _render_logo_layer(layer, width, height)
    mask = np.zeros((height, width), dtype=np.float32)
    stamp_arr = np.asarray(stamp, dtype=np.float32) / 255.0
    if layer.tile:
        sh, sw = stamp_arr.shape
        # 从负偏移起铺，保证边缘也有水印
        for y in range(-sh, height, max(sh, 1)):
            for x in range(-sw, width, max(sw, 1)):
                y0, x0 = max(y, 0), max(x, 0)
                y1, x1 = min(y + sh, height), min(x + sw, width)
                if y1 <= y0 or x1 <= x0:
                    continue
                sub = stamp_arr[y0 - y:y1 - y, x0 - x:x1 - x]
                region = mask[y0:y1, x0:x1]
                mask[y0:y1, x0:x1] = np.maximum(region, sub)
    else:
        x, y = _place_single(stamp, layer, width, height)
        sh, sw = stamp_arr.shape
        y0, x0 = max(y, 0), max(x, 0)
        y1, x1 = min(y + sh, height), min(x + sw, width)
        if y1 > y0 and x1 > x0:
            sub = stamp_arr[y0 - y:y1 - y, x0 - x:x1 - x]
            mask[y0:y1, x0:x1] = np.maximum(mask[y0:y1, x0:x1], sub)
    return mask


def render_spec_masks(
    spec: WatermarkSpec, width: int, height: int
) -> list[np.ndarray]:
    """渲染 spec 全部 Layer -> [(H, W) float32, ...]，顺序与 layers 一致。"""
    return [render_layer_mask(layer, width, height) for layer in spec.layers]


def hash_masks(masks: list[np.ndarray]) -> str:
    """对量化到 8bit 的 mask 求 sha256（供往返一致性校验）。"""
    hasher = hashlib.sha256()
    for mask in masks:
        quantized = np.round(
            np.clip(mask, 0.0, 1.0) * (2 ** MASK_QUANT_BITS - 1)
        ).astype(np.uint8)
        hasher.update(quantized.tobytes())
        hasher.update(str(quantized.shape).encode("ascii"))
    return hasher.hexdigest()
