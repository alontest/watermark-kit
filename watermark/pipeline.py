"""流水线：add_watermark / remove_watermark / inspect_spec / save。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from watermark import blend, meta, render
from watermark.spec import WatermarkSpec

# 单图层无自定义颜色时的默认水印色（深灰，黑白图上都可读）
DEFAULT_COLOR = (90, 90, 90)
# 保存 PNG 时是否保留原 alpha
_MAX_DIM_LOG = 4  # report 里 mask 指纹只记前几位避免冗长


def _load_image_array(path: str | Path) -> tuple[Image.Image, np.ndarray]:
    """读图并归正 EXIF 方向，返回 (PIL 图, float32 数组)。"""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return img, arr


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """sRGB -> 线性光（值域 0..255）。"""
    xn = x / 255.0
    linear = np.where(xn <= 0.04045, xn / 12.92, ((xn + 0.055) / 1.055) ** 2.4)
    return linear * 255.0


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """线性光 -> sRGB（值域 0..255）。"""
    xn = np.clip(x / 255.0, 0.0, 1.0)
    srgb = np.where(xn <= 0.0031308, xn * 12.92, 1.055 * xn ** (1 / 2.4) - 0.055)
    return srgb * 255.0


def _to_blend_space(arr: np.ndarray, blend_space: str) -> np.ndarray:
    if blend_space == "linear":
        return _srgb_to_linear(arr)
    return arr


def _from_blend_space(arr: np.ndarray, blend_space: str) -> np.ndarray:
    if blend_space == "linear":
        return _linear_to_srgb(arr)
    return arr


def _blend_layers(
    arr: np.ndarray,
    masks: list[np.ndarray],
    spec: WatermarkSpec,
    *,
    reverse: bool,
) -> tuple[np.ndarray, float, list[str]]:
    """按 spec 逐层混合或还原，返回 (结果数组, 累计不可逆占比, warnings)。"""
    work = _to_blend_space(arr, spec.blend_space)
    warnings: list[str] = []
    total_unrecoverable = 0.0
    for layer, mask in zip(spec.layers, masks):
        # 加与去必须用同一个 C：颜色随 spec 持久化，正反自动一致
        color = np.asarray(
            layer.color if layer.color is not None else DEFAULT_COLOR,
            dtype=np.float32,
        )
        if reverse:
            work, unrecoverable = blend.remove_layer(work, mask, color, layer.opacity)
        else:
            work, _ = blend.apply_layer(work, mask, color, layer.opacity)
            unrecoverable = 0.0
        total_unrecoverable = max(total_unrecoverable, unrecoverable)
    out = _from_blend_space(work, spec.blend_space)
    return np.clip(out, 0.0, 255.0), total_unrecoverable, warnings


def _check_mask_shape(masks: list[np.ndarray], width: int, height: int) -> None:
    for i, mask in enumerate(masks):
        if mask.shape != (height, width):
            raise ValueError(
                f"mask[{i}] 形状 {mask.shape} 与图像 {(height, width)} 不一致"
            )


def _mask_short(fingerprint: str) -> str:
    return fingerprint[: 2 * _MAX_DIM_LOG]


def add_watermark(
    image: Image.Image | str | Path,
    spec: WatermarkSpec,
) -> dict:
    """加水印。返回 JSON 安全的 report；`image` 结果在 report["image"]。

    混合颜色按图层取自 spec（Layer.color 为 None 时用 DEFAULT_COLOR），
    并随 spec 一起持久化，去水印时从 spec 读回同一颜色，保证可逆。
    """
    if isinstance(image, (str, Path)):
        img, arr = _load_image_array(image)
    else:
        img = ImageOps.exif_transpose(image) if image.getexif() else image.copy()
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
    height, width = arr.shape[:2]
    masks = render.render_spec_masks(spec, width, height)
    _check_mask_shape(masks, width, height)
    blended, unrecoverable, warnings = _blend_layers(
        arr, masks, spec, reverse=False
    )
    mode = img.mode
    channels = arr.shape[2]
    out_arr = blended[..., 0] if channels == 1 else blended
    out_img = Image.fromarray(
        np.round(out_arr).astype(np.uint8),
        mode=mode if channels in (1, 3, 4) else "RGB",
    )
    mask_fp = render.hash_masks(masks)
    report = {
        "mode": mode,
        "width": width,
        "height": height,
        "blend_space": spec.blend_space,
        "layers": len(spec.layers),
        "mask_sha256": mask_fp,
        "unrecoverable_ratio": unrecoverable,
        "predicted_psnr": None,
        "warnings": warnings,
        "image": out_img,
    }
    return report


def remove_watermark(
    image: Image.Image | str | Path,
    spec: WatermarkSpec,
    *,
    expected_mask_sha256: str | None = None,
) -> dict:
    """去水印。expected_mask_sha256 用于校验 spec 与图上水印一致。"""
    if isinstance(image, (str, Path)):
        img, arr = _load_image_array(image)
    else:
        img = ImageOps.exif_transpose(image) if image.getexif() else image.copy()
        arr = np.asarray(img, dtype=np.float32)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
    height, width = arr.shape[:2]
    masks = render.render_spec_masks(spec, width, height)
    _check_mask_shape(masks, width, height)
    mask_fp = render.hash_masks(masks)
    mask_match = expected_mask_sha256 is None or mask_fp == expected_mask_sha256
    warnings: list[str] = []
    if not mask_match:
        warnings.append("mask 指纹不匹配：spec 渲染结果与 expected_mask_sha256 不同")
    blended, unrecoverable, layer_warnings = _blend_layers(
        arr, masks, spec, reverse=True
    )
    warnings.extend(layer_warnings)
    channels = arr.shape[2]
    out_arr = blended[..., 0] if channels == 1 else blended
    mode = "L" if channels == 1 else (img.mode if channels in (3, 4) else "RGB")
    out_img = Image.fromarray(np.round(out_arr).astype(np.uint8), mode=mode)
    report = {
        "mode": mode,
        "width": width,
        "height": height,
        "blend_space": spec.blend_space,
        "layers": len(spec.layers),
        "mask_sha256": mask_fp,
        "mask_match": mask_match,
        "unrecoverable_ratio": unrecoverable,
        "predicted_psnr": None,
        "warnings": warnings,
        "image": out_img,
    }
    return report


def inspect_spec(spec: WatermarkSpec, width: int, height: int) -> dict:
    """渲染 spec 的 mask 并返回摘要（不混合）。"""
    masks = render.render_spec_masks(spec, width, height)
    return {
        "width": width,
        "height": height,
        "blend_space": spec.blend_space,
        "layers": len(spec.layers),
        "mask_sha256": render.hash_masks(masks),
        "masks": masks,
    }


def save(
    image: Image.Image,
    path: str | Path,
    spec: WatermarkSpec,
    **save_kwargs,
) -> None:
    """图片 + spec 一起落盘（PNG tEXt / JPEG EXIF / 其他格式 sidecar）。"""
    meta.save_with_spec(image, path, spec, **save_kwargs)


def _json_safe(report: dict) -> dict:
    """去掉 PIL 对象，返回 JSON 安全的 report 副本。"""
    return {k: v for k, v in report.items() if k != "image"}


def report_to_json(report: dict) -> str:
    return json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True)
