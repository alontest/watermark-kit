"""元数据读写：PNG tEXt / JPEG EXIF 0x010E / .wm.json sidecar。"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from watermark.spec import SpecError, WatermarkSpec

# PNG tEXt 键
SPEC_TEXT_KEY = "watermark.spec"
# JPEG EXIF ImageDescription 标签 ID
JPEG_EXIF_TAG = 0x010E
# sidecar 文件后缀
SIDECAR_SUFFIX = ".wm.json"

# JPEG EXIF 只能存纯 ASCII JSON
_ASCII_REPLACEMENTS = {
    "：": ": ", "，": ", ", "（": " (", "）": " )",
}


def _to_ascii_json(text: str) -> str:
    """把 JSON 压缩成 EXIF 可存的纯 ASCII 字符串。"""
    compact = json.dumps(
        json.loads(text), ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    return compact


def save_with_spec(
    image: Image.Image,
    path: str | Path,
    spec: WatermarkSpec,
    **save_kwargs,
) -> None:
    """把图片连同 spec 一起写入文件。"""
    dest = Path(path)
    suffix = dest.suffix.lower()
    spec_json = spec.to_json()
    if suffix == ".png":
        info = PngInfo()
        info.add_text(SPEC_TEXT_KEY, spec_json)
        image.save(dest, format="PNG", pnginfo=info, **save_kwargs)
    elif suffix in (".jpg", ".jpeg"):
        exif = Image.Exif()
        exif[JPEG_EXIF_TAG] = _to_ascii_json(spec_json)
        image.save(dest, format="JPEG", exif=exif, **save_kwargs)
    else:
        # 其他格式落 sidecar
        image.save(dest, **save_kwargs)
        dump_sidecar(spec, dest)


def read_spec(path: str | Path) -> WatermarkSpec | None:
    """从图片文件读取 spec；找不到返回 None。"""
    source = Path(path)
    with Image.open(source) as img:
        if source.suffix.lower() == ".png":
            spec_json = img.text.get(SPEC_TEXT_KEY)
        elif source.suffix.lower() in (".jpg", ".jpeg"):
            exif = img.getexif()
            spec_json = exif.get(JPEG_EXIF_TAG)
        else:
            spec_json = None
    if spec_json is None:
        sidecar = sidecar_path_for(source)
        if sidecar.is_file():
            return WatermarkSpec.from_json(sidecar.read_text(encoding="utf-8"))
        return None
    try:
        return WatermarkSpec.from_json(spec_json)
    except SpecError:
        raise


def sidecar_path_for(path: str | Path) -> Path:
    """给定图片路径，返回对应 sidecar 路径（photo.png -> photo.png.wm.json）。"""
    return Path(path).with_name(Path(path).name + SIDECAR_SUFFIX)


def dump_sidecar(spec: WatermarkSpec, image_path: str | Path) -> Path:
    """spec 写到图片同目录的 .wm.json sidecar。"""
    out = sidecar_path_for(image_path)
    out.write_text(spec.to_json(), encoding="utf-8")
    return out
