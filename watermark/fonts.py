"""字体解析：路径 / 字体名模糊匹配 / 平台候选列表。

结果固化进 spec：只记录解析后的绝对路径，不在 spec 里存"字体名"这种
需要二次解析的字段。
"""

from __future__ import annotations

import difflib
from pathlib import Path
from typing import Union

from PIL import ImageFont

from watermark.spec import SpecError

FONT_EXTENSIONS = (".ttf", ".otf", ".ttc")

# 按平台的候选字体列表（顺序即优先级）
PLATFORM_FONT_CANDIDATES: dict[str, tuple[str, ...]] = {
    "Darwin": (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ),
    "Windows": (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ),
    "Linux": (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ),
}


def _font_candidates(name: str) -> tuple[Path, ...]:
    """把字体名展开为按平台搜索的候选路径。"""
    candidates: list[Path] = []
    for entry in PLATFORM_FONT_CANDIDATES.get(_platform_name(), ()):
        candidates.append(Path(entry))
    if name:
        candidates.insert(0, Path(name))
    return tuple(candidates)


def _platform_name() -> str:
    import platform

    return platform.system()


def resolve_font(
    font: Union[str, Path, None],
    *,
    text: str | None = None,
    size: int = 36,
) -> Path:
    """解析字体为实际可用的 .ttf/.otf/.ttc 文件路径。

    - font 为路径且存在 -> 直接返回
    - font 为名字 -> 在平台字体目录中模糊匹配文件名
    - font 为 None 或匹配失败 -> 返回平台候选列表中第一个可用字体
    全部失败时抛 SpecError。
    """
    if font is not None:
        font_path = Path(str(font)).expanduser()
        if font_path.is_file():
            return font_path.resolve()
        matched = _fuzzy_match(font_path.name)
        if matched is not None:
            return matched

    fallback = _platform_fallback()
    if fallback is not None:
        return fallback
    raise SpecError(
        f"找不到可用字体：font={font!r}，且平台候选字体均不可用"
        "（可通过 font= 指定确切的 .ttf/.otf/.ttc 路径）"
    )


def _search_roots() -> tuple[Path, ...]:
    """字体名模糊匹配时的搜索目录。"""
    import platform

    system = platform.system()
    if system == "Darwin":
        return (Path("/System/Library/Fonts"), Path("/Library/Fonts"),
                Path.home() / "Library" / "Fonts")
    if system == "Windows":
        return (Path("C:/Windows/Fonts"),)
    return (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"))


def _iter_font_files() -> list[Path]:
    files: list[Path] = []
    for root in _search_roots():
        if not root.is_dir():
            continue
        try:
            files.extend(p for p in root.rglob("*") if p.suffix.lower() in FONT_EXTENSIONS)
        except OSError:
            continue
    return files


def _fuzzy_match(name: str) -> Path | None:
    """按文件名在平台字体目录里找最接近的字体文件。"""
    if not name:
        return None
    files = _iter_font_files()
    if not files:
        return None
    by_path = {str(p): p for p in files}
    matches = difflib.get_close_matches(name, by_path.keys(), n=1, cutoff=0.6)
    if matches:
        return by_path[matches[0]]
    # 再按文件主名（不含扩展名）宽松匹配一次
    stem = Path(name).stem.lower()
    for path_str, path in by_path.items():
        if path.stem.lower() == stem:
            return path
    return None


def _platform_fallback() -> Path | None:
    """平台候选列表中第一个可用的字体。"""
    import platform

    for entry in PLATFORM_FONT_CANDIDATES.get(platform.system(), ()):
        path = Path(entry)
        if path.is_file():
            return path
    # 候选全失败时，从搜索目录里随便取一个可用字体
    files = _iter_font_files()
    return files[0] if files else None


def load_font(font_path: str | Path, size: int) -> ImageFont.FreeTypeFont:
    """加载指定路径与字号的具体字体对象。"""
    try:
        return ImageFont.truetype(str(font_path), size)
    except OSError as exc:
        raise SpecError(f"字体加载失败: {font_path} -> {exc}") from exc
