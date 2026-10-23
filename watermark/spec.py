"""水印规格：Layer / WatermarkSpec dataclass 与 JSON 往返、字段校验。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

# 允许的九宫格锚点
POSITIONS = (
    "top-left", "top-center", "top-right",
    "middle-left", "center", "middle-right",
    "bottom-left", "bottom-center", "bottom-right",
)
# 允许的混合空间
BLEND_SPACES = ("srgb", "linear")
# A -> 1 时反向除零；上限 0.95 留出安全余量
MIN_OPACITY = 0.05
MAX_OPACITY = 0.95


def _check_color(color: tuple[int, int, int] | None) -> None:
    """校验 layer.color：None 或 3 个 0..255 整数。"""
    if color is None:
        return
    if isinstance(color, (str, bytes)) or not isinstance(color, (tuple, list)):
        raise SpecError(f"color 必须是 (R, G, B) 三元组或 None，当前为 {color!r}")
    if len(color) != 3:
        raise SpecError(f"color 必须是 3 个分量，当前为 {color!r}")
    for v in color:
        if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 255:
            raise SpecError(f"color 分量需为 0..255 整数，当前为 {color!r}")


class SpecError(ValueError):
    """spec 字段非法。"""


def _check_layer_common(*, opacity: float, position: str, margin: float, offset: tuple[float, float]) -> None:
    """校验 Layer 的公共字段（blend_space 在 WatermarkSpec 层校验）。"""
    if not (MIN_OPACITY <= opacity <= MAX_OPACITY):
        raise SpecError(
            f"opacity 必须在 ({MIN_OPACITY}, {MAX_OPACITY}] 内，当前为 {opacity}"
        )
    if position not in POSITIONS:
        raise SpecError(f"position 必须是 {POSITIONS} 之一，当前为 {position!r}")
    if margin < 0:
        raise SpecError(f"margin 不能为负，当前为 {margin}")
    if len(offset) != 2 or offset[0] < 0 or offset[1] < 0:
        raise SpecError(f"offset 必须是非负 (dx, dy)，当前为 {offset!r}")


@dataclass(frozen=True)
class Layer:
    """单个水印图层：文字或 logo。

    kind="text" 时 content 为文字，font / font_size / stroke / rotate 生效；
    kind="logo" 时 content 为图片路径，text 参数忽略。
    """

    kind: Literal["text", "logo"]
    content: str
    opacity: float = 0.25
    position: str = "bottom-right"
    margin: float = 0.03
    offset: tuple[float, float] = (0.0, 0.0)
    font: str | None = None
    font_size: int = 36
    stroke_width: int = 0
    rotate: float = 0.0
    tile: bool = False
    tile_gap: int = 0
    # None 表示使用引擎默认色；加水印与去水印必须用同一个 C 才可逆
    color: tuple[int, int, int] | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("text", "logo"):
            raise SpecError(f"kind 必须是 'text' 或 'logo'，当前为 {self.kind!r}")
        if not self.content:
            raise SpecError("content 不能为空")
        if self.kind == "text" and self.font_size <= 0:
            raise SpecError(f"font_size 必须大于 0，当前为 {self.font_size}")
        if self.kind == "logo" and self.tile:
            raise SpecError("logo 图层不支持 tile")
        if self.tile_gap < 0:
            raise SpecError(f"tile_gap 不能为负，当前为 {self.tile_gap}")
        _check_color(self.color)
        # 统一为 tuple，保证 from_dict(列表) 与直接构造可比较
        if self.color is not None:
            object.__setattr__(self, "color", tuple(self.color))
        _check_layer_common(
            opacity=self.opacity,
            position=self.position,
            margin=self.margin,
            offset=self.offset,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["offset"] = list(self.offset)
        d["color"] = list(self.color) if self.color is not None else None
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layer:
        if not isinstance(data, dict):
            raise SpecError(f"Layer JSON 必须是对象，当前为 {type(data).__name__}")
        offset = data.get("offset", (0.0, 0.0))
        if isinstance(offset, (list, tuple)) and len(offset) == 2:
            offset = (float(offset[0]), float(offset[1]))
        else:
            raise SpecError(f"offset 必须是二元数组，当前为 {data.get('offset')!r}")
        kwargs = dict(data)
        kwargs["offset"] = offset
        color = kwargs.get("color")
        if isinstance(color, (list, tuple)) and len(color) == 3:
            kwargs["color"] = tuple(int(c) for c in color)
        try:
            return cls(**kwargs)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SpecError(f"Layer 字段不合法: {exc}") from exc


@dataclass(frozen=True)
class WatermarkSpec:
    """整张图的水印规格：1..n 个图层 + 全局混合空间。"""

    layers: tuple[Layer, ...]
    blend_space: Literal["srgb", "linear"] = "srgb"
    version: int = 1

    def __post_init__(self) -> None:
        if not self.layers:
            raise SpecError("layers 不能为空")
        for i, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                raise SpecError(f"layers[{i}] 不是 Layer 实例")
        if self.blend_space not in BLEND_SPACES:
            raise SpecError(
                f"blend_space 必须是 {BLEND_SPACES} 之一，当前为 {self.blend_space!r}"
            )
        if self.version != 1:
            raise SpecError(f"仅支持 version=1，当前为 {self.version}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "blend_space": self.blend_space,
            "layers": [layer.to_dict() for layer in self.layers],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WatermarkSpec:
        if not isinstance(data, dict):
            raise SpecError(f"spec JSON 必须是对象，当前为 {type(data).__name__}")
        if data.get("version") != 1:
            raise SpecError(f"仅支持 version=1，当前为 {data.get('version')!r}")
        blend_space = data.get("blend_space", "srgb")
        if blend_space not in BLEND_SPACES:
            raise SpecError(
                f"blend_space 必须是 {BLEND_SPACES} 之一，当前为 {blend_space!r}"
            )
        raw_layers = data.get("layers")
        if not isinstance(raw_layers, (list, tuple)) or not raw_layers:
            raise SpecError("layers 必须是非空数组")
        layers = tuple(Layer.from_dict(item) for item in raw_layers)
        return cls(layers=layers, blend_space=blend_space, version=1)

    @classmethod
    def from_json(cls, text: str) -> WatermarkSpec:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SpecError(f"spec JSON 解析失败: {exc}") from exc
        return cls.from_dict(data)


def dump_spec(spec: WatermarkSpec, path: str | Path) -> None:
    """spec 写入 .wm.json sidecar 文件（ensure_ascii=False）。"""
    Path(path).write_text(spec.to_json(), encoding="utf-8")


def load_spec(path: str | Path) -> WatermarkSpec:
    """从 .wm.json sidecar 文件读取 spec。"""
    return WatermarkSpec.from_json(Path(path).read_text(encoding="utf-8"))
