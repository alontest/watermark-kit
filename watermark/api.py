"""FastAPI 本地 Web 界面：拖拽上传 -> 加/去水印 -> 预览 -> 下载。

全程本地运行，上传的图片只存在内存里；全分辨率结果按 token 存于
进程内 FIFO 字典（容量上限 MAX_STORE，超出淘汰最旧条目）。
"""

from __future__ import annotations

import base64
import io
import os
import secrets
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image, ImageOps
from PIL import UnidentifiedImageError

from watermark import __version__, meta, pipeline
from watermark.spec import BLEND_SPACES, POSITIONS, Layer, WatermarkSpec

# 服务端结果缓存容量（FIFO 淘汰）
MAX_STORE = 24
# 预览图最长边
MAX_PREVIEW = 1400
# 指纹在指标里展示的截断长度
_FP_DISPLAY = 16

_WEB_DIR = Path(__file__).parent / "web"
_INDEX_HTML = _WEB_DIR / "index.html"

# 引擎当前固定以默认颜色混合（remove 无 color 入口），自定义颜色仅记录
_COLOR_NOTE = (
    "引擎当前固定以默认颜色 (90,90,90) 混合以保证可逆还原，"
    "自定义颜色仅记录、暂不生效"
)


# ---------------------------------------------------------------- 结果存储

class _ResultStore:
    """token -> 结果条目的 FIFO 缓存。"""

    def __init__(self, capacity: int = MAX_STORE) -> None:
        self._data: OrderedDict[str, dict] = OrderedDict()
        self._capacity = capacity

    def put(self, entry: dict) -> str:
        token = secrets.token_urlsafe(16)
        self._data[token] = entry
        self._data.move_to_end(token)
        while len(self._data) > self._capacity:
            self._data.popitem(last=False)
        return token

    def get(self, token: str) -> dict | None:
        return self._data.get(token)


_STORE = _ResultStore()


# ---------------------------------------------------------------- 解析辅助

def _parse_color(text: str) -> tuple[int, int, int]:
    """支持 '#RRGGBB' 或 'R,G,B' 两种写法。"""
    raw = (text or "").strip()
    if raw.startswith("#"):
        hexpart = raw[1:]
        if len(hexpart) != 6:
            raise ValueError(f"颜色需为 #RRGGBB，当前为 {raw!r}")
        try:
            return tuple(int(hexpart[i:i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]
        except ValueError:
            raise ValueError(f"颜色需为 #RRGGBB，当前为 {raw!r}") from None
    parts = raw.split(",")
    if len(parts) != 3:
        raise ValueError(f"颜色需为 R,G,B 或 #RRGGBB，当前为 {raw!r}")
    try:
        values = tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise ValueError(f"颜色需为整数 R,G,B，当前为 {raw!r}") from None
    if any(not (0 <= v <= 255) for v in values):
        raise ValueError(f"颜色分量需在 0..255，当前为 {values}")
    return values  # type: ignore[return-value]


def _check_non_negative(name: str, value: float) -> float:
    if value < 0:
        raise ValueError(f"{name} 不能为负，当前为 {value}")
    return value


def _load_upload_image(upload: UploadFile) -> Image.Image:
    """读上传图片并归正 EXIF 方向。"""
    data = upload.file.read()
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError:
        raise ValueError(
            f"无法识别的图片文件: {upload.filename!r}，请上传 PNG/JPEG 等常见格式"
        ) from None
    return ImageOps.exif_transpose(img) if img.getexif() else img


def _read_spec_from_image(img: Image.Image) -> WatermarkSpec | None:
    """按图片实际格式读内嵌配方（PNG tEXt / JPEG EXIF）。"""
    spec_json = None
    if img.format == "PNG":
        spec_json = img.text.get(meta.SPEC_TEXT_KEY)
    elif img.format == "JPEG":
        raw = img.getexif().get(meta.JPEG_EXIF_TAG)
        spec_json = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    if not spec_json:
        return None
    return WatermarkSpec.from_json(spec_json)


def _make_preview(img: Image.Image) -> str:
    """生成最长边 <= MAX_PREVIEW 的 PNG dataURL。"""
    preview = img.copy()
    preview.thumbnail((MAX_PREVIEW, MAX_PREVIEW), Image.LANCZOS)
    buf = io.BytesIO()
    preview.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@contextmanager
def _temp_file_from_upload(upload: UploadFile | None) -> Iterator[str | None]:
    """把上传文件落到临时文件（logo 引擎按路径读取），请求结束删除。"""
    if upload is None:
        yield None
        return
    suffix = Path(upload.filename or "logo.png").suffix or ".png"
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(upload.file.read())
        yield path
    finally:
        os.unlink(path)


# ---------------------------------------------------------------- 应用

def create_app() -> FastAPI:
    """创建 FastAPI 应用。"""
    app = FastAPI(title="watermark-kit", version=__version__,
                  docs_url=None, redoc_url=None)

    @app.exception_handler(ValueError)
    def _value_error_handler(request, exc: ValueError):  # noqa: ANN001
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/")
    def index() -> Response:
        if not _INDEX_HTML.is_file():
            raise HTTPException(500, "页面文件缺失: watermark/web/index.html")
        return Response(_INDEX_HTML.read_bytes(), media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    # ------------------------------------------------ add

    @app.post("/api/add")
    def api_add(
        image: UploadFile = File(...),
        text: str = Form(""),
        logo: UploadFile | None = File(None),
        size: int = Form(36),
        opacity: float = Form(0.25),
        color: str = Form("#5a5a5a"),
        position: str = Form("bottom-right"),
        margin: float = Form(0.03),
        rotate: float = Form(0.0),
        stroke: int = Form(0),
        tile: bool = Form(False),
        tile_gap: int = Form(0),
        blend_space: str = Form("srgb"),
    ) -> dict:
        img = _load_upload_image(image)
        color_rgb = _parse_color(color)  # 校验并记录；引擎当前固定默认色
        for name, value in (("stroke", stroke), ("tile_gap", tile_gap)):
            _check_non_negative(name, value)
        with _temp_file_from_upload(logo) as logo_path:
            if logo_path is not None:
                layer = Layer(
                    kind="logo", content=logo_path, opacity=opacity,
                    position=position, margin=margin,
                    font_size=size, stroke_width=stroke, rotate=rotate,
                )
                notes = ["logo 图层不支持平铺，tile 参数已忽略"]
            else:
                if not text.strip():
                    raise ValueError("请提供水印文字，或上传 logo 图片")
                layer = Layer(
                    kind="text", content=text, opacity=opacity,
                    position=position, margin=margin,
                    font_size=size, stroke_width=stroke, rotate=rotate,
                    tile=tile,
                )
                notes = [] if not tile else []  # tile_gap 提示见下
                if tile and tile_gap:
                    notes.append("tile_gap 暂以字号近似参与布局，独立间隔未生效")
            spec = WatermarkSpec(layers=(layer,), blend_space=blend_space)
            report = pipeline.add_watermark(img, spec)

        notes.append(_COLOR_NOTE)
        entry = {
            "image": report["image"], "spec": spec,
            "report": report, "color": color_rgb,
        }
        token = _STORE.put(entry)
        return {
            "ok": True,
            "token": token,
            "preview": _make_preview(report["image"]),
            "spec": spec.to_dict(),
            "report": pipeline._json_safe(report),
            "color": list(color_rgb),
            "notes": notes,
        }

    # ------------------------------------------------ remove

    @app.post("/api/remove")
    def api_remove(
        image: UploadFile = File(...),
        spec: str = Form(""),
        expected_mask_sha256: str = Form(""),
    ) -> dict:
        img = _load_upload_image(image)
        if spec.strip():
            watermark_spec = WatermarkSpec.from_json(spec)
        else:
            watermark_spec = _read_spec_from_image(img)
        if watermark_spec is None:
            raise HTTPException(
                422,
                "图片元数据里没有水印配方，请求也未携带配方 JSON；"
                "请上传加水印时导出的配方 .wm.json",
            )
        report = pipeline.remove_watermark(
            img, watermark_spec,
            expected_mask_sha256=expected_mask_sha256 or None,
        )
        entry = {"image": report["image"], "spec": watermark_spec, "report": report}
        token = _STORE.put(entry)
        return {
            "ok": True,
            "token": token,
            "preview": _make_preview(report["image"]),
            "spec": watermark_spec.to_dict(),
            "report": pipeline._json_safe(report),
        }

    # ------------------------------------------------ inspect

    @app.post("/api/inspect")
    def api_inspect(image: UploadFile = File(...)) -> dict:
        img = _load_upload_image(image)
        found_spec = _read_spec_from_image(img)
        if found_spec is None:
            raise HTTPException(
                422,
                "图片元数据里没有水印配方；请上传配方 .wm.json 后再执行去水印",
            )
        layers = [
            {
                "index": i,
                "kind": layer.kind,
                "content": layer.content if layer.kind == "text" else "(logo)",
                "opacity": layer.opacity,
                "position": layer.position,
                "font_size": layer.font_size,
                "rotate": layer.rotate,
                "stroke_width": layer.stroke_width,
                "tile": layer.tile,
            }
            for i, layer in enumerate(found_spec.layers)
        ]
        return {
            "found": True,
            "width": img.width,
            "height": img.height,
            "format": img.format,
            "blend_space": found_spec.blend_space,
            "layers": layers,
            "spec": found_spec.to_dict(),
            "mask_sha256": None,  # 不渲染 mask，指纹以 remove 报告为准
        }

    # ------------------------------------------------ download

    @app.get("/api/download/{token}")
    def api_download(token: str, as_json: int = Query(0, alias="as_json")) -> Response:
        entry = _STORE.get(token)
        if entry is None:
            raise HTTPException(404, "结果不存在或已被淘汰（容量 24），请重新处理")
        if as_json:
            spec = entry["spec"]
            if spec is None:
                raise HTTPException(404, "该结果没有关联配方")
            return Response(
                content=spec.to_json(),
                media_type="application/json",
                headers={
                    "Content-Disposition":
                        'attachment; filename="watermark.wm.json"'
                },
            )
        buf = io.BytesIO()
        entry["image"].save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": 'attachment; filename="result.png"'},
        )

    return app
