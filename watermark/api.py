"""FastAPI 本地 Web 界面：口令登录 -> 拖拽上传 -> 加/去水印 -> 预览 -> 下载。

全程本地运行，上传的图片只存在内存里；全分辨率结果按 token 存于
进程内 FIFO 字典（容量上限 MAX_STORE，超出淘汰最旧条目）。

鉴权是单用户口令 + 签名 Cookie 会话，实现在 ``watermark.auth``；
本模块只做请求/响应适配：``/``、``/api/*`` 受保护，``/login``、
``POST /api/login``、``/api/health`` 公开。
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
from urllib.parse import urlsplit

from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from PIL import Image, ImageOps
from PIL import UnidentifiedImageError

from watermark import __version__, auth, catalog, meta, pipeline
from watermark.spec import BLEND_SPACES, POSITIONS, Layer, WatermarkSpec

# 服务端结果缓存容量（FIFO 淘汰）
MAX_STORE = 24
# 预览图最长边
MAX_PREVIEW = 1400
# 指纹在指标里展示的截断长度
_FP_DISPLAY = 16

_WEB_DIR = Path(__file__).parent / "web"
_INDEX_HTML = _WEB_DIR / "index.html"
_LOGIN_HTML = _WEB_DIR / "login.html"

# 登录失败/限流的中文提示
_MSG_BAD_PASSWORD = "密码不正确"
_MSG_RATE_LIMITED = "登录失败次数过多，请 5 分钟后再试"
_MSG_LOCAL_ONLY = "只能在运行服务的本机上设置访问密码"
_MSG_UNAUTHORIZED = "请先登录"

# 会校验来源的写方法
_UNSAFE_METHODS = frozenset({"POST", "PUT", "DELETE", "PATCH"})

# catalog 查询接口的分页与过滤边界
_CATALOG_MIN_LIMIT = 1
_CATALOG_MAX_LIMIT = 200
_CATALOG_DEFAULT_LIMIT = 50
_CATALOG_ACTIONS = frozenset({"add", "remove"})
_CATALOG_STATUSES = frozenset({"ok", "failed"})


def _is_secure_request(request: Request) -> bool:
    """HTTPS 判定：直连 TLS，或反向代理声明 https。"""
    if request.url.scheme == "https":
        return True
    forwarded = request.headers.get("x-forwarded-proto", "")
    return forwarded.split(",")[0].strip().lower() == "https"


def _host_matches(origin: str, host: str) -> bool:
    """Origin 与 Host 同源判断（只比较 host:port，忽略 scheme 差异）。"""
    parsed = urlsplit(origin)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return False
    origin_host = parsed.netloc.lower()
    host = host.lower()
    if origin_host == host:
        return True
    # 默认端口归一：example.com == example.com:80/443
    default_port = "443" if parsed.scheme == "https" else "80"
    return origin_host == f"{host}:{default_port}" or f"{origin_host}:{default_port}" == host


def _check_same_origin(request: Request) -> bool:
    """写请求必须有同源 Origin/Referer；两者都缺失视为不可信。"""
    host = request.headers.get("host", "")
    if not host:
        return False
    origin = request.headers.get("origin")
    if origin and origin != "null":
        return _host_matches(origin, host)
    referer = request.headers.get("referer")
    if referer:
        return _host_matches(referer, host)
    return False


def _client_ip(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _session_cookie_kwargs(request: Request, max_age: int) -> dict:
    return {
        "key": auth.COOKIE_NAME,
        "max_age": max_age,
        "path": "/",
        "httponly": True,
        "samesite": "lax",
        "secure": _is_secure_request(request),
    }


def _login_page_mode(auth_service: auth.AuthService, request: Request) -> str:
    """页面模式：setup（首次设置）/ login / blocked（远端且尚未配置）。"""
    if auth_service.is_configured:
        return "login"
    return "setup" if auth.is_loopback_host(_client_ip(request)) else "blocked"


def _read_login_page(auth_service: auth.AuthService, request: Request) -> HTMLResponse:
    if not _LOGIN_HTML.is_file():
        raise HTTPException(500, "页面文件缺失: watermark/web/login.html")
    html = _LOGIN_HTML.read_text(encoding="utf-8")
    return HTMLResponse(html.replace("{{MODE}}", _login_page_mode(auth_service, request)))


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


def _catalog_limit(value: int) -> int:
    if not _CATALOG_MIN_LIMIT <= value <= _CATALOG_MAX_LIMIT:
        raise ValueError(
            f"limit 需在 {_CATALOG_MIN_LIMIT}..{_CATALOG_MAX_LIMIT}，当前为 {value}"
        )
    return value


def _catalog_offset(value: int) -> int:
    if value < 0:
        raise ValueError(f"offset 不能为负，当前为 {value}")
    return value


def _catalog_filter_value(name: str, value: str, allowed: frozenset[str]) -> str:
    if value not in allowed:
        raise ValueError(f"{name} 需为 {'/'.join(sorted(allowed))} 之一，当前为 {value!r}")
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

def create_app(auth_service: auth.AuthService | None = None) -> FastAPI:
    """创建 FastAPI 应用；``auth_service`` 便于测试注入临时凭据文件。"""
    service = auth_service if auth_service is not None else auth.AuthService()
    app = FastAPI(title="watermark-kit", version=__version__,
                  docs_url=None, redoc_url=None)
    app.state.auth = service

    @app.exception_handler(ValueError)
    def _value_error_handler(request, exc: ValueError):  # noqa: ANN001
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.middleware("http")
    async def _auth_guard(request: Request, call_next):  # noqa: ANN001
        """统一鉴权：公开路径放行；写请求校验同源。"""
        path = request.url.path
        public = (
            path == "/login"
            or path == "/api/login"
            or path == "/api/health"
        )
        if not public and not service.validate_session(
            request.cookies.get(auth.COOKIE_NAME)
        ):
            if path.startswith("/api/"):
                return JSONResponse(
                    status_code=401,
                    content={"ok": False, "detail": _MSG_UNAUTHORIZED},
                )
            return RedirectResponse("/login", status_code=302)
        # 鉴权之后再做 CSRF 同源校验：未登录一律 401，已登录的跨站写请求 403
        if path.startswith("/api/") and request.method in _UNSAFE_METHODS:
            if not _check_same_origin(request):
                return JSONResponse(
                    status_code=403,
                    content={"ok": False, "detail": "跨站请求被拒绝"},
                )
        return await call_next(request)

    def _setup_allowed(request: Request) -> bool:
        """首次设置仅允许回环来源，避免 0.0.0.0 暴露时被抢注。"""
        return auth.is_loopback_host(_client_ip(request))

    # ------------------------------------------------ 登录

    @app.get("/login")
    def login_page(request: Request) -> Response:
        if service.is_configured and service.validate_session(
            request.cookies.get(auth.COOKIE_NAME)
        ):
            return RedirectResponse("/", status_code=302)
        return _read_login_page(service, request)

    @app.post("/api/login")
    def api_login(
        request: Request,
        password: str = Form(""),
        confirm: str = Form(""),
    ) -> Response:
        ip = _client_ip(request)
        setup_mode = not service.is_configured
        if setup_mode and not _setup_allowed(request):
            return JSONResponse(
                status_code=403, content={"ok": False, "detail": _MSG_LOCAL_ONLY}
            )
        if service.limiter.is_blocked(ip):
            return JSONResponse(
                status_code=429, content={"ok": False, "detail": _MSG_RATE_LIMITED}
            )

        if setup_mode:
            if confirm and confirm != password:
                return JSONResponse(
                    status_code=400, content={"ok": False, "detail": "两次输入的密码不一致"}
                )
            error = auth.validate_password_policy(password)
            if error:
                return JSONResponse(
                    status_code=400, content={"ok": False, "detail": error}
                )
            try:
                service.set_password(password)
            except auth.AuthError as exc:
                return JSONResponse(
                    status_code=400, content={"ok": False, "detail": str(exc)}
                )
        elif not service.verify(password):
            service.limiter.record_failure(ip)
            return JSONResponse(
                status_code=401, content={"ok": False, "detail": _MSG_BAD_PASSWORD}
            )

        service.limiter.reset(ip)
        response = JSONResponse(
            content={"ok": True, "setup": setup_mode, "auth": True}
        )
        response.set_cookie(
            value=service.issue_session(),
            **_session_cookie_kwargs(request, auth.SESSION_TTL_SECONDS),
        )
        return response

    @app.post("/api/logout")
    def api_logout(request: Request) -> Response:
        response = JSONResponse(content={"ok": True})
        response.set_cookie(
            value="",
            **_session_cookie_kwargs(request, 0),
        )
        return response

    @app.get("/")
    def index() -> Response:
        if not _INDEX_HTML.is_file():
            raise HTTPException(500, "页面文件缺失: watermark/web/index.html")
        return Response(_INDEX_HTML.read_bytes(), media_type="text/html; charset=utf-8")

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__, "auth": service.is_configured}

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
        color_rgb = _parse_color(color)
        for name, value in (("stroke", stroke), ("tile_gap", tile_gap)):
            _check_non_negative(name, value)
        with _temp_file_from_upload(logo) as logo_path:
            if logo_path is not None:
                layer = Layer(
                    kind="logo", content=logo_path, opacity=opacity,
                    position=position, margin=margin,
                    font_size=size, stroke_width=stroke, rotate=rotate,
                    color=color_rgb,
                )
                notes = ["logo 图层不支持平铺，tile 参数已忽略"]
            else:
                if not text.strip():
                    raise ValueError("请提供水印文字，或上传 logo 图片")
                layer = Layer(
                    kind="text", content=text, opacity=opacity,
                    position=position, margin=margin,
                    font_size=size, stroke_width=stroke, rotate=rotate,
                    tile=tile, tile_gap=tile_gap, color=color_rgb,
                )
                notes = []
                if tile and tile_gap:
                    notes.append("tile_gap 暂以字号近似参与布局，独立间隔未生效")
            spec = WatermarkSpec(layers=(layer,), blend_space=blend_space)
            report = pipeline.add_watermark(img, spec)

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

    # ------------------------------------------------ catalog（只读查询）

    def _catalog_error(status_code: int, message: str) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"error": message})

    @app.get("/api/catalog")
    def api_catalog(
        request: Request,
        limit: str = Query(str(_CATALOG_DEFAULT_LIMIT)),
        offset: str = Query("0"),
        action: str | None = Query(None),
        status: str | None = Query(None),
    ) -> Response:
        try:
            limit_value = _catalog_limit(int(limit))
            offset_value = _catalog_offset(int(offset))
        except ValueError:
            return _catalog_error(
                400,
                f"limit 需为 {_CATALOG_MIN_LIMIT}..{_CATALOG_MAX_LIMIT} 的整数"
                f"（默认 {_CATALOG_DEFAULT_LIMIT}），offset 需为不小于 0 的整数",
            )
        if action is not None:
            try:
                _catalog_filter_value("action", action, _CATALOG_ACTIONS)
            except ValueError as exc:
                return _catalog_error(400, str(exc))
        if status is not None:
            try:
                _catalog_filter_value("status", status, _CATALOG_STATUSES)
            except ValueError as exc:
                return _catalog_error(400, str(exc))
        items = catalog.list_entries(
            limit=limit_value, offset=offset_value, action=action, status=status
        )
        return {
            "total": catalog.stats()["total"],
            "limit": limit_value,
            "offset": offset_value,
            "items": items,
        }

    @app.get("/api/catalog/{entry_id}")
    def api_catalog_entry(entry_id: str) -> Response:
        try:
            numeric_id = int(entry_id)
        except ValueError:
            return _catalog_error(400, f"entry_id 需为整数，当前为 {entry_id!r}")
        entry = catalog.get_entry(numeric_id)
        if entry is None:
            return _catalog_error(404, "not found")
        return entry

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
