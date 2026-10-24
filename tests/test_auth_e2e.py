"""认证边界独立验收：真实 uvicorn 进程 + 真实 HTTP，不改产品代码。

与 ``test_auth.py`` 的差别：那里用 TestClient 在进程内直调 ASGI，本文件
每个用例都启动独立子进程服务，经 TCP 走完整中间件、Cookie、限流与 CSRF
链路，避免「实现者自证」。

覆盖看板卡 t_6222b553 的 9 项：
1. 未认证边界  2. 登录成功  3. 凭据错误不可区分  4. 会话篡改/过期
5. 限流        6. CSRF     7. 凭据文件与 auth status  8. 登录态全链路回归
9. 退出登录

发现的产线问题只记录在完成报告中，不在测试里放宽断言。
"""

from __future__ import annotations

import io
import json
import os
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from watermark import auth
from watermark.blend import diff_stats, psnr

from conftest import AUTH_PASSWORD

WRONG_PASSWORD = "{{SECRET_9k2m4q7x}}"
REPO_ROOT = Path(__file__).resolve().parents[1]

# 验收阈值（与 test_e2e.py / test_roundtrip.py 一致）
MIN_PSNR_DB = 45.0
MAX_PIXEL_DIFF = 1.0  # 1/255 * 255

# 服务启动与单请求超时
SERVER_START_TIMEOUT = 25.0
HTTP_TIMEOUT = 30.0


# ---------------------------------------------------------------- 工具

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _png_bytes(seed: int = 2026, w: int = 200, h: int = 150) -> bytes:
    """numpy 合成 PNG（渐变 + 色块 + 噪声），返回字节。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([
        xx / (w - 1) * 255.0,
        yy / (h - 1) * 255.0,
        np.full((h, w), 128.0),
    ], axis=-1)
    circle = (yy - h // 2) ** 2 + (xx - w // 3) ** 2 <= (h // 4) ** 2
    base[circle] = [220.0, 80.0, 60.0]
    base += rng.normal(0, 4, base.shape)
    buf = io.BytesIO()
    Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").save(buf, "PNG")
    return buf.getvalue()


def _array_from_png(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


class LiveServer:
    """一个真实 uvicorn 子进程 + 指向它的 HTTP 客户端工厂。"""

    def __init__(self, base_url: str, config_dir: Path, proc: subprocess.Popen) -> None:
        self.base_url = base_url
        self.config_dir = config_dir
        self.auth_path = config_dir / "auth.json"
        self.proc = proc

    def client(self) -> httpx.Client:
        """不跟随重定向、默认不带 Origin（需要时显式传）。"""
        return httpx.Client(
            base_url=self.base_url,
            follow_redirects=False,
            timeout=HTTP_TIMEOUT,
        )

    def origin(self) -> dict[str, str]:
        return {"Origin": self.base_url}

    def login(self, client: httpx.Client, password: str = AUTH_PASSWORD) -> httpx.Response:
        return client.post(
            "/api/login", data={"password": password}, headers=self.origin()
        )

    def secret(self) -> bytes:
        config = auth.load_auth_config(self.auth_path)
        assert config is not None, "凭据文件应存在"
        return config.secret


def _wait_until_healthy(base_url: str, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    last_error = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(f"uvicorn 提前退出({proc.returncode}): {output}")
        try:
            resp = httpx.get(f"{base_url}/api/health", timeout=2.0)
            if resp.status_code == 200:
                return
            last_error = f"health={resp.status_code}"
        except httpx.HTTPError as exc:
            last_error = repr(exc)
        time.sleep(0.1)
    raise RuntimeError(f"服务 {SERVER_START_TIMEOUT}s 内未就绪: {last_error}")


def _start_server(config_dir: Path) -> LiveServer:
    port = _free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = {**os.environ, auth.CONFIG_DIR_ENV: str(config_dir)}
    env.pop(auth.PASSWORD_ENV, None)
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn", "watermark.api:create_app",
            "--factory", "--host", "127.0.0.1", "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        _wait_until_healthy(base_url, proc)
    except BaseException:
        proc.terminate()
        proc.wait(timeout=10)
        raise
    return LiveServer(base_url, config_dir, proc)


def _stop_server(server: LiveServer) -> None:
    if server.proc.poll() is None:
        server.proc.terminate()
        try:
            server.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.proc.kill()
            server.proc.wait(timeout=5)


@pytest.fixture()
def server(tmp_path):
    """未配置口令的真实服务（首次设置场景）。"""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    live = _start_server(config_dir)
    try:
        yield live
    finally:
        _stop_server(live)


@pytest.fixture()
def configured_server(tmp_path):
    """已写入口令的真实服务（凭据文件由产品代码生成）。"""
    config_dir = tmp_path / "config"
    auth.AuthService(path=config_dir / "auth.json").set_password(AUTH_PASSWORD)
    live = _start_server(config_dir)
    try:
        yield live
    finally:
        _stop_server(live)


@pytest.fixture()
def logged_in(configured_server):
    """已完成登录的客户端（Cookie 在真实 HTTP 链路上签发）。"""
    client = configured_server.client()
    resp = configured_server.login(client)
    assert resp.status_code == 200, resp.text
    try:
        yield configured_server, client
    finally:
        client.close()


def _run_cli(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    merged = {**os.environ, **(env or {})}
    return subprocess.run(
        [sys.executable, "-m", "watermark.cli", *args],
        cwd=str(REPO_ROOT), env=merged,
        capture_output=True, text=True, timeout=120,
    )


# ---------------------------------------------------------------- 1. 未认证边界

def test_unauthenticated_boundaries(server):
    with server.client() as client:
        resp = client.get("/")
        assert resp.status_code == 302
        assert resp.headers["location"] == "/login"

        for path in ("/api/add", "/api/remove", "/api/inspect"):
            resp = client.post(path)
            assert resp.status_code == 401, f"{path} 未登录应 401，实际 {resp.status_code}"
            assert resp.json() == {"ok": False, "detail": "请先登录"}

        resp = client.get("/api/download/anything")
        assert resp.status_code == 401
        assert resp.json()["ok"] is False

        health = client.get("/api/health")
        assert health.status_code == 200
        assert set(health.json()) == {"ok", "version", "auth"}, health.text


# ---------------------------------------------------------------- 2. 登录成功

def test_login_success_sets_cookie_and_authorizes_api(server):
    with server.client() as client:
        resp = client.post(
            "/api/login",
            data={"password": AUTH_PASSWORD, "confirm": AUTH_PASSWORD},
            headers=server.origin(),
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"ok": True, "setup": True, "auth": True}

        cookie = resp.headers["set-cookie"]
        assert auth.COOKIE_NAME in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
        assert "Path=/" in cookie
        assert "Secure" not in cookie, "HTTP 请求不应设置 Secure"
        assert client.cookies.get(auth.COOKIE_NAME)

        resp = client.post(
            "/api/add",
            files={"image": ("img.png", _png_bytes(), "image/png")},
            data={"text": "登录后写入", "color": "#00cc44"},
            headers=server.origin(),
        )
        assert resp.status_code == 200, resp.text
        payload = resp.json()
        assert payload["ok"] is True
        assert payload["token"]
        assert payload["spec"]["layers"][0]["color"] == [0, 204, 68]


# ---------------------------------------------------------------- 3. 凭据错误

def test_wrong_password_is_indistinguishable(configured_server):
    with configured_server.client() as client:
        resp = configured_server.login(client, WRONG_PASSWORD)
        assert resp.status_code == 401
        body = resp.text
        assert resp.json() == {"ok": False, "detail": "密码不正确"}
        for leak in ("用户不存在", "无此用户", "未注册", "账号", "username"):
            assert leak not in body, f"单用户场景不应泄露区分信息: {leak}"
        assert "set-cookie" not in {k.lower() for k in resp.headers}
        assert not client.cookies.get(auth.COOKIE_NAME)


# ---------------------------------------------------------------- 4. 会话篡改

def _tampered_tokens(server: LiveServer, token: str) -> dict[str, str]:
    payload, signature = token.split(".")
    forged_payload = auth.encode_token(json.dumps(
        {"v": auth.AUTH_VERSION, "exp": 9_999_999_999, "iat": 1}
    ).encode())
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    expired = auth.encode_session(
        server.secret(), now=time.time() - auth.SESSION_TTL_SECONDS - 60
    )
    return {
        "payload": f"{forged_payload}.{signature}",
        "signature": f"{payload}.{flipped}",
        "truncated": token[: len(token) // 2],
        "expired": expired,
    }


@pytest.mark.parametrize("kind", ["payload", "signature", "truncated", "expired"])
def test_tampered_session_rejected(logged_in, kind):
    server, client = logged_in
    token = client.cookies.get(auth.COOKIE_NAME)
    assert token
    forged = _tampered_tokens(server, token)[kind]

    with server.client() as probe:
        resp = probe.get("/", headers={"Cookie": f"{auth.COOKIE_NAME}={forged}"})
        assert resp.status_code == 302, f"{kind} 篡改后页面应重定向"
        assert resp.headers["location"] == "/login"

        resp = probe.post(
            "/api/add", headers={"Cookie": f"{auth.COOKIE_NAME}={forged}"}
        )
        assert resp.status_code == 401, f"{kind} 篡改后 API 应 401，实际 {resp.status_code}"


# ---------------------------------------------------------------- 5. 限流

def test_rate_limit_blocks_sixth_attempt(configured_server):
    with configured_server.client() as client:
        for i in range(auth.MAX_LOGIN_FAILURES):
            resp = configured_server.login(client, WRONG_PASSWORD)
            assert resp.status_code == 401, f"第 {i + 1} 次失败应为 401"
        resp = configured_server.login(client, WRONG_PASSWORD)
        assert resp.status_code == 429, f"第 6 次应 429，实际 {resp.status_code}"
        assert "5 分钟" in resp.json()["detail"]
        # 限流窗口内正确口令也被拒绝
        assert configured_server.login(client, AUTH_PASSWORD).status_code == 429


def test_successful_login_resets_failure_counter(configured_server):
    with configured_server.client() as client:
        for _ in range(auth.MAX_LOGIN_FAILURES - 1):
            assert configured_server.login(client, WRONG_PASSWORD).status_code == 401
        assert configured_server.login(client, AUTH_PASSWORD).status_code == 200
        # 计数已清零：再失败 5 次仍是 401，第 6 次才 429
        for i in range(auth.MAX_LOGIN_FAILURES):
            resp = configured_server.login(client, WRONG_PASSWORD)
            assert resp.status_code == 401, f"清零后第 {i + 1} 次失败应为 401"
        assert configured_server.login(client, WRONG_PASSWORD).status_code == 429


# ---------------------------------------------------------------- 6. CSRF

def test_csrf_origin_rejected_and_same_origin_allowed(logged_in):
    server, client = logged_in
    files = {"image": ("img.png", _png_bytes(seed=7), "image/png")}
    data = {"text": "csrf"}

    evil = client.post(
        "/api/add", files=files, data=data,
        headers={"Origin": "http://evil.example"},
    )
    assert evil.status_code == 403, f"跨站写请求应 403，实际 {evil.status_code}"
    assert "跨站" in evil.json()["detail"]

    missing = client.post("/api/add", files=files, data=data)
    assert missing.status_code == 403, "无 Origin/Referer 的写请求应拒绝"

    ok = client.post("/api/add", files=files, data=data, headers=server.origin())
    assert ok.status_code == 200, ok.text
    assert ok.json()["ok"] is True


# ---------------------------------------------------------------- 7. 凭据文件

def test_credentials_file_and_auth_status(server):
    with server.client() as client:
        resp = client.post(
            "/api/login",
            data={"password": AUTH_PASSWORD, "confirm": AUTH_PASSWORD},
            headers=server.origin(),
        )
        assert resp.status_code == 200

    path = server.auth_path
    assert path.is_file(), f"首次设置应写入凭据文件: {path}"
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"凭据文件权限应为 0600，实际 {oct(mode)}"

    raw = path.read_text(encoding="utf-8")
    assert AUTH_PASSWORD not in raw, "凭据文件不得含明文密码"
    data = json.loads(raw)
    assert set(data) == {"version", "salt", "hash", "secret"}
    assert "password" not in raw.lower()

    proc = _run_cli(
        "auth", "status", "--json",
        env={auth.CONFIG_DIR_ENV: str(server.config_dir)},
    )
    assert proc.returncode == 0, proc.stderr
    status = json.loads(proc.stdout)
    assert status["configured"] is True
    assert status["mode_text"] == "0600"
    assert status["permission_ok"] is True
    text = json.dumps(status, ensure_ascii=False)
    for forbidden in ("salt", "hash", "secret"):
        assert forbidden not in text, f"auth status 不应输出 {forbidden}"
    for value in (data["salt"], data["hash"], data["secret"]):
        assert value not in proc.stdout, "auth status 不应输出凭据材料"


# ---------------------------------------------------------------- 8. 全链路回归

def _assert_reversible(original: np.ndarray, restored: bytes) -> float:
    restored_arr = _array_from_png(restored)
    assert restored_arr.shape == original.shape
    value = psnr(original, restored_arr)
    stats = diff_stats(original, restored_arr)
    assert value >= MIN_PSNR_DB, f"PSNR={value:.2f}dB < {MIN_PSNR_DB}"
    assert stats["max"] <= MAX_PIXEL_DIFF, f"最大差 {stats['max']} > {MAX_PIXEL_DIFF}/255"
    return value


def test_authenticated_roundtrip_regression(logged_in, tmp_path):
    """登录态下 API add→remove→compare 与 CLI 路径都过可逆验收线。"""
    server, client = logged_in
    src_bytes = _png_bytes(seed=11)
    original = _array_from_png(src_bytes)

    # -- API 路径（带会话 Cookie）
    add = client.post(
        "/api/add", files={"image": ("src.png", src_bytes, "image/png")},
        data={"text": "认证回归", "color": "#ff8800", "opacity": "0.3",
              "size": "24", "position": "center"},
        headers=server.origin(),
    )
    assert add.status_code == 200, add.text
    token = add.json()["token"]

    marked = client.get(f"/api/download/{token}")
    assert marked.status_code == 200
    marked_path = tmp_path / "api_marked.png"
    marked_path.write_bytes(marked.content)

    spec_text = client.get(f"/api/download/{token}", params={"as_json": 1}).text
    remove = client.post(
        "/api/remove",
        files={"image": ("wm.png", marked.content, "image/png")},
        data={"spec": spec_text},
        headers=server.origin(),
    )
    assert remove.status_code == 200, remove.text
    assert remove.json()["report"]["mask_match"] is True
    restored = client.get(f"/api/download/{remove.json()['token']}")
    assert restored.status_code == 200

    api_psnr = _assert_reversible(original, restored.content)

    # -- CLI 路径（真实子进程，同一张原图）
    src_path = tmp_path / "src.png"
    src_path.write_bytes(src_bytes)
    cli_marked = tmp_path / "cli_marked.png"
    cli_restored = tmp_path / "cli_restored.png"
    proc = _run_cli(
        "add", str(src_path), "-o", str(cli_marked),
        "--text", "认证回归", "--color", "#ff8800", "--opacity", "0.3",
        "--size", "24", "--json",
    )
    assert proc.returncode == 0, proc.stderr
    proc = _run_cli("remove", str(cli_marked), "-o", str(cli_restored), "--json")
    assert proc.returncode == 0, proc.stderr
    cli_psnr = _assert_reversible(original, cli_restored.read_bytes())

    proc = _run_cli("compare", str(src_path), str(cli_restored), "--json")
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout)
    assert report["psnr_db"] >= MIN_PSNR_DB
    assert report["max_diff"] <= MAX_PIXEL_DIFF

    print(f"认证态回归 PSNR：API {api_psnr:.2f} dB / CLI {cli_psnr:.2f} dB")


# ---------------------------------------------------------------- 9. 退出登录

def test_logout_clears_cookie(logged_in):
    server, client = logged_in
    old_token = client.cookies.get(auth.COOKIE_NAME)
    assert old_token

    resp = client.post("/api/logout", headers=server.origin())
    assert resp.status_code == 200
    cookie = resp.headers["set-cookie"]
    assert "Max-Age=0" in cookie
    assert auth.COOKIE_NAME in cookie

    # 客户端按 Set-Cookie 丢弃 Cookie 后，后续请求不再被授权
    assert not client.cookies.get(auth.COOKIE_NAME)
    assert client.get("/").status_code == 302
    assert client.post("/api/add").status_code == 401


@pytest.mark.xfail(
    strict=True,
    reason="产品缺口：会话为无状态签名 Cookie，服务端无吊销列表，退出后重放旧 Cookie 仍有效",
)
def test_logout_replayed_cookie_rejected(logged_in):
    """看板卡要求「退出后重放旧 cookie → 401」；当前实现不满足，故 xfail。"""
    server, client = logged_in
    old_token = client.cookies.get(auth.COOKIE_NAME)
    assert old_token
    assert client.post("/api/logout", headers=server.origin()).status_code == 200

    with server.client() as probe:
        resp = probe.get("/", headers={"Cookie": f"{auth.COOKIE_NAME}={old_token}"})
        assert resp.status_code == 302, "退出后旧 Cookie 重放应失效"
        resp = probe.post(
            "/api/add", headers={"Cookie": f"{auth.COOKIE_NAME}={old_token}"}
        )
        assert resp.status_code == 401, "退出后旧 Cookie 重放 API 应 401"
