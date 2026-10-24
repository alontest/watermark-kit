"""认证测试：scrypt 哈希、会话签名/篡改/过期、限流、凭据文件权限、Web 鉴权链路。"""

from __future__ import annotations

import json
import os
import stat
import time

import pytest
from fastapi.testclient import TestClient

from watermark import auth
from watermark.api import create_app

from conftest import AUTH_PASSWORD, TEST_ORIGIN

WRONG_PASSWORD = "definitely-wrong-2026"


# ---------------------------------------------------------------- 口令哈希

def test_hash_password_roundtrip():
    salt, digest = auth.hash_password(AUTH_PASSWORD)
    assert len(salt) == auth.SALT_BYTES
    assert len(digest) == auth.HASH_BYTES
    assert auth.verify_password(AUTH_PASSWORD, salt, digest) is True


def test_hash_password_uses_scrypt_parameters():
    salt, digest = auth.hash_password(AUTH_PASSWORD, salt=b"\x01" * auth.SALT_BYTES)
    import hashlib

    expected = hashlib.scrypt(
        AUTH_PASSWORD.encode("utf-8"), salt=salt,
        n=auth.SCRYPT_N, r=auth.SCRYPT_R, p=auth.SCRYPT_P,
        dklen=auth.HASH_BYTES,
    )
    assert digest == expected


def test_hash_password_random_salt_differs():
    salt_a, digest_a = auth.hash_password(AUTH_PASSWORD)
    salt_b, digest_b = auth.hash_password(AUTH_PASSWORD)
    assert salt_a != salt_b
    assert digest_a != digest_b


def test_verify_password_rejects_wrong_and_empty():
    salt, digest = auth.hash_password(AUTH_PASSWORD)
    assert auth.verify_password(WRONG_PASSWORD, salt, digest) is False
    assert auth.verify_password("", salt, digest) is False


def test_password_policy():
    assert auth.validate_password_policy("short") is not None
    assert auth.validate_password_policy(AUTH_PASSWORD) is None


# ---------------------------------------------------------------- 会话签名

def _secret() -> bytes:
    return b"\x02" * auth.SECRET_BYTES


def test_session_roundtrip_contains_only_meta():
    token = auth.encode_session(_secret(), now=1_000_000)
    assert token.count(".") == 1
    payload = auth.decode_session(_secret(), token, now=1_000_000)
    assert payload is not None
    assert set(payload) == {"v", "exp", "iat"}
    assert payload["exp"] == 1_000_000 + auth.SESSION_TTL_SECONDS


def test_session_expired_rejected():
    token = auth.encode_session(_secret(), now=1_000_000, ttl=60)
    assert auth.decode_session(_secret(), token, now=1_000_061) is None
    assert auth.decode_session(_secret(), token, now=1_000_059) is not None


def test_session_tampered_payload_rejected():
    token = auth.encode_session(_secret(), now=1_000_000)
    payload, signature = token.split(".")
    forged = auth.encode_token(json.dumps(
        {"v": 1, "exp": 9_999_999_999, "iat": 1_000_000}
    ).encode())
    assert auth.decode_session(_secret(), f"{forged}.{signature}", now=1_000_000) is None


def test_session_tampered_signature_rejected():
    token = auth.encode_session(_secret(), now=1_000_000)
    payload, signature = token.split(".")
    flipped = ("A" if signature[0] != "A" else "B") + signature[1:]
    assert auth.decode_session(_secret(), f"{payload}.{flipped}", now=1_000_000) is None


def test_session_wrong_secret_rejected():
    token = auth.encode_session(_secret(), now=1_000_000)
    assert auth.decode_session(b"\x03" * auth.SECRET_BYTES, token, now=1_000_000) is None


@pytest.mark.parametrize("bad", ["", "no-dot", "a.b.c", ".", "abc.", ".abc", "!!!.???"])
def test_session_malformed_rejected(bad):
    assert auth.decode_session(_secret(), bad, now=1_000_000) is None


def test_session_payload_has_no_password_material():
    token = auth.encode_session(_secret(), now=1_000_000)
    assert AUTH_PASSWORD not in token
    payload = token.split(".")[0]
    assert "password" not in auth.decode_token(payload).decode("utf-8").lower()


# ---------------------------------------------------------------- 限流

def test_rate_limiter_blocks_after_five_failures():
    now = [0.0]
    limiter = auth.LoginRateLimiter(clock=lambda: now[0])
    for _ in range(auth.MAX_LOGIN_FAILURES):
        assert limiter.is_blocked("1.2.3.4") is False
        limiter.record_failure("1.2.3.4")
    assert limiter.is_blocked("1.2.3.4") is True
    assert limiter.is_blocked("5.6.7.8") is False


def test_rate_limiter_window_expires():
    now = [0.0]
    limiter = auth.LoginRateLimiter(clock=lambda: now[0])
    for _ in range(auth.MAX_LOGIN_FAILURES):
        limiter.record_failure("1.2.3.4")
    assert limiter.is_blocked("1.2.3.4") is True
    now[0] = auth.LOGIN_WINDOW_SECONDS + 1
    assert limiter.is_blocked("1.2.3.4") is False


def test_rate_limiter_reset_on_success():
    limiter = auth.LoginRateLimiter()
    for _ in range(auth.MAX_LOGIN_FAILURES - 1):
        limiter.record_failure("1.2.3.4")
    limiter.reset("1.2.3.4")
    for _ in range(auth.MAX_LOGIN_FAILURES - 1):
        limiter.record_failure("1.2.3.4")
    assert limiter.is_blocked("1.2.3.4") is False


# ---------------------------------------------------------------- 凭据文件

def test_save_auth_config_permissions_0600(tmp_path):
    path = tmp_path / "nested" / "auth.json"
    salt, digest = auth.hash_password(AUTH_PASSWORD)
    config = auth.AuthConfig(salt=salt, password_hash=digest, secret=os.urandom(32))
    auth.save_auth_config(path, config)

    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, f"凭据文件权限应为 0600，实际 {oct(mode)}"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert auth.load_auth_config(path) == config
    assert not list(path.parent.glob("*.tmp")), "不应残留临时文件"


def test_auth_config_has_no_plaintext_password(tmp_path):
    path = tmp_path / "auth.json"
    salt, digest = auth.hash_password(AUTH_PASSWORD)
    auth.save_auth_config(
        path, auth.AuthConfig(salt=salt, password_hash=digest, secret=os.urandom(32))
    )
    raw = path.read_text(encoding="utf-8")
    assert AUTH_PASSWORD not in raw
    data = json.loads(raw)
    assert set(data) == {"version", "salt", "hash", "secret"}
    assert "password" not in raw.lower()


def test_load_missing_config_returns_none(tmp_path):
    assert auth.load_auth_config(tmp_path / "absent.json") is None


def test_load_corrupt_config_fails_closed(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(auth.AuthError):
        auth.load_auth_config(path)


def test_load_wrong_version_rejected(tmp_path):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps({"version": 99}), encoding="utf-8")
    with pytest.raises(auth.AuthError):
        auth.load_auth_config(path)


def test_service_set_password_rotates_secret(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    old_token = auth_service.issue_session()
    assert auth_service.validate_session(old_token) is True
    auth_service.set_password(AUTH_PASSWORD + "-new")
    assert auth_service.validate_session(old_token) is False, "改密后旧会话应失效"


def test_service_status_has_no_secrets(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    payload = auth_service.status()
    assert payload["configured"] is True
    assert payload["mode_text"] == "0600"
    assert payload["permission_ok"] is True
    text = json.dumps(payload)
    assert "hash" not in text and "secret" not in text


def test_env_password_overrides_file_and_disables_setup(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    salt, digest = auth.hash_password("file-password-1")
    auth.save_auth_config(
        path, auth.AuthConfig(salt=salt, password_hash=digest, secret=os.urandom(32))
    )
    monkeypatch.setenv("WATERMARK_PASSWORD", AUTH_PASSWORD)
    service = auth.AuthService(path=path)
    assert service.source == "env"
    assert service.verify(AUTH_PASSWORD) is True
    assert service.verify("file-password-1") is False
    with pytest.raises(auth.AuthError):
        service.set_password("another-password")


def test_env_password_does_not_write_file(tmp_path, monkeypatch):
    path = tmp_path / "auth.json"
    monkeypatch.setenv("WATERMARK_PASSWORD", AUTH_PASSWORD)
    auth.AuthService(path=path)
    assert not path.exists()


def test_loopback_detection():
    assert auth.is_loopback_host("127.0.0.1") is True
    assert auth.is_loopback_host("::1") is True
    assert auth.is_loopback_host("localhost") is True
    assert auth.is_loopback_host("0.0.0.0") is False
    assert auth.is_loopback_host("192.168.1.10") is False
    assert auth.is_loopback_host(None) is False


# ---------------------------------------------------------------- Web 链路

def _client(app) -> TestClient:
    """默认模拟本机浏览器：回环来源 + 同源 Origin。"""
    return TestClient(
        app, headers={"Origin": TEST_ORIGIN}, client=("127.0.0.1", 50000)
    )


def _setup_and_login(client, password: str = AUTH_PASSWORD):
    return client.post(
        "/api/login", data={"password": password, "confirm": password}
    )


def test_unauthenticated_page_redirects_to_login(auth_app):
    resp = _client(auth_app).get("/", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/login"


@pytest.mark.parametrize(
    "path", ["/api/add", "/api/remove", "/api/inspect", "/api/download/x", "/api/logout"]
)
def test_unauthenticated_api_returns_401_json(auth_app, path):
    client = _client(auth_app)
    method = client.get if path.startswith("/api/download") else client.post
    resp = method(path)
    assert resp.status_code == 401
    assert resp.json() == {"ok": False, "detail": "请先登录"}


def test_login_page_public_and_shows_setup_when_unconfigured(auth_app):
    resp = _client(auth_app).get("/login")
    assert resp.status_code == 200
    assert "设置密码" in resp.text
    assert '"setup"' in resp.text


def test_login_page_blocked_for_remote_before_setup(auth_app):
    client = TestClient(
        auth_app, client=("192.168.1.50", 50000), headers={"Origin": TEST_ORIGIN}
    )
    resp = client.get("/login")
    assert resp.status_code == 200
    assert '"blocked"' in resp.text
    assert "本机" in resp.text


def test_login_page_shows_login_mode_when_configured(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    resp = _client(create_app(auth_service)).get("/login")
    assert resp.status_code == 200
    assert '"login"' in resp.text


def test_first_setup_sets_cookie_flags(auth_app):
    client = _client(auth_app)
    resp = _setup_and_login(client)
    assert resp.status_code == 200
    assert resp.json()["setup"] is True
    cookie = resp.headers["set-cookie"]
    assert auth.COOKIE_NAME in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/" in cookie
    assert f"Max-Age={auth.SESSION_TTL_SECONDS}" in cookie
    assert "Secure" not in cookie, "HTTP 请求不应加 Secure"
    assert client.get("/").status_code == 200


def test_secure_cookie_flag_on_https(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = TestClient(create_app(auth_service), base_url="https://testserver")
    resp = client.post(
        "/api/login", data={"password": AUTH_PASSWORD},
        headers={"Origin": "https://testserver"},
    )
    assert resp.status_code == 200
    assert "Secure" in resp.headers["set-cookie"]


def test_second_setup_attempt_rejected(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    resp = client.post(
        "/api/login", data={"password": "brand-new-password", "confirm": "brand-new-password"}
    )
    assert resp.status_code == 401


def test_login_success_and_health(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    assert client.get("/api/health").json()["auth"] is True
    assert client.post("/api/login", data={"password": AUTH_PASSWORD}).status_code == 200
    assert client.get("/api/health").status_code == 200


def test_wrong_password_401_and_message(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    resp = client.post("/api/login", data={"password": WRONG_PASSWORD})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "密码不正确"
    assert "set-cookie" not in resp.headers


def test_setup_password_policy_enforced(auth_app):
    resp = _setup_and_login(_client(auth_app), password="short")
    assert resp.status_code == 400
    assert "至少" in resp.json()["detail"]


def test_setup_confirm_mismatch_rejected(auth_app):
    resp = _client(auth_app).post(
        "/api/login", data={"password": AUTH_PASSWORD, "confirm": AUTH_PASSWORD + "x"}
    )
    assert resp.status_code == 400
    assert "不一致" in resp.json()["detail"]


def test_setup_rejected_from_non_loopback(auth_service):
    app = create_app(auth_service)
    client = TestClient(
        app, client=("192.168.1.50", 50000), headers={"Origin": TEST_ORIGIN}
    )
    resp = client.post(
        "/api/login", data={"password": AUTH_PASSWORD, "confirm": AUTH_PASSWORD}
    )
    assert resp.status_code == 403
    assert "本机" in resp.json()["detail"]
    assert auth_service.is_configured is False, "非回环来源不得写入口令"


def test_tampered_cookie_401(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    client.post("/api/login", data={"password": AUTH_PASSWORD})
    token = client.cookies.get(auth.COOKIE_NAME)
    payload, signature = token.split(".")
    client.cookies.set(auth.COOKIE_NAME, f"{payload}.{'A' * len(signature)}")
    assert client.get("/", follow_redirects=False).status_code == 302
    assert client.post("/api/add").status_code == 401


def test_expired_cookie_401(tmp_path):
    salt, digest = auth.hash_password(AUTH_PASSWORD)
    secret = os.urandom(32)
    path = tmp_path / "auth.json"
    auth.save_auth_config(
        path, auth.AuthConfig(salt=salt, password_hash=digest, secret=secret)
    )
    service = auth.AuthService(path=path)
    client = _client(create_app(service))
    expired = auth.encode_session(secret, now=time.time() - auth.SESSION_TTL_SECONDS - 10)
    client.cookies.set(auth.COOKIE_NAME, expired)
    assert client.get("/", follow_redirects=False).status_code == 302
    assert client.post("/api/inspect").status_code == 401


def test_sixth_failed_attempt_returns_429(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    for i in range(auth.MAX_LOGIN_FAILURES):
        resp = client.post("/api/login", data={"password": WRONG_PASSWORD})
        assert resp.status_code == 401, f"第 {i + 1} 次失败应为 401"
    resp = client.post("/api/login", data={"password": WRONG_PASSWORD})
    assert resp.status_code == 429
    assert "5 分钟" in resp.json()["detail"]
    # 限流期间即使密码正确也拒绝
    assert client.post("/api/login", data={"password": AUTH_PASSWORD}).status_code == 429


def test_successful_login_resets_failure_counter(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    for _ in range(auth.MAX_LOGIN_FAILURES - 1):
        client.post("/api/login", data={"password": WRONG_PASSWORD})
    assert client.post("/api/login", data={"password": AUTH_PASSWORD}).status_code == 200
    for _ in range(auth.MAX_LOGIN_FAILURES - 1):
        resp = client.post("/api/login", data={"password": WRONG_PASSWORD})
        assert resp.status_code == 401


def test_cross_origin_post_403(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    client.post("/api/login", data={"password": AUTH_PASSWORD})
    resp = client.post(
        "/api/logout", headers={"Origin": "http://evil.example.com"}
    )
    assert resp.status_code == 403
    assert "跨站" in resp.json()["detail"]


def test_missing_origin_and_referer_403(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = TestClient(create_app(auth_service))  # 不带默认 Origin
    client.post(
        "/api/login", data={"password": AUTH_PASSWORD},
        headers={"Origin": TEST_ORIGIN},
    )
    assert client.post("/api/logout").status_code == 403


def test_same_origin_referer_allowed(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = TestClient(create_app(auth_service), client=("127.0.0.1", 50000))
    client.post(
        "/api/login", data={"password": AUTH_PASSWORD},
        headers={"Referer": f"{TEST_ORIGIN}/login"},
    )
    resp = client.post("/api/logout", headers={"Referer": f"{TEST_ORIGIN}/"})
    assert resp.status_code == 200


def test_logout_clears_cookie(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    client.post("/api/login", data={"password": AUTH_PASSWORD})
    resp = client.post("/api/logout")
    assert resp.status_code == 200
    cookie = resp.headers["set-cookie"]
    assert "Max-Age=0" in cookie
    assert client.get("/", follow_redirects=False).status_code == 302


def test_login_page_redirects_when_already_logged_in(auth_service):
    auth_service.set_password(AUTH_PASSWORD)
    client = _client(create_app(auth_service))
    client.post("/api/login", data={"password": AUTH_PASSWORD})
    resp = client.get("/login", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["location"] == "/"


def test_env_password_disables_setup_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("WATERMARK_PASSWORD", AUTH_PASSWORD)
    service = auth.AuthService(path=tmp_path / "auth.json")
    client = _client(create_app(service))
    resp = client.post(
        "/api/login", data={"password": AUTH_PASSWORD, "confirm": AUTH_PASSWORD}
    )
    assert resp.status_code == 200
    assert resp.json()["setup"] is False
    assert not (tmp_path / "auth.json").exists()
