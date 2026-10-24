"""单用户口令认证：scrypt 口令哈希、HMAC 签名会话 Cookie、登录限流。

设计边界（纯本地/私有部署）：
- 只有一个口令，没有注册、没有多用户、没有角色。
- 凭据文件默认 ``~/.config/watermark-kit/auth.json``，权限 0600；
  可用 ``WATERMARK_CONFIG_DIR`` 覆盖配置目录（容器/测试用）。
- ``WATERMARK_PASSWORD`` 环境变量优先于凭据文件：启动时哈希进内存，
  不落盘，并禁用首次设置流程。
- 会话 Cookie 为 ``base64url(payload).HMAC-SHA256(secret, payload)``，
  payload 只含版本、签发时间和过期时间，不含任何口令信息。

本模块不依赖 Web 框架，便于单独测试；API 层只做请求/响应适配。
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import ipaddress
import json
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# ---------------------------------------------------------------- 常量

# 凭据文件版本，字段或 KDF 参数变更时递增
AUTH_VERSION = 1
# scrypt 参数：内存约 16 MiB，单次校验几十毫秒，足够本地口令
SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16
HASH_BYTES = 32
SECRET_BYTES = 32

# 会话 Cookie
COOKIE_NAME = "wm_session"
SESSION_TTL_SECONDS = 7 * 24 * 3600

# 登录限流：同一 IP 在窗口内失败达到上限后拒绝
MAX_LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 5 * 60

# 口令策略
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 1024

# 凭据文件权限
FILE_MODE = 0o600
DIR_MODE = 0o700

CONFIG_DIR_ENV = "WATERMARK_CONFIG_DIR"
PASSWORD_ENV = "WATERMARK_PASSWORD"
CONFIG_DIR_NAME = "watermark-kit"


class AuthError(ValueError):
    """凭据文件或认证配置错误（消息中不包含哈希内容）。"""


# ---------------------------------------------------------------- 编解码

def encode_token(raw: bytes) -> str:
    """无 padding 的 base64url。"""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_token(text: str) -> bytes:
    """解码无 padding 的 base64url；非法输入抛 ValueError。"""
    padded = text + "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, UnicodeEncodeError, ValueError) as exc:
        raise ValueError("非法 base64url") from exc


# ---------------------------------------------------------------- 口令哈希

def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    """返回 (salt, scrypt 哈希)。salt 未提供时随机生成 16 字节。"""
    if not isinstance(password, str) or not password:
        raise ValueError("密码不能为空")
    if salt is None:
        salt = secrets.token_bytes(SALT_BYTES)
    if len(salt) != SALT_BYTES:
        raise ValueError(f"salt 必须为 {SALT_BYTES} 字节")
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P,
        dklen=HASH_BYTES,
    )
    return salt, digest


def verify_password(password: str, salt: bytes, expected_hash: bytes) -> bool:
    """恒定时间比较口令哈希；任何异常都视为校验失败。"""
    if not isinstance(password, str) or not password:
        return False
    try:
        _, candidate = hash_password(password, salt=salt)
    except (ValueError, TypeError):
        return False
    return hmac.compare_digest(candidate, expected_hash)


def validate_password_policy(password: str) -> str | None:
    """返回错误提示；None 表示通过。"""
    if not isinstance(password, str) or not password:
        return "密码不能为空"
    if len(password) < PASSWORD_MIN_LENGTH:
        return f"密码至少 {PASSWORD_MIN_LENGTH} 位"
    if len(password) > PASSWORD_MAX_LENGTH:
        return f"密码不能超过 {PASSWORD_MAX_LENGTH} 位"
    return None


# ---------------------------------------------------------------- 凭据文件

@dataclass(frozen=True)
class AuthConfig:
    """持久化的认证材料（全部为原始字节，序列化时才编码）。"""

    salt: bytes
    password_hash: bytes
    secret: bytes
    version: int = AUTH_VERSION

    def to_json_dict(self) -> dict:
        return {
            "version": self.version,
            "salt": encode_token(self.salt),
            "hash": encode_token(self.password_hash),
            "secret": encode_token(self.secret),
        }

    @classmethod
    def from_json_dict(cls, data: object) -> "AuthConfig":
        if not isinstance(data, dict):
            raise ValueError("凭据文件应为 JSON 对象")
        version = data.get("version")
        if version != AUTH_VERSION:
            raise ValueError(f"不支持的凭据版本: {version!r}")
        salt = decode_token(str(data["salt"]))
        password_hash = decode_token(str(data["hash"]))
        secret = decode_token(str(data["secret"]))
        if len(salt) != SALT_BYTES:
            raise ValueError("salt 长度不正确")
        if len(password_hash) != HASH_BYTES:
            raise ValueError("hash 长度不正确")
        if len(secret) != SECRET_BYTES:
            raise ValueError("secret 长度不正确")
        return cls(salt=salt, password_hash=password_hash, secret=secret)


def default_auth_path() -> Path:
    """凭据文件路径：默认 ~/.config/watermark-kit/auth.json。"""
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser() / "auth.json"
    return Path.home() / ".config" / CONFIG_DIR_NAME / "auth.json"


def load_auth_config(path: Path) -> AuthConfig | None:
    """读取凭据文件；不存在返回 None，损坏时抛 AuthError（fail closed）。"""
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuthError(f"无法读取凭据文件 {path}: {exc}") from None
    try:
        return AuthConfig.from_json_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthError(f"凭据文件格式不正确 {path}: {exc}") from None


def save_auth_config(path: Path, config: AuthConfig) -> None:
    """原子写入凭据文件，创建时即为 0600。"""
    directory = path.parent
    if not directory.is_dir():
        directory.mkdir(parents=True, mode=DIR_MODE)
        os.chmod(directory, DIR_MODE)
    payload = json.dumps(config.to_json_dict(), indent=2).encode("utf-8")
    tmp_path = directory / f".{path.name}.{secrets.token_hex(8)}.tmp"
    fd = os.open(tmp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


# ---------------------------------------------------------------- 会话签名

def _session_payload(exp: int, iat: int) -> bytes:
    return json.dumps(
        {"v": AUTH_VERSION, "exp": exp, "iat": iat}, separators=(",", ":")
    ).encode("utf-8")


def _sign(secret: bytes, payload: str) -> str:
    signature = hmac.new(secret, payload.encode("ascii"), hashlib.sha256).digest()
    return encode_token(signature)


def encode_session(
    secret: bytes,
    *,
    now: float | None = None,
    ttl: int = SESSION_TTL_SECONDS,
) -> str:
    """生成 ``payload.signature`` 形式的会话值。"""
    issued = int(now if now is not None else time.time())
    payload = encode_token(_session_payload(issued + ttl, issued))
    return f"{payload}.{_sign(secret, payload)}"


def decode_session(
    secret: bytes,
    token: str,
    *,
    now: float | None = None,
) -> dict | None:
    """校验签名与过期时间；任何异常或篡改返回 None。"""
    if not isinstance(token, str) or token.count(".") != 1:
        return None
    payload, signature = token.split(".")
    if not payload or not signature:
        return None
    expected = _sign(secret, payload)
    if not hmac.compare_digest(signature, expected):
        return None
    try:
        data = json.loads(decode_token(payload))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    exp = data.get("exp")
    if not isinstance(exp, int) or isinstance(exp, bool):
        return None
    current = now if now is not None else time.time()
    if exp <= current:
        return None
    return data


# ---------------------------------------------------------------- 限流

class LoginRateLimiter:
    """内存登录限流：窗口内失败达到上限即拒绝，成功清零。"""

    def __init__(
        self,
        max_failures: int = MAX_LOGIN_FAILURES,
        window_seconds: int = LOGIN_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_failures = max_failures
        self.window_seconds = window_seconds
        self._clock = clock
        self._failures: dict[str, list[float]] = {}

    def _recent(self, key: str) -> list[float]:
        now = self._clock()
        cutoff = now - self.window_seconds
        stamps = [t for t in self._failures.get(key, []) if t > cutoff]
        if stamps:
            self._failures[key] = stamps
        else:
            self._failures.pop(key, None)
        return stamps

    def is_blocked(self, key: str) -> bool:
        return len(self._recent(key)) >= self.max_failures

    def record_failure(self, key: str) -> None:
        self._recent(key)
        self._failures.setdefault(key, []).append(self._clock())

    def reset(self, key: str) -> None:
        self._failures.pop(key, None)


# ---------------------------------------------------------------- 服务

def is_loopback_host(host: str | None) -> bool:
    """判断客户端地址是否为本机回环。"""
    if not host:
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.split("%", 1)[0]).is_loopback
    except ValueError:
        return False


class AuthService:
    """认证状态与操作入口；API 层每个应用持有一个实例。"""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        password: str | None = None,
        limiter: LoginRateLimiter | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(path) if path is not None else default_auth_path()
        self._clock = clock
        self.limiter = limiter if limiter is not None else LoginRateLimiter()
        env_password = os.environ.get(PASSWORD_ENV) if password is None else password
        self._from_env = bool(env_password)
        if self._from_env:
            salt, digest = hash_password(env_password)
            # 环境变量模式不落盘，secret 仅存内存，重启后旧会话失效
            self._config: AuthConfig | None = AuthConfig(
                salt=salt,
                password_hash=digest,
                secret=secrets.token_bytes(SECRET_BYTES),
            )
        else:
            self._config = load_auth_config(self.path)

    # -- 状态

    @property
    def is_configured(self) -> bool:
        return self._config is not None

    @property
    def uses_env_password(self) -> bool:
        return self._from_env

    @property
    def source(self) -> str | None:
        if self._from_env:
            return "env"
        return "file" if self._config is not None else None

    def status(self) -> dict:
        """对外可展示的状态：绝不含哈希或 secret。"""
        exists = self.path.is_file()
        mode = self.path.stat().st_mode & 0o777 if exists else None
        return {
            "configured": self.is_configured,
            "source": self.source,
            "path": str(self.path),
            "exists": exists,
            "mode": mode,
            "mode_text": f"{mode:04o}" if mode is not None else None,
            "permission_ok": (mode == FILE_MODE) if mode is not None else None,
        }

    # -- 口令

    def verify(self, password: str) -> bool:
        if self._config is None:
            return False
        return verify_password(password, self._config.salt, self._config.password_hash)

    def set_password(self, password: str) -> None:
        """写入新口令并轮换会话 secret（旧会话全部失效）。"""
        if self._from_env:
            raise AuthError("已通过 WATERMARK_PASSWORD 提供口令，不能写入凭据文件")
        error = validate_password_policy(password)
        if error:
            raise AuthError(error)
        salt, digest = hash_password(password)
        config = AuthConfig(
            salt=salt,
            password_hash=digest,
            secret=secrets.token_bytes(SECRET_BYTES),
        )
        save_auth_config(self.path, config)
        self._config = config

    # -- 会话

    def issue_session(self, *, now: float | None = None) -> str:
        if self._config is None:
            raise AuthError("尚未设置访问密码")
        return encode_session(self._config.secret, now=now)

    def validate_session(self, token: str | None) -> bool:
        if self._config is None or not token:
            return False
        return decode_session(
            self._config.secret, token, now=self._clock()
        ) is not None
