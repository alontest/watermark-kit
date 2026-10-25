"""SQLite 档案库存储层：记录每次加水印/去水印的处理历史。

只负责存储，不涉及 CLI 与 API。默认库文件为 ``~/.watermark-kit/catalog.db``，
可用环境变量 ``WATERMARK_CATALOG_DB`` 覆盖（测试用它指向临时文件）。
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

# 默认库文件位置
DEFAULT_DB_DIR = Path.home() / ".watermark-kit"
DEFAULT_DB_NAME = "catalog.db"
# 环境变量名，用于覆盖库文件位置
DB_PATH_ENV = "WATERMARK_CATALOG_DB"

# 当前 schema 版本，写入 catalog_meta
SCHEMA_VERSION = "1"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS catalog_entries (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_sha256 TEXT    NOT NULL,
    source_path   TEXT    NOT NULL,
    output_path   TEXT,
    action        TEXT    NOT NULL CHECK (action IN ('add','remove')),
    spec_json     TEXT    NOT NULL,
    params_json   TEXT,
    status        TEXT    NOT NULL DEFAULT 'ok' CHECK (status IN ('ok','failed')),
    error         TEXT,
    duration_ms   INTEGER,
    created_at    INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_catalog_sha     ON catalog_entries(source_sha256);
CREATE INDEX IF NOT EXISTS idx_catalog_created ON catalog_entries(created_at DESC);
CREATE TABLE IF NOT EXISTS catalog_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

_ENTRY_COLUMNS = (
    "id",
    "source_sha256",
    "source_path",
    "output_path",
    "action",
    "spec_json",
    "params_json",
    "status",
    "error",
    "duration_ms",
    "created_at",
)


def catalog_path() -> Path:
    """返回当前生效的库文件路径（环境变量优先，否则默认路径）。"""
    env_path = os.environ.get(DB_PATH_ENV)
    if env_path:
        return Path(env_path)
    return DEFAULT_DB_DIR / DEFAULT_DB_NAME


def init_db() -> sqlite3.Connection:
    """打开连接并确保表和索引存在；幂等，可重复调用。"""
    db_path = catalog_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    with conn:
        conn.executescript(_SCHEMA_SQL)
        conn.execute(
            "INSERT OR REPLACE INTO catalog_meta (key, value) VALUES (?, ?)",
            ("schema_version", SCHEMA_VERSION),
        )
    return conn


def record(
    *,
    source_sha256: str,
    source_path: str,
    action: str,
    spec_json: str,
    output_path: str | None = None,
    params_json: str | None = None,
    status: str = "ok",
    error: str | None = None,
    duration_ms: int | None = None,
) -> int:
    """写入一条处理记录，返回新行 id。"""
    with contextlib.closing(init_db()) as conn, conn:
        cursor = conn.execute(
            """
            INSERT INTO catalog_entries (
                source_sha256, source_path, output_path, action,
                spec_json, params_json, status, error, duration_ms, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_sha256,
                source_path,
                output_path,
                action,
                spec_json,
                params_json,
                status,
                error,
                duration_ms,
                _now_ms(),
            ),
        )
        return int(cursor.lastrowid)


def list_entries(
    *,
    limit: int = 50,
    offset: int = 0,
    action: str | None = None,
    status: str | None = None,
    since: int | None = None,
) -> list[dict[str, Any]]:
    """按 created_at 倒序列出记录；过滤参数为 None 即不过滤。"""
    clauses: list[str] = []
    params: list[Any] = []
    if action is not None:
        clauses.append("action = ?")
        params.append(action)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if since is not None:
        clauses.append("created_at >= ?")
        params.append(since)

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    params.extend([limit, offset])
    with contextlib.closing(init_db()) as conn:
        rows = conn.execute(
            f"SELECT {', '.join(_ENTRY_COLUMNS)} FROM catalog_entries"
            f"{where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def get_entry(entry_id: int) -> dict[str, Any] | None:
    """按 id 取单条记录，不存在返回 None。"""
    with contextlib.closing(init_db()) as conn:
        row = conn.execute(
            f"SELECT {', '.join(_ENTRY_COLUMNS)} FROM catalog_entries WHERE id = ?",
            (entry_id,),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def stats() -> dict[str, Any]:
    """汇总统计：总数、按 action 分组计数、最近一条的时间戳。"""
    with contextlib.closing(init_db()) as conn:
        total = int(conn.execute("SELECT COUNT(*) FROM catalog_entries").fetchone()[0])
        by_action_rows = conn.execute(
            "SELECT action, COUNT(*) FROM catalog_entries GROUP BY action"
        ).fetchall()
        last_row = conn.execute(
            "SELECT MAX(created_at) FROM catalog_entries"
        ).fetchone()
    by_action = {row[0]: int(row[1]) for row in by_action_rows}
    last_created_at = last_row[0] if last_row is not None else None
    return {
        "total": total,
        "by_action": by_action,
        "last_created_at": int(last_created_at) if last_created_at is not None else None,
    }


def export_json(dest: str | Path) -> int:
    """把全部记录导成 JSON 数组写入 dest，返回导出的行数。"""
    with contextlib.closing(init_db()) as conn:
        rows = conn.execute(
            f"SELECT {', '.join(_ENTRY_COLUMNS)} FROM catalog_entries ORDER BY id ASC"
        ).fetchall()
    entries = [_row_to_dict(row) for row in rows]
    Path(dest).write_text(
        json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return len(entries)


def _now_ms() -> int:
    """当前 Unix 时间戳（毫秒）。"""
    return int(time.time() * 1000)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    """把查询行转成普通 dict。"""
    return {key: row[key] for key in row.keys()}
