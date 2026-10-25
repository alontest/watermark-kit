"""watermark.catalog 存储层公开契约测试。

用例全部通过 ``WATERMARK_CATALOG_DB`` 指向 tmp_path，不触碰真实库文件。
"""

from __future__ import annotations

import json
import sqlite3
import time

import pytest

from watermark import catalog

# ---------------------------------------------------------------- 夹具


@pytest.fixture()
def catalog_env(tmp_path, monkeypatch):
    """把库文件指到临时目录，保证用例之间、用例与真实库完全隔离。"""
    db_path = tmp_path / "catalog" / "catalog.db"
    monkeypatch.setenv("WATERMARK_CATALOG_DB", str(db_path))
    return db_path


def _add(**overrides) -> int:
    """写入一条记录并返回 id；默认 add/ok，可按需覆盖字段。"""
    params = dict(
        source_sha256="a" * 64,
        source_path="/tmp/src.png",
        action="add",
        spec_json='{"kind":"text","text":"hi"}',
        status="ok",
    )
    params.update(overrides)
    entry_id = catalog.record(**params)
    # 保证相邻两条 created_at（毫秒）不同，使排序/过滤断言确定性成立
    time.sleep(0.005)
    return entry_id


# ---------------------------------------------------------------- init_db


def test_init_db_idempotent_and_schema_version(catalog_env):
    conn1 = catalog.init_db()
    conn1.close()
    conn2 = catalog.init_db()  # 第二次调用不应报错
    try:
        row = conn2.execute(
            "SELECT value FROM catalog_meta WHERE key = 'schema_version'"
        ).fetchone()
    finally:
        conn2.close()
    assert row["value"] == "1"


def test_catalog_path_env_override(catalog_env):
    assert catalog.catalog_path() == catalog_env


# ---------------------------------------------------------------- record / list


def test_record_and_list_desc_with_full_fields(catalog_env):
    _add(source_path="/tmp/one.png")
    _add(source_path="/tmp/two.png", action="remove")
    _add(source_path="/tmp/three.png")

    entries = catalog.list_entries()

    assert len(entries) == 3
    # created_at 倒序
    created = [e["created_at"] for e in entries]
    assert created == sorted(created, reverse=True)
    # 插入顺序与倒序对应
    assert [e["source_path"] for e in entries] == [
        "/tmp/three.png",
        "/tmp/two.png",
        "/tmp/one.png",
    ]
    # 字段完整
    expected_keys = {
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
    }
    assert all(expected_keys <= set(e) for e in entries)
    assert all(isinstance(e["id"], int) for e in entries)


# ---------------------------------------------------------------- 过滤


def test_filter_by_action(catalog_env):
    _add(action="add")
    _add(action="remove")
    _add(action="add")

    entries = catalog.list_entries(action="remove")
    assert len(entries) == 1
    assert entries[0]["action"] == "remove"


def test_filter_by_status(catalog_env):
    _add(status="ok")
    _add(status="failed", error="boom")
    _add(status="ok")

    entries = catalog.list_entries(status="failed")
    assert len(entries) == 1
    assert entries[0]["status"] == "failed"
    assert entries[0]["error"] == "boom"


def test_filter_by_since(catalog_env):
    first_id = _add()
    first_created = catalog.get_entry(first_id)["created_at"]
    _add()

    entries = catalog.list_entries(since=first_created + 1)
    assert len(entries) == 1  # 只剩后写入的那条
    entries = catalog.list_entries(since=first_created)
    assert len(entries) == 2  # >= 语义，边界也计入


def test_filter_combined(catalog_env):
    _add(action="add", status="ok")
    _add(action="add", status="failed", error="x")
    _add(action="remove", status="ok")

    entries = catalog.list_entries(action="add", status="failed")
    assert len(entries) == 1
    assert entries[0]["action"] == "add"
    assert entries[0]["status"] == "failed"


# ---------------------------------------------------------------- 分页


def test_pagination_limit_and_offset(catalog_env):
    for i in range(3):
        _add(source_path=f"/tmp/{i}.png")

    assert [e["source_path"] for e in catalog.list_entries(limit=1)] == ["/tmp/2.png"]

    page2 = catalog.list_entries(limit=1, offset=1)
    assert [e["source_path"] for e in page2] == ["/tmp/1.png"]

    # offset 超出总数返回空列表
    assert catalog.list_entries(offset=10) == []
    assert catalog.list_entries(limit=1, offset=3) == []


# ---------------------------------------------------------------- get_entry


def test_get_entry_hit_and_miss(catalog_env):
    entry_id = _add(output_path="/tmp/out.png", duration_ms=12)

    entry = catalog.get_entry(entry_id)
    assert entry is not None
    assert entry["id"] == entry_id
    assert entry["output_path"] == "/tmp/out.png"
    assert entry["duration_ms"] == 12
    assert entry["status"] == "ok"

    assert catalog.get_entry(999999) is None


# ---------------------------------------------------------------- stats


def test_stats_empty_db(catalog_env):
    s = catalog.stats()
    assert s["total"] == 0
    assert s["by_action"] == {}
    assert s["last_created_at"] is None


def test_stats_totals_and_by_action(catalog_env):
    _add(action="add")
    _add(action="add")
    _add(action="remove", status="failed", error="x")

    s = catalog.stats()
    assert s["total"] == 3
    assert s["by_action"] == {"add": 2, "remove": 1}
    assert isinstance(s["last_created_at"], int)


# ---------------------------------------------------------------- export_json


def test_export_json_writes_file_and_returns_count(catalog_env, tmp_path):
    _add(source_path="/tmp/中文 '引号'.png")
    _add(action="remove")

    # 注：export_json 不创建父目录，dest 需指向已存在的目录
    dest = tmp_path / "catalog.json"
    count = catalog.export_json(dest)

    assert count == 2
    data = json.loads(dest.read_text(encoding="utf-8"))
    assert len(data) == count
    assert data[0]["source_path"] == "/tmp/中文 '引号'.png"


def test_export_json_empty_db(catalog_env, tmp_path):
    dest = tmp_path / "empty.json"
    assert catalog.export_json(dest) == 0
    assert json.loads(dest.read_text(encoding="utf-8")) == []


# ---------------------------------------------------------------- 约束


def test_record_rejects_bogus_action(catalog_env):
    with pytest.raises(sqlite3.IntegrityError):
        _add(action="bogus")


def test_record_rejects_bogus_status(catalog_env):
    with pytest.raises(sqlite3.IntegrityError):
        _add(status="bogus")


# ---------------------------------------------------------------- 特殊字符 source_path


@pytest.mark.parametrize(
    "source_path",
    [
        "/tmp/it's a 'quoted' file.png",
        "/tmp/中文 目录/图片.png",
        "/tmp/混合 '单引号' 与 中文.png",
    ],
)
def test_record_special_chars_roundtrip(catalog_env, source_path):
    entry_id = _add(source_path=source_path)
    entry = catalog.get_entry(entry_id)
    assert entry["source_path"] == source_path
    # 列表读取同样无损
    assert catalog.list_entries(limit=10)[0]["source_path"] == source_path
