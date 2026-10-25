"""catalog Web API 端点回归测试。

复用 conftest 的 ``auth_client``（已登录 TestClient），库文件通过
``WATERMARK_CATALOG_DB`` 指向 tmp_path。重点回归 fix-1：
过滤条件下 ``total`` 必须等于过滤后的条数，而不是全库总数。
"""

from __future__ import annotations

import time

import pytest

from watermark import catalog


@pytest.fixture()
def catalog_env(tmp_path, monkeypatch):
    """库文件指到临时目录，保证与真实库和其他用例隔离。"""
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
    # 保证相邻两条 created_at（毫秒）不同，使列表/过滤断言确定性成立
    time.sleep(0.005)
    return entry_id


# ---------------------------------------------------------------- 鉴权


def test_catalog_requires_login(catalog_env):
    from fastapi.testclient import TestClient

    from watermark.api import create_app

    # 全新 client，不带登录 cookie
    client = TestClient(create_app())
    resp = client.get("/api/catalog")
    assert resp.status_code == 401


# ---------------------------------------------------------------- 列表


def test_catalog_list_total_matches_items(auth_client, catalog_env):
    ids = [_add(source_path=f"/tmp/{i}.png") for i in range(3)]

    resp = auth_client.get("/api/catalog")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(data["items"]) == 3
    assert sorted(item["id"] for item in data["items"]) == sorted(ids)


def test_catalog_filter_action_total_is_filtered_count(auth_client, catalog_env):
    """回归 fix-1：过滤后 total 必须等于过滤条数，不得返回全库总数。"""
    _add(action="add")
    _add(action="remove")
    _add(action="remove")

    resp = auth_client.get("/api/catalog", params={"action": "remove"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(data["items"]) == 2
    assert all(item["action"] == "remove" for item in data["items"])


def test_catalog_filter_status_total_is_filtered_count(auth_client, catalog_env):
    _add(status="ok")
    _add(status="failed", error="boom")
    _add(status="failed", error="bang")

    resp = auth_client.get("/api/catalog", params={"status": "failed"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(data["items"]) == 2
    assert all(item["status"] == "failed" for item in data["items"])


def test_catalog_filter_combined(auth_client, catalog_env):
    _add(action="add", status="ok")
    _add(action="add", status="failed", error="x")
    _add(action="remove", status="ok")

    resp = auth_client.get(
        "/api/catalog", params={"action": "add", "status": "failed"}
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(data["items"]) == 1
    assert data["items"][0]["action"] == "add"
    assert data["items"][0]["status"] == "failed"


# ---------------------------------------------------------------- 分页校验


@pytest.mark.parametrize("params", [{"limit": "0"}, {"limit": "999"}, {"offset": "-1"}])
def test_catalog_rejects_invalid_paging(auth_client, catalog_env, params):
    resp = auth_client.get("/api/catalog", params=params)
    assert resp.status_code == 400


# ---------------------------------------------------------------- 单条


def test_catalog_entry_not_found(auth_client, catalog_env):
    resp = auth_client.get("/api/catalog/99999")
    assert resp.status_code == 404
    assert resp.json() == {"error": "not found"}


def test_catalog_entry_invalid_id(auth_client, catalog_env):
    resp = auth_client.get("/api/catalog/abc")
    assert resp.status_code == 400


def test_catalog_entry_hit(auth_client, catalog_env):
    entry_id = _add(output_path="/tmp/out.png")
    resp = auth_client.get(f"/api/catalog/{entry_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == entry_id
