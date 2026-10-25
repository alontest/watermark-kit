"""catalog CLI 子命令回归测试。

直接进程内调用 ``watermark.cli.main([...])``（不经 subprocess）：
``main`` 的返回值就是退出码，``monkeypatch.setenv`` 对同进程立即生效，
比 subprocess 启动解释器快且无路径依赖。库文件通过 ``WATERMARK_CATALOG_DB``
指向 tmp_path，不触碰真实库。
"""

from __future__ import annotations

import json
import time

import pytest

from watermark import catalog, cli

# ---------------------------------------------------------------- 夹具


@pytest.fixture()
def catalog_env(tmp_path, monkeypatch):
    """库文件指到临时目录，保证与真实库和其他用例隔离。"""
    db_path = tmp_path / "catalog" / "catalog.db"
    monkeypatch.setenv("WATERMARK_CATALOG_DB", str(db_path))
    return db_path


def _run(*args: str) -> int:
    """执行 catalog 子命令，返回退出码。"""
    return cli.main(["catalog", *args])


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
    # 保证相邻两条 created_at（毫秒）不同，使倒序断言确定性成立
    time.sleep(0.005)
    return entry_id


# ---------------------------------------------------------------- list


def test_list_empty_db_prints_hint(catalog_env, capsys):
    code = _run("list")

    assert code == 0
    assert "档案库为空" in capsys.readouterr().out


def test_list_table_contains_expected_ids_in_desc_order(catalog_env, capsys):
    ids = [_add(source_path=f"/tmp/{i}.png") for i in range(3)]

    code = _run("list")

    assert code == 0
    out = capsys.readouterr().out
    # 表头存在
    assert "ID" in out
    # 各 id 都出现在表格里，且行顺序为 created_at 倒序（最后插入的最先）
    data_lines = [line for line in out.splitlines() if line.strip() and not line.startswith(("ID", "--"))]
    assert len(data_lines) == 3
    assert [str(entry_id) for entry_id in reversed(ids)] == [
        line.split()[0] for line in data_lines
    ]


def test_list_json_flag_outputs_json(catalog_env, capsys):
    first_id = _add()
    _add()

    code = _run("list", "--json")

    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert [row["id"] for row in rows] == [first_id + 1, first_id]


# ---------------------------------------------------------------- stats


def test_stats_json_reports_total(catalog_env, capsys):
    _add()
    _add(action="remove")

    code = _run("stats", "--json")

    assert code == 0
    data = json.loads(capsys.readouterr().out)
    assert data["total"] == 2
    assert data["by_action"] == {"add": 1, "remove": 1}


# ---------------------------------------------------------------- show


def test_show_existing_entry_exits_zero(catalog_env, capsys):
    entry_id = _add(output_path="/tmp/out.png", duration_ms=12)

    code = _run("show", str(entry_id))

    assert code == 0
    out = capsys.readouterr().out
    assert f"id: {entry_id}" in out
    assert "/tmp/out.png" in out


def test_show_missing_entry_exits_two(catalog_env, capsys):
    code = _run("show", "99999")

    assert code == 2
    assert "错误" in capsys.readouterr().err


# ---------------------------------------------------------------- 分页校验（依赖 fix-2）


def test_list_rejects_negative_limit(catalog_env, capsys):
    code = _run("list", "--limit", "-5")

    assert code == 2
    assert "--limit" in capsys.readouterr().err


def test_list_rejects_negative_offset(catalog_env, capsys):
    code = _run("list", "--offset", "-1")

    assert code == 2
    assert "--offset" in capsys.readouterr().err


# ---------------------------------------------------------------- prune


def test_prune_rejects_non_positive_days(catalog_env, capsys):
    code = _run("prune", "--days", "0")

    assert code == 2
    assert "--days" in capsys.readouterr().err


def test_prune_without_yes_is_dry_run(catalog_env):
    _add()
    _add(action="remove")
    before = catalog.stats()

    code = _run("prune", "--days", "30")

    assert code == 0
    after = catalog.stats()
    assert after == before, "不带 --yes 的 prune 只允许统计，不得删数据"
