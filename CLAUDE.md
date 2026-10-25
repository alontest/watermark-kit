# test-A

与全局及 `~/.claude/rules/vault/` 规范冲突时，以本文件为准；本文件没写的，遵守全局与 Vault 标准。

## 定位

水印 CLI + FastAPI 后端（`watermark/` 包：`catalog.py`、`cli.py`、`api.py`；`tests/` 单测）。

## 验证命令

- 全部测试：`.venv/bin/python -m pytest tests/ -q`（基线 229 passed / 1 skipped / 1 xfailed，2026-10-25 实测）
- CLI 冒烟：`.venv/bin/python -m watermark.cli catalog list`
- 注意：系统 python3 无依赖，必须走 `.venv/bin/python`

## 本项目铁律

- catalog 接口的 `total` 必须随过滤条件计数，不返回全表总数
- CLI/API 对非法参数必须非零退出，不允许静默降级
- 只在 `.venv` 就绪后运行测试；不要把依赖装进系统 python

## 禁区

- 不要动 `docs/`（探针卡工作区）
- 不要引入新依赖
