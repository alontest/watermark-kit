"""CLI 占位入口：子命令由后续卡实现。"""

from __future__ import annotations


def main() -> int:
    """watermark 命令入口，核心引擎卡只保留占位。"""
    print("watermark-kit：CLI 尚未实现，请使用 Python API。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
