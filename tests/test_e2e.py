"""端到端验收测试：真实子进程 CLI + 真实 HTTP 请求。

全链路钉死「加水印 -> 去水印」可逆性保证，不改产品代码：
- CLI: numpy 合成图 -> add(自定义色/平铺/旋转) -> inspect -> remove -> compare
- API: add -> spec 回传 -> remove -> download，与原图比阈值
- 验收线与 tests/test_roundtrip.py 口径一致：PSNR >= 45dB 且最大差 <= 1/255

全部使用 tmp_path，测试互相独立、可并行。
"""

from __future__ import annotations

import io
import json
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from watermark.api import create_app
from watermark.blend import diff_stats, psnr

# 验收阈值（与 test_roundtrip.py 一致）
MIN_PSNR_DB = 45.0
MAX_PIXEL_DIFF = 1.0  # 即 1/255 * 255
# 合成测试图尺寸
IMG_W, IMG_H = 200, 150


# ---------------------------------------------------------------- 工具

def _run_cli(*args: str) -> tuple[int, dict | None, str]:
    """跑真实子进程 CLI（python -m watermark.cli），返回 (退出码, JSON, stdout)。"""
    proc = subprocess.run(
        [sys.executable, "-m", "watermark.cli", *args],
        capture_output=True, text=True, timeout=120,
    )
    return proc.returncode, None, proc.stdout + proc.stderr


def _run_cli_json(*args: str) -> tuple[int, dict]:
    """跑子进程 CLI 并解析 --json stdout。

    已知产品侧问题：`batch --json` 内部逐文件调 _run_add(json=False)，
    stdout 会先混入「已加水印: …」行再输出 JSON，这里从首个 '{' 起解析。
    """
    proc = subprocess.run(
        [sys.executable, "-m", "watermark.cli", *args],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, f"CLI 失败({proc.returncode}): {proc.stderr}\n{proc.stdout}"
    stdout = proc.stdout
    if "{" in stdout and not stdout.lstrip().startswith("{"):
        stdout = stdout[stdout.index("{"):]
    return proc.returncode, json.loads(stdout)


def _make_synth_png(path, seed: int = 2026) -> None:
    """numpy 合成测试图：渐变底 + 彩色圆 + 噪声，写入 path。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W]
    base = np.stack([
        xx / (IMG_W - 1) * 255.0,
        yy / (IMG_H - 1) * 255.0,
        np.full((IMG_H, IMG_W), 128.0),
    ], axis=-1)
    circle = (yy - IMG_H // 2) ** 2 + (xx - IMG_W // 3) ** 2 <= (IMG_H // 4) ** 2
    base[circle] = [220.0, 80.0, 60.0]
    noise = rng.normal(0, 4, base.shape)
    img = Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")
    img.save(path)


def _load_rgb(path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float32)


def _assert_reversible(original_arr: np.ndarray, restored_path) -> float:
    """断言还原图对原图过验收线，返回实测 PSNR。"""
    restored_arr = _load_rgb(restored_path)
    assert restored_arr.shape == original_arr.shape
    value = psnr(original_arr, restored_arr)
    stats = diff_stats(original_arr, restored_arr)
    assert value >= MIN_PSNR_DB, f"PSNR={value:.2f}dB < {MIN_PSNR_DB}"
    assert stats["max"] <= MAX_PIXEL_DIFF, f"最大差 {stats['max']} > {MAX_PIXEL_DIFF}/255"
    return value


def _png_upload(path, name: str = "img.png") -> tuple:
    return (name, path.read_bytes(), "image/png")


# ---------------------------------------------------------------- CLI 全链路

def test_cli_full_roundtrip(tmp_path):
    """add(自定义色+平铺+旋转) -> inspect 含配方 -> remove -> compare 过阈值。"""
    src = tmp_path / "photo.png"
    _make_synth_png(src)
    original_arr = _load_rgb(src)
    marked = tmp_path / "marked.png"
    restored = tmp_path / "restored.png"

    # 1. add：自定义颜色 + 平铺 + 旋转
    add_args = (
        "add", str(src), "-o", str(marked),
        "--text", "E2E 2026", "--color", "#ff8800",
        "--opacity", "0.3", "--size", "24",
        "--tile", "--rotate", "15", "--position", "center",
        "--json",
    )
    code, add_report = _run_cli_json(*add_args)
    assert code == 0
    assert add_report["layers"] == 1
    assert add_report["mask_sha256"]
    assert marked.is_file()

    # 2. inspect：读回配方
    code, inspect = _run_cli_json("inspect", str(marked), "--json")
    assert code == 0
    assert inspect["spec_source"] == "元数据"
    assert inspect["width"] == IMG_W and inspect["height"] == IMG_H
    assert inspect["layers"][0]["kind"] == "text"
    assert inspect["layers"][0]["content"] == "E2E 2026"
    assert inspect["layers"][0]["tile"] is True
    assert inspect["layers"][0]["rotate"] == 15.0
    assert inspect["mask_sha256"] == add_report["mask_sha256"], "inspect 指纹与 add 不一致"

    # 3. remove：从元数据拿配方去水印
    code, remove_report = _run_cli_json(
        "remove", str(marked), "-o", str(restored), "--json",
    )
    assert code == 0
    assert remove_report["mask_match"] is True
    assert restored.is_file()

    # 4. compare：CLI 报告 + 本地复算双重验证
    code, cmp = _run_cli_json("compare", str(src), str(restored), "--json")
    assert code == 0
    assert cmp["psnr_db"] is not None
    assert cmp["psnr_db"] >= MIN_PSNR_DB, f"compare PSNR {cmp['psnr_db']} < {MIN_PSNR_DB}"
    assert cmp["max_diff"] <= MAX_PIXEL_DIFF, f"compare 最大差 {cmp['max_diff']} > 1/255"
    value = _assert_reversible(original_arr, restored)
    print(f"CLI 全链路实测 PSNR: {value:.2f} dB (compare 报告 {cmp['psnr_db']} dB)")


def test_cli_remove_without_spec_fails(tmp_path):
    """没有配方元数据/sidecar 的图 remove 必须失败（退出码 2）。"""
    src = tmp_path / "plain.png"
    _make_synth_png(src, seed=99)
    code, _, output = _run_cli("remove", str(src), "-o", str(tmp_path / "r.png"))
    assert code == 2, f"无配方 remove 应退出码 2，实际 {code}；输出: {output}"
    assert "配方" in output


# ---------------------------------------------------------------- API 全链路

@pytest.fixture()
def api_client():
    return TestClient(create_app())


def test_api_full_roundtrip(api_client, tmp_path):
    """add(自定义色) -> spec 原样回传 remove -> download 拉回成品过阈值。"""
    src = tmp_path / "api.png"
    _make_synth_png(src, seed=7)
    original_arr = _load_rgb(src)

    # 1. add：自定义颜色
    resp = api_client.post(
        "/api/add",
        files={"image": _png_upload(src)},
        data={"text": "API 往返", "color": "#00cc44", "opacity": "0.3",
              "size": "24", "position": "bottom-right"},
    )
    assert resp.status_code == 200, resp.text
    add_data = resp.json()
    assert add_data["ok"] is True
    assert add_data["spec"]["layers"][0]["color"] == [0, 204, 68], "自定义颜色应写入 spec"

    # 2. download 拉回加水印成品（PNG 无元数据），spec 原样回传 remove
    marked_resp = api_client.get(f"/api/download/{add_data['token']}")
    assert marked_resp.status_code == 200
    assert marked_resp.headers["content-type"] == "image/png"
    spec_text = api_client.get(
        f"/api/download/{add_data['token']}", params={"as_json": 1}
    ).text
    marked_file = tmp_path / "api_marked.png"
    marked_file.write_bytes(marked_resp.content)

    resp = api_client.post(
        "/api/remove",
        files={"image": _png_upload(marked_file, "wm.png")},
        data={"spec": spec_text},
    )
    assert resp.status_code == 200, resp.text
    remove_data = resp.json()
    assert remove_data["report"]["mask_match"] is True

    # 3. download 拉回还原图，与原图比阈值
    restored_resp = api_client.get(f"/api/download/{remove_data['token']}")
    assert restored_resp.status_code == 200
    restored_file = tmp_path / "api_restored.png"
    restored_file.write_bytes(restored_resp.content)
    value = _assert_reversible(original_arr, restored_file)
    print(f"API 全链路实测 PSNR: {value:.2f} dB")


def test_api_remove_without_spec_returns_4xx(api_client, tmp_path):
    """不带配方且图里也没有元数据的 remove 必须返回 4xx。"""
    src = tmp_path / "plain.png"
    _make_synth_png(src, seed=42)
    resp = api_client.post(
        "/api/remove", files={"image": _png_upload(src, "plain.png")},
    )
    assert 400 <= resp.status_code < 500
    assert "配方" in resp.json()["detail"]


# ---------------------------------------------------------------- 颜色生效回归

def test_color_takes_effect_and_roundtrips(tmp_path):
    """同参数不同颜色产出不同图；两种颜色均能过可逆验收线。"""
    colors = ("#ff0000", "#00ff00")
    marked_paths = []
    for i, color in enumerate(colors):
        src = tmp_path / f"src_{i}.png"
        _make_synth_png(src, seed=2026)  # 同一张图
        marked = tmp_path / f"marked_{i}.png"
        restored = tmp_path / f"restored_{i}.png"
        _run_cli_json(
            "add", str(src), "-o", str(marked),
            "--text", "颜色", "--color", color, "--opacity", "0.4",
            "--size", "24", "--json",
        )
        marked_paths.append(marked)
        _run_cli_json("remove", str(marked), "-o", str(restored), "--json")
        _assert_reversible(_load_rgb(src), restored)

    a, b = (_load_rgb(p) for p in marked_paths)
    stats = diff_stats(a, b)
    assert stats["max"] > 0, "不同颜色的加水印结果应不同（颜色链路生效）"
    print(f"颜色回归：两色加水印最大差 {stats['max']:.1f}/255")


def test_jpeg_lossy_roundtrip_reports_amplified_error(tmp_path):
    """JPEG 有损路径：往返误差被放大——PSNR 报告机制工作、明确低于 PNG 路径。"""
    src = tmp_path / "src.png"
    _make_synth_png(src, seed=11)
    original_arr = _load_rgb(src)

    psnrs = {}
    max_diffs = {}
    for fmt in ("png", "jpg"):
        marked = tmp_path / f"marked.{fmt}"
        restored = tmp_path / f"restored.{fmt}"
        _run_cli_json(
            "add", str(src), "-o", str(marked),
            "--text", "lossy", "--color", "#5a5a5a",
            "--opacity", "0.3", "--size", "24", "--json",
        )
        _run_cli_json("remove", str(marked), "-o", str(restored), "--json")
        code, cmp = _run_cli_json("compare", str(src), str(restored), "--json")
        assert code == 0
        assert cmp["psnr_db"] is not None and cmp["psnr_db"] > 0, \
            f"{fmt} 路径 PSNR 报告应给出有限值"
        psnrs[fmt] = cmp["psnr_db"]
        max_diffs[fmt] = cmp["max_diff"]

    assert max_diffs["jpg"] > MAX_PIXEL_DIFF, \
        "JPEG 量化误差经反解放大，最大差应超过 1/255"
    assert psnrs["jpg"] < psnrs["png"], \
        f"JPEG 往返 PSNR ({psnrs['jpg']}) 应低于 PNG ({psnrs['png']})"
    print(f"JPEG 有损放大：PNG {psnrs['png']:.2f} dB vs JPEG {psnrs['jpg']:.2f} dB")


# ---------------------------------------------------------------- batch 冒烟

def test_batch_add_smoke(tmp_path):
    """3 张图批量 add，输出目录 3 个成品。"""
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    for i in range(3):
        _make_synth_png(in_dir / f"img_{i}.png", seed=100 + i)
    out_dir = tmp_path / "out"

    code, payload = _run_cli_json(
        "batch", str(in_dir), "-o", str(out_dir), "--mode", "add", "--json",
    )
    assert code == 0
    assert payload["total"] == 3
    assert len(payload["ok"]) == 3, f"失败项: {payload['failed']}"
    assert payload["failed"] == []
    outputs = sorted(p for p in out_dir.iterdir() if p.suffix == ".png")
    assert len(outputs) == 3, f"输出目录应有 3 个成品，实际 {outputs}"
    # 成品应带配方元数据，可被 remove 识别
    for out in outputs:
        code, inspect = _run_cli_json("inspect", str(out), "--json")
        assert code == 0
        assert inspect["layers"][0]["content"] == "BATCH"
