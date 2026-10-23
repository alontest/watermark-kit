"""CLI 测试：通过 main(argv) 驱动各子命令。"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from watermark.cli import main
from watermark.meta import read_spec, sidecar_path_for


@pytest.fixture()
def src_png(tmp_path):
    """随机 RGB 测试图文件。"""
    rng = np.random.default_rng(2026)
    img = Image.fromarray(
        rng.integers(0, 256, (120, 160, 3), dtype=np.uint8), "RGB"
    )
    path = tmp_path / "photo.png"
    img.save(path)
    return path


def _read_json(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


# ---------------------------------------------------------------- add

def test_add_text_default_output(src_png):
    assert main(["add", str(src_png), "--text", "机密"]) == 0
    out = src_png.with_name("photo.wm.png")
    assert out.is_file()
    from watermark.meta import read_spec
    assert read_spec(out) is not None, "PNG 元数据应带 spec"


def test_add_text_output_and_no_sidecar(src_png):
    out = src_png.with_name("marked.png")
    assert main(["add", str(src_png), "--text", "hi", "-o", str(out)]) == 0
    assert out.is_file()
    assert not sidecar_path_for(out).exists()


def test_add_json_report(src_png, capsys):
    assert main(["add", str(src_png), "--text", "hi", "--json"]) == 0
    payload = _read_json(capsys)
    assert payload["layers"] == 1
    assert payload["output"].endswith("photo.wm.png")
    assert "mask_sha256" in payload


def test_add_logo(src_png, tmp_path):
    logo = tmp_path / "logo.png"
    rng = np.random.default_rng(7)
    alpha = np.full((12, 24, 1), 200, dtype=np.uint8)
    rgb = rng.integers(0, 256, (12, 24, 3), dtype=np.uint8)
    Image.fromarray(
        np.concatenate([rgb, alpha], axis=-1), "RGBA"
    ).save(logo)
    assert main(["add", str(src_png), "--logo", str(logo), "-o",
                 str(tmp_path / "out.png")]) == 0
    assert (tmp_path / "out.png").is_file()


def test_add_text_and_logo_mutually_exclusive(src_png):
    with pytest.raises(SystemExit) as exc:
        main(["add", str(src_png), "--text", "a", "--logo", "b"])
    assert exc.value.code == 2


def test_add_missing_input(tmp_path):
    assert main(["add", str(tmp_path / "nope.png"), "--text", "a"]) == 2


def test_add_bad_color(src_png):
    assert main(["add", str(src_png), "--text", "a", "--color", "1,2"]) == 2


def test_add_bad_position(src_png):
    assert main(["add", str(src_png), "--text", "a", "--position", "nope"]) == 2


def test_add_bad_opacity(src_png):
    assert main(["add", str(src_png), "--text", "a", "--opacity", "3"]) == 2


# ---------------------------------------------------------------- remove

def test_add_inspect_remove_compare_roundtrip(src_png, tmp_path, capsys):
    """验收 1：全链路 PSNR >= 45dB。"""
    out = tmp_path / "marked.png"
    restored = tmp_path / "restored.png"
    assert main(["add", str(src_png), "--text", "机密文件",
                 "-o", str(out)]) == 0
    assert main(["inspect", str(out)]) == 0
    assert "元数据" in capsys.readouterr().out
    assert main(["remove", str(out), "-o", str(restored)]) == 0
    capsys.readouterr()  # 丢弃 remove 的人类可读输出
    assert main(["compare", str(src_png), str(restored), "--json"]) == 0
    payload = _read_json(capsys)
    assert payload["psnr_db"] >= 45.0


def test_remove_from_sidecar(src_png, tmp_path, capsys):
    """无元数据图 + sidecar 也能还原。"""
    marked = tmp_path / "marked.png"
    assert main(["add", str(src_png), "--text", "sc", "-o", str(marked),
                 "--sidecar"]) == 0
    assert sidecar_path_for(marked).is_file()
    bare = tmp_path / "bare.png"
    Image.open(marked).save(bare)  # 去掉元数据，只留 sidecar
    import shutil
    shutil.copy(sidecar_path_for(marked), sidecar_path_for(bare))
    restored = tmp_path / "restored.png"
    assert main(["remove", str(bare), "-o", str(restored)]) == 0
    capsys.readouterr()  # 丢弃 remove 的人类可读输出
    assert main(["compare", str(src_png), str(restored), "--json"]) == 0
    assert _read_json(capsys)["psnr_db"] >= 45.0


def test_remove_without_recipe_errors_with_hint(src_png, tmp_path, capsys):
    """验收 3：元数据被清掉时 remove 报错并提示 --spec。"""
    marked = tmp_path / "marked.png"
    assert main(["add", str(src_png), "--text", "sc", "-o", str(marked)]) == 0
    bare = tmp_path / "bare.png"
    Image.open(marked).save(bare)  # 去掉元数据，也不留 sidecar
    assert main(["remove", str(bare)]) == 2
    err = capsys.readouterr().err
    assert "--spec" in err


def test_remove_with_explicit_spec(src_png, tmp_path):
    marked = tmp_path / "marked.png"
    assert main(["add", str(src_png), "--text", "sc", "-o", str(marked),
                 "--sidecar"]) == 0
    bare = tmp_path / "bare.png"
    Image.open(marked).save(bare)
    spec_copy = tmp_path / "recipe.wm.json"
    import shutil
    shutil.copy(sidecar_path_for(marked), spec_copy)
    restored = tmp_path / "restored.png"
    assert main(["remove", str(bare), "--spec", str(spec_copy),
                 "-o", str(restored)]) == 0
    assert restored.is_file()


def test_remove_save_spec(src_png, tmp_path):
    from watermark.spec import load_spec
    marked = tmp_path / "marked.png"
    assert main(["add", str(src_png), "--text", "sc", "-o", str(marked)]) == 0
    saved = tmp_path / "saved.wm.json"
    restored = tmp_path / "restored.png"
    assert main(["remove", str(marked), "-o", str(restored),
                 "--save-spec", str(saved)]) == 0
    assert saved.is_file()
    assert load_spec(saved) is not None


# ---------------------------------------------------------------- compare

def test_compare_identical_images(src_png, capsys):
    assert main(["compare", str(src_png), str(src_png), "--json"]) == 0
    payload = _read_json(capsys)
    assert payload["psnr_inf"] is True


def test_compare_size_mismatch(src_png, tmp_path):
    other = tmp_path / "other.png"
    Image.new("RGB", (32, 32), (0, 0, 0)).save(other)
    assert main(["compare", str(src_png), str(other)]) == 2


# ---------------------------------------------------------------- batch

def test_batch_add_and_remove(src_png, tmp_path):
    in_dir = src_png.parent
    src_png2 = in_dir / "photo2.jpg"
    Image.open(src_png).convert("RGB").save(src_png2, format="JPEG")
    out_add = tmp_path / "added"
    assert main(["batch", str(in_dir), "-o", str(out_add),
                 "--mode", "add"]) == 0
    added = sorted(p.name for p in out_add.iterdir() if p.suffix == ".png")
    assert "photo.wm.png" in added
    out_rm = tmp_path / "removed"
    assert main(["batch", str(out_add), "-o", str(out_rm),
                 "--mode", "remove"]) == 0
    assert (out_rm / "photo.wm.restored.png").is_file()


def test_batch_continues_on_failure(src_png, tmp_path):
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    Image.open(src_png).save(in_dir / "good.png")
    (in_dir / "broken.png").write_bytes(b"not-an-image")
    out_dir = tmp_path / "out"
    assert main(["batch", str(in_dir), "-o", str(out_dir),
                 "--mode", "add", "--json"]) == 1
    # 单张失败不中断：good.png 仍成功
    assert (out_dir / "good.wm.png").is_file()


def test_batch_pattern(src_png, tmp_path):
    in_dir = src_png.parent
    photo2 = in_dir / "photo2.jpg"
    Image.open(src_png).convert("RGB").save(photo2, format="JPEG")
    out = tmp_path / "out"
    assert main(["batch", str(in_dir), "-o", str(out), "--mode", "add",
                 "--pattern", "photo2.*"]) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == ["photo2.wm.jpg"]


def test_batch_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["batch", str(empty), "-o", str(tmp_path / "out"),
                 "--mode", "add"]) == 2


# ---------------------------------------------------------------- demo

def test_demo_creates_files_and_passes(tmp_path, capsys):
    """验收 2：demo 生成三张图且结论为 ✅。"""
    out_dir = tmp_path / "demo"
    assert main(["demo", "--out", str(out_dir)]) == 0
    assert (out_dir / "original.png").is_file()
    assert (out_dir / "watermarked.png").is_file()
    assert (out_dir / "restored.png").is_file()
    text = capsys.readouterr().out
    assert "✅" in text
    assert main(["demo", "--out", str(tmp_path / "demo2"), "--json"]) == 0
    assert _read_json(capsys)["conclusion"] == "✅ 往返无损"


# ---------------------------------------------------------------- serve

def test_serve_friendly_error_when_api_missing(capsys):
    try:
        import watermark.api  # noqa: F401
        pytest.skip("watermark.api 已存在，跳过占位测试")
    except ImportError:
        pass
    assert main(["serve"]) == 2
    assert "watermark.api" in capsys.readouterr().err


# ---------------------------------------------------------------- 杂项

def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_no_command_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
