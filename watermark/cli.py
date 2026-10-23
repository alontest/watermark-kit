"""命令行入口：watermark add/remove/inspect/compare/batch/demo/serve。

argparse 实现，`main(argv=None) -> int` 供 tests 和 console script 调用；
所有报告类子命令支持 --json 输出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from watermark import __version__, blend, meta, pipeline
from watermark.pipeline import report_to_json
from watermark.render import hash_masks, render_spec_masks
from watermark.spec import (
    BLEND_SPACES,
    POSITIONS,
    Layer,
    SpecError,
    WatermarkSpec,
    dump_spec,
    load_spec,
)

# 往返质量合格线（与 tests/test_roundtrip.py 口径一致）
MIN_PSNR_DB = 45.0
# demo 用的合成测试图尺寸
DEMO_WIDTH = 320
DEMO_HEIGHT = 240
# serve 未实现时的占位提示
SERVE_HINT = "watermark.api:create_app 尚未实现（Web 卡未完成），serve 暂不可用"

IMG_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


class CliError(Exception):
    """可预期的用户错误：打印消息、退出码 2，不带 traceback。"""


# ---------------------------------------------------------------- add

def _build_add_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", help="输出路径（默认 <stem>.wm<后缀>）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="水印文字（与 --logo 二选一）")
    group.add_argument("--logo", help="logo 图片路径（与 --text 二选一）")
    parser.add_argument("--font", help="字体路径或名称")
    parser.add_argument("--size", type=int, default=36, help="字号（logo 为最大边，默认 36）")
    parser.add_argument("--opacity", type=float, default=0.25, help="不透明度 0.05..0.95（默认 0.25）")
    parser.add_argument("--color", default="90,90,90", help="水印颜色 R,G,B（默认 90,90,90）")
    parser.add_argument("--position", default="bottom-right", help=f"九宫格锚点（默认 bottom-right，可选 {POSITIONS}）")
    parser.add_argument("--margin", type=float, default=0.03, help="边距比例（默认 0.03）")
    parser.add_argument("--offset", default="0,0", metavar="X,Y", help="像素偏移（默认 0,0）")
    parser.add_argument("--rotate", type=float, default=0.0, help="旋转角度（度）")
    parser.add_argument("--stroke", type=int, default=0, help="描边宽度（px）")
    parser.add_argument("--tile", action="store_true", help="平铺水印（仅 --text）")
    parser.add_argument("--tile-gap", type=int, default=0, help="平铺间隔（px，暂以字号近似参与布局）")
    parser.add_argument("--blend-space", choices=BLEND_SPACES, default="srgb", help="混合空间（默认 srgb）")
    parser.add_argument("--quality", type=int, default=None, help="JPEG 保存质量（1-95）")
    parser.add_argument("--spec-out", help="把 spec 另存到该 .wm.json 路径")
    parser.add_argument("--sidecar", action="store_true", help="额外输出 .wm.json sidecar")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出报告")


def _parse_color(text: str) -> tuple[int, int, int]:
    parts = text.split(",")
    if len(parts) != 3:
        raise CliError(f"--color 需要形如 R,G,B，当前为 {text!r}")
    try:
        values = tuple(int(p.strip()) for p in parts)
    except ValueError:
        raise CliError(f"--color 需要整数 R,G,B，当前为 {text!r}") from None
    if any(not (0 <= v <= 255) for v in values):
        raise CliError(f"--color 分量需在 0..255，当前为 {values}")
    return values  # type: ignore[return-value]


def _parse_offset(text: str) -> tuple[float, float]:
    parts = text.split(",")
    if len(parts) != 2:
        raise CliError(f"--offset 需要形如 x,y，当前为 {text!r}")
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        raise CliError(f"--offset 需要数字 x,y，当前为 {text!r}") from None


def _default_add_output(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + ".wm" + input_path.suffix)


def _run_add(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.is_file():
        raise CliError(f"输入文件不存在: {src}")
    color = _parse_color(args.color)
    offset = _parse_offset(args.offset)
    kind = "text" if args.text is not None else "logo"
    try:
        layer = Layer(
            kind=kind,
            content=args.text if kind == "text" else args.logo,
            opacity=args.opacity,
            position=args.position,
            margin=args.margin,
            offset=offset,
            font=args.font,
            font_size=args.size,
            stroke_width=args.stroke,
            rotate=args.rotate,
            tile=args.tile,
        )
        spec = WatermarkSpec(layers=(layer,), blend_space=args.blend_space)
    except SpecError as exc:
        raise CliError(str(exc)) from None

    out_path = Path(args.output) if args.output else _default_add_output(src)
    save_kwargs = {"quality": args.quality} if args.quality is not None else {}
    report = pipeline.add_watermark(src, spec)
    report["input"] = str(src)
    report["output"] = str(out_path)
    if args.quality is not None:
        report["quality"] = args.quality
    pipeline.save(report["image"], out_path, spec, **save_kwargs)
    if args.spec_out:
        dump_spec(spec, args.spec_out)
    if args.sidecar:
        report["sidecar"] = str(meta.dump_sidecar(spec, out_path))

    if args.json:
        print(report_to_json(report))
    else:
        print(f"已加水印: {src} -> {out_path}")
        print(f"  图层: {report['layers']}  mask指纹: {report['mask_sha256'][:16]}…")
        if args.quality is not None:
            print(f"  quality: {args.quality}")
    return 0


# ---------------------------------------------------------------- remove

def _build_remove_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("-o", "--output", help="输出路径（默认 <stem>.restored<后缀>）")
    parser.add_argument("--spec", help="手动指定配方（.wm.json）")
    parser.add_argument("--allow-scale", action="store_true", help="图片尺寸与 spec 不一致时强行近似还原")
    parser.add_argument("--save-spec", help="把最终使用的 spec 另存到该 .wm.json 路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出报告")


def _load_spec_for_remove(args: argparse.Namespace, src: Path) -> tuple[WatermarkSpec | None, str]:
    """配方来源：--spec > 图片元数据 > sidecar（inspect 亦复用）。"""
    manual = getattr(args, "spec", None)
    if manual:
        return load_spec(manual), "--spec"
    spec = meta.read_spec(src)
    if spec is not None:
        return spec, "元数据"
    return None, "未找到"


def _run_remove(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.is_file():
        raise CliError(f"输入文件不存在: {src}")
    spec, source = _load_spec_for_remove(args, src)
    if spec is None:
        raise CliError(
            f"{src} 未找到水印配方：图片元数据里没有 spec，也没有 "
            f"{meta.sidecar_path_for(src).name} sidecar；"
            "请用 --spec 手动指定配方文件"
        )
    with Image.open(src) as img:
        width, height = img.size
    try:
        render_spec_masks(spec, width, height)
    except SpecError as exc:
        if args.allow_scale:
            # 强行还原：尺寸不一致时按比例缩放字号近似
            spec = _rescale_spec(spec, width, height)
            source += "+allow-scale"
        else:
            raise CliError(
                f"{exc}；如需强行近似还原请加 --allow-scale"
            ) from None

    out_path = Path(args.output) if args.output else (
        src.with_name(src.stem + ".restored" + src.suffix)
    )
    report = pipeline.remove_watermark(src, spec)
    report["input"] = str(src)
    report["output"] = str(out_path)
    report["spec_source"] = source
    if not report["mask_match"] and "mask 指纹不匹配" not in " ".join(report["warnings"]):
        report["warnings"] = list(report["warnings"]) + ["mask 指纹不匹配：spec 与图上水印可能不一致"]
    if report["unrecoverable_ratio"] > 0:
        report["warnings"] = list(report["warnings"]) + [
            f"{report['unrecoverable_ratio']:.2%} 像素不可逆，已按原样保留"
        ]
    pipeline.save(report["image"], out_path, spec)
    if args.save_spec:
        dump_spec(spec, args.save_spec)

    if args.json:
        print(report_to_json(report))
    else:
        print(f"已去水印: {src} -> {out_path}")
        print(f"  配方来源: {source}  图层: {report['layers']}")
        if report["warnings"]:
            for w in report["warnings"]:
                print(f"  ⚠ {w}")
    return 0


# ---------------------------------------------------------------- inspect

def _build_inspect_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="输入图片路径")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")


def _run_inspect(args: argparse.Namespace) -> int:
    src = Path(args.input)
    if not src.is_file():
        raise CliError(f"输入文件不存在: {src}")
    spec, source = _load_spec_for_remove(args, src)
    if spec is None:
        raise CliError(
            f"{src} 未找到水印配方：图片元数据里没有 spec，也没有 "
            f"{meta.sidecar_path_for(src).name} sidecar"
        )
    with Image.open(src) as img:
        width, height = img.size
    masks = render_spec_masks(spec, width, height)
    mask_fp = hash_masks(masks)
    entries = []
    for i, layer in enumerate(spec.layers):
        entry = {
            "index": i,
            "kind": layer.kind,
            "content": layer.content,
            "opacity": layer.opacity,
            "position": layer.position,
            "margin": layer.margin,
            "offset": list(layer.offset),
            "font": layer.font,
            "font_size": layer.font_size,
            "stroke_width": layer.stroke_width,
            "rotate": layer.rotate,
            "tile": layer.tile,
        }
        if layer.kind == "logo":
            entry["font_size"] = None  # logo 用 font_size 作最大边，标注区分
        entries.append(entry)
    payload = {
        "input": str(src),
        "spec_source": source,
        "width": width,
        "height": height,
        "blend_space": spec.blend_space,
        "version": spec.version,
        "layers": entries,
        "mask_sha256": mask_fp,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    print(f"图片: {src} ({width}x{height})")
    print(f"配方来源: {source}")
    print(f"混合空间: {spec.blend_space}  版本: {spec.version}")
    print(f"mask 指纹: {mask_fp[:16]}…")
    print(f"图层数: {len(spec.layers)}")
    for e in entries:
        size_note = "" if e["kind"] == "text" else " (logo 最大边)"
        print(
            f"  [{e['index']}] {e['kind']}: {e['content']!r}"
            f"  opacity={e['opacity']}  position={e['position']}"
            f"  size={e['font_size']}{size_note}  rotate={e['rotate']}"
            f"  tile={e['tile']}  stroke={e['stroke_width']}"
        )
    return 0


# ---------------------------------------------------------------- compare

def _build_compare_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("a", help="参考图（通常为原图）")
    parser.add_argument("b", help="对比图（通常为去水印后的图）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")


def _load_arrays(a: Path, b: Path):
    """读两幅图为同形状 float32 数组（灰度统一升为 3 通道）。"""
    img_a = pipeline._load_image_array(a)[1]
    img_b = pipeline._load_image_array(b)[1]
    if img_a.shape[-1] == 1:
        img_a = np.repeat(img_a, 3, axis=-1)
    if img_b.shape[-1] == 1:
        img_b = np.repeat(img_b, 3, axis=-1)
    if img_a.shape != img_b.shape:
        raise CliError(
            f"两幅图尺寸不一致: {img_a.shape[1]}x{img_a.shape[0]} vs "
            f"{img_b.shape[1]}x{img_b.shape[0]}，无法对比"
        )
    return img_a, img_b


def _run_compare(args: argparse.Namespace) -> int:
    path_a, path_b = Path(args.a), Path(args.b)
    for p in (path_a, path_b):
        if not p.is_file():
            raise CliError(f"输入文件不存在: {p}")
    arr_a, arr_b = _load_arrays(path_a, path_b)
    value = blend.psnr(arr_a, arr_b)
    stats = blend.diff_stats(arr_a, arr_b)
    diff_mask = np.abs(arr_a - arr_b).max(axis=-1) > 0.5
    diff_ratio = float(diff_mask.mean())
    payload = {
        "a": str(path_a),
        "b": str(path_b),
        "psnr_db": None if value == float("inf") else round(value, 2),
        "psnr_inf": value == float("inf"),
        "max_diff": round(stats["max"], 4),
        "mean_diff": round(stats["mean"], 6),
        "rmse": round(stats["rmse"], 6),
        "diff_pixel_ratio": round(diff_ratio, 6),
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    psnr_text = "inf（完全一致）" if payload["psnr_inf"] else f"{payload['psnr_db']:.2f} dB"
    print(f"PSNR: {psnr_text}")
    print(f"最大差: {payload['max_diff']:.4f}/255")
    print(f"差异像素占比: {payload['diff_pixel_ratio']:.4%}")
    return 0


# ---------------------------------------------------------------- batch

def _build_batch_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="输入目录")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--mode", choices=("add", "remove"), required=True, help="批处理模式")
    parser.add_argument("--pattern", default="*", help="文件通配符（默认 *）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出汇总")


def _iter_images(directory: Path, pattern: str) -> list[Path]:
    return sorted(
        p for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in IMG_SUFFIXES and not p.name.endswith(".wm.json")
    )


def _run_batch(args: argparse.Namespace) -> int:
    in_dir = Path(args.input)
    if not in_dir.is_dir():
        raise CliError(f"输入目录不存在: {in_dir}")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _iter_images(in_dir, args.pattern)
    if not files:
        raise CliError(f"{in_dir} 里没有匹配 {args.pattern!r} 的图片")

    ok: list[str] = []
    failed: list[dict] = []
    for src in files:
        try:
            if args.mode == "add":
                ns = argparse.Namespace(
                    input=str(src),
                    output=str(out_dir / _default_add_output(src).name),
                    text="BATCH",
                    logo=None,
                    font=None,
                    size=36,
                    opacity=0.25,
                    color="90,90,90",
                    position="bottom-right",
                    margin=0.03,
                    offset="0,0",
                    rotate=0.0,
                    stroke=0,
                    tile=False,
                    tile_gap=0,
                    blend_space="srgb",
                    quality=None,
                    spec_out=None,
                    sidecar=False,
                    json=False,
                )
                _run_add(ns)
            else:
                ns = argparse.Namespace(
                    input=str(src),
                    output=str(out_dir / (src.stem + ".restored" + src.suffix)),
                    spec=None,
                    allow_scale=False,
                    save_spec=None,
                    json=False,
                )
                _run_remove(ns)
            ok.append(src.name)
        except (CliError, SpecError, ValueError, OSError) as exc:
            failed.append({"file": src.name, "error": str(exc)})
    payload = {
        "mode": args.mode,
        "input": str(in_dir),
        "output": str(out_dir),
        "total": len(files),
        "ok": ok,
        "failed": failed,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"批处理完成: 成功 {len(ok)} / 失败 {len(failed)}（共 {len(files)}）")
        for item in failed:
            print(f"  ✗ {item['file']}: {item['error']}")
    return 0 if not failed else 1


# ---------------------------------------------------------------- demo

def _build_demo_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", default="demo", help="输出目录（默认 demo）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")


def _make_demo_image(width: int = DEMO_WIDTH, height: int = DEMO_HEIGHT) -> Image.Image:
    """numpy 合成测试图：渐变底 + 彩色圆 + 横竖条纹。"""
    rng = np.random.default_rng(2026)
    yy, xx = np.mgrid[0:height, 0:width]
    base = np.stack([
        (xx / max(width - 1, 1) * 255),
        (yy / max(height - 1, 1) * 255),
        np.full((height, width), 128.0),
    ], axis=-1)
    # 彩色圆
    cy, cx = height // 2, width // 3
    circle = (yy - cy) ** 2 + (xx - cx) ** 2 <= (height // 4) ** 2
    base[circle] = [220.0, 80.0, 60.0]
    # 横条纹
    base[(yy % 24) < 6, 2] = np.clip(base[(yy % 24) < 6, 2] + 60, 0, 255)
    # 轻噪声，避免大片同值像素
    noise = rng.normal(0, 3, base.shape)
    return Image.fromarray(np.clip(base + noise, 0, 255).astype(np.uint8), "RGB")


def _run_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    original_path = out_dir / "original.png"
    marked_path = out_dir / "watermarked.png"
    restored_path = out_dir / "restored.png"

    original = _make_demo_image()
    original.save(original_path)

    layer = Layer(kind="text", content="watermark-kit demo", opacity=0.25,
                  position="bottom-right", font_size=28)
    spec = WatermarkSpec(layers=(layer,), blend_space="srgb")
    report_add = pipeline.add_watermark(original_path, spec)
    report_add["output"] = str(marked_path)
    pipeline.save(report_add["image"], marked_path, spec)

    spec_back = meta.read_spec(marked_path)
    if spec_back is None:
        raise CliError("demo 异常：加水印后读不到 spec（引擎 bug）")
    report_rm = pipeline.remove_watermark(
        marked_path, spec_back,
        expected_mask_sha256=report_add["mask_sha256"],
    )
    report_rm["output"] = str(restored_path)
    pipeline.save(report_rm["image"], restored_path, spec_back)

    arr_orig = np.asarray(original, dtype=np.float32)
    arr_restored = np.asarray(report_rm["image"], dtype=np.float32)
    value = blend.psnr(arr_orig, arr_restored)
    stats = blend.diff_stats(arr_orig, arr_restored)
    payload = {
        "out_dir": str(out_dir),
        "original": str(original_path),
        "watermarked": str(marked_path),
        "restored": str(restored_path),
        "psnr_db": None if value == float("inf") else round(value, 2),
        "psnr_inf": value == float("inf"),
        "max_diff": round(stats["max"], 4),
        "mask_match": report_rm["mask_match"],
        "conclusion": "✅ 往返无损" if value >= MIN_PSNR_DB else "⚠ 有可见差异",
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(f"原图:      {original_path}")
        print(f"加水印后:  {marked_path}")
        print(f"去水印后:  {restored_path}")
        psnr_text = "inf（完全一致）" if payload["psnr_inf"] else f"{payload['psnr_db']:.2f} dB"
        print(f"往返 PSNR: {psnr_text}")
        print(f"最大差:    {payload['max_diff']:.4f}/255")
        print(payload["conclusion"])
    return 0 if value >= MIN_PSNR_DB else 1


# ---------------------------------------------------------------- serve

def _build_serve_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="监听端口")
    parser.add_argument("--open", action="store_true", help="启动后打开浏览器")


def _run_serve(args: argparse.Namespace) -> int:
    """转发给后续 Web 卡的 watermark.api:create_app；未实现时给友好提示。"""
    try:
        from watermark.api import create_app  # noqa: F401
    except ImportError:
        print(SERVE_HINT, file=sys.stderr)
        return 2
    import threading
    import webbrowser

    import uvicorn

    if args.open:
        threading.Timer(
            1.0, webbrowser.open,
            args=(f"http://{args.host}:{args.port}",),
        ).start()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")
    return 0


# ---------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="watermark",
        description="可逆图像水印：加水印后可按记录参数近乎无损还原",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="给图片加水印")
    _build_add_parser(add_parser)
    add_parser.set_defaults(func=_run_add)

    remove_parser = subparsers.add_parser("remove", help="去水印")
    _build_remove_parser(remove_parser)
    remove_parser.set_defaults(func=_run_remove)

    inspect_parser = subparsers.add_parser("inspect", help="查看图片上的水印配方")
    _build_inspect_parser(inspect_parser)
    inspect_parser.set_defaults(func=_run_inspect)

    compare_parser = subparsers.add_parser("compare", help="对比两幅图差异")
    _build_compare_parser(compare_parser)
    compare_parser.set_defaults(func=_run_compare)

    batch_parser = subparsers.add_parser("batch", help="批量加/去水印")
    _build_batch_parser(batch_parser)
    batch_parser.set_defaults(func=_run_batch)

    demo_parser = subparsers.add_parser("demo", help="生成往返演示与 PSNR 报告")
    _build_demo_parser(demo_parser)
    demo_parser.set_defaults(func=_run_demo)

    serve_parser = subparsers.add_parser("serve", help="启动本地 Web 界面")
    _build_serve_parser(serve_parser)
    serve_parser.set_defaults(func=_run_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    """watermark 命令入口。"""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except CliError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2
    except SpecError as exc:
        print(f"错误: spec 不合法: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
