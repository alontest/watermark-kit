# watermark-kit

可逆图像水印工具：给图片加水印，之后按记录的配方（spec）近乎无损地去掉水印。

## 原理

水印是一层半透明覆盖：逐通道做凸组合

```
I_out = I_in × (1 − A) + C × A
```

`A` 为有效 alpha（不透明度 mask），`C` 为水印颜色。去水印时用同一配方反解：

```
I_in = (I_out − C × A) / (1 − A)
```

因为凸组合的取值天然落在 [0, 255] 区间内、不发生截断，PNG 无损往返时反解可精确恢复原像素（实测 PSNR > 70 dB，残差来自 `uint8` 取整）。当 `A → 1` 时 `(1 − A) → 0`，除法放大误差直至不可逆，因此 `opacity` 被限制在 **0.05 – 0.95**。

## 安装

需要 Python ≥ 3.10，使用 [uv](https://docs.astral.sh/uv/)：

```bash
uv venv
uv pip install -e ".[dev]"        # 核心 + pytest；要 Web 界面则安装 ".[web]"
```

安装后得到 `watermark` 命令（源码目录内也可用 `.venv/bin/watermark`）。

## CLI 用法

```bash
# 加水印：右下角 30% 不透明度文字，同时写图片元数据和 .wm.json sidecar
watermark add photo.png --text "© 2026 Alice" --opacity 0.3 --position bottom-right --sidecar

# 去水印：优先读图片元数据，其次读旁挂的 .wm.json，也可 --spec 手动指定
watermark remove photo.wm.png -o restored.png

# 查看（无需去水印即可看到配方）
watermark inspect photo.wm.png

# 对比两幅图（PSNR / 最大差 / 差异像素占比）
watermark compare original.png restored.png

# 批量加水印（--mode remove 批量去水印）
watermark batch ./photos --mode add -o ./done --pattern "*.png"

# 生成一张合成图的加水印 → 去水印演示和 PSNR 报告（输出到 demo/）
watermark demo

# 启动本地 Web 界面
watermark serve
```

常用选项：`--logo`（图片水印，与 `--text` 二选一）、`--tile`（平铺）、`--rotate`、`--blend-space linear`、`--json`（所有子命令支持 JSON 输出）。

## Web 用法

```bash
watermark serve --open
```

启动后自动打开浏览器（默认 `http://127.0.0.1:8000`），拖入图片即可加水印 / 去水印并预览下载。需要安装 `.[web]` 依赖（FastAPI + uvicorn）。

## 重要限制

- **只能去掉自己加的、配方还在的水印。** 没有去水印魔法：remove 必须拿到当初 add 时记录的 spec（图片元数据或 `.wm.json`）。第三方图片上的水印无法去除。
- **元数据会被平台剥离。** 微信、微博等平台压缩图片时会丢弃 PNG tEXt / EXIF 元数据，到手的图就还原不了了——发布前请另存 `.wm.json` sidecar（`add --sidecar` 或 `--spec-out`）。
- **建议 PNG，不要 JPEG。** JPEG 有损压缩会在量化时破坏像素值，反解误差被放大；实测对同一张图，PNG 往返 PSNR > 70 dB，JPEG 往返仅约 12 dB，基本不可用。
- **图片被缩放后不再精确。** 尺寸变化后原 spec 的渲染 mask 与实际水印对不上，只能近似还原（实测缩放 0.5× 后 PSNR 约 40 dB）。`remove --allow-scale` 允许这种近似还原（按比例缩放字号），但结果会明显偏离原图。
- 原图必须能被无损地拿到同一状态：经过截断（暗角、强烈调色）或另一次有损处理的图，即使配方还在也只能近似还原。

## 项目结构

```
watermark/
├── spec.py       # WatermarkSpec / Layer 数据类，JSON 往返与字段校验（opacity 上限等）
├── render.py     # spec → 与原图同尺寸的有效 alpha mask（文字/ logo/平铺/九宫格定位）
├── blend.py      # 正向混合与反解还原，PSNR / 差异统计
├── meta.py       # 配方元数据读写：PNG tEXt / JPEG EXIF / .wm.json sidecar
├── pipeline.py   # add / remove / inspect 流水线与报告组装
├── fonts.py      # 字体解析与加载
├── cli.py        # argparse 命令行入口（add/remove/inspect/compare/batch/demo/serve）
├── api.py        # FastAPI 应用（serve 后端）
└── web/
    └── index.html  # Web 界面（拖拽上传、预览、下载）
```

## 开发

```bash
uv pip install -e ".[dev]"
pytest
```
