"""可逆水印核心引擎：spec / fonts / render / blend / meta / pipeline。"""

from watermark.spec import Layer, WatermarkSpec
from watermark.pipeline import add_watermark, inspect_spec, remove_watermark, save

__version__ = "0.1.0"

__all__ = [
    "Layer",
    "WatermarkSpec",
    "add_watermark",
    "inspect_spec",
    "remove_watermark",
    "save",
    "__version__",
]
