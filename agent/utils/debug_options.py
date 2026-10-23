"""读取 MFAAvalonia 的调试截图保存选项。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DebugCaptureOptions:
    """保存调试截图的开关。"""

    save_draw: bool = False


def load_debug_capture_options(config_dir: Path) -> DebugCaptureOptions:
    """从 config/maa_option.json 读取 UI 保存的截图设置，缺失时采用 UI 默认值。"""
    defaults = DebugCaptureOptions()
    try:
        payload: Any = json.loads((config_dir / "maa_option.json").read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return defaults
    if not isinstance(payload, dict):
        return defaults
    save_draw = payload.get("save_draw", defaults.save_draw)
    return DebugCaptureOptions(
        save_draw=save_draw if isinstance(save_draw, bool) else defaults.save_draw,
    )
