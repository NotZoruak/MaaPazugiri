"""在预选点局部确认新增选中描边，不定位其他饿鬼或生成路线。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from .color_checks import prepare_body_color_support
from .image_io import Image
from .model import COLORS
from .recognizer import HSV_RANGES


@dataclass(frozen=True)
class RanbuGrabResult:
    """只有当前帧颜色和新增选中描边同时可信，才确认游戏接入。"""

    status: str
    reason: str
    color: str | None = None
    cyan_pixels: int = 0
    added_cyan_pixels: int = 0
    glow_quadrants: int = 0
    state: str | None = None
    state_checked: bool = False


def inspect_ranbu_grab(
    before: Image, current: Image, point: tuple[int, int], body_size: tuple[float, float], color: str | None
) -> RanbuGrabResult:
    """比较按下前后同点局部变化；旧截图只用作状态基线，不用于搜索对象。"""
    if (
        color not in COLORS
        or before.shape != current.shape
        or current.ndim != 3
        or current.shape[2] != 3
        or current.dtype != np.uint8
        or before.dtype != np.uint8
        or not current.size
        or not all(math.isfinite(value) for value in (*point, *body_size))
        or min(body_size) <= 0
        or body_size[0] > current.shape[1]
        or body_size[1] > current.shape[0]
    ):
        return RanbuGrabResult("unknown", "起点局部输入无效")
    px, py = point
    bw, bh = body_size
    support = prepare_body_color_support((0, 0), body_size)
    if support.count < 16:
        return RanbuGrabResult("unknown", "身体颜色采样不足")
    left, top = px + support.left, py + support.top
    height, width = support.mask.shape
    if left < 0 or top < 0 or left + width > current.shape[1] or top + height > current.shape[0]:
        return RanbuGrabResult("unknown", "起点身体采样不完整")
    patch = current[top : top + height, left : left + width]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 排除青色选中外缘，用身体两侧与下部的色相核对实际选中颜色。
    glow = (patch[:, :, 0] >= 180) & (patch[:, :, 1] >= 180) & (patch[:, :, 2] <= 140)
    counts = []
    for name in COLORS:
        selected = np.zeros(support.mask.shape, np.uint8)
        for lower, upper in HSV_RANGES[name]:
            selected = cast(
                Image,
                cv2.bitwise_or(
                    selected, cv2.inRange(hsv, np.array((lower[0], 35, 65)), np.array((upper[0], 255, 255)))
                ),
            )
        selected[glow] = 0
        selected &= support.mask
        counts.append(cv2.countNonZero(selected))
    best = max(range(len(COLORS)), key=lambda index: counts[index])
    observed = COLORS[best]
    if (
        observed != color
        or counts[best] < support.count * 0.15
        or any(count > counts[best] * 0.65 for index, count in enumerate(counts) if index != best)
    ):
        return RanbuGrabResult("unknown", "当前起点颜色不足以确认预选颜色已接入", observed)
    x1, y1 = max(0, math.floor(px - bw * 0.8)), max(0, math.floor(py - bh * 0.8))
    x2, y2 = min(current.shape[1], math.ceil(px + bw * 0.8)), min(current.shape[0], math.ceil(py + bh * 0.8))

    def cyan(image: Image) -> np.ndarray:
        """当前局部的青色高亮，不复用旧图中的对象或位置。"""
        crop = image[y1:y2, x1:x2]
        return (crop[:, :, 0] >= 220) & (crop[:, :, 1] >= 210) & (crop[:, :, 2] <= 100)

    previous_glow, current_glow = cyan(before), cyan(current)
    added_glow = current_glow & ~previous_glow
    total = int(np.count_nonzero(current_glow))
    added = int(np.count_nonzero(added_glow))
    cx, cy = px - x1, py - y1
    quadrants = sum(
        int(np.count_nonzero(region) >= bw * bh * 0.005)
        for region in (added_glow[:cy, :cx], added_glow[:cy, cx:], added_glow[cy:, :cx], added_glow[cy:, cx:])
    )
    confirmed = total >= bw * bh * 0.08 and added >= bw * bh * 0.04 and quadrants >= 3
    return RanbuGrabResult(
        "confirmed" if confirmed else "unknown",
        "预选颜色与新增选中描边同时成立" if confirmed else "未确认起点出现足够新增选中描边",
        observed,
        total,
        added,
        quadrants,
    )
