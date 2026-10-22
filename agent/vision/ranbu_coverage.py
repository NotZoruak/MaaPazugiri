"""以当前按住画面的颜色像素统计发现强残留证据，不定位饿鬼。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, cast

import cv2
import numpy as np

from .image_io import Image
from .recognizer import HSV_RANGES


@dataclass(frozen=True)
class RanbuCoverageResult:
    """全区域像素证据；未出现强阳性不表示已经连接全部饿鬼。"""

    status: Literal["present", "unknown"]
    reason: str
    sampled_pixels: int = 0
    pale_pixels: int = 0
    saturated_pixels: int = 0
    cyan_pixels: int = 0
    thin_arc_area: float = 0.0
    thin_arc_fraction: float = 0.0
    pale_to_saturated_ratio: float = 0.0
    local_residual_arc: bool = False


def inspect_ranbu_coverage(
    image: Image, board_region: tuple[float, float, float, float], color: str | None
) -> RanbuCoverageResult:
    """只在褪白细长外缘与目标色亮白比例均明显时返回残留存在。

    当前只验证蓝、绿、黄的乱舞按住画面。只统计本帧目标色亮白像素的
    形状分布与青色发光证据，不提取对象、对象数量、中心或触控坐标。
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return RanbuCoverageResult("unknown", "当前图像无效")
    if color not in ("blue", "green", "yellow"):
        return RanbuCoverageResult("unknown", "该起点颜色尚无残留像素实机验证")
    if not all(math.isfinite(value) for value in board_region):
        return RanbuCoverageResult("unknown", "棋盘区域无效")
    x, y, width, height = board_region
    left, top = max(0, math.ceil(x)), max(0, math.ceil(y))
    right, bottom = min(image.shape[1], math.floor(x + width)), min(image.shape[0], math.floor(y + height))
    if width <= 0 or height <= 0 or left >= right or top >= bottom:
        return RanbuCoverageResult("unknown", "棋盘区域不在截图内")
    patch = image[top:bottom, left:right]
    # 限制分析像素数；这是颜色统计分辨率，不是棋子尺寸或棋盘格子假设。
    if patch.shape[1] > 360:
        size = (360, max(1, round(patch.shape[0] * 360 / patch.shape[1])))
        patch = cast(Image, cv2.resize(patch, size, interpolation=cv2.INTER_AREA))
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    hue_support = np.zeros(hue.shape, dtype=np.uint8)
    for lower, upper in HSV_RANGES[color]:
        hue_support = cast(Image, cv2.bitwise_or(hue_support, cv2.inRange(hue, np.array(lower[0]), np.array(upper[0]))))
    target = hue_support != 0
    pale = (target & (saturation >= 5) & (saturation < 80) & (value > 225)).astype(np.uint8)
    saturated_pixels = int(np.count_nonzero(target & (saturation >= 120) & (value >= 65)))
    cyan_pixels = int(np.count_nonzero((patch[:, :, 0] >= 220) & (patch[:, :, 1] >= 210) & (patch[:, :, 2] <= 100)))
    pale_pixels = int(np.count_nonzero(pale))
    sampled_pixels = patch.shape[0] * patch.shape[1]
    contours, _ = cv2.findContours(pale, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thin_arc_area = 0.0
    local_residual_arc = False
    near_cyan = cv2.dilate(
        ((patch[:, :, 0] >= 220) & (patch[:, :, 1] >= 210) & (patch[:, :, 2] <= 100)).astype(np.uint8),
        np.ones((5, 5), np.uint8),
    )
    scale = patch.shape[1] / 360
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        # 白眼和嘴部通常紧凑；褪白身体外缘形成较长的窄弧，圆度明显较低。
        if perimeter > 0 and 4 * math.pi * area / (perimeter * perimeter) < 0.20:
            thin_arc_area += area
        # 少量未连接白边不再除以全盘白色像素；只核对形状和周围是否已有青色描边。
        if (
            perimeter >= 65 * scale
            and area >= 75 * scale * scale
            and 4 * math.pi * area / (perimeter * perimeter) <= 0.24
            and max(cv2.boundingRect(contour)[2:]) >= 24 * scale
        ):
            boundary = np.zeros(pale.shape, np.uint8)
            cv2.drawContours(boundary, [contour], -1, 1, 1)
            boundary_pixels = cv2.countNonZero(boundary)
            if boundary_pixels and np.count_nonzero((boundary != 0) & (near_cyan != 0)) <= boundary_pixels * 0.10:
                local_residual_arc = True
    thin_arc_fraction = thin_arc_area / pale_pixels if pale_pixels else 0.0
    pale_ratio = pale_pixels / saturated_pixels if saturated_pixels else 0.0
    # 仅将两个独立比例均明显、且存在按住青色发光的情况作为强阳性。
    present = (
        cyan_pixels >= sampled_pixels * 0.005
        and pale_pixels >= sampled_pixels * 0.01
        and saturated_pixels >= sampled_pixels * 0.01
        and thin_arc_fraction >= 0.25
        and pale_ratio >= 0.20
    ) or (local_residual_arc and cyan_pixels >= sampled_pixels * 0.005 and saturated_pixels >= sampled_pixels * 0.005)
    return RanbuCoverageResult(
        "present" if present else "unknown",
        "当前帧存在明显目标色褪白细长外缘" if present else "未发现强残留证据，不能据此确认已经全部接入",
        sampled_pixels,
        pale_pixels,
        saturated_pixels,
        cyan_pixels,
        thin_arc_area,
        thin_arc_fraction,
        pale_ratio,
        local_residual_arc,
    )
