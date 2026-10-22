"""从当前菱形上方提取乱舞连线数字，独立校验可见数字位数。"""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np

from .image_io import Image
from .model import ChainEndpointMarker


@dataclass(frozen=True)
class RanbuDigits:
    """保存原图数字区域及独立OCR图；未知时不生成可采信的数字位数。"""

    roi: tuple[int, int, int, int] | None
    filtered: Image | None
    digit_count: int | None
    component_boxes: tuple[tuple[int, int, int, int], ...]
    reason: str


def prepare_ranbu_digits(image: Image, marker: ChainEndpointMarker) -> RanbuDigits:
    """按本帧菱形尺寸取样，只保留具有蓝色描边的亮白数字笔画。

    输出为放大两倍、带留白的黑字白底BGR图，交给原生OCR独立识别。
    不按固定棋子数量限制数值；文字位数必须再与可见独立部件核对。
    """
    unknown = RanbuDigits(None, None, None, (), "invalid_input")
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        return unknown
    if not all(math.isfinite(value) for value in (*marker.box, *marker.center)):
        return unknown
    x, y, width, height = marker.box
    if min(width, height) <= 0:
        return unknown
    source_height, source_width = image.shape[:2]
    left, top = max(0, round(x - width * 0.6)), max(0, round(y - height * 0.75))
    right = min(source_width, left + round(width * 2.2))
    bottom = min(source_height, top + round(height * 0.8))
    if right <= left or bottom <= top:
        return unknown
    roi = (left, top, right - left, bottom - top)
    patch = image[top:bottom, left:right]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    # 七张现场图中，数字为亮白核心和青蓝描边；身体反光与紫色连线不同时满足。
    white = ((np.min(patch, axis=2) > 200) & (hsv[:, :, 1] < 70)).astype(np.uint8)
    blue = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 115) & (hsv[:, :, 1] >= 100) & (hsv[:, :, 2] > 130)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(white)
    kernel_size = max(3, round(height * 0.06) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    selected: list[tuple[int, int, int, int, int]] = []
    for label, stat in enumerate(stats[1:], start=1):
        cx, cy, cw, ch, area = (int(value) for value in stat)
        if not (
            max(2, width * 0.035) <= cw <= width * 0.5
            and height * 0.34 <= ch <= height * 0.6
            and area >= height * height * 0.012
            and height * 0.08 <= cy <= height * 0.3
        ):
            continue
        component = (labels == label).astype(np.uint8)
        border = (cv2.dilate(component, kernel) != 0) & (component == 0)
        border_count = int(np.count_nonzero(border))
        if border_count and int(np.count_nonzero(blue & border)) >= border_count * 0.5:
            selected.append((cx, cy, cw, ch, label))
    if not selected:
        return RanbuDigits(roi, None, None, (), "no_digit_components")
    selected.sort()
    component_boxes = tuple((left + cx, top + cy, cw, ch) for cx, cy, cw, ch, _ in selected)
    min_x = min(item[0] for item in selected)
    max_x = max(item[0] + item[2] for item in selected)
    min_y = min(item[1] for item in selected)
    max_y = max(item[1] + item[3] for item in selected)
    if (
        min_x == 0
        or max_x == right - left
        or abs((min_x + max_x) * 0.5 + left - marker.center[0]) > width * 0.3
        or max(item[1] for item in selected) - min_y > height * 0.08
        or max_y - min(item[1] + item[3] for item in selected) > height * 0.08
        or any(a[0] + a[2] >= b[0] for a, b in zip(selected, selected[1:], strict=False))
    ):
        return RanbuDigits(roi, None, None, component_boxes, "ambiguous_digit_geometry")
    mask = np.isin(labels, [item[4] for item in selected]).astype(np.uint8) * 255
    core = mask[min_y:max_y, min_x:max_x]
    padding = max(4, round(height * 0.09))
    filtered = cv2.copyMakeBorder(core, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
    filtered = cv2.resize(255 - filtered, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2BGR)
    return RanbuDigits(
        (left + min_x, top + min_y, max_x - min_x, max_y - min_y),
        cast(Image, filtered),
        len(selected),
        component_boxes,
        "digit_components_verified",
    )


def parse_ranbu_digit_text(prepared: RanbuDigits, text: str) -> int | None:
    """仅接受与独立可见部件位数一致的正整数；额外OCR笔画不能变成新数字。"""
    normalized = unicodedata.normalize("NFKC", text).strip()
    if (
        prepared.filtered is None
        or prepared.digit_count is None
        or not normalized.isascii()
        or not normalized.isdecimal()
        or len(normalized) != prepared.digit_count
    ):
        return None
    value = int(normalized)
    return value if value > 0 else None
