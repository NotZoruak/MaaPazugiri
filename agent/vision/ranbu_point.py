"""直接核对乱舞预选触点附近的身体颜色，不依赖后续整盘对象框。"""

from __future__ import annotations

import math
from typing import cast

import cv2
import numpy as np

from .color_checks import prepare_body_color_support
from .image_io import Image
from .model import COLORS
from .recognizer import HSV_RANGES

# 乱舞起点判色单独收紧红黄交界，不改变棋盘整体识别范围。
_RANBU_POINT_HSV_RANGES = {
    "red": (((0, 115, 65), (14, 255, 255)), ((165, 115, 65), (179, 255, 255))),
    "yellow": (((15, 100, 80), (37, 255, 255)),),
}


def ranbu_point_color_margin(image: Image, point: tuple[int, int], body_size: tuple[float, float], color: str) -> int:
    """只用当前帧评估选点的颜色余量，不搜索对象或更换实际起点。

    在四个方向各偏移身体尺寸的百分之八，统计仍满足原颜色条件的次数。
    该分数仅供首次同色候选排序；后续帧仍须在原点重新满足完整颜色条件。
    """
    if color not in COLORS or not all(math.isfinite(size) and size > 0 for size in body_size):
        return 0
    dx, dy = max(1, round(body_size[0] * 0.08)), max(1, round(body_size[1] * 0.08))
    px, py = point
    return sum(
        identify_ranbu_point_color(image, candidate, body_size) == color
        for candidate in ((px - dx, py), (px + dx, py), (px, py - dy), (px, py + dy))
    )


def identify_ranbu_point_color(image: Image, point: tuple[int, int], body_size: tuple[float, float]) -> str | None:
    """按预选对象尺寸核对当前帧局部主色；遮挡、双色或边界不确定时返回未知。

    调用方须先从当前帧可信普通饿鬼选点，并自行排除保留色。本函数只核对
    颜色，不宣称能从局部颜色独立区分炸弹等对象，也不搜索或追踪移动对象。
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return None
    px, py = point
    bw, bh = body_size
    if not all(math.isfinite(value) for value in (px, py, bw, bh)) or min(bw, bh) <= 0:
        return None
    height, width = image.shape[:2]
    if bw > width or bh > height or not (0 <= px < width and 0 <= py < height):
        return None
    # 两侧避开眼睛和嘴部，下部补充身体证据；形状随本次预选对象尺寸缩放。
    support = prepare_body_color_support((0.0, 0.0), body_size)
    left, top = round(px) + support.left, round(py) + support.top
    mask_height, mask_width = support.mask.shape
    if support.count < 16 or left < 0 or top < 0 or left + mask_width > width or top + mask_height > height:
        return None
    patch = image[top : top + mask_height, left : left + mask_width]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    color_masks: list[Image] = []
    counts: list[int] = []
    for color in COLORS:
        color_mask = np.zeros(support.mask.shape, dtype=np.uint8)
        for lower, upper in _RANBU_POINT_HSV_RANGES.get(color, HSV_RANGES[color]):
            color_mask = cast(Image, cv2.bitwise_or(color_mask, cv2.inRange(hsv, np.array(lower), np.array(upper))))
        color_mask &= support.mask
        color_masks.append(color_mask)
        counts.append(cv2.countNonZero(color_mask))
    best = max(range(len(COLORS)), key=lambda index: counts[index])
    # 主色须占身体采样区域多数，且不得同时存在较强的另一普通颜色。
    if counts[best] < support.count * 0.55 or any(
        count > support.count * 0.20 for index, count in enumerate(counts) if index != best
    ):
        return None
    yy, xx = np.ogrid[support.top : support.top + mask_height, support.left : support.left + mask_width]
    for segment, minimum_fraction in ((xx < -bw * 0.21, 0.35), (xx > bw * 0.21, 0.35), (yy > bh * 0.22, 0.25)):
        selected = (support.mask != 0) & segment
        area = int(np.count_nonzero(selected))
        if area == 0 or np.count_nonzero((color_masks[best] != 0) & selected) < area * minimum_fraction:
            return None
    return COLORS[best]
