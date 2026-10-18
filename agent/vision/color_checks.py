"""缓存模板的相对身体形状，逐帧核对强化饿鬼的主色。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .image_io import Image


@dataclass(frozen=True)
class BodyColorSupport:
    """相对于模板起点的静态掩码，不含任何截图位置。"""

    left: int
    top: int
    mask: Image
    count: int


def prepare_body_color_support(center_offset: tuple[float, float], body_size: tuple[float, float]) -> BodyColorSupport:
    """加载时制备身体两侧和下部的颜色核对形状。"""
    cx, cy = center_offset
    bw, bh = body_size
    left, right = math.floor(cx - bw * 0.41), math.ceil(cx + bw * 0.41)
    top, bottom = math.floor(cy - bh * 0.29), math.ceil(cy + bh * 0.42)
    yy, xx = np.ogrid[top:bottom, left:right]
    selected = ((xx - cx) / (bw * 0.4)) ** 2 + ((yy - cy - bh * 0.08) / (bh * 0.33)) ** 2 <= 1
    selected |= (np.abs(xx - cx) < bw * 0.4) & (yy >= cy - bh * 0.28) & (yy < cy + bh * 0.1)
    selected &= ~((np.abs(xx - cx) < bw * 0.21) & (yy < cy + bh * 0.22))
    mask = selected.astype(np.uint8) * 255
    return BodyColorSupport(left, top, mask, cv2.countNonZero(mask))


def enhanced_color_compatible(
    palette: Image, support: BodyColorSupport, origin: tuple[int, int], color_index: int
) -> bool:
    """以当前帧四通道二值颜色图核对占比及主色，保持原有裁切规则。"""
    left, top = origin[0] + support.left, origin[1] + support.top
    height, width = support.mask.shape
    x1, y1 = max(0, left), max(0, top)
    x2, y2 = min(palette.shape[1], left + width), min(palette.shape[0], top + height)
    if x1 >= x2 or y1 >= y2:
        return False
    mask = support.mask[y1 - top : y2 - top, x1 - left : x2 - left]
    area = support.count if mask.shape == support.mask.shape else cv2.countNonZero(mask)
    if area == 0:
        return False
    means = cv2.mean(palette[y1:y2, x1:x2], mask=mask)
    counts = [round(mean * area / 255.0) for mean in means]
    return counts[color_index] >= area * 0.06 and all(
        counts[color_index] > count for index, count in enumerate(counts) if index != color_index
    )
