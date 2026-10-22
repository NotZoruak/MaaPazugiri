"""用当前帧的白色数字、品红描边和小数点核对乱舞倒计时。"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from vision.image_io import Image

# 区域来自720×1280实机截图，覆盖数字本身，不依赖会随倒计时变化的扇形。
RANBU_COUNTDOWN_ROI = (562, 198, 83, 49)


def _has_magenta_border(labels: NDArray[np.int32], label: int, magenta: NDArray[np.bool_]) -> bool:
    """白色笔画外沿必须同时具有品红色发光，排除普通白色界面文字。"""
    component = (labels == label).astype(np.uint8)
    border = (cv2.dilate(component, np.ones((5, 5), np.uint8)) != 0) & (component == 0)
    count = int(np.count_nonzero(border))
    return count > 0 and int(np.count_nonzero(magenta & border)) >= count * 0.6


def matches_ranbu_countdown(image: Image) -> bool:
    """只接受已核对布局中的两位发光数字和低位小数点，不缓存历史结论。

    现场十张截图的数字高度为28～29像素，白色笔画的品红外沿占比为
    68%～86%；检查保留少量抗锯齿余量，同时要求三个部件的相对位置。
    此函数确认倒计时图案是否存在，不推断数字对应的剩余时间。
    """
    if image.shape != (1280, 720, 3) or image.dtype != np.uint8:
        return False
    x, y, width, height = RANBU_COUNTDOWN_ROI
    patch = image[y : y + height, x : x + width]
    white = (np.min(patch, axis=2) > 220).astype(np.uint8)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    magenta = cv2.inRange(hsv, np.array((135, 85, 190), np.uint8), np.array((170, 255, 255), np.uint8)) != 0
    _, raw_labels, stats, _ = cv2.connectedComponentsWithStats(white)
    labels = np.asarray(raw_labels, dtype=np.int32)
    digits: list[tuple[int, int, int, int]] = []
    dots: list[tuple[int, int, int, int]] = []
    for label, stat in enumerate(stats[1:], start=1):
        left, top, body_width, body_height, area = (int(value) for value in stat)
        if 7 <= body_width <= 27 and 24 <= body_height <= 34 and 100 <= area <= 500 and 6 <= top <= 16:
            if _has_magenta_border(labels, label, magenta):
                digits.append((left, top, body_width, body_height))
        elif 3 <= body_width <= 7 and 3 <= body_height <= 8 and 8 <= area <= 40 and 29 <= top <= 39:
            dots.append((left, top, body_width, body_height))
    if len(digits) != 2:
        return False
    first, second = sorted(digits)
    if not (7 <= first[0] <= 26 and 43 <= second[0] <= 53):
        return False
    if abs(first[1] - second[1]) > 3 or abs(first[1] + first[3] - second[1] - second[3]) > 3:
        return False
    return any(
        first[0] + first[2] < dot_x < dot_x + dot_width < second[0]
        and 33 <= dot_x <= 44
        and abs(dot_y + dot_height - first[1] - first[3]) <= 5
        for dot_x, dot_y, dot_width, dot_height in dots
    )
