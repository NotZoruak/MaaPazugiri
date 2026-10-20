"""检查已核对布局中乱舞计时圈顶部偏左竖条的固定颜色。"""

from __future__ import annotations

import numpy as np
from vision.image_io import Image
from vision.model import COLORS, BoardObject, BoardResult

# 开发参数来自010、017两张720×1280实机图，不依赖倒计时数字。
RANBU_ROI = (595, 166, 4, 14)
RANBU_LOWER_BGR = (246, 89, 89)
RANBU_UPPER_BGR = (255, 112, 114)


def matches_ranbu_marker(image: Image) -> bool:
    """全部56个像素处于开发色域才命中，不缓存前一帧结论。"""
    if image.shape != (1280, 720, 3) or image.dtype != np.uint8:
        return False
    x, y, width, height = RANBU_ROI
    patch = image[y : y + height, x : x + width]
    return bool(np.all((patch >= RANBU_LOWER_BGR) & (patch <= RANBU_UPPER_BGR)))


def choose_ranbu_start(board: BoardResult, reserved_color: str | None) -> tuple[int, int] | None:
    """从本帧识别结果选择一个未保留颜色的有效起点。"""
    candidates: list[tuple[int, BoardObject]] = []
    for item in board.objects:
        if item.kind == "ghoul":
            if item.color not in COLORS or item.color == reserved_color:
                continue
            point = item.candidate_point or item.center
        elif item.kind == "special_ghoul" and item.subtype == "black":
            point = item.center
        else:
            continue
        if item.quality not in ("identified", "single_feature"):
            continue
        x, y, width, height = board.board_region
        if not (x <= point[0] < x + width and y <= point[1] < y + height):
            continue
        candidates.append((item.id, item))
    if not candidates:
        return None
    chosen = min(candidates, key=lambda entry: entry[0])[1]
    point = chosen.candidate_point if chosen.kind == "ghoul" and chosen.candidate_point is not None else chosen.center
    return round(point[0]), round(point[1])


def ranbu_sweep_points(board: BoardResult) -> tuple[tuple[int, int], ...]:
    """按当前帧棋盘边界生成固定的九道往返扫动轨迹。"""
    x, y, width, height = board.board_region
    left = round(x + width * 0.03)
    right = round(x + width * 0.97)
    points: list[tuple[int, int]] = []
    for index in range(9):
        lane_y = round(y + height * (index + 1) / 10)
        lane = ((left, lane_y), (right, lane_y))
        points.extend(lane if index % 2 == 0 else reversed(lane))
    return tuple(points)
