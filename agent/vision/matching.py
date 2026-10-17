"""缓存静态模板，并在当前帧内用保守颜色筛选减少精细匹配。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .image_io import Image

ScoreMap = NDArray[np.float32]
ColorIntegral = NDArray[np.int32]


@dataclass(frozen=True)
class PreparedTemplate:
    """保存模板派生数据，不保存截图或棋盘对象位置。"""

    image: Image
    mask: Image
    full_mask: bool
    coarse_image: Image | None = None
    coarse_mask: Image | None = None
    coarse_full_mask: bool = False


def prepare_template(template: Image, mask: Image) -> PreparedTemplate:
    """加载时生成灰度缩图与有效掩码，避免每帧重复生成。"""
    full_mask = bool(np.all(mask == 255))
    if min(template.shape[:2]) < 20:
        return PreparedTemplate(template, mask, full_mask)
    height, width = template.shape[:2]
    size = (max(2, round(width / 2)), max(2, round(height / 2)))
    small_template = cast(Image, cv2.resize(cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), size))
    support = cv2.resize((mask > 0).astype(np.float32), size)
    small_mask = np.where(support >= 1.0 - 1e-6, 255, 0).astype(np.uint8)
    valid = small_template[small_mask > 0]
    if valid.size < 8 or float(np.var(valid.astype(np.float32))) < 1:
        return PreparedTemplate(template, mask, full_mask)
    return PreparedTemplate(template, mask, full_mask, small_template, small_mask, bool(np.all(small_mask == 255)))


def prepare_board(board: Image) -> Image:
    """每帧独立生成一次灰度缩图，仅在该帧的模板间共享。"""
    height, width = board.shape[:2]
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    return cast(Image, cv2.resize(gray, (max(2, round(width / 2)), max(2, round(height / 2)))))


def color_integral(mask: Image) -> ColorIntegral:
    """为当前帧建立颜色积分图，区域计数只需四次数组读取。"""
    return cast(ColorIntegral, cv2.integral((mask > 0).astype(np.uint8), sdepth=cv2.CV_32S))


def rectangle_count(integral: ColorIntegral, x1: int, y1: int, x2: int, y2: int) -> int:
    """统计左闭右开区域的颜色像素数。"""
    if x1 >= x2 or y1 >= y2:
        return 0
    return int(integral[y2, x2]) - int(integral[y1, x2]) - int(integral[y2, x1]) + int(integral[y1, x1])


@dataclass(frozen=True)
class ColorGate:
    """仅拒绝整个精细搜索窗口均不可能通过颜色条件的候选。"""

    integral: ColorIntegral
    offset: tuple[float, float]
    body_size: tuple[float, float]
    minimum_fraction: float

    def allows(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """以颜色数量上界和主体面积下界保守放行，并覆盖极值邻域。"""
        if self.minimum_fraction <= 0:
            return True
        height, width = self.integral.shape[0] - 1, self.integral.shape[1] - 1
        # 扩展两像素，保留可能抑制邻近有效峰值的响应。
        low_x = max(0.0, x1 - 2 + self.offset[0])
        low_y = max(0.0, y1 - 2 + self.offset[1])
        high_x = min(width - 1e-9, x2 + 1 + self.offset[0])
        high_y = min(height - 1e-9, y2 + 1 + self.offset[1])
        if low_x > high_x or low_y > high_y:
            return False
        bw, bh = self.body_size
        left, top = max(0, math.floor(low_x - bw / 2) - 1), max(0, math.floor(low_y - bh / 2) - 1)
        right, bottom = min(width, math.ceil(high_x + bw / 2) + 1), min(height, math.ceil(high_y + bh / 2) + 1)
        count_upper = rectangle_count(self.integral, left, top, right, bottom)
        # 两端取整与边缘裁切只会进一步放宽这个面积下界。
        min_width = max(1.0, min(bw, width, low_x + bw / 2, width - high_x + bw / 2) - 2)
        min_height = max(1.0, min(bh, height, low_y + bh / 2, height - high_y + bh / 2) - 2)
        return count_upper >= self.minimum_fraction * min_width * min_height


def exact_response(image: Image, template: Image, mask: Image, *, full_mask: bool | None = None) -> ScoreMap:
    """全白掩码走快速分支，其余掩码保持有效像素约束。"""
    complete = bool(np.all(mask == 255)) if full_mask is None else full_mask
    if complete:
        response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    else:
        response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    return cast(ScoreMap, np.nan_to_num(response, nan=-1.0, posinf=-1.0, neginf=-1.0))


def local_response(
    board: Image,
    pattern: PreparedTemplate,
    full_search: bool,
    coarse_board: Image | None,
    gate: ColorGate | None = None,
) -> ScoreMap:
    """在当前帧原图中精细匹配，不使用其他帧的位置信息。"""
    template, mask = pattern.image, pattern.mask
    if full_search or pattern.coarse_image is None or pattern.coarse_mask is None:
        return exact_response(board, template, mask, full_mask=pattern.full_mask)
    small_board = prepare_board(board) if coarse_board is None else coarse_board
    coarse = exact_response(small_board, pattern.coarse_image, pattern.coarse_mask, full_mask=pattern.coarse_full_mask)
    peaks = cv2.dilate(coarse, np.ones((3, 3), dtype=np.uint8))
    ys, xs = np.where((coarse >= 0.55) & (coarse >= peaks - 1e-7))
    positions = sorted(zip(xs.tolist(), ys.tolist(), strict=True), key=lambda point: -float(coarse[point[1], point[0]]))
    height, width = template.shape[:2]
    board_height, board_width = board.shape[:2]
    response = np.full((board_height - height + 1, board_width - width + 1), -1.0, dtype=np.float32)
    sx, sy = board_width / small_board.shape[1], board_height / small_board.shape[0]
    for px, py in positions[:120]:
        cx, cy = round(px * sx), round(py * sy)
        x1, y1 = max(0, cx - 5), max(0, cy - 5)
        x2, y2 = min(response.shape[1], cx + 6), min(response.shape[0], cy + 6)
        if x1 >= x2 or y1 >= y2 or (gate is not None and not gate.allows(x1, y1, x2, y2)):
            continue
        refined = exact_response(
            board[y1 : y2 + height - 1, x1 : x2 + width - 1], template, mask, full_mask=pattern.full_mask
        )
        response[y1:y2, x1:x2] = np.maximum(response[y1:y2, x1:x2], refined)
    return response
