"""识别已核对布局的乱舞状态，并从当前帧选择乱舞触点。"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from vision.image_io import Image
from vision.model import COLORS, BoardObject, BoardResult
from vision.ranbu_point import identify_ranbu_point_color, ranbu_point_color_margin

from .candidates import preferred_ghoul_touch_point


def _ghoul_touch_point(item: BoardObject, board: BoardResult) -> tuple[float, float]:
    """复用普通连线的当前帧身体落点及炸弹覆盖排除。"""
    return preferred_ghoul_touch_point(item, board) if item.candidate_point is not None else item.center


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


def _start_point(item: BoardObject, board: BoardResult) -> tuple[float, float] | None:
    """起点优先身体中心；异色身体中央重叠时，回退自身可见部位。"""
    if item.kind != "ghoul":
        return item.center
    if item.candidate_point is None:
        return None

    def overlaps_other_color(point: tuple[float, float]) -> bool:
        """只排除本帧异色对象的中央区域，外接框边缘重叠不代表遮挡。"""
        for other in board.objects:
            if other is item or other.kind not in ("ghoul", "special_ghoul") or other.color == item.color:
                continue
            width, height = other.body_box[2:]
            if not all(math.isfinite(value) for value in (*other.center, width, height)) or min(width, height) <= 0:
                return True
            if ((point[0] - other.center[0]) / (width * 0.3)) ** 2 + (
                (point[1] - other.center[1]) / (height * 0.3)
            ) ** 2 <= 1:
                return True
        return False

    for point in (_ghoul_touch_point(item, board), item.candidate_point):
        if all(math.isfinite(value) for value in point) and not overlaps_other_color(point):
            return point
    return None


def _start_candidates(board: BoardResult, reserved_color: str | None) -> list[tuple[int, str | None, tuple[int, int]]]:
    """只收集本帧内具有有效落点的未保留对象，不使用历史坐标。"""
    candidates: list[tuple[int, str | None, tuple[int, int]]] = []
    for item in board.objects:
        if item.kind == "ghoul":
            if item.color not in COLORS or item.color == reserved_color:
                continue
            point = _start_point(item, board)
        elif item.kind == "special_ghoul" and item.subtype == "black":
            point = item.center
        else:
            continue
        if item.quality not in ("identified", "single_feature"):
            continue
        x, y, width, height = board.board_region
        if point is None or not all(math.isfinite(value) for value in point):
            continue
        px, py = round(point[0]), round(point[1])
        source_width, source_height = board.source_size
        if not (0 <= px < source_width and 0 <= py < source_height and x <= px < x + width and y <= py < y + height):
            continue
        candidates.append((item.id, item.color if item.kind == "ghoul" else None, (px, py)))
    return candidates


def ranbu_start_color(board: BoardResult, start: tuple[int, int]) -> str | None:
    """从本帧唯一普通起点读取颜色；黑色或位置歧义不套用普通颜色规则。"""
    matches = [color for _, color, point in _start_candidates(board, None) if point == start]
    return matches[0] if len(matches) == 1 else None


def _ordered_start_candidates(
    board: BoardResult, reserved_color: str | None, *, previous_color: str | None = None
) -> list[tuple[int, str | None, tuple[int, int]]]:
    """为当前帧全部起点排序，避免首个不可读对象长期阻塞其他颜色。"""
    candidates = _start_candidates(board, reserved_color)
    if not candidates:
        return []
    ordinary = [entry for entry in candidates if entry[1] in COLORS]
    # 已确认的普通颜色优先；黑色仅作独立回退，不因编号靠前占用每一轮。
    candidates = ordinary or candidates
    if previous_color in COLORS:
        offset = COLORS.index(previous_color) + 1
        order = (*COLORS[offset:], *COLORS[:offset])
        candidates.sort(key=lambda entry: (order.index(entry[1]) if entry[1] in order else len(order), entry[0]))
    else:
        candidates.sort(key=lambda entry: entry[0])
    return candidates


def choose_ranbu_start(
    board: BoardResult, reserved_color: str | None, *, previous_color: str | None = None
) -> tuple[int, int] | None:
    """先轮换本帧可用颜色，只有同色可选时才重复；同色内按编号确定起点。"""
    candidates = _ordered_start_candidates(board, reserved_color, previous_color=previous_color)
    return candidates[0][2] if candidates else None


def _choose_pixel_start(
    board: BoardResult, image: Image, reserved_color: str | None, previous_color: str | None
) -> tuple[tuple[int, int], str, tuple[float, float]] | None:
    """同色内优先本帧颜色余量更大的点，确认帧仍只核对原落点。"""
    objects = {item.id: item for item in board.objects}
    best: tuple[tuple[int, int], str, tuple[float, float]] | None = None
    best_margin = -1
    for object_id, color, point in _ordered_start_candidates(board, reserved_color, previous_color=previous_color):
        if best is not None and color != best[1]:
            break
        if color not in COLORS or ghoul_color_at_point(board, point) != color:
            continue
        size = objects[object_id].body_box[2:]
        if identify_ranbu_point_color(image, point, size) == color:
            margin = ranbu_point_color_margin(image, point, size, color)
            if margin > best_margin:
                best, best_margin = (point, color, size), margin
            if margin == 4:
                return best
    return best


def ghoul_color_at_point(board: BoardResult, point: tuple[int, int]) -> str | None:
    """读取当前帧中预选落点上的普通饿鬼颜色，证据不足时返回未知。"""
    px, py = point
    width, height = board.source_size
    bx, by, bw, bh = board.board_region
    if not (0 <= px < width and 0 <= py < height and bx <= px < bx + bw and by <= py < by + bh):
        return None
    colors: set[str] = set()
    for item in board.objects:
        if not all(math.isfinite(value) for value in (*item.center, *item.body_box)):
            continue
        ow, oh = item.body_box[2:]
        if min(ow, oh) <= 0:
            continue
        central = ((px - item.center[0]) / (ow * 0.3)) ** 2 + ((py - item.center[1]) / (oh * 0.3)) ** 2 <= 1
        visible = (
            item.candidate_point is not None
            and all(math.isfinite(value) for value in item.candidate_point)
            and point == tuple(round(value) for value in item.candidate_point)
        )
        if not central and not visible:
            continue
        if (
            item.kind != "ghoul"
            or item.color is None
            or item.color not in COLORS
            or item.quality not in ("identified", "single_feature")
            or item.candidate_point is None
        ):
            return None
        colors.add(item.color)
    return next(iter(colors)) if len(colors) == 1 else None


class RanbuStartConfirmation:
    """只保存待确认点与颜色，不将上一帧对象提供给识别器。"""

    def __init__(self) -> None:
        self._pending: tuple[tuple[int, int], str] | None = None
        self._pending_size: tuple[float, float] | None = None
        self._frame_id: str | None = None
        self._layout: tuple[tuple[int, int], tuple[float, float, float, float], str] | None = None
        self._last_color: str | None = None
        self.last_report: dict[str, object] = {}
        self.confirmed_body_size: tuple[float, float] | None = None

    def reset(self) -> None:
        """状态中断或动作后清空待确认点。"""
        self._pending = None
        self._pending_size = None
        self._frame_id = None
        self._layout = None
        self.confirmed_body_size = None

    def pending_layout_for(self, image: Image) -> tuple[tuple[float, float, float, float], str] | None:
        """只提供已核对的布局元数据，确认帧不重新识别整盘或复用对象。"""
        if self._pending is None or self._layout is None or self._pending_size is None:
            return None
        source_size, region, profile = self._layout
        if (image.shape[1], image.shape[0]) != source_size:
            self.reset()
            return None
        return region, profile

    def observe(
        self, board: BoardResult | None, reserved_color: str | None, *, image: Image | None = None
    ) -> tuple[tuple[int, int], str] | None:
        """要求同一预选点在连续两个独立帧中均命中同一非保留色。"""
        self.confirmed_body_size = None
        if board is None:
            self.reset()
            self.last_report = {"stable": False, "reason": "当前帧没有棋盘识别结果"}
            return None
        layout = (board.source_size, board.board_region, board.profile)
        if self._layout != layout:
            self._pending = None
            self._pending_size = None
        if self._frame_id == board.frame_id:
            self.reset()
            self.last_report = {"stable": False, "reason": "重复使用同一帧，等待独立颜色证据"}
            return None
        self._layout = layout
        self._frame_id = board.frame_id
        previous = self._pending
        current_color = None
        if previous is not None:
            current_color = (
                identify_ranbu_point_color(image, previous[0], self._pending_size)
                if image is not None and self._pending_size is not None
                else ghoul_color_at_point(board, previous[0])
                if image is None
                else None
            )
        if previous is not None and current_color == previous[1] and current_color != reserved_color:
            self.confirmed_body_size = self._pending_size
            self._pending = None
            self._pending_size = None
            self._last_color = current_color
            self.last_report = {
                "stable": True,
                "reason": "同一预选点连续两帧命中同一非保留色",
                "point": previous[0],
                "color": current_color,
            }
            return previous
        if image is not None and previous is not None and current_color in COLORS and current_color != reserved_color:
            self._pending = (previous[0], current_color)
            self.last_report = {
                "stable": False,
                "reason": "预选点变为另一非保留色，重新累计连续颜色证据",
                "point": previous[0],
                "color": current_color,
            }
            return None
        start = choose_ranbu_start(board, reserved_color, previous_color=self._last_color)
        color = ghoul_color_at_point(board, start) if start is not None else None
        size = None
        if image is not None:
            selected = _choose_pixel_start(board, image, reserved_color, self._last_color)
            if selected is not None:
                start, color, size = selected
            else:
                start, color = None, None
        self._pending = (
            (start, color)
            if start is not None and color is not None and color in COLORS and color != reserved_color
            else None
        )
        self.last_report = {
            "stable": False,
            "reason": "等待预选点的下一帧同色证据" if self._pending else "当前帧没有可信的非保留色起点",
            "previous_point": previous[0] if previous else None,
            "previous_color": previous[1] if previous else None,
            "observed_color": current_color,
            "point": self._pending[0] if self._pending else None,
            "color": self._pending[1] if self._pending else None,
        }
        self._pending_size = size if self._pending else None
        return None


def ranbu_short_link_points(
    board: BoardResult, reserved_color: str | None
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """普通饿鬼全部为保留色时，选当前帧内距离最近的两个可信触点。"""
    if reserved_color not in COLORS or any(item.kind == "special_ghoul" for item in board.objects):
        return None
    ghouls = sorted((item for item in board.objects if item.kind == "ghoul"), key=lambda item: item.id)
    if len(ghouls) < 2 or len({item.id for item in ghouls}) != len(ghouls):
        return None
    x, y, width, height = board.board_region
    source_width, source_height = board.source_size
    points: list[tuple[int, int]] = []
    for item in ghouls:
        if item.color != reserved_color or item.quality not in ("identified", "single_feature"):
            return None
        point = item.candidate_point or item.center
        if not all(math.isfinite(value) for value in point):
            return None
        px, py = round(point[0]), round(point[1])
        if not (0 <= px < source_width and 0 <= py < source_height and x <= px < x + width and y <= py < y + height):
            return None
        points.append((px, py))
    pairs = [pair for pair in combinations(points, 2) if pair[0] != pair[1]]
    return min(pairs, key=lambda pair: math.dist(*pair)) if pairs else None


def ranbu_sweep_points(board: BoardResult) -> tuple[tuple[int, int], ...]:
    """只按棋盘边界生成密集完整折返，不读取任何饿鬼位置或颜色。"""
    x, y, width, height = board.board_region
    source_width, source_height = board.source_size
    if (
        not all(math.isfinite(value) for value in board.board_region)
        or min(width, height, source_width, source_height) <= 0
    ):
        raise ValueError("乱舞棋盘边界必须为有限正尺寸")
    left, right = max(0, math.ceil(x)), min(source_width - 1, math.ceil(x + width) - 1)
    top, bottom = max(0, math.ceil(y)), min(source_height - 1, math.ceil(y + height) - 1)
    if left >= right or top >= bottom:
        raise ValueError("乱舞棋盘必须在截图内具有有效扫动范围")
    # 扫描间距按画面短边缩放，覆盖上下边缘；这只是手势采样，不代表固定棋盘行列。
    spacing = max(1.0, min(source_width, source_height) * 0.05)
    lanes = math.ceil((bottom - top) / spacing) + 1
    lane_heights = [round(top + (bottom - top) * index / (lanes - 1)) for index in range(lanes)]
    points: list[tuple[int, int]] = []
    for index, lane_y in enumerate(lane_heights):
        lane = ((left, lane_y), (right, lane_y))
        points.extend(lane if index % 2 == 0 else reversed(lane))
    return tuple(points)
