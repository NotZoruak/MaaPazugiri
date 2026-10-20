"""比较独立识别的相邻棋盘，仅为普通连线提供稳定性约束。"""

from __future__ import annotations

import math

from vision.model import BoardObject, BoardResult


def _valid_board(board: BoardResult) -> bool:
    """空棋盘、重复编号或无效几何不能构成稳定证据。"""
    width, height = board.source_size
    bx, by, bw, bh = board.board_region
    if (
        width <= 0
        or height <= 0
        or not all(math.isfinite(value) for value in board.board_region)
        or bw <= 0
        or bh <= 0
        or not board.objects
        or len({item.id for item in board.objects}) != len(board.objects)
    ):
        return False
    for item in board.objects:
        if not all(math.isfinite(value) for value in (*item.center, *item.body_box)):
            return False
        if item.body_box[2] <= 0 or item.body_box[3] <= 0:
            return False
        for point in (item.center, item.candidate_point):
            if point is not None and (
                not all(math.isfinite(value) for value in point)
                or not (0 <= point[0] < width and 0 <= point[1] < height)
                or not (bx <= point[0] < bx + bw and by <= point[1] < by + bh)
            ):
                return False
    return True


def _can_pair(first: BoardObject, second: BoardObject, screen_tolerance: float) -> bool:
    """忽略单帧编号及模板分数波动，核对对象类别、位置、尺寸和触点。"""
    if (first.kind, first.color, first.bomb_level, first.subtype, first.enhanced) != (
        second.kind,
        second.color,
        second.bomb_level,
        second.subtype,
        second.enhanced,
    ):
        return False
    dimensions = (*first.body_box[2:], *second.body_box[2:])
    # 位移容差同时受截图短边的0.5%和对象短边的8%约束，最低容纳一个像素取整误差。
    tolerance = max(1.0, min(screen_tolerance, min(dimensions) * 0.08))
    if math.dist(first.center, second.center) > tolerance:
        return False
    if any(abs(a - b) > min(a, b) * 0.1 for a, b in zip(first.body_box[2:], second.body_box[2:], strict=True)):
        return False
    if first.candidate_point is None or second.candidate_point is None:
        return first.candidate_point is second.candidate_point
    return math.dist(first.candidate_point, second.candidate_point) <= tolerance


def boards_are_similar(previous: BoardResult, current: BoardResult) -> bool:
    """要求同范围的两张独立结果中全部对象一一对应，不能用总数或平均位置代替。"""
    if (
        previous.frame_id == current.frame_id
        or previous.source_size != current.source_size
        or previous.board_region != current.board_region
        or previous.profile != current.profile
        or previous.coverage != current.coverage
        or len(previous.objects) != len(current.objects)
        or not _valid_board(previous)
        or not _valid_board(current)
    ):
        return False
    tolerance = min(current.source_size) * 0.005
    candidates = [
        [index for index, target in enumerate(current.objects) if _can_pair(item, target, tolerance)]
        for item in previous.objects
    ]
    matched: dict[int, int] = {}

    def pair(index: int, visited: set[int]) -> bool:
        """为每个旧对象寻找独立对应项，避免多个对象重复匹配同一个目标。"""
        for target in candidates[index]:
            if target in visited:
                continue
            visited.add(target)
            if target not in matched or pair(matched[target], visited):
                matched[target] = index
                return True
        return False

    return all(pair(index, set()) for index in sorted(range(len(candidates)), key=lambda index: len(candidates[index])))


class NormalLinkStability:
    """仅保留上一份独立结果作比较，不向识别器提供旧坐标或旧对象。"""

    def __init__(self) -> None:
        self._previous: BoardResult | None = None

    def reset(self) -> None:
        """状态中断或操作后，必须重新取得两帧稳定证据。"""
        self._previous = None

    def observe(self, board: BoardResult | None) -> bool:
        """每次以当前帧替换比较基准，只判断相邻两次结果。"""
        if board is None or not _valid_board(board):
            self.reset()
            return False
        stable = self._previous is not None and boards_are_similar(self._previous, board)
        self._previous = board
        return stable
