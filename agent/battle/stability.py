"""比较独立识别的相邻棋盘，为普通连线提供全盘稳定性约束。"""

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
        if item.candidate_point is not None:
            x, y = item.candidate_point
            ox, oy, ow, oh = item.body_box
            if not (ox <= x <= ox + ow and oy <= y <= oy + oh):
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
    """比较对象类别与几何，不把眼睛和嘴巴之间的合法触点切换当作位移。"""
    if (first.kind, first.color, first.bomb_level, first.subtype, first.enhanced) != (
        second.kind,
        second.color,
        second.bomb_level,
        second.subtype,
        second.enhanced,
    ):
        return False
    dimensions = (*first.body_box[2:], *second.body_box[2:])
    # 截图短边720像素时允许8像素抖动；对象短边的8%仍是上限，最低容纳一个像素取整误差。
    tolerance = max(1.0, min(screen_tolerance, min(dimensions) * 0.08))
    if math.dist(first.center, second.center) > tolerance:
        return False
    # 模板尺寸为量化估计，相邻0.9/1.0/1.1档允许切换；按较大值计10%，并容纳浮点边界。
    if any(abs(a - b) > max(a, b) * 0.1 + 1e-6 for a, b in zip(first.body_box[2:], second.body_box[2:], strict=True)):
        return False
    if first.candidate_point is None or second.candidate_point is None:
        return first.candidate_point is second.candidate_point
    # 两帧触点已分别通过当前对象范围校验，触点不必落在同一部位。
    return True


def _compare_boards(previous: BoardResult, current: BoardResult) -> tuple[bool, str]:
    """逐项比较并保留拒绝原因，不能用总数或平均位置代替对象匹配。"""
    if previous.frame_id == current.frame_id:
        return False, "重复使用同一帧，尚未取得两帧独立结果"
    if (
        previous.source_size != current.source_size
        or previous.board_region != current.board_region
        or previous.profile != current.profile
        or previous.coverage != current.coverage
    ):
        return False, "截图尺寸、棋盘区域或识别范围发生变化"
    if len(previous.objects) != len(current.objects):
        return False, f"棋盘对象数量变化：{len(previous.objects)} → {len(current.objects)}"
    if not _valid_board(previous) or not _valid_board(current):
        return False, "存在无效对象几何或触点超出当前对象范围"
    tolerance = min(current.source_size) * 8.0 / 720.0
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

    for index in sorted(range(len(candidates)), key=lambda index: len(candidates[index])):
        if not pair(index, set()):
            item = previous.objects[index]
            return False, f"上一帧对象 {item.id}（{item.label}）未找到位置、类别和尺寸均相近的独立对应项"
    return True, "连续两帧对象几何稳定"


def boards_are_similar(previous: BoardResult, current: BoardResult) -> bool:
    """要求同范围的两张独立结果中全部对象一一对应。"""
    return _compare_boards(previous, current)[0]


def _board_summary(board: BoardResult | None) -> dict[str, object] | None:
    """只保存排障所需的几何与类别，不复制模板证据或图像。"""
    if board is None:
        return None

    def coordinates(values: tuple[float, ...] | None) -> list[float | None] | None:
        """无效坐标记为空值，诊断文件不能因非有限值而阻断安全等待。"""
        return None if values is None else [value if math.isfinite(value) else None for value in values]

    return {
        "frame_id": board.frame_id,
        "source_size": board.source_size,
        "board_region": coordinates(board.board_region),
        "coverage": board.coverage,
        "objects": [
            {
                "id": item.id,
                "kind": item.kind,
                "color": item.color,
                "bomb_level": item.bomb_level,
                "subtype": item.subtype,
                "enhanced": item.enhanced,
                "center": coordinates(item.center),
                "body_box": coordinates(item.body_box),
                "candidate_point": coordinates(item.candidate_point),
            }
            for item in board.objects
        ],
    }


class NormalLinkStability:
    """仅保留上一份独立结果作比较，不向识别器提供旧坐标或旧对象。"""

    def __init__(self) -> None:
        self._previous: BoardResult | None = None
        self.last_report: dict[str, object] = {}

    def reset(self) -> None:
        """状态中断或操作后，必须重新取得两帧稳定证据。"""
        self._previous = None

    def observe(self, board: BoardResult | None) -> bool:
        """每次以当前帧替换比较基准，只判断相邻两次结果。"""
        previous = self._previous
        if board is None or not _valid_board(board):
            self.reset()
            stable, reason = False, "当前棋盘缺失或存在无效对象几何、触点"
        else:
            stable, reason = (
                _compare_boards(previous, board) if previous is not None else (False, "仅取得一帧有效棋盘，等待下一帧")
            )
            self._previous = board
        self.last_report = {
            "stable": stable,
            "reason": reason,
            "previous": _board_summary(previous),
            "current": _board_summary(board),
        }
        return stable
