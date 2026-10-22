"""比较独立识别的相邻棋盘，为普通连线提供全盘稳定性约束。"""

from __future__ import annotations

import math
from dataclasses import replace
from statistics import median

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


def _bottom_band_top(board: BoardResult) -> float:
    """根据本帧饿鬼尺寸限定底部容差带，不预设角色位置或棋盘格。"""
    heights = [item.body_box[3] for item in board.objects if item.kind == "ghoul"]
    _, by, _, bh = board.board_region
    depth = min(bh * 0.2, median(heights) * 1.25) if heights else 0.0
    return by + bh - depth


def _compare_boards(previous: BoardResult, current: BoardResult) -> tuple[bool, str, float | None]:
    """逐项比较并保留拒绝原因，不能用总数或平均位置代替对象匹配。"""
    if previous.frame_id == current.frame_id:
        return False, "重复使用同一帧，尚未取得两帧独立结果", None
    if (
        previous.source_size != current.source_size
        or previous.board_region != current.board_region
        or previous.profile != current.profile
        or previous.coverage != current.coverage
    ):
        return False, "截图尺寸、棋盘区域或识别范围发生变化", None
    if not _valid_board(previous) or not _valid_board(current):
        return False, "存在无效对象几何或触点超出当前对象范围", None
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
        pair(index, set())
    paired_previous = set(matched.values())
    missing = [item for index, item in enumerate(previous.objects) if index not in paired_previous]
    added = [item for index, item in enumerate(current.objects) if index not in matched]
    if not missing and not added:
        return True, "连续两帧对象几何稳定", None

    band_top = _bottom_band_top(current)
    # 每帧至多一个底部普通四色饿鬼存在差异，且必须有其余对象的独立稳定证据。
    tolerated = (
        len(matched) >= 2
        and len(missing) <= 1
        and len(added) <= 1
        and all(
            item.kind == "ghoul" and item.color in {"red", "yellow", "green", "blue"} and item.center[1] >= band_top
            for item in (*missing, *added)
        )
    )
    if tolerated and missing and added:
        # 不能把底部相隔很远的消失与新增当作同一只饿鬼的检测抖动。
        tolerated = (
            math.dist(missing[0].center, added[0].center) <= min(*missing[0].body_box[2:], *added[0].body_box[2:]) * 0.5
        )
    if tolerated:
        return True, "底部单只饿鬼识别差异，其余对象稳定；本轮连线避开底部", band_top
    if len(previous.objects) != len(current.objects):
        return False, f"棋盘对象数量变化：{len(previous.objects)} → {len(current.objects)}", None
    item = missing[0]
    return False, f"上一帧对象 {item.id}（{item.label}）未找到位置、类别和尺寸均相近的独立对应项", None


def boards_are_similar(previous: BoardResult, current: BoardResult) -> bool:
    """比较同范围独立结果，仅允许底部单只普通饿鬼的局部误差。"""
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
        self.route_board: BoardResult | None = None
        self.last_report: dict[str, object] = {}

    def reset(self) -> None:
        """状态中断或操作后，必须重新取得两帧稳定证据。"""
        self._previous = None
        self.route_board = None

    def observe(self, board: BoardResult | None) -> bool:
        """每次以当前帧替换比较基准，只判断相邻两次结果。"""
        previous = self._previous
        self.route_board = None
        band_top = None
        if board is None or not _valid_board(board):
            self.reset()
            stable, reason = False, "当前棋盘缺失或存在无效对象几何、触点"
        else:
            stable, reason, band_top = (
                _compare_boards(previous, board)
                if previous is not None
                else (False, "仅取得一帧有效棋盘，等待下一帧", None)
            )
            self._previous = board
            if stable:
                self.route_board = board
                if band_top is not None:
                    bx, by, bw, _ = board.board_region
                    # 只从本帧生成规划视图；裁剪区域保证路线的直线段也不会穿入不确定带。
                    self.route_board = replace(
                        board,
                        board_region=(bx, by, bw, band_top - by),
                        objects=[
                            item
                            for item in board.objects
                            if item.kind != "ghoul" or item.body_box[1] + item.body_box[3] < band_top
                        ],
                    )
        self.last_report = {
            "stable": stable,
            "reason": reason,
            "bottom_exclusion_y": band_top,
            "previous": _board_summary(previous),
            "current": _board_summary(board),
        }
        return stable
