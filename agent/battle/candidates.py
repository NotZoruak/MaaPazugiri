"""从当前帧识别对象生成自由布局连接候选与有界路线预演。"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations, pairwise

from vision.connectability import ConnectabilityResult, ConnectabilityStatus, assess_ghoul_connectability
from vision.model import BoardObject, BoardResult, is_black_ghoul, is_linkable_ghoul


@dataclass(frozen=True)
class CandidateRoute:
    """保存尚未经游戏确认的同色候选路线。"""

    color: str
    object_ids: tuple[int, ...]
    points: tuple[tuple[float, float], ...]
    minimum_clearance: float
    total_distance: float


@dataclass
class RoutePreview:
    """记录同帧候选图、拟选路线及其未验证范围。"""

    frame_id: str
    status: str = "no_route"
    candidate_edges: list[ConnectabilityResult] = field(default_factory=list)
    uncertain_edges: list[ConnectabilityResult] = field(default_factory=list)
    rejected_pairs: int = 0
    excluded_objects: list[dict[str, object]] = field(default_factory=list)
    routes: list[CandidateRoute] = field(default_factory=list)
    selected_route: CandidateRoute | None = None
    search_expansions: int = 0
    search_truncated: bool = False
    automatic_action_allowed: bool = False
    game_connection_verified: bool = False
    path_clearance_verified: bool = False
    issues: list[str] = field(default_factory=lambda: ["仅为几何候选，未确认游戏接入和障碍物穿越规则"])


def _point_inside(point: tuple[float, float] | None, board: BoardResult) -> bool:
    """只接受当前帧有效棋盘范围内的有限坐标。"""
    if point is None or len(point) != 2 or not all(math.isfinite(value) for value in point):
        return False
    x, y = point
    width, height = board.source_size
    bx, by, bw, bh = board.board_region
    return (
        0 <= x < width
        and 0 <= y < height
        and bx <= x < bx + bw
        and by <= y < by + bh
        and not any(rx <= x <= rx + rw and ry <= y <= ry + rh for rx, ry, rw, rh in board.route_exclusion_regions)
    )


def _segment_avoids_exclusions(first: tuple[float, float], second: tuple[float, float], board: BoardResult) -> bool:
    """检查整条线段，端点在区域外仍不能穿越或擦过不确定区域。"""
    for x, y, width, height in board.route_exclusion_regions:
        begin, end = 0.0, 1.0
        for origin, delta, lower, upper in (
            (first[0], second[0] - first[0], x, x + width),
            (first[1], second[1] - first[1], y, y + height),
        ):
            if abs(delta) < 1e-9:
                if not lower <= origin <= upper:
                    break
            else:
                enter, leave = sorted(((lower - origin) / delta, (upper - origin) / delta))
                begin, end = max(begin, enter), min(end, leave)
                if begin > end:
                    break
        else:
            return False
    return True


def _route_rank(route: CandidateRoute) -> tuple[int, int, float, float, tuple[int, ...]]:
    """黑色优先，再比较对象数量、几何余量、路线长度和编号顺序。"""
    return (
        0 if route.color == "black" else 1,
        -len(route.object_ids),
        -route.minimum_clearance,
        route.total_distance,
        route.object_ids,
    )


def preferred_ghoul_touch_point(item: BoardObject, board: BoardResult) -> tuple[float, float]:
    """优先当前身体中心；被本帧炸弹外框覆盖时保留识别器已排除炸弹的可见触点。"""
    assert item.candidate_point is not None
    if not _point_inside(item.center, board) or not _point_inside(
        (round(item.center[0]), round(item.center[1])), board
    ):
        return item.candidate_point
    for other in board.objects:
        if other.kind != "bomb":
            continue
        x, y, width, height = other.body_box
        if not all(math.isfinite(value) for value in other.body_box) or min(width, height) <= 0:
            return item.candidate_point
        if x <= item.center[0] < x + width and y <= item.center[1] < y + height:
            return item.candidate_point
    return item.center


def build_route_preview(
    board: BoardResult, *, beam_width: int = 32, max_expansions: int = 4096, black_only: bool = False
) -> RoutePreview:
    """按剩余连接数量优先有限回溯；宽度限制每步分支，结果不保证全局最长。"""
    if beam_width < 1 or max_expansions < 1:
        raise ValueError("候选搜索宽度和总扩展上限必须为正整数")
    result = RoutePreview(board.frame_id)
    if any(
        len(region) != 4 or not all(math.isfinite(value) for value in region) or min(region[2:]) <= 0
        for region in board.route_exclusion_regions
    ):
        result.status = "invalid_board"
        result.issues.append("规划避让区域无效，本帧不生成路线")
        return result
    if len({item.id for item in board.objects}) != len(board.objects):
        result.status = "invalid_board"
        result.issues.append("当前帧对象编号不唯一，不能关联候选路线")
        return result

    eligible: dict[int, BoardObject] = {}
    points: dict[int, tuple[float, float]] = {}
    for item in sorted(board.objects, key=lambda obj: obj.id):
        reason = None
        if not is_linkable_ghoul(item) or black_only and not is_black_ghoul(item):
            reason = "不是当前检查范围内已确认连接规则的饿鬼"
        elif item.quality not in ("identified", "single_feature"):
            reason = "对象识别质量不足"
        elif not _point_inside(item.center, board) or not _point_inside(item.candidate_point, board):
            reason = "缺少位于当前棋盘范围内的有效中心或候选触点"
        if reason is not None:
            result.excluded_objects.append({"object_id": item.id, "reason": reason})
            continue
        assert item.candidate_point is not None
        point = preferred_ghoul_touch_point(item, board)
        if not _point_inside((round(point[0]), round(point[1])), board):
            result.excluded_objects.append({"object_id": item.id, "reason": "实际触点取整后不在截图或棋盘内"})
            continue
        eligible[item.id] = item
        points[item.id] = point

    adjacency: dict[int, set[int]] = {identifier: set() for identifier in eligible}
    clearances: dict[frozenset[int], float] = {}
    for first, second in combinations(eligible.values(), 2):
        if first.color != second.color:
            continue
        assessment = assess_ghoul_connectability(board, first.id, second.id)
        if assessment.status == ConnectabilityStatus.CANDIDATE:
            if not _segment_avoids_exclusions(
                (round(points[first.id][0]), round(points[first.id][1])),
                (round(points[second.id][0]), round(points[second.id][1])),
                board,
            ):
                result.rejected_pairs += 1
                continue
            result.candidate_edges.append(assessment)
            adjacency[first.id].add(second.id)
            adjacency[second.id].add(first.id)
            clearances[frozenset((first.id, second.id))] = min(assessment.edge_clearances)
        elif assessment.status == ConnectabilityStatus.UNCERTAIN:
            result.uncertain_edges.append(assessment)
        else:
            result.rejected_pairs += 1

    def make_route(path: tuple[int, ...]) -> CandidateRoute:
        """把搜索路线整理为当前帧触点，不推断触控有效性。"""
        path = min(path, tuple(reversed(path)))
        pairs = list(pairwise(path))
        color = eligible[path[0]].color
        assert color is not None
        return CandidateRoute(
            color=color,
            object_ids=path,
            points=tuple(points[identifier] for identifier in path),
            minimum_clearance=min(clearances[frozenset(pair)] for pair in pairs),
            total_distance=round(sum(math.dist(points[first], points[second]) for first, second in pairs), 3),
        )

    def choices(endpoint: int, used: frozenset[int]) -> Iterator[int]:
        """先访问剩余连接少的对象，避免提前封死端点和窄分支。"""
        neighbors = sorted(adjacency[endpoint] - used, key=lambda node: (len(adjacency[node] - used), node))
        result.search_truncated |= len(neighbors) > beam_width
        return iter(neighbors[:beam_width])

    components: list[set[int]] = []
    remaining = {identifier for identifier, neighbors in adjacency.items() if neighbors}
    while remaining:
        pending = [min(remaining)]
        component: set[int] = set()
        while pending:
            identifier = pending.pop()
            if identifier in component:
                continue
            component.add(identifier)
            pending.extend(sorted(adjacency[identifier] - component, reverse=True))
        remaining.difference_update(component)
        components.append(component)
    for component in sorted(
        components, key=lambda nodes: (0 if eligible[min(nodes)].color == "black" else 1, -len(nodes), min(nodes))
    ):
        seeds = sorted(
            {(first, second) for first in component for second in adjacency[first] if first < second},
            key=lambda path: _route_rank(make_route(path)),
        )
        best = make_route(seeds[0])
        # 单条路线至多拥有两个度数为一的对象，其余叶端至少要舍弃。
        leaves = sum(len(adjacency[node]) == 1 for node in component)
        upper_bound = len(component) - max(0, leaves - 2)
        for start in sorted(component, key=lambda node: (len(adjacency[node]), node)):
            if len(best.object_ids) >= upper_bound or result.search_expansions >= max_expansions:
                break
            used = frozenset((start,))
            stack: list[tuple[tuple[int, ...], frozenset[int], Iterator[int]]] = [
                ((start,), used, choices(start, used))
            ]
            while stack and result.search_expansions < max_expansions:
                path, used, neighbors = stack[-1]
                neighbor = next(neighbors, None)
                if neighbor is None:
                    stack.pop()
                    continue
                result.search_expansions += 1
                extension = (*path, neighbor)
                if len(extension) >= len(best.object_ids):
                    best = min(best, make_route(extension), key=_route_rank)
                if len(best.object_ids) >= upper_bound:
                    break
                next_used = used | {neighbor}
                stack.append((extension, next_used, choices(neighbor, next_used)))
        result.search_truncated |= result.search_expansions >= max_expansions and len(best.object_ids) < upper_bound
        result.routes.append(best)

    result.routes.sort(key=_route_rank)
    if result.routes:
        result.status = "candidate"
        result.selected_route = result.routes[0]
    return result
