"""从当前帧识别对象生成自由布局连接候选与有界路线预演。"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import combinations, pairwise

from vision.connectability import ConnectabilityResult, ConnectabilityStatus, assess_ghoul_connectability
from vision.model import COLORS, BoardObject, BoardResult


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
    return 0 <= x < width and 0 <= y < height and bx <= x < bx + bw and by <= y < by + bh


def _route_rank(route: CandidateRoute) -> tuple[int, float, float, tuple[int, ...]]:
    """先比较对象数量，再比较几何余量、路线长度和编号顺序。"""
    return (-len(route.object_ids), -route.minimum_clearance, route.total_distance, route.object_ids)


def build_route_preview(board: BoardResult, *, beam_width: int = 32, max_expansions: int = 4096) -> RoutePreview:
    """按剩余连接数量优先有限回溯；宽度限制每步分支，结果不保证全局最长。"""
    if beam_width < 1 or max_expansions < 1:
        raise ValueError("候选搜索宽度和总扩展上限必须为正整数")
    result = RoutePreview(board.frame_id)
    if len({item.id for item in board.objects}) != len(board.objects):
        result.status = "invalid_board"
        result.issues.append("当前帧对象编号不唯一，不能关联候选路线")
        return result

    eligible: dict[int, BoardObject] = {}
    points: dict[int, tuple[float, float]] = {}
    for item in sorted(board.objects, key=lambda obj: obj.id):
        reason = None
        if item.kind != "ghoul" or item.color not in COLORS:
            reason = "不是已确认颜色的普通饿鬼"
        elif item.quality not in ("identified", "single_feature"):
            reason = "对象识别质量不足"
        elif not _point_inside(item.center, board) or not _point_inside(item.candidate_point, board):
            reason = "缺少位于当前棋盘范围内的有效中心或候选触点"
        if reason is not None:
            result.excluded_objects.append({"object_id": item.id, "reason": reason})
            continue
        assert item.candidate_point is not None
        eligible[item.id] = item
        points[item.id] = item.candidate_point

    adjacency: dict[int, set[int]] = {identifier: set() for identifier in eligible}
    clearances: dict[frozenset[int], float] = {}
    for first, second in combinations(eligible.values(), 2):
        if first.color != second.color:
            continue
        assessment = assess_ghoul_connectability(board, first.id, second.id)
        if assessment.status == ConnectabilityStatus.CANDIDATE:
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
    for component in sorted(components, key=lambda nodes: (-len(nodes), min(nodes))):
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
