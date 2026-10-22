"""根据当前帧饿鬼的位置估计连线候选关系。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .model import COLORS, BoardObject, BoardResult, is_black_ghoul


class ConnectabilityStatus(StrEnum):
    """表示静态几何判定的结论。"""

    CANDIDATE = "candidate"
    REJECTED = "rejected"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class ConnectabilityParameters:
    """保存由当前开发样例校准的八边形参数。

    支撑上限根据三张截图中的已标注样例拟合，后续仍需实战样本校准。
    """

    rotation_degrees: float = 7.8
    support_limits: tuple[float, float, float, float] = (1.463, 1.058, 1.284, 1.380)
    uncertainty_band: float = 0.04


@dataclass(frozen=True, slots=True)
class ConnectabilityResult:
    """保存一对饿鬼的静态连线几何结论与诊断值。"""

    frame_id: str
    first_id: int
    second_id: int
    status: ConnectabilityStatus
    reason: str
    normalized_offset: tuple[float, float] | None = None
    direction_degrees: float | None = None
    edge_clearances: tuple[float, ...] = ()
    polygons_overlap: bool | None = None


DEFAULT_CONNECTABILITY_PARAMETERS = ConnectabilityParameters()


def _result(
    board: BoardResult,
    first_id: int,
    second_id: int,
    status: ConnectabilityStatus,
    reason: str,
) -> ConnectabilityResult:
    """构造不含几何测量值的判定结果。"""
    return ConnectabilityResult(board.frame_id, first_id, second_id, status, reason)


def _find_object(board: BoardResult, identifier: int) -> BoardObject | None:
    """仅在当前帧查找唯一对象。"""
    matches = [item for item in board.objects if item.id == identifier]
    if len(matches) != 1:
        return None
    return matches[0]


def _finite_positive_size(value: object) -> float | None:
    """读取有限的正尺寸。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _geometry(object_: BoardObject) -> tuple[float, float, float, float] | None:
    """读取当前对象中心和尺寸，并拒绝无效数值。"""
    if len(object_.center) != 2:
        return None
    center_x, center_y = object_.center
    if not math.isfinite(center_x) or not math.isfinite(center_y):
        return None
    width = _finite_positive_size(object_.size.get("width"))
    height = _finite_positive_size(object_.size.get("height"))
    if width is None or height is None:
        return None
    return float(center_x), float(center_y), width, height


def assess_ghoul_connectability(
    board: BoardResult,
    first_id: int,
    second_id: int,
    *,
    parameters: ConnectabilityParameters = DEFAULT_CONNECTABILITY_PARAMETERS,
) -> ConnectabilityResult:
    """用相对位移八边形判断同帧两只普通饿鬼是否可作为连线候选。

    八边形边界代表两只饿鬼各自扩大后的连接范围是否重叠。边界不确定带
    会把识别误差附近的组合保留为未知；候选结论仍需在拖动时由目标上的
    大菱形确认。
    """
    if first_id == second_id:
        return _result(board, first_id, second_id, ConnectabilityStatus.REJECTED, "起点和终点是同一个对象")
    first = _find_object(board, first_id)
    second = _find_object(board, second_id)
    if first is None or second is None:
        return _result(board, first_id, second_id, ConnectabilityStatus.UNCERTAIN, "当前帧缺少唯一的指定对象")
    black_pair = is_black_ghoul(first) and is_black_ghoul(second)
    if not black_pair and (first.kind != "ghoul" or second.kind != "ghoul"):
        return _result(board, first_id, second_id, ConnectabilityStatus.REJECTED, "只处理普通同色饿鬼或黑黑连接")
    if not black_pair and (first.color not in COLORS or second.color not in COLORS):
        return _result(board, first_id, second_id, ConnectabilityStatus.UNCERTAIN, "饿鬼颜色未确认")
    if first.color != second.color:
        return _result(board, first_id, second_id, ConnectabilityStatus.REJECTED, "两只饿鬼颜色不同")
    if first.quality not in ("identified", "single_feature") or second.quality not in ("identified", "single_feature"):
        return _result(board, first_id, second_id, ConnectabilityStatus.UNCERTAIN, "对象识别质量不足")

    first_geometry = _geometry(first)
    second_geometry = _geometry(second)
    if first_geometry is None or second_geometry is None:
        return _result(board, first_id, second_id, ConnectabilityStatus.UNCERTAIN, "对象中心或尺寸无效")

    first_x, first_y, first_width, first_height = first_geometry
    second_x, second_y, second_width, second_height = second_geometry
    average_width = (first_width + second_width) / 2
    average_height = (first_height + second_height) / 2
    offset_x = (second_x - first_x) / average_width
    offset_y = (second_y - first_y) / average_height
    direction = math.degrees(math.atan2(offset_y, offset_x)) % 180

    if (
        len(parameters.support_limits) != 4
        or not math.isfinite(parameters.rotation_degrees)
        or not math.isfinite(parameters.uncertainty_band)
        or parameters.uncertainty_band < 0
        or any(not math.isfinite(limit) or limit <= 0 for limit in parameters.support_limits)
    ):
        return _result(board, first_id, second_id, ConnectabilityStatus.UNCERTAIN, "连线几何参数无效")
    clearances = tuple(
        limit
        - abs(
            offset_x * math.cos(math.radians(parameters.rotation_degrees + index * 45))
            + offset_y * math.sin(math.radians(parameters.rotation_degrees + index * 45))
        )
        for index, limit in enumerate(parameters.support_limits)
    )
    minimum_clearance = min(clearances)
    overlaps = minimum_clearance >= 0
    if minimum_clearance > parameters.uncertainty_band:
        status = ConnectabilityStatus.CANDIDATE
        reason = "中心位移位于八边形连接范围内且离边界较远"
    elif minimum_clearance < -parameters.uncertainty_band:
        status = ConnectabilityStatus.REJECTED
        reason = "中心位移超出八边形连接范围"
    else:
        status = ConnectabilityStatus.UNCERTAIN
        reason = "中心位移靠近八边形边界"
    return ConnectabilityResult(
        board.frame_id,
        first_id,
        second_id,
        status,
        reason,
        normalized_offset=(round(offset_x, 4), round(offset_y, 4)),
        direction_degrees=round(direction, 2),
        edge_clearances=tuple(round(value, 4) for value in clearances),
        polygons_overlap=overlaps,
    )
