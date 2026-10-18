"""定义离线识别结果和模板命中证据。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

COLORS = ("red", "yellow", "blue", "green")
OBSTACLE_TYPES = {"shell": "贝壳", "rock": "石头", "box": "箱子", "branch": "树枝", "bronze_bell": "青铜钟"}
SPECIAL_GHOUL_TYPES = {"black": "特殊黑色饿鬼"}
EXPECTED_LABELS = (
    *("ghoul_" + color for color in COLORS),
    "bomb_1",
    "bomb_2",
    "bomb_3",
    *("obstacle_" + subtype for subtype in OBSTACLE_TYPES),
    *("special_ghoul_" + subtype for subtype in SPECIAL_GHOUL_TYPES),
)
LABEL_NAMES = {
    "ghoul_red": "红色饿鬼",
    "ghoul_yellow": "黄色饿鬼",
    "ghoul_blue": "蓝色饿鬼",
    "ghoul_green": "绿色饿鬼",
    "bomb_1": "一级炸弹",
    "bomb_2": "二级炸弹",
    "bomb_3": "三级炸弹",
    "bomb_unknown": "炸弹等级未知",
    **{"obstacle_" + subtype: name for subtype, name in OBSTACLE_TYPES.items()},
    **{"special_ghoul_" + subtype: name for subtype, name in SPECIAL_GHOUL_TYPES.items()},
}


@dataclass
class Hit:
    """保存参考尺寸坐标中的局部命中。"""

    template_id: str
    kind: str
    color: str | None
    bomb_level: int | None
    part: str
    center: tuple[float, float]
    body_size: tuple[float, float]
    anchor: tuple[float, float]
    score: float
    color_fraction: float | None = None
    accepted: bool = True
    subtype: str | None = None
    enhanced: bool = False
    minimum_parts: int = 1


@dataclass
class BoardObject:
    """保存对象估计和可见证据，不表示已经允许自动操作。"""

    id: int
    kind: str
    color: str | None
    bomb_level: int | None
    label: str
    name: str
    center: tuple[float, float]
    body_box: tuple[float, float, float, float]
    size: dict[str, Any]
    evidence: list[dict[str, Any]]
    candidate_point: tuple[float, float] | None
    quality: str
    subtype: str | None = None
    enhanced: bool | None = None


@dataclass
class BoardResult:
    """单张截图的识别结果，场景稳定性默认未验证。"""

    frame_id: str
    source_size: tuple[int, int]
    board_region: tuple[float, float, float, float]
    profile: str
    coverage: dict[str, Any]
    scene_state: str = "unverified"
    objects: list[BoardObject] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    automatic_action_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """返回可直接序列化的诊断数据。"""
        return asdict(self)
