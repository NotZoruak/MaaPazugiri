"""读取 MFAAvalonia v1 作战策略并生成安全的单步决策。"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any


class StrategyFormatError(ValueError):
    """表示策略缺失、损坏或不符合已确认的格式版本。"""


class BattlePhase(StrEnum):
    """策略允许评估的战斗阶段。"""

    OPERABLE = "operable"
    RANBU = "ranbu"


@dataclass(frozen=True)
class UltimateRule:
    """一个奥义位置及其前置条件。"""

    slot: int
    enabled: bool
    condition: dict[str, Any] | None = None


@dataclass(frozen=True)
class NormalLinkSettings:
    """普通连线的颜色优先级配置。"""

    prioritize_colors: bool
    long_link_threshold: int
    color_priority: tuple[str, ...]
    color_priority_count: int = 4


@dataclass(frozen=True)
class StageActions:
    """阶段内的奥义、炸弹和普通连线设置。"""

    priority: tuple[str, ...]
    bomb_levels: tuple[int, ...]
    ultimates: tuple[UltimateRule, ...]
    normal_link: NormalLinkSettings
    priority_count: int = 4
    prioritize_actions: bool = True


@dataclass(frozen=True)
class FrenzySettings:
    """乱舞阶段额外允许的行动和保留颜色。"""

    allow_ultimates: bool
    allow_bombs: bool
    reserved_color: str | None
    continue_on_unknown_color: bool = False


@dataclass(frozen=True)
class StageTransition:
    """按数组顺序检查的阶段转变规则。"""

    id: str
    target_stage_id: str
    condition: dict[str, Any]


@dataclass(frozen=True)
class StrategyStage:
    """一组行动、乱舞与阶段转变配置。"""

    id: str
    name: str
    actions: StageActions
    frenzy: FrenzySettings
    transitions: tuple[StageTransition, ...]


@dataclass(frozen=True)
class StrategyDocument:
    """已校验的 v1 策略文档。"""

    id: str
    name: str
    initial_stage_id: str
    stages: tuple[StrategyStage, ...]

    @classmethod
    def builtin_default(cls) -> StrategyDocument:
        """构造与 MFAAvalonia 默认值一致的内置策略，不读取 JSON 文件。"""
        stage_id = "00000000000000000000000000000001"
        return cls(
            "default",
            "默认策略",
            stage_id,
            (
                StrategyStage(
                    stage_id,
                    "全程作战",
                    StageActions(
                        ("ultimate", "bomb", "enhanced_ghoul", "black_ghoul"),
                        (3,),
                        tuple(UltimateRule(slot, True) for slot in (1, 2, 3)),
                        NormalLinkSettings(False, 7, ("red", "blue", "green", "yellow")),
                    ),
                    FrenzySettings(True, True, None),
                    (),
                ),
            ),
        )

    @property
    def stage_by_id(self) -> dict[str, StrategyStage]:
        """以稳定标识查找阶段。"""
        return {stage.id: stage for stage in self.stages}


@dataclass
class StrategyMetrics:
    """当前可确认的策略条件值；None 表示未知。"""

    battle_normal_links: int = 0
    stage_normal_links: int = 0
    enemy_hp_percent: float | None = None
    ally_hp_percent: float | None = None
    ultimate_ready: Mapping[int, bool | None] = field(default_factory=dict)
    frenzy_active: bool | None = None


@dataclass(frozen=True)
class StrategyDecision:
    """策略本次提出的单一步骤，不代表游戏已确认或允许触控。"""

    kind: str
    stage_id: str
    slot: int | None = None
    level: int | None = None
    route: Any | None = None
    reserved_color: str | None = None
    transition_chain: tuple[str, ...] = ()
    transition_loop: bool = False
    reason: str = ""
    ranbu_start: tuple[int, int] | None = None
    ranbu_color: str | None = None
    ranbu_frame_id: str | None = None
    ranbu_body_size: tuple[float, float] | None = None
    ranbu_continue_on_unknown_color: bool = False


@dataclass(frozen=True)
class StageResolution:
    """保存一次观察对应的阶段评估结果，供后续行动选择直接使用。"""

    previous_stage_id: str
    stage: StrategyStage
    transition_chain: tuple[str, ...] = ()
    transition_loop: bool = False
    applied_transitions: tuple[StageTransition, ...] = ()


_ID_PATTERN = re.compile(r"^[0-9a-fA-F]{32}$")
_HP_PATTERN = re.compile(r"^\s*(\d+)\s*/\s*(\d+)\s*$")
_COLORS = ("red", "blue", "green", "yellow")
_ACTION_PRIORITIES = {"ultimate", "bomb", "enhanced_ghoul", "black_ghoul"}
_COMPARISONS = {
    "greater_or_equal",
    "less_or_equal",
    "greater",
    "less",
    "equal",
    "not_equal",
}


def _object(value: Any, required: set[str], optional: set[str], where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StrategyFormatError(f"{where}必须是对象")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise StrategyFormatError(f"{where}缺少字段：{', '.join(sorted(missing))}")
    if unknown:
        raise StrategyFormatError(f"{where}包含未知字段：{', '.join(sorted(unknown))}")
    return value


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise StrategyFormatError(f"{where}必须是非空字符串")
    return value


def _boolean(value: Any, where: str) -> bool:
    if not isinstance(value, bool):
        raise StrategyFormatError(f"{where}必须是布尔值")
    return value


def _integer(value: Any, where: str, *, minimum: int = 0, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise StrategyFormatError(f"{where}必须是大于等于{minimum}的整数")
    if maximum is not None and value > maximum:
        raise StrategyFormatError(f"{where}不能大于{maximum}")
    return value


def _guid(value: Any, where: str) -> str:
    identifier = _string(value, where)
    if not _ID_PATTERN.fullmatch(identifier):
        raise StrategyFormatError(f"{where}必须是32位GUID N格式")
    return identifier.lower()


def _validate_condition(
    condition: Any, where: str, *, depth: int = 0, allow_empty_root: bool = False, owner_slot: int | None = None
) -> dict[str, Any]:
    if depth >= 16:
        raise StrategyFormatError(f"{where}的条件嵌套超过16层")
    node = _object(
        condition,
        {"type"},
        {"match", "children", "metric", "comparison", "value", "slots", "slot_match", "expected"},
        where,
    )
    kind = _string(node["type"], f"{where}.type")
    if kind == "group":
        group = _object(node, {"type", "match", "children"}, set(), where)
        match = _string(group["match"], f"{where}.match")
        children = group["children"]
        if match not in ("all", "any") or not isinstance(children, list):
            raise StrategyFormatError(f"{where}的条件组格式无效")
        if not children and not (depth == 0 and allow_empty_root):
            raise StrategyFormatError(f"{where}不能是空条件组")
        for index, child in enumerate(children):
            _validate_condition(child, f"{where}.children[{index}]", depth=depth + 1, owner_slot=owner_slot)
        return group
    if kind != "predicate":
        raise StrategyFormatError(f"{where}.type必须是group或predicate")
    metric = _string(node.get("metric"), f"{where}.metric")
    common_numeric = {"type", "metric", "comparison", "value"}
    if metric in ("battle_normal_links", "stage_normal_links", "enemy_hp_percent", "ally_hp_percent"):
        predicate = _object(node, common_numeric, set(), where)
        comparison = _string(predicate["comparison"], f"{where}.comparison")
        if comparison not in _COMPARISONS:
            raise StrategyFormatError(f"{where}.comparison无效")
        value = predicate["value"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise StrategyFormatError(f"{where}.value必须是有限的非负数")
        if metric in ("enemy_hp_percent", "ally_hp_percent") and value > 100:
            raise StrategyFormatError(f"{where}.value血量百分比必须位于0到100之间")
        if metric in ("battle_normal_links", "stage_normal_links") and int(value) != value:
            raise StrategyFormatError(f"{where}.value手数必须是整数")
        return predicate
    if metric == "ultimate_ready":
        predicate = _object(node, {"type", "metric", "slots", "slot_match", "expected"}, set(), where)
        slots = predicate["slots"]
        if (
            not isinstance(slots, list)
            or not slots
            or any(isinstance(slot, bool) or slot not in (1, 2, 3) for slot in slots)
            or len(set(slots)) != len(slots)
        ):
            raise StrategyFormatError(f"{where}.slots无效")
        if predicate["slot_match"] != "all" or not isinstance(predicate["expected"], bool):
            raise StrategyFormatError(f"{where}必须指定所有位置的布尔状态")
        if owner_slot in slots:
            raise StrategyFormatError(f"{where}不能把所属奥义自身就绪作为前置条件")
        return predicate
    if metric == "frenzy_active":
        predicate = _object(node, {"type", "metric", "expected"}, set(), where)
        _boolean(predicate["expected"], f"{where}.expected")
        return predicate
    raise StrategyFormatError(f"{where}.metric不受支持：{metric}")


def _parse_document(data: Any, requested_id: str) -> StrategyDocument:
    root = _object(data, {"schema_version", "id", "name", "initial_stage_id", "stages"}, set(), "策略")
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise StrategyFormatError(f"不支持策略格式版本：{root['schema_version']}")
    identifier = _guid(root["id"], "策略.id")
    if identifier != requested_id.lower():
        raise StrategyFormatError("策略标识与所选文件名不一致")
    name = _string(root["name"], "策略.name")
    initial_id = _guid(root["initial_stage_id"], "策略.initial_stage_id")
    raw_stages = root["stages"]
    if not isinstance(raw_stages, list) or not raw_stages:
        raise StrategyFormatError("策略至少需要一个阶段")

    stages: list[StrategyStage] = []
    stage_ids: set[str] = set()
    transition_ids: set[str] = set()
    for stage_index, raw_stage in enumerate(raw_stages):
        prefix = f"策略.stages[{stage_index}]"
        stage = _object(raw_stage, {"id", "name", "actions", "frenzy", "transitions"}, set(), prefix)
        stage_id = _guid(stage["id"], f"{prefix}.id")
        if stage_id in stage_ids:
            raise StrategyFormatError("阶段标识重复")
        stage_ids.add(stage_id)
        stage_name = _string(stage["name"], f"{prefix}.name")

        raw_actions = _object(
            stage["actions"],
            {"priority", "bomb_levels", "ultimates", "normal_link"},
            {"priority_count", "prioritize_actions"},
            f"{prefix}.actions",
        )
        priority = raw_actions["priority"]
        if not isinstance(priority, list):
            raise StrategyFormatError(f"{prefix}.actions.priority必须是行动优先级数组")
        if (
            len(priority) == 2
            and all(isinstance(action, str) for action in priority)
            and set(priority) == {"ultimate", "bomb"}
        ):
            priority.extend(("enhanced_ghoul", "black_ghoul"))
        if len(priority) != len(_ACTION_PRIORITIES) or set(priority) != _ACTION_PRIORITIES:
            raise StrategyFormatError(f"{prefix}.actions.priority必须完整排列四种行动类别")
        priority_count = _integer(
            raw_actions.get("priority_count", 4), f"{prefix}.actions.priority_count", maximum=4
        )
        prioritize_actions = _boolean(
            raw_actions.get("prioritize_actions", True), f"{prefix}.actions.prioritize_actions"
        )
        bomb_levels = raw_actions["bomb_levels"]
        if (
            not isinstance(bomb_levels, list)
            or any(type(level) is not int or level not in (1, 2, 3) for level in bomb_levels)
            or len(set(bomb_levels)) != len(bomb_levels)
        ):
            raise StrategyFormatError(f"{prefix}.actions.bomb_levels无效")

        raw_ultimates = raw_actions["ultimates"]
        if not isinstance(raw_ultimates, list) or len(raw_ultimates) != 3:
            raise StrategyFormatError(f"{prefix}.actions.ultimates必须包含三个位置")
        ultimates: list[UltimateRule] = []
        for rule_index, raw_rule in enumerate(raw_ultimates):
            rule_prefix = f"{prefix}.actions.ultimates[{rule_index}]"
            rule = _object(raw_rule, {"slot", "enabled"}, {"condition"}, rule_prefix)
            slot = _integer(rule["slot"], f"{rule_prefix}.slot", minimum=1, maximum=3)
            if any(existing.slot == slot for existing in ultimates):
                raise StrategyFormatError("奥义位置重复")
            enabled = _boolean(rule["enabled"], f"{rule_prefix}.enabled")
            condition = None
            if "condition" in rule:
                condition = _validate_condition(
                    rule["condition"], f"{rule_prefix}.condition", allow_empty_root=True, owner_slot=slot
                )
                if condition["type"] != "group":
                    raise StrategyFormatError(f"{rule_prefix}.condition根节点必须是条件组")
            ultimates.append(UltimateRule(slot, enabled, condition))
        if {rule.slot for rule in ultimates} != {1, 2, 3}:
            raise StrategyFormatError(f"{prefix}.actions.ultimates必须各包含位置1、2、3")

        raw_normal = _object(
            raw_actions["normal_link"],
            {"prioritize_colors", "long_link_threshold", "color_priority"},
            {"color_priority_count"},
            f"{prefix}.actions.normal_link",
        )
        colors = raw_normal["color_priority"]
        threshold = _integer(raw_normal["long_link_threshold"], f"{prefix}.actions.normal_link.long_link_threshold")
        if threshold != 7 or not isinstance(colors, list) or len(colors) != 4 or set(colors) != set(_COLORS):
            raise StrategyFormatError(f"{prefix}.actions.normal_link颜色顺序或七连门槛无效")
        normal = NormalLinkSettings(
            _boolean(raw_normal["prioritize_colors"], f"{prefix}.actions.normal_link.prioritize_colors"),
            threshold,
            tuple(colors),
            _integer(
                raw_normal.get("color_priority_count", 4),
                f"{prefix}.actions.normal_link.color_priority_count",
                maximum=4,
            ),
        )

        raw_frenzy = _object(
            stage["frenzy"],
            {"allow_ultimates", "allow_bombs"},
            {"reserved_color", "continue_on_unknown_color"},
            f"{prefix}.frenzy",
        )
        reserved = raw_frenzy.get("reserved_color")
        if reserved is not None and reserved not in _COLORS:
            raise StrategyFormatError(f"{prefix}.frenzy.reserved_color无效")
        frenzy = FrenzySettings(
            _boolean(raw_frenzy["allow_ultimates"], f"{prefix}.frenzy.allow_ultimates"),
            _boolean(raw_frenzy["allow_bombs"], f"{prefix}.frenzy.allow_bombs"),
            reserved,
            _boolean(raw_frenzy.get("continue_on_unknown_color", False), f"{prefix}.frenzy.continue_on_unknown_color"),
        )

        raw_transitions = stage["transitions"]
        if not isinstance(raw_transitions, list):
            raise StrategyFormatError(f"{prefix}.transitions必须是数组")
        transitions: list[StageTransition] = []
        for transition_index, raw_transition in enumerate(raw_transitions):
            transition_prefix = f"{prefix}.transitions[{transition_index}]"
            transition = _object(raw_transition, {"id", "target_stage_id", "condition"}, set(), transition_prefix)
            transition_id = _guid(transition["id"], f"{transition_prefix}.id")
            if transition_id in transition_ids:
                raise StrategyFormatError("转变规则标识重复")
            transition_ids.add(transition_id)
            condition = _validate_condition(transition["condition"], f"{transition_prefix}.condition")
            if condition["type"] != "group":
                raise StrategyFormatError(f"{transition_prefix}.condition根节点必须是条件组")
            transitions.append(
                StageTransition(
                    transition_id,
                    _guid(transition["target_stage_id"], f"{transition_prefix}.target_stage_id"),
                    condition,
                )
            )
        stages.append(
            StrategyStage(
                stage_id,
                stage_name,
                StageActions(
                    tuple(priority), tuple(bomb_levels), tuple(ultimates), normal, priority_count, prioritize_actions
                ),
                frenzy,
                tuple(transitions),
            )
        )

    if initial_id not in stage_ids:
        raise StrategyFormatError("初始阶段不存在")
    for stage in stages:
        if any(transition.target_stage_id not in stage_ids for transition in stage.transitions):
            raise StrategyFormatError(f"阶段“{stage.name}”引用了不存在的转变目标")
    return StrategyDocument(identifier, name, initial_id, tuple(stages))


def load_strategy(strategy_id: object, config_dir: Path) -> StrategyDocument:
    """从配置目录读取选中策略；default 始终使用内置策略。"""
    if strategy_id == "default":
        return StrategyDocument.builtin_default()
    if not isinstance(strategy_id, str) or not _ID_PATTERN.fullmatch(strategy_id):
        raise StrategyFormatError("策略 ID 必须是default或32位GUID N格式")
    path = config_dir / "strategies" / f"{strategy_id}.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise StrategyFormatError(f"找不到策略文件：{path.name}") from error
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StrategyFormatError(f"策略文件无法读取：{path.name}：{error}") from error
    try:
        return _parse_document(data, strategy_id)
    except (TypeError, KeyError, InvalidOperation) as error:
        raise StrategyFormatError(f"策略格式无效：{error}") from error


@dataclass
class StrategyRuntime:
    """维护单场战斗的阶段和成功普通连线手数。"""

    strategy: StrategyDocument
    metrics: StrategyMetrics = field(default_factory=StrategyMetrics)
    current_stage_id: str = ""

    def __post_init__(self) -> None:
        if not self.current_stage_id:
            self.current_stage_id = self.strategy.initial_stage_id

    def record_normal_link_success(self) -> None:
        """只在新画面确认普通非乱舞连线成功后调用。"""
        self.metrics.battle_normal_links += 1
        self.metrics.stage_normal_links += 1

    def required_visual_metrics(self, phase: BattlePhase) -> frozenset[str]:
        """收集当前及所有可达阶段的视觉依赖，不改变条件评估或阶段转变。"""
        visual_metrics = {"enemy_hp_percent", "ally_hp_percent", "ultimate_ready"}
        required: set[str] = set()

        def collect(condition: dict[str, Any]) -> None:
            """递归检查已经校验过的条件树，不根据当前值提前裁掉分支。"""
            if condition["type"] == "group":
                for item in condition["children"]:
                    collect(item)
            elif condition["metric"] in visual_metrics:
                required.add(condition["metric"])

        stage_by_id = self.strategy.stage_by_id
        pending = [self.current_stage_id]
        visited: set[str] = set()
        while pending:
            identifier = pending.pop()
            if identifier in visited:
                continue
            visited.add(identifier)
            stage = stage_by_id[identifier]
            for transition in stage.transitions:
                collect(transition.condition)
                pending.append(transition.target_stage_id)
            if phase != BattlePhase.RANBU or stage.frenzy.allow_ultimates:
                for rule in stage.actions.ultimates:
                    if rule.enabled:
                        required.add("ultimate_ready")
                        if rule.condition is not None:
                            collect(rule.condition)
        return frozenset(required)

    def evaluate_stage(self, metrics: StrategyMetrics | None = None) -> StageResolution:
        """独立评估并应用当前观察中成立的阶段转变，阻断同帧循环。"""
        context = replace(metrics if metrics is not None else self.metrics)
        stage_by_id = self.strategy.stage_by_id
        stage = stage_by_id[self.current_stage_id]
        previous_stage_id = stage.id
        seen = {stage.id}
        chain: list[str] = []
        applied: list[StageTransition] = []
        for _ in range(len(stage_by_id)):
            transition = next(
                (item for item in stage.transitions if evaluate_condition(item.condition, context) is True), None
            )
            if transition is None:
                return StageResolution(previous_stage_id, stage, tuple(chain), False, tuple(applied))
            if transition.target_stage_id in seen:
                return StageResolution(previous_stage_id, stage, tuple(chain), True, tuple(applied))
            chain.append(transition.target_stage_id)
            applied.append(transition)
            seen.add(transition.target_stage_id)
            self.current_stage_id = transition.target_stage_id
            if metrics is None:
                self.metrics.stage_normal_links = 0
            context = replace(context, stage_normal_links=0)
            stage = stage_by_id[self.current_stage_id]
        looped = any(item.target_stage_id in seen for item in stage.transitions)
        return StageResolution(previous_stage_id, stage, tuple(chain), looped, tuple(applied))

    def resolve_stage(self, metrics: StrategyMetrics | None = None) -> tuple[StrategyStage, tuple[str, ...], bool]:
        """兼容旧调用方，返回阶段、转变链和循环标志。"""
        result = self.evaluate_stage(metrics)
        return result.stage, result.transition_chain, result.transition_loop


def _compare(actual: float | int, operator: str, expected: float | int) -> bool:
    return {
        "greater_or_equal": actual >= expected,
        "less_or_equal": actual <= expected,
        "greater": actual > expected,
        "less": actual < expected,
        "equal": actual == expected,
        "not_equal": actual != expected,
    }[operator]


def evaluate_condition(condition: Mapping[str, Any], metrics: StrategyMetrics, *, depth: int = 0) -> bool | None:
    """以 True、False、None 三态计算策略条件，未知不会因否定而成立。"""
    if depth >= 16:
        return None
    if condition.get("type") == "group":
        values = [evaluate_condition(child, metrics, depth=depth + 1) for child in condition.get("children", [])]
        if condition.get("match") == "all":
            if False in values:
                return False
            return None if None in values else True
        if True in values:
            return True
        return None if None in values else False

    metric = condition.get("metric")
    if metric in ("battle_normal_links", "stage_normal_links", "enemy_hp_percent", "ally_hp_percent"):
        actual = getattr(metrics, metric, None)
        if actual is None:
            return None
        return _compare(actual, condition["comparison"], condition["value"])
    if metric == "frenzy_active":
        actual = metrics.frenzy_active
        return None if actual is None else actual is condition["expected"]
    if metric == "ultimate_ready":
        values = [metrics.ultimate_ready.get(slot) for slot in condition["slots"]]
        expected = condition["expected"]
        if any(value is not None and value is not expected for value in values):
            return False
        return None if any(value is None for value in values) else True
    return None


def _value(target: Any, name: str) -> Any:
    return target.get(name) if isinstance(target, Mapping) else getattr(target, name, None)


def choose_normal_route(
    routes: list[Any] | tuple[Any, ...],
    *,
    prioritize_colors: bool,
    long_link_threshold: int,
    color_priority: tuple[str, ...],
    color_priority_count: int = 4,
) -> Any | None:
    """按已确认的七连门槛与颜色顺序，从当前帧候选路线中选择一路。"""
    if not routes:
        return None
    candidates = list(routes)
    enhanced = [
        route
        for route in candidates
        if len(_value(route, "enhanced_object_ids") or ()) > 0
        and len(_value(route, "object_ids") or ()) >= 2
    ]
    if enhanced:
        candidates = [
            route
            for route in enhanced
            if len(_value(route, "enhanced_object_ids") or ())
            == max(len(_value(candidate, "enhanced_object_ids") or ()) for candidate in enhanced)
        ]
    else:
        black = [
            route
            for route in candidates
            if _value(route, "color") == "black" and len(_value(route, "object_ids") or ()) >= 2
        ]
        if black:
            return max(black, key=lambda route: len(_value(route, "object_ids") or ()))
    if prioritize_colors:
        long_routes = [route for route in candidates if len(_value(route, "object_ids") or ()) >= long_link_threshold]
        if long_routes:
            candidates = long_routes
        color_rank = {color: index for index, color in enumerate(color_priority[:color_priority_count])}
        candidates.sort(
            key=lambda route: (
                color_rank.get(_value(route, "color"), len(color_rank)),
                -len(_value(route, "object_ids") or ()),
            )
        )
    else:
        candidates.sort(key=lambda route: -len(_value(route, "object_ids") or ()))
    return candidates[0]


def _ready_value(value: Any) -> bool | None:
    if value is None:
        return None
    state = getattr(value, "value", value)
    if state is True or state == "ready":
        return True
    if state is False or state in ("charging", "unknown"):
        return False if state in (False, "charging") else None
    return None


def decide_action(
    runtime: StrategyRuntime,
    phase: BattlePhase | str,
    *,
    metrics: StrategyMetrics | None = None,
    ultimate_states: Mapping[int, Any] | None = None,
    available_bomb_levels: list[int] | tuple[int, ...] = (),
    routes: list[Any] | tuple[Any, ...] = (),
    route_provider: Callable[[], list[Any] | tuple[Any, ...]] | None = None,
    enhanced_routes: list[Any] | tuple[Any, ...] = (),
    enhanced_route_ready: bool = True,
    black_routes: list[Any] | tuple[Any, ...] = (),
    black_route_ready: bool = True,
    stage_resolution: StageResolution | None = None,
) -> StrategyDecision:
    """使用已评估阶段选择一步建议，仅在需要普通连线时请求路线。"""
    try:
        battle_phase = BattlePhase(phase)
    except ValueError:
        return StrategyDecision("wait", runtime.current_stage_id, reason="战斗阶段未确认")
    states = {slot: _ready_value(value) for slot, value in (ultimate_states or {}).items()}
    supplied = metrics if metrics is not None else runtime.metrics
    context = replace(
        supplied,
        ultimate_ready=states if states else supplied.ultimate_ready,
        frenzy_active=(battle_phase == BattlePhase.RANBU),
    )
    resolution = stage_resolution or runtime.evaluate_stage(context)
    stage = resolution.stage
    transitions = resolution.transition_chain
    if resolution.transition_loop:
        return StrategyDecision(
            "wait", stage.id, transition_chain=transitions, transition_loop=True, reason="同一观察内阶段转变形成循环"
        )

    enhanced = [
        route
        for route in enhanced_routes
        if len(_value(route, "enhanced_object_ids") or ()) > 0
        and len(_value(route, "object_ids") or ()) >= 2
    ]
    black = [
        route
        for route in (*black_routes, *routes)
        if _value(route, "color") == "black" and len(_value(route, "object_ids") or ()) >= 2
    ]
    ranked_actions = (
        stage.actions.priority[: stage.actions.priority_count] if stage.actions.prioritize_actions else ()
    )
    action_priority = (
        *ranked_actions,
        *(action for action in ("ultimate", "bomb", "enhanced_ghoul", "black_ghoul") if action not in ranked_actions),
    )
    for action in action_priority:
        if action == "ultimate":
            if battle_phase == BattlePhase.RANBU and not stage.frenzy.allow_ultimates:
                continue
            for rule in stage.actions.ultimates:
                if not rule.enabled or states.get(rule.slot) is not True:
                    continue
                if rule.condition is not None and evaluate_condition(rule.condition, context) is not True:
                    continue
                return StrategyDecision(
                    "ultimate",
                    stage.id,
                    slot=rule.slot,
                    transition_chain=transitions,
                    reason="策略条件允许且当前帧确认奥义就绪",
                )
        elif action == "bomb":
            if battle_phase == BattlePhase.RANBU and not stage.frenzy.allow_bombs:
                continue
            usable = [
                level for level in available_bomb_levels if level in stage.actions.bomb_levels and level in (1, 2, 3)
            ]
            if usable:
                return StrategyDecision(
                    "bomb",
                    stage.id,
                    level=max(usable),
                    transition_chain=transitions,
                    reason="按策略允许等级选择当前帧可用炸弹",
                )
        elif action == "enhanced_ghoul" and enhanced:
            if not enhanced_route_ready:
                return StrategyDecision(
                    "wait", stage.id, transition_chain=transitions, reason="等待强化饿鬼连线棋盘稳定"
                )
            route = choose_normal_route(
                enhanced,
                prioritize_colors=False,
                long_link_threshold=stage.actions.normal_link.long_link_threshold,
                color_priority=stage.actions.normal_link.color_priority,
                color_priority_count=stage.actions.normal_link.color_priority_count,
            )
            assert route is not None
            return StrategyDecision(
                "normal_link",
                stage.id,
                route=route,
                transition_chain=transitions,
                reason="按行动优先级选择强化饿鬼路线",
            )
        elif action == "black_ghoul" and black:
            if not black_route_ready:
                return StrategyDecision("wait", stage.id, transition_chain=transitions, reason="等待黑色连线棋盘稳定")
            return StrategyDecision(
                "black_link",
                stage.id,
                route=max(black, key=lambda route: len(_value(route, "object_ids") or ())),
                transition_chain=transitions,
                reason="按行动优先级选择黑色饿鬼路线",
            )

    if battle_phase == BattlePhase.RANBU:
        return StrategyDecision(
            "ranbu_link",
            stage.id,
            reserved_color=stage.frenzy.reserved_color,
            ranbu_continue_on_unknown_color=(
                stage.frenzy.reserved_color is not None and stage.frenzy.continue_on_unknown_color
            ),
            transition_chain=transitions,
            reason="乱舞按当前帧颜色选点并扫动",
        )
    if route_provider is not None:
        routes = route_provider()
    route = choose_normal_route(
        routes,
        prioritize_colors=stage.actions.normal_link.prioritize_colors,
        long_link_threshold=stage.actions.normal_link.long_link_threshold,
        color_priority=stage.actions.normal_link.color_priority,
        color_priority_count=stage.actions.normal_link.color_priority_count,
    )
    if route is None:
        return StrategyDecision("wait", stage.id, transition_chain=transitions, reason="当前帧没有可信候选路线")
    return StrategyDecision(
        "normal_link", stage.id, route=route, transition_chain=transitions, reason="策略选择当前帧普通连线候选"
    )


def parse_hp_percent(text: str | None) -> float | None:
    """把 OCR 的 A/B 读数转成百分比；格式异常时返回未知。"""
    if not isinstance(text, str):
        return None
    match = _HP_PATTERN.fullmatch(text)
    if match is None:
        return None
    try:
        current = Decimal(match.group(1))
        maximum = Decimal(match.group(2))
    except InvalidOperation:
        return None
    if maximum <= 0 or current > maximum:
        return None
    percentage = current * Decimal(100) / maximum
    return float(percentage)
