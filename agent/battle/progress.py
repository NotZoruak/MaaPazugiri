"""把内部战斗决策整理成 GUI 中的简短中文进度。"""

from __future__ import annotations

from typing import Any

from utils import logger

from .strategy import BattlePhase, StageResolution, StrategyDecision, StrategyDocument

_COLORS = {"red": "红色", "blue": "蓝色", "green": "绿色", "yellow": "黄色", "black": "黑色"}
_METRICS = {
    "battle_normal_links": "本场普通连线手数",
    "stage_normal_links": "本阶段普通连线手数",
    "enemy_hp_percent": "敌方血量",
    "ally_hp_percent": "我方血量",
}
_COMPARISONS = {
    "greater_or_equal": "≥",
    "less_or_equal": "≤",
    "greater": ">",
    "less": "<",
    "equal": "=",
    "not_equal": "≠",
}


def _condition_text(condition: dict[str, Any]) -> str:
    """保留已校验条件树的组合关系、比较方式和布尔含义。"""
    if condition["type"] == "group":
        mode = "全部满足" if condition["match"] == "all" else "任一满足"
        children = "；".join(_condition_text(child) for child in condition["children"])
        return f"{mode}（{children}）"
    metric = condition["metric"]
    if metric == "frenzy_active":
        return "处于乱舞" if condition["expected"] else "不处于乱舞"
    if metric == "ultimate_ready":
        slots = "、".join(str(slot) for slot in condition["slots"])
        state = "就绪" if condition["expected"] else "未就绪"
        return f"{slots} 号位奥义全部{state}"
    suffix = "%" if metric in ("enemy_hp_percent", "ally_hp_percent") else ""
    return f"{_METRICS[metric]} {_COMPARISONS[condition['comparison']]} {condition['value']:g}{suffix}"


def _operation_text(decision: StrategyDecision) -> str:
    """描述当前准备执行的操作，不把候选目标数量当作消除结果。"""
    if decision.kind == "ultimate":
        return f"释放 {decision.slot} 号位奥义"
    if decision.kind == "bomb":
        return f"点击 {decision.level} 级炸弹"
    if decision.kind == "normal_link" and decision.route is not None:
        color = _COLORS.get(decision.route.color, "未知颜色")
        return f"普通连线（{color}，目标 {len(decision.route.object_ids)} 个）"
    if decision.kind == "black_link" and decision.route is not None:
        return f"优先连接黑色饿鬼（目标 {len(decision.route.object_ids)} 个）"
    if decision.kind == "ranbu_link":
        color = _COLORS.get(decision.reserved_color or "")
        return f"乱舞扫动（保留{color}）" if color else "乱舞扫动"
    if decision.kind == "ranbu_short_link":
        return "乱舞短连后松手（当前棋盘全为保留色）"
    if decision.kind == "wait" and decision.reason in ("等待棋盘稳定", "等待起点颜色确认", "等待黑色连线棋盘稳定"):
        return decision.reason
    return "等待"


class BattleProgressReporter:
    """合并连续相同的等待、奥义和乱舞摘要，实际转阶段时展示命中条件。"""

    def __init__(self, strategy: StrategyDocument) -> None:
        self._stages = strategy.stage_by_id
        self._last_summary: tuple[BattlePhase, str, str, int | None, str | None, str] | None = None

    def report(self, phase: BattlePhase | str, resolution: StageResolution, decision: StrategyDecision) -> None:
        """仅展示可操作与乱舞进度，合并重复提示但保留普通连线与炸弹记录。"""
        try:
            confirmed_phase = BattlePhase(phase)
        except ValueError:
            return
        previous = self._stages[resolution.previous_stage_id]
        if resolution.applied_transitions:
            self._last_summary = None
        for transition in resolution.applied_transitions:
            target = self._stages[transition.target_stage_id]
            logger.info(
                "阶段切换：{} → {}；条件：{}。", previous.name, target.name, _condition_text(transition.condition)
            )
            previous = target
        if decision.kind in ("wait", "ultimate", "ranbu_link", "ranbu_short_link"):
            signature = (
                confirmed_phase,
                resolution.stage.id,
                decision.kind,
                decision.slot,
                decision.reserved_color,
                decision.reason if decision.kind == "wait" else "",
            )
            if signature == self._last_summary:
                return
            self._last_summary = signature
        else:
            self._last_summary = None
        state = "可操作" if confirmed_phase == BattlePhase.OPERABLE else "乱舞"
        logger.info(
            "当前状态：{}；所属阶段：{}；进行操作：{}。", state, resolution.stage.name, _operation_text(decision)
        )
