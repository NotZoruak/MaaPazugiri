"""验证 GUI 只收到用户关心的状态、阶段、操作和实际转阶段条件。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.candidates import CandidateRoute  # noqa: E402
from battle.progress import BattleProgressReporter  # noqa: E402
from battle.strategy import (  # noqa: E402
    BattlePhase,
    StageResolution,
    StageTransition,
    StrategyDecision,
    StrategyDocument,
    StrategyMetrics,
    StrategyRuntime,
)
from utils import logger  # noqa: E402


class BattleProgressTests(unittest.TestCase):
    """通过真实日志通道检查用户能看到的内容。"""

    def setUp(self) -> None:
        self.messages: list[str] = []
        sink = logger.add(lambda message: self.messages.append(message.record["message"]), level="INFO")
        self.addCleanup(logger.remove, sink)
        self.strategy = StrategyDocument.builtin_default()
        self.stage = replace(self.strategy.stages[0], name="先攒红色饿鬼")
        self.strategy = replace(self.strategy, stages=(self.stage,))
        self.resolution = StageResolution(self.stage.id, self.stage)
        self.reporter = BattleProgressReporter(self.strategy)

    def test_normal_action_uses_user_stage_name_and_hides_internal_objects(self) -> None:
        """一条摘要包含中文状态、用户阶段名和目标数量，不泄露原始对象。"""
        route = CandidateRoute("red", (1, 2, 3, 4, 5, 6, 7, 8), ((10, 20),) * 8, 0.5, 123.0)
        decision = StrategyDecision("normal_link", self.stage.id, route=route)
        self.reporter.report(BattlePhase.OPERABLE, self.resolution, decision)
        self.assertEqual(len(self.messages), 1)
        for value in ("可操作", "先攒红色饿鬼", "普通连线", "红色", "8"):
            self.assertIn(value, self.messages[0])
        for value in (self.stage.id, "StrategyDecision", "CandidateRoute", "(10, 20)", "条件"):
            self.assertNotIn(value, self.messages[0])

    def test_non_operable_states_do_not_produce_progress(self) -> None:
        """未知、动画和胜负状态不混入可操作状态日志。"""
        decision = StrategyDecision("wait", self.stage.id, reason="等待识别")
        for phase in ("unknown", "inoperable", "victory", "defeat"):
            self.reporter.report(phase, self.resolution, decision)
        self.assertEqual(self.messages, [])

    def test_ranbu_and_each_action_are_readable(self) -> None:
        """奥义、炸弹和乱舞使用可读操作名，颜色采用中文。"""
        for decision, expected in (
            (StrategyDecision("ultimate", self.stage.id, slot=2), ("2", "奥义")),
            (StrategyDecision("bomb", self.stage.id, level=3), ("3", "炸弹")),
            (StrategyDecision("ranbu_link", self.stage.id, reserved_color="green"), ("乱舞扫动", "绿色")),
            (StrategyDecision("ranbu_short_link", self.stage.id, reserved_color="red"), ("乱舞短连", "松手")),
        ):
            with self.subTest(kind=decision.kind):
                self.messages.clear()
                self.reporter.report(BattlePhase.RANBU, self.resolution, decision)
                self.assertEqual(len(self.messages), 1)
                self.assertIn("乱舞", self.messages[0])
                for text in expected:
                    self.assertIn(text, self.messages[0])

    def test_repeated_wait_and_ultimate_are_suppressed(self) -> None:
        """等待原因与奥义槽位不变时不重复显示进度。"""
        wait = StrategyDecision("wait", self.stage.id, reason="当前帧没有可信候选路线")
        for _ in range(3):
            self.reporter.report(BattlePhase.OPERABLE, self.resolution, wait)
        self.assertEqual(len(self.messages), 1)
        action = StrategyDecision("ultimate", self.stage.id, slot=1)
        self.reporter.report(BattlePhase.OPERABLE, self.resolution, action)
        self.reporter.report(BattlePhase.OPERABLE, self.resolution, action)
        self.assertEqual(len(self.messages), 2)

    def test_repeated_ranbu_is_suppressed(self) -> None:
        """同一阶段连续乱舞扫动和短连分别只显示一次。"""
        for kind in ("ranbu_link", "ranbu_short_link"):
            decision = StrategyDecision(kind, self.stage.id, reserved_color="red")
            for _ in range(3):
                self.reporter.report(BattlePhase.RANBU, self.resolution, decision)
        self.assertEqual(len(self.messages), 2)

    def test_changed_ultimate_slot_and_intervening_action_are_reported(self) -> None:
        """切换奥义槽位或经过其他操作后，同一槽位仍能重新提示。"""
        first = StrategyDecision("ultimate", self.stage.id, slot=1)
        second = replace(first, slot=2)
        wait = StrategyDecision("wait", self.stage.id, reason="等待棋盘稳定")
        for decision in (first, first, second, second, wait, second):
            self.reporter.report(BattlePhase.OPERABLE, self.resolution, decision)
        self.assertEqual(len(self.messages), 4)
        self.assertIn("释放 1 号位奥义", self.messages[0])
        self.assertIn("释放 2 号位奥义", self.messages[1])
        self.assertIn("释放 2 号位奥义", self.messages[3])

    def test_ranbu_phase_stage_and_reserved_color_changes_are_reported(self) -> None:
        """状态、阶段或保留色改变时不能被连续提示去重吞掉。"""
        next_stage = replace(self.stage, id="输出", name="集中输出")
        strategy = replace(self.strategy, stages=(self.stage, next_stage))
        reporter = BattleProgressReporter(strategy)
        decision = StrategyDecision("ranbu_link", self.stage.id, reserved_color="red")
        reporter.report(BattlePhase.OPERABLE, self.resolution, decision)
        reporter.report(BattlePhase.RANBU, self.resolution, decision)
        reporter.report(BattlePhase.RANBU, self.resolution, replace(decision, reserved_color="green"))
        reporter.report(BattlePhase.RANBU, StageResolution(next_stage.id, next_stage), decision)
        self.assertEqual(len(self.messages), 4)

    def test_repeated_normal_links_and_bombs_are_kept(self) -> None:
        """相同摘要也可能代表不同目标，普通连线和炸弹继续逐次记录。"""
        route = CandidateRoute("red", (1, 2, 3), ((10, 20),) * 3, 0.5, 123.0)
        for decision in (
            StrategyDecision("normal_link", self.stage.id, route=route),
            StrategyDecision("bomb", self.stage.id, level=1),
        ):
            self.reporter.report(BattlePhase.OPERABLE, self.resolution, decision)
            self.reporter.report(BattlePhase.OPERABLE, self.resolution, decision)
        self.assertEqual(len(self.messages), 4)

    def test_wait_hides_parenthesized_explanations(self) -> None:
        """等待日志只显示操作名，不附带原始原因或括号说明。"""
        for reason, operation in (
            ("等待棋盘稳定", "等待棋盘稳定"),
            ("当前帧没有可信候选路线", "等待"),
        ):
            with self.subTest(reason=reason):
                self.messages.clear()
                self.reporter.report(
                    BattlePhase.OPERABLE, self.resolution, StrategyDecision("wait", self.stage.id, reason=reason)
                )
                self.assertEqual(self.messages, [f"当前状态：可操作；所属阶段：先攒红色饿鬼；进行操作：{operation}。"])

    def test_transition_reports_the_rule_that_actually_matched(self) -> None:
        """同目标多规则及同帧连续切换时，仅显示真正命中的条件。"""
        never = {"type": "predicate", "metric": "stage_normal_links", "comparison": "less", "value": 0}
        first = {"type": "predicate", "metric": "stage_normal_links", "comparison": "greater_or_equal", "value": 2}
        second = {"type": "predicate", "metric": "stage_normal_links", "comparison": "equal", "value": 0}
        a = replace(
            self.stage,
            transitions=(StageTransition("未命中", "收尾", never), StageTransition("命中", "收尾", first)),
        )
        b = replace(self.stage, id="收尾", name="集中输出", transitions=(StageTransition("继续", "完成", second),))
        c = replace(self.stage, id="完成", name="最后清场", transitions=())
        strategy = replace(self.strategy, stages=(a, b, c))
        runtime = StrategyRuntime(strategy)
        resolution = runtime.evaluate_stage(StrategyMetrics(stage_normal_links=2))
        reporter = BattleProgressReporter(strategy)
        decision = StrategyDecision("ultimate", c.id, slot=3)
        reporter.report(BattlePhase.OPERABLE, resolution, decision)
        self.assertEqual(len(self.messages), 3)
        self.assertIn("先攒红色饿鬼 → 集中输出", self.messages[0])
        self.assertIn("本阶段普通连线手数 ≥ 2", self.messages[0])
        self.assertIn("集中输出 → 最后清场", self.messages[1])
        self.assertIn("本阶段普通连线手数 = 0", self.messages[1])
        self.assertNotIn("< 0", "\n".join(self.messages))
        self.messages.clear()
        reporter.report(BattlePhase.OPERABLE, runtime.evaluate_stage(), decision)
        self.assertEqual(self.messages, [])

    def test_nested_conditions_keep_all_any_and_boolean_meaning(self) -> None:
        """条件组保留全部或任一语义，血量、奥义和乱舞均可阅读。"""
        condition = {
            "type": "group",
            "match": "all",
            "children": [
                {"type": "predicate", "metric": "enemy_hp_percent", "comparison": "less_or_equal", "value": 30},
                {
                    "type": "group",
                    "match": "any",
                    "children": [
                        {
                            "type": "predicate",
                            "metric": "ultimate_ready",
                            "slots": [1, 2],
                            "slot_match": "all",
                            "expected": True,
                        },
                        {"type": "predicate", "metric": "frenzy_active", "expected": True},
                    ],
                },
            ],
        }
        a = replace(self.stage, transitions=(StageTransition("转变", "输出", condition),))
        b = replace(self.stage, id="输出", name="结束阶段", transitions=())
        strategy = replace(self.strategy, stages=(a, b))
        resolution = StrategyRuntime(strategy).evaluate_stage(
            StrategyMetrics(enemy_hp_percent=25, ultimate_ready={1: True, 2: True}, frenzy_active=False)
        )
        BattleProgressReporter(strategy).report(BattlePhase.OPERABLE, resolution, StrategyDecision("wait", b.id))
        text = self.messages[0]
        for value in ("全部满足", "任一满足", "敌方血量 ≤ 30%", "1、2", "奥义", "乱舞"):
            self.assertIn(value, text)


if __name__ == "__main__":
    unittest.main()
