"""验证战斗仅跳过策略所有可达阶段均用不到的视觉指标。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.observation import BattleObserver  # noqa: E402
from battle.strategy import (  # noqa: E402
    BattlePhase,
    StageTransition,
    StrategyDocument,
    StrategyMetrics,
    StrategyRuntime,
    decide_action,
)


def predicate(metric: str) -> dict[str, object]:
    """构造用于需求扫描的条件。"""
    return {"type": "predicate", "metric": metric, "comparison": "less", "value": 50}


class VisualMetricNeedsTests(unittest.TestCase):
    """可达阶段、条件组和可用奥义均参与需求判断。"""

    def setUp(self) -> None:
        document = StrategyDocument.builtin_default()
        stage = document.stages[0]
        self.stage = replace(stage, frenzy=replace(stage.frenzy, allow_ultimates=False))
        self.document = replace(document, stages=(self.stage,))

    def test_disabled_ultimate_and_unused_hp_need_no_visual_metrics(self) -> None:
        """乱舞禁奥义且没有视觉条件时，不需要血量或奥义。"""
        runtime = StrategyRuntime(self.document)
        self.assertEqual(runtime.required_visual_metrics(BattlePhase.RANBU), frozenset())
        self.assertEqual(
            runtime.required_visual_metrics(BattlePhase.OPERABLE),
            frozenset(("ultimate_ready",)),
        )

    def test_normal_disabled_ultimates_do_not_require_readiness_or_rule_hp(self) -> None:
        """普通阶段禁用的奥义及其条件不应触发额外识别。"""
        rules = tuple(
            replace(rule, enabled=False, condition=predicate("enemy_hp_percent"))
            for rule in self.stage.actions.ultimates
        )
        stage = replace(self.stage, actions=replace(self.stage.actions, ultimates=rules))
        runtime = StrategyRuntime(replace(self.document, stages=(stage,)))
        self.assertEqual(runtime.required_visual_metrics(BattlePhase.OPERABLE), frozenset())

    def test_normal_reachable_stage_keeps_ultimate_readiness_and_all_conditions(self) -> None:
        """普通阶段同帧可能转入的阶段保留全部依赖，不受禁乱舞奥义开关影响。"""
        rule = replace(self.stage.actions.ultimates[0], condition=predicate("ally_hp_percent"))
        second = replace(self.stage, id="后续", actions=replace(self.stage.actions, ultimates=(rule,)))
        first = replace(
            self.stage,
            actions=replace(self.stage.actions, ultimates=()),
            transitions=(StageTransition("进入", second.id, predicate("enemy_hp_percent")),),
        )
        runtime = StrategyRuntime(replace(self.document, stages=(first, second)))
        self.assertEqual(
            runtime.required_visual_metrics(BattlePhase.OPERABLE),
            frozenset(("enemy_hp_percent", "ally_hp_percent", "ultimate_ready")),
        )

    def test_nested_conditions_in_reachable_stages_are_kept(self) -> None:
        """同帧可能继续转入的阶段也必须取得新指标。"""
        downstream = replace(
            self.stage,
            id="后续",
            transitions=(StageTransition("回转", self.stage.id, predicate("ally_hp_percent")),),
        )
        first = replace(
            self.stage,
            transitions=(
                StageTransition(
                    "进入",
                    downstream.id,
                    {"type": "group", "match": "any", "children": [predicate("enemy_hp_percent")]},
                ),
            ),
        )
        runtime = StrategyRuntime(replace(self.document, stages=(first, downstream)))
        self.assertEqual(
            runtime.required_visual_metrics(BattlePhase.RANBU), frozenset(("enemy_hp_percent", "ally_hp_percent"))
        )

    def test_unreachable_stage_does_not_force_unused_ocr(self) -> None:
        """无路径可达的阶段不会让当前乱舞多做识别。"""
        other = replace(
            self.stage, id="不可达", transitions=(StageTransition("自环", "不可达", predicate("enemy_hp_percent")),)
        )
        runtime = StrategyRuntime(replace(self.document, stages=(self.stage, other)))
        self.assertEqual(runtime.required_visual_metrics(BattlePhase.RANBU), frozenset())

    def test_transition_ultimate_condition_is_kept_even_when_casting_is_disabled(self) -> None:
        """禁用释放奥义不能跳过阶段条件依赖的就绪状态。"""
        stage = replace(
            self.stage, transitions=(StageTransition("状态条件", self.stage.id, predicate("ultimate_ready")),)
        )
        runtime = StrategyRuntime(replace(self.document, stages=(stage,)))
        self.assertEqual(runtime.required_visual_metrics(BattlePhase.RANBU), frozenset(("ultimate_ready",)))

    def test_allowed_ultimate_keeps_readiness_and_its_prerequisites(self) -> None:
        """允许的奥义保留就绪检查及其血量前置。"""
        rule = replace(self.stage.actions.ultimates[0], condition=predicate("enemy_hp_percent"))
        stage = replace(
            self.stage,
            frenzy=replace(self.stage.frenzy, allow_ultimates=True),
            actions=replace(self.stage.actions, ultimates=(rule,)),
        )
        runtime = StrategyRuntime(replace(self.document, stages=(stage,)))
        self.assertEqual(
            runtime.required_visual_metrics(BattlePhase.RANBU), frozenset(("enemy_hp_percent", "ultimate_ready"))
        )

    def test_trimmed_metrics_preserve_actions_across_reachable_transitions(self) -> None:
        """全量与精简指标在未知值、条件边界和同帧转阶段时给出同一动作。"""
        rule = replace(self.stage.actions.ultimates[0], condition=predicate("enemy_hp_percent"))
        target = replace(self.stage, id="后续", actions=replace(self.stage.actions, ultimates=(rule,)))
        first = replace(
            self.stage,
            actions=replace(self.stage.actions, ultimates=()),
            transitions=(StageTransition("进入", target.id, predicate("ally_hp_percent")),),
        )
        for document in (self.document, replace(self.document, stages=(first, target))):
            for phase in BattlePhase:
                for enemy in (None, 0.0, 49.0, 50.0, 100.0):
                    for ally in (None, 0.0, 49.0, 50.0, 100.0):
                        for ready in (None, False, True):
                            full = StrategyMetrics(
                                enemy_hp_percent=enemy, ally_hp_percent=ally, ultimate_ready={1: ready}
                            )
                            runtime = StrategyRuntime(document)
                            required = runtime.required_visual_metrics(phase)
                            trimmed = replace(
                                full,
                                enemy_hp_percent=enemy if "enemy_hp_percent" in required else None,
                                ally_hp_percent=ally if "ally_hp_percent" in required else None,
                                ultimate_ready=full.ultimate_ready if "ultimate_ready" in required else {},
                            )
                            expected = decide_action(runtime, phase, metrics=full, ultimate_states=full.ultimate_ready)
                            actual = decide_action(
                                StrategyRuntime(document),
                                phase,
                                metrics=trimmed,
                                ultimate_states=trimmed.ultimate_ready,
                            )
                            self.assertEqual(actual, expected)


class RequiredMetricObservationTests(unittest.TestCase):
    """未检查的指标保持未知，不从旧帧复制。"""

    def test_ranbu_point_frame_skips_only_explicitly_unused_metrics(self) -> None:
        """空需求不调用血量OCR和奥义识别，按需读取的一方仍来自当前帧。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        direct, preview, ultimate = Mock(), Mock(), Mock()
        ultimate.analyze.return_value = SimpleNamespace(slots=[])
        observer = BattleObserver(direct, preview, ultimate, "阶段")
        layout = ((0, 210, 720, 780), "已核对布局")
        with patch("battle.observation.recognize_hp_percent", return_value=42.0) as hp:
            first = observer.observe(image, BattlePhase.RANBU, ranbu_layout=layout, required_metrics=frozenset())
            hp.assert_not_called()
            ultimate.analyze.assert_not_called()
            self.assertIsNone(first.metrics.enemy_hp_percent)
            self.assertIsNone(first.metrics.ally_hp_percent)
            self.assertEqual(first.metrics.ultimate_ready, {})
            self.assertEqual(set(first.unchecked_metrics), {"enemy_hp_percent", "ally_hp_percent", "ultimate_ready"})
            second = observer.observe(
                image, BattlePhase.RANBU, ranbu_layout=layout, required_metrics=frozenset(("enemy_hp_percent",))
            )
            self.assertEqual(hp.call_count, 1)
            self.assertIs(hp.call_args.args[1], image)
            self.assertEqual(hp.call_args.args[2], "enemy")
            self.assertEqual(second.metrics.enemy_hp_percent, 42.0)
            self.assertIsNone(second.metrics.ally_hp_percent)
            self.assertNotEqual(first.frame_id, second.frame_id)
            self.assertNotIn("enemy_hp_percent", second.unchecked_metrics)

    def test_normal_phase_keeps_board_and_skips_explicitly_unused_metrics(self) -> None:
        """普通阶段空需求仍识别本帧棋盘，未检查的血量和奥义保持未知。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        preview, ultimate = Mock(), Mock()
        preview.recognize_board.return_value = SimpleNamespace(objects=[])
        ultimate.analyze.return_value = SimpleNamespace(slots=[])
        observer = BattleObserver(Mock(), preview, ultimate, "阶段")
        with patch("battle.observation.recognize_hp_percent", return_value=33.0) as hp:
            result = observer.observe(image, BattlePhase.OPERABLE, required_metrics=frozenset())
        hp.assert_not_called()
        ultimate.analyze.assert_not_called()
        preview.recognize_board.assert_called_once()
        self.assertIsNone(result.metrics.enemy_hp_percent)
        self.assertIsNone(result.metrics.ally_hp_percent)
        self.assertEqual(result.metrics.ultimate_ready, {})
        self.assertEqual(set(result.unchecked_metrics), {"enemy_hp_percent", "ally_hp_percent", "ultimate_ready"})

    def test_missing_needs_keeps_full_metrics_for_existing_callers(self) -> None:
        """没有显式需求的调用者继续读取完整指标。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        preview, ultimate = Mock(), Mock()
        preview.recognize_board.return_value = SimpleNamespace(objects=[])
        ultimate.analyze.return_value = SimpleNamespace(slots=[])
        observer = BattleObserver(Mock(), preview, ultimate, "阶段")
        with patch("battle.observation.recognize_hp_percent", return_value=33.0) as hp:
            result = observer.observe(image, BattlePhase.OPERABLE)
        self.assertEqual(hp.call_count, 2)
        ultimate.analyze.assert_called_once()
        self.assertEqual(result.unchecked_metrics, ())


if __name__ == "__main__":
    unittest.main()
