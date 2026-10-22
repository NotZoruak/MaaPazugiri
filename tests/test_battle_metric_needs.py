"""验证乱舞仅跳过策略所有可达阶段均用不到的视觉指标。"""

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
    StrategyRuntime,
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

    def test_normal_phase_keeps_full_metrics_even_when_empty_needs_are_passed(self) -> None:
        """普通阶段不启用乱舞的按需精简。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        preview, ultimate = Mock(), Mock()
        preview.recognize_board.return_value = SimpleNamespace(objects=[])
        ultimate.analyze.return_value = SimpleNamespace(slots=[])
        observer = BattleObserver(Mock(), preview, ultimate, "阶段")
        with patch("battle.observation.recognize_hp_percent", return_value=33.0) as hp:
            result = observer.observe(image, BattlePhase.OPERABLE, required_metrics=frozenset())
        self.assertEqual(hp.call_count, 2)
        ultimate.analyze.assert_called_once()
        self.assertEqual(result.unchecked_metrics, ())


if __name__ == "__main__":
    unittest.main()
