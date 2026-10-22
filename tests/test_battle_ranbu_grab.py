"""验证乱舞按下后直接扫动，不等待起点选中描边。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import test_battle_execution as fixtures  # noqa: E402
from battle.execution import MaaBattleExecutor  # noqa: E402
from battle.session import BattleOutcome  # noqa: E402
from battle.strategy import StrategyDecision  # noqa: E402


class RanbuDirectStartTests(unittest.TestCase):
    """本帧起点判色完成后，不再用按下后的选中外观阻断扫动。"""

    def test_live_action_does_not_install_selection_outline_check(self) -> None:
        """实际动作入口不得重新接入导致短按即松手的描边回调。"""
        from custom.action.battle import BattleAction

        context = SimpleNamespace(
            tasker=SimpleNamespace(controller=Mock(), stopping=False), run_recognition_direct=Mock()
        )
        with (
            patch("custom.action.battle.BattleFrameRecognizer"),
            patch("custom.action.battle.BoardPreviewer"),
            patch("custom.action.battle.UltimateChargeRecognizer"),
            patch("custom.action.battle.BattleObserver"),
            patch("custom.action.battle.BattleSession") as session,
            patch("custom.action.battle.MaaBattleExecutor") as executor,
        ):
            session.return_value.run.return_value = BattleOutcome.VICTORY
            self.assertTrue(BattleAction().run(context, SimpleNamespace(custom_action_param="{}")).success)
            self.assertNotIn("inspect_ranbu_start", executor.call_args.kwargs)
            self.assertNotIn("inspect_ranbu_hold", executor.call_args.kwargs)

    def test_full_sweep_without_start_size_never_waits_for_capture(self) -> None:
        """未提供描边尺寸也完成完整轨迹，不追加按住截图。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        clock, preview = fixtures.ReplayClock(), fixtures.MarkerReplay(0)
        capture_moves = []

        class CaptureOrderController(fixtures.ReplayController):
            """记录截图时已派发的移动次数。"""

            def post_screencap(self):
                """回放没有任何选中描边的纯黑截图。"""
                capture_moves.append(self.moves)
                return super().post_screencap()

        controller = CaptureOrderController()
        executor = MaaBattleExecutor(
            controller,
            preview,
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        executor.execute_once(
            fixture.upper_yellow_observation(), fixture.resolution, StrategyDecision("ranbu_link", "阶段")
        )
        self.assertGreater(controller.moves, 200)
        self.assertEqual(capture_moves, [])
        self.assertEqual(controller.actions[0][0], "down")
        self.assertEqual(controller.actions[1][0], "move")
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertTrue(preview.ranbu_reports[0]["dispatch_completed"])


if __name__ == "__main__":
    unittest.main()
