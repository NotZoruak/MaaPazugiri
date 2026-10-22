"""验证乱舞保持同一笔触控，根据当前画面有界补扫。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import test_battle_execution as fixtures  # noqa: E402
from battle.ranbu_feedback import RanbuHoldFeedback  # noqa: E402
from battle.session import BattleState  # noqa: E402
from battle.strategy import StrategyDecision  # noqa: E402


class RanbuResweepTests(unittest.TestCase):
    """数量增长、停滞、模式退出和用户停止均有明确边界。"""

    def replay(self, feedback, *, fail_move=None, stop_after=10000):
        """只替换设备和画面反馈，运行实际乱舞执行器。"""
        from battle.execution import MaaBattleExecutor

        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        controller, clock, preview = (
            fixtures.ReplayController(fail_move=fail_move),
            fixtures.ReplayClock(),
            fixtures.MarkerReplay(0),
        )
        counts = iter(feedback)
        calls = []

        def inspect(image, board, color):
            """反馈必须发生在触点按住期间。"""
            self.assertNotIn(("up", ()), controller.actions)
            calls.append(controller.moves)
            value = next(counts)
            if isinstance(value, Exception):
                raise value
            return value

        executor = MaaBattleExecutor(
            controller,
            preview,
            lambda: None,
            lambda: True,
            lambda: controller.moves >= stop_after,
            clock=clock.clock,
            sleep=clock.sleep,
            inspect_ranbu_hold=inspect,
        )
        observation = fixture.upper_yellow_observation()
        executor.execute_once(observation, fixture.resolution, StrategyDecision("ranbu_link", "阶段"))
        return controller, preview, calls

    def test_growth_does_not_add_a_third_pass(self) -> None:
        """明确残留只允许一次补扫，数字增长不能再增加一遍。"""
        controller, preview, calls = self.replay([RanbuHoldFeedback(BattleState.RANBU, n, "present") for n in (4, 12)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(sum(kind == "down" for kind, _ in controller.actions), 1)
        self.assertEqual(sum(kind == "up" for kind, _ in controller.actions), 1)
        self.assertEqual(controller.actions[-1], ("up", ()))
        report = preview.ranbu_reports[0]
        self.assertEqual([p["count"] for p in report["passes"]], [4, None])
        self.assertEqual(report["stop_reason"], "pass_limit")
        self.assertFalse(report["connection_progress_observed"])
        self.assertFalse(report["game_connection_verified"])
        times = [e["sent_ms"] for e in report["events"]]
        self.assertEqual(times, sorted(times))

    def test_numbers_do_not_control_the_reverse_pass(self) -> None:
        """数量相同不影响当前帧明确残留触发的唯一一次反向补扫。"""
        controller, preview, calls = self.replay([RanbuHoldFeedback(BattleState.RANBU, 4, "present")] * 2)
        self.assertEqual(len(calls), 1)
        moves = [point for kind, point in controller.actions if kind == "move"]
        self.assertEqual(moves[-1], (0, 210))
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "pass_limit")

    def test_unknown_count_does_not_trigger_a_supplement(self) -> None:
        """数量缺失不能作为需要补扫的证据。"""
        _, preview, calls = self.replay([RanbuHoldFeedback(BattleState.RANBU, None)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "residual_unknown")

    def test_mode_exit_does_not_start_a_supplement(self) -> None:
        """当前画面不再确认乱舞，立即松手。"""
        _, preview, calls = self.replay([RanbuHoldFeedback(BattleState.OPERABLE, None)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "phase_exit")

    def test_unchecked_state_never_permits_a_supplement(self) -> None:
        """即使像素提示残留，也必须在当前帧确认乱舞后才能补扫。"""
        _, preview, calls = self.replay([RanbuHoldFeedback(BattleState.RANBU, None, "present", state_checked=False)])
        self.assertEqual(len(calls), 1)
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "phase_unconfirmed")

    def test_feedback_exception_releases_without_a_supplement(self) -> None:
        """原生反馈异常时不继续扫动，也不能留下按住状态。"""
        controller, preview, calls = self.replay([RuntimeError("回放反馈失败")])
        self.assertEqual(len(calls), 1)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "feedback_exception")

    def test_stop_during_the_reverse_pass_releases_without_another_capture(self) -> None:
        """用户停止可中断补扫，也不继续读取画面。"""
        controller, preview, calls = self.replay([RanbuHoldFeedback(BattleState.RANBU, 4, "present")], stop_after=300)
        self.assertEqual(len(calls), 1)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])

    def test_reverse_move_failure_releases_and_reports_incomplete(self) -> None:
        """补扫失败不能沿用上一遍的完成标志。"""
        from battle.execution import MaaBattleExecutor

        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        controller, clock, preview = (
            fixtures.ReplayController(fail_move=300),
            fixtures.ReplayClock(),
            fixtures.MarkerReplay(0),
        )
        executor = MaaBattleExecutor(
            controller,
            preview,
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
            inspect_ranbu_hold=lambda *a: RanbuHoldFeedback(BattleState.RANBU, 4, "present"),
        )
        with self.assertRaisesRegex(RuntimeError, "移动连线触点失败"):
            executor.execute_once(
                fixture.upper_yellow_observation(), fixture.resolution, StrategyDecision("ranbu_link", "阶段")
            )
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "input_failed")


if __name__ == "__main__":
    unittest.main()
