"""验证当前乱舞单遍扫动不再等待截图、残留反馈或反向补扫。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

import test_battle_execution as fixtures
from battle.execution import MaaBattleExecutor
from battle.ranbu import ranbu_sweep_points
from battle.strategy import StrategyDecision


class SingleSweepTests(unittest.TestCase):
    """检查不追加识别的单遍契约及停止、输入失败的释放边界。"""

    def replay(self, *, fail_move=None, stopping=None, allowed=None, release_ok=True):
        """回放实际执行器，截图和反馈一旦被调用就报错。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        controller = fixtures.ReplayController(fail_move=fail_move)
        controller.post_screencap = Mock(side_effect=AssertionError("单遍乱舞不应截图"))
        original_up = controller.post_touch_up

        def release():
            """保留释放事件并回放设备返回的释放状态。"""
            original_up()
            return fixtures.ReplayJob(release_ok)

        controller.post_touch_up = release
        clock, preview = fixtures.ReplayClock(), fixtures.MarkerReplay(0)
        feedback = Mock(side_effect=AssertionError("单遍乱舞不应检查反馈"))
        executor = MaaBattleExecutor(
            controller,
            preview,
            lambda: None,
            (lambda: allowed(controller)) if allowed else lambda: True,
            (lambda: stopping(controller)) if stopping else lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
            inspect_ranbu_hold=feedback,
        )
        observation = fixture.upper_yellow_observation()
        return controller, clock, preview, feedback, executor, observation, fixture.resolution

    def test_full_sweep_releases_without_capture_or_legacy_feedback(self) -> None:
        """完整轨迹只派发一遍，松手后报告不冒充游戏已经全部连上。"""
        controller, clock, preview, feedback, executor, observation, resolution = self.replay()
        executor.execute_once(observation, resolution, StrategyDecision("ranbu_link", "阶段"))
        controller.post_screencap.assert_not_called()
        feedback.assert_not_called()
        self.assertEqual(controller.actions[-2:], [("move", ranbu_sweep_points(observation.board)[-1]), ("up", ())])
        report = preview.ranbu_reports[0]
        self.assertEqual(len(report["passes"]), 1)
        self.assertEqual(report["stop_reason"], "single_sweep_complete")
        self.assertEqual(report["capture_status"], "not_attempted")
        self.assertFalse(report["passes"][0]["feedback_checked"])
        self.assertTrue(report["dispatch_completed"])
        self.assertFalse(report["game_connection_verified"])
        self.assertAlmostEqual(clock.now, 0.4)

    def test_stopping_during_sweep_releases_without_feedback(self) -> None:
        """用户停止后不再移动、截图或读取残留。"""
        controller, _, preview, feedback, executor, observation, resolution = self.replay(
            stopping=lambda device: device.moves >= 3
        )
        executor.execute_once(observation, resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertEqual(controller.moves, 3)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "stopped")
        feedback.assert_not_called()

    def test_phase_permission_loss_does_not_add_a_reverse_pass(self) -> None:
        """当前许可撤销后立即结束这笔手势，不能开启反向补扫。"""
        controller, _, preview, feedback, executor, observation, resolution = self.replay(
            allowed=lambda device: device.moves < 3
        )
        executor.execute_once(observation, resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertEqual(controller.moves, 3)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])
        feedback.assert_not_called()

    def test_input_failure_is_reported_and_released(self) -> None:
        """设备移动失败与用户停止必须保留不同的诊断原因。"""
        controller, _, preview, _, executor, observation, resolution = self.replay(fail_move=3)
        with self.assertRaisesRegex(RuntimeError, "移动连线触点失败"):
            executor.execute_once(observation, resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "input_failed")
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])

    def test_release_failure_never_counts_as_completed_dispatch(self) -> None:
        """整条移动完成也不能掩盖触点释放失败。"""
        _, _, preview, _, executor, observation, resolution = self.replay(release_ok=False)
        with self.assertRaisesRegex(RuntimeError, "释放连线触点失败"):
            executor.execute_once(observation, resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])
        self.assertFalse(preview.ranbu_reports[0]["events"][-1]["succeeded"])
