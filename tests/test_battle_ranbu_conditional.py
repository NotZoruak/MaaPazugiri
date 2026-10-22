"""验证补扫只由当前帧明确残留触发，实时不读取连线数字。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import test_battle_execution as fixtures  # noqa: E402
from battle.execution import MaaBattleExecutor  # noqa: E402
from battle.ranbu_feedback import RanbuHoldFeedback  # noqa: E402
from battle.session import BattleOutcome, BattleState  # noqa: E402
from battle.strategy import StrategyDecision  # noqa: E402
from vision.ranbu_coverage import RanbuCoverageResult  # noqa: E402


class ConditionalRanbuTests(unittest.TestCase):
    """数量不是补扫依据；未知、完成与停止都有明确释放边界。"""

    def test_final_sweep_captures_then_releases_without_another_feedback_check(self) -> None:
        """补扫结束后不再等待无法改变动作的状态与残留检查。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        clock, preview = fixtures.ReplayClock(), fixtures.MarkerReplay(0)
        checked = []

        class CaptureController(fixtures.ReplayController):
            """记录截图顺序，并回放十毫秒的截图耗时。"""

            def post_screencap(self):
                self.actions.append(("capture", ()))
                clock.now += 0.01
                return fixtures.ReplayJob()

        controller = CaptureController()

        def inspect(image, board, color):
            """首遍明确残留，反馈识别耗时三十毫秒。"""
            checked.append(controller.moves)
            clock.now += 0.03
            return RanbuHoldFeedback(BattleState.RANBU, None, "present")

        executor = MaaBattleExecutor(
            controller,
            preview,
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
            inspect_ranbu_hold=inspect,
        )
        executor.execute_once(
            fixture.upper_yellow_observation(), fixture.resolution, StrategyDecision("ranbu_link", "阶段")
        )
        self.assertEqual(len(checked), 1)
        self.assertEqual(controller.actions[-2:], [("capture", ()), ("up", ())])
        self.assertAlmostEqual(clock.now, 0.85, places=6)
        last = preview.ranbu_reports[0]["passes"][-1]
        self.assertFalse(last["feedback_checked"])
        self.assertEqual(last["feedback_skip_reason"], "pass_limit")
        self.assertEqual(last["feedback_elapsed_ms"], 0.0)
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "pass_limit")

    def test_final_capture_failure_still_releases_without_feedback(self) -> None:
        """最后一遍截图失败时仍直接释放，不调用尾部识别或开始新动作。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        clock, preview = fixtures.ReplayClock(), fixtures.MarkerReplay(0)
        checks = []

        class CaptureFailureController(fixtures.ReplayController):
            """仅令补扫结束后的第二次截图失败。"""

            def __init__(self):
                super().__init__()
                self.captures = 0

            def post_screencap(self):
                self.captures += 1
                return fixtures.ReplayJob(self.captures == 1)

        controller = CaptureFailureController()

        def inspect(image, board, color):
            """首遍残留触发唯一一次补扫。"""
            checks.append(controller.moves)
            return RanbuHoldFeedback(BattleState.RANBU, None, "present")

        executor = MaaBattleExecutor(
            controller,
            preview,
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
            inspect_ranbu_hold=inspect,
        )
        executor.execute_once(
            fixture.upper_yellow_observation(), fixture.resolution, StrategyDecision("ranbu_link", "阶段")
        )
        self.assertEqual(len(checks), 1)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "capture_failed")
        self.assertFalse(preview.ranbu_reports[0]["held_image_available"])

    def test_live_callback_does_not_locate_markers_or_read_numbers(self) -> None:
        """真实入口仅调用状态检查与像素统计，不引入数字原生识别调用。"""
        from custom.action.battle import BattleAction

        direct = Mock()
        context = SimpleNamespace(
            tasker=SimpleNamespace(controller=Mock(), stopping=False),
            run_recognition_direct=direct,
        )
        with (
            patch("custom.action.battle.BattleFrameRecognizer") as frame,
            patch("custom.action.battle.BoardPreviewer") as preview,
            patch("custom.action.battle.UltimateChargeRecognizer"),
            patch("custom.action.battle.BattleObserver"),
            patch("custom.action.battle.MaaBattleExecutor") as executor,
            patch("custom.action.battle.BattleSession") as session,
            patch(
                "custom.action.battle.inspect_ranbu_coverage", return_value=RanbuCoverageResult("present", "当前帧残留")
            ) as coverage,
        ):
            session.return_value.run.return_value = BattleOutcome.VICTORY
            frame.return_value.analyze.return_value = BattleState.RANBU
            self.assertTrue(BattleAction().run(context, SimpleNamespace(custom_action_param="{}")).success)
            inspect = executor.call_args.kwargs["inspect_ranbu_hold"]
            fixture = fixtures.BattleExecutionTests()
            fixture.setUp()
            observation = fixture.upper_yellow_observation()
            result = inspect(observation.image, observation.board, "yellow")
            self.assertEqual(result.residual_status, "present")
            self.assertIsNone(result.count)
            coverage.assert_called_once()
            preview.return_value.recognize_chain_endpoint.assert_not_called()
            direct.assert_not_called()
            frame.return_value.analyze.reset_mock()
            coverage.return_value = RanbuCoverageResult("unknown", "没有强残留证据")
            result = inspect(observation.image, observation.board, "yellow")
            self.assertFalse(result.state_checked)
            frame.return_value.analyze.assert_not_called()

    def run_case(self, residuals, counts=(), *, stop_after=10000):
        """运行实际执行器，回放残留证据和旧报告兼容字段。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        controller, clock, preview = fixtures.ReplayController(), fixtures.ReplayClock(), fixtures.MarkerReplay(0)
        feedback = iter(residuals)
        numbers = iter(counts)
        inspected = []

        def inspect(image, board, color):
            """只回放与当前截图对应的残留结论。"""
            self.assertNotIn(("up", ()), controller.actions)
            inspected.append(color)
            return RanbuHoldFeedback(BattleState.RANBU, next(numbers), residual_status=next(feedback))

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
        executor.execute_once(
            fixture.upper_yellow_observation(), fixture.resolution, StrategyDecision("ranbu_link", "阶段")
        )
        return controller, clock, preview, inspected

    def test_no_residual_releases_after_the_first_sweep(self) -> None:
        """未见明确残留时一遍就松手，不能因数字未知无条件补扫。"""
        controller, clock, preview, checks = self.run_case(["absent"], [None])
        self.assertEqual(len(checks), 1)
        self.assertLess(clock.now, 0.85)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "no_residual_evidence")
        self.assertFalse(preview.ranbu_reports[0]["game_connection_verified"])

    def test_unknown_residual_does_not_trigger_a_blind_supplement(self) -> None:
        """残留未知不当作有残留，也不宣称已经全部接入。"""
        _, _, preview, checks = self.run_case(["unknown"], [14])
        self.assertEqual(len(checks), 1)
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "residual_unknown")
        self.assertFalse(preview.ranbu_reports[0]["game_connection_verified"])

    def test_present_residual_allows_only_one_reverse_sweep(self) -> None:
        """两遍都见残留也不得执行第三遍，数字缺失不影响已经确认的残留。"""
        controller, clock, preview, checks = self.run_case(["present", "present"], [None, 12])
        self.assertEqual(len(checks), 1)
        self.assertLess(clock.now, 1.65)
        self.assertEqual(sum(k == "down" for k, _ in controller.actions), 1)
        self.assertEqual(sum(k == "up" for k, _ in controller.actions), 1)
        self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "pass_limit")
        self.assertEqual([p["residual_status"] for p in preview.ranbu_reports[0]["passes"]], ["present", "unknown"])

    def test_decreasing_numbers_are_conflicting_diagnostics(self) -> None:
        """数值下降不能当作可靠的数量停滞或改变补扫次数。"""
        _, _, preview, checks = self.run_case(["present", "absent"], [141, 14])
        self.assertEqual(len(checks), 1)
        self.assertEqual(preview.ranbu_reports[0]["count_trend"], "unknown")
        self.assertNotEqual(preview.ranbu_reports[0]["stop_reason"], "no_growth")

    def test_stop_skips_number_ocr_and_releases_immediately(self) -> None:
        """停止中的未完成手势不再进入耗时数字诊断。"""
        controller, _, preview, checks = self.run_case([], [], stop_after=3)
        self.assertEqual(checks, [])
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])


if __name__ == "__main__":
    unittest.main()
