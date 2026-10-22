"""验证奥义提示识别、专用手势与主循环的去重分流。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
import test_battle_execution as fixtures
from battle.execution import MaaBattleExecutor
from battle.recognition import BattleFrameRecognizer
from battle.session import BattleOutcome, BattleSession, BattleState
from custom.action.battle import BattleAction


class UltimateConfirmationTests(unittest.TestCase):
    """专用提示只能进入对应手势，不得进入普通棋盘连线。"""

    def test_prompt_text_distinguishes_tap_swipe_and_ambiguity(self) -> None:
        """点击和滑动文案分别分流，冲突或不完整的选点提示保持未知。"""
        for text, expected in (
            ("タップで位置を決めて", BattleState.ULTIMATE_TAP_CONFIRMATION),
            ("スワイプで位置を決めて", BattleState.ULTIMATE_SWIPE_CONFIRMATION),
            ("タップ スワイプ", BattleState.UNKNOWN),
            ("位置を決めて", BattleState.UNKNOWN),
            ("", BattleState.OPERABLE),
        ):
            with self.subTest(text=text):
                native = Mock(return_value=SimpleNamespace(all_results=[SimpleNamespace(text=text)]))
                recognizer = BattleFrameRecognizer(native)
                with patch.object(recognizer, "_hit", side_effect=lambda name, image: name in {"gear", "brightness"}):
                    self.assertEqual(recognizer.analyze(np.zeros((1280, 720, 3), np.uint8)), expected)
                native.assert_called_once()

    def test_terminal_frame_does_not_run_confirmation_ocr(self) -> None:
        """终局已确认时跳过奥义提示OCR。"""
        for hit, expected in (
            ("victory", BattleState.VICTORY),
            ("defeat", BattleState.DEFEAT),
        ):
            native = Mock()
            recognizer = BattleFrameRecognizer(native)
            with patch.object(recognizer, "_hit", side_effect=lambda name, image, hit=hit: name == hit):
                self.assertEqual(recognizer.analyze(np.zeros((1280, 720, 3), np.uint8)), expected)
            native.assert_not_called()

    def test_missing_gear_and_missing_prompt_remain_unknown(self) -> None:
        """齿轮缺失时仍检查奥义提示，但空文案不能误判为可操作。"""
        native = Mock(return_value=SimpleNamespace(all_results=[]))
        recognizer = BattleFrameRecognizer(native)
        with patch.object(recognizer, "_hit", return_value=False):
            self.assertEqual(recognizer.analyze(np.zeros((1280, 720, 3), np.uint8)), BattleState.UNKNOWN)
        native.assert_called_once()

    def executor(self, stopping=lambda: False):
        """仅记录手势，不连接设备。"""
        controller = SimpleNamespace(
            post_click=Mock(return_value=fixtures.ReplayJob()), post_swipe=Mock(return_value=fixtures.ReplayJob())
        )
        return controller, MaaBattleExecutor(controller, Mock(), lambda: None, lambda: True, stopping)

    def test_tap_and_swipe_dispatch_distinct_gestures(self) -> None:
        """按已定义的三点点击和纵横滑动顺序派发，不混用手势。"""
        controller, executor = self.executor()
        executor.execute_ultimate_confirmation(BattleState.ULTIMATE_TAP_CONFIRMATION)
        self.assertEqual(controller.post_click.call_args_list, [call(360, 450), call(200, 730), call(520, 730)])
        controller.post_swipe.assert_not_called()
        executor.execute_ultimate_confirmation(BattleState.ULTIMATE_SWIPE_CONFIRMATION)
        self.assertEqual(
            controller.post_swipe.call_args_list, [call(360, 350, 360, 930, 500), call(70, 640, 650, 640, 500)]
        )

    def test_stop_prevents_remaining_taps(self) -> None:
        """首个点击后收到停止信号，不继续发送余下两次点击。"""
        controller, executor = self.executor(stopping=Mock(side_effect=[False, True]))
        executor.execute_ultimate_confirmation(BattleState.ULTIMATE_TAP_CONFIRMATION)
        self.assertEqual(controller.post_click.call_count, 1)

    def test_confirmation_failure_propagates(self) -> None:
        """输入失败不能被当成已处理的奥义提示。"""
        controller, executor = self.executor()
        controller.post_swipe.return_value = fixtures.ReplayJob(False)
        with self.assertRaisesRegex(RuntimeError, "奥义十字滑动失败"):
            executor.execute_ultimate_confirmation(BattleState.ULTIMATE_SWIPE_CONFIRMATION)
        self.assertEqual(controller.post_swipe.call_count, 1)

    def test_session_routes_confirmation_and_honors_stop_after_callback(self) -> None:
        """确认提示独立于普通和乱舞回调，回调后的停止信号立即生效。"""
        clock = fixtures.ReplayClock()
        session = BattleSession(clock=clock.clock, sleep=clock.sleep)
        stopped = [False]
        callback = Mock(side_effect=lambda state: stopped.__setitem__(0, True))
        normal, ranbu = Mock(), Mock()
        result = session.run(
            lambda: BattleState.ULTIMATE_TAP_CONFIRMATION,
            stopped=lambda: stopped[0],
            on_operable=normal,
            on_ranbu=ranbu,
            on_ultimate_confirmation=callback,
        )
        self.assertEqual(result, BattleOutcome.CANCELLED)
        callback.assert_called_once_with(BattleState.ULTIMATE_TAP_CONFIRMATION)
        normal.assert_not_called()
        ranbu.assert_not_called()

    def test_live_action_deduplicates_prompt_through_unknown_frames(self) -> None:
        """同一提示间歇识别未知时不重发，离开提示后允许处理下一次奥义。"""
        clock = fixtures.ReplayClock()
        image = np.zeros((1280, 720, 3), np.uint8)
        job = fixtures.ReplayJob()
        job.get = lambda: image
        context = SimpleNamespace(
            tasker=SimpleNamespace(controller=SimpleNamespace(post_screencap=lambda: job), stopping=False),
            get_node_data=lambda name: None,
            run_recognition_direct=Mock(),
        )
        states = [
            BattleState.ULTIMATE_TAP_CONFIRMATION,
            BattleState.UNKNOWN,
            BattleState.ULTIMATE_TAP_CONFIRMATION,
            BattleState.INOPERABLE,
            BattleState.ULTIMATE_TAP_CONFIRMATION,
            BattleState.ULTIMATE_SWIPE_CONFIRMATION,
            BattleState.VICTORY,
        ]
        with (
            patch("custom.action.battle.BattleFrameRecognizer.analyze", side_effect=states),
            patch("custom.action.battle.MaaBattleExecutor") as executor,
            patch("custom.action.battle.BoardPreviewer") as preview,
            patch(
                "custom.action.battle.BattleSession", return_value=BattleSession(clock=clock.clock, sleep=clock.sleep)
            ),
        ):
            result = BattleAction().run(context, SimpleNamespace(custom_action_param="{}"))
        self.assertTrue(result.success)
        self.assertEqual(
            executor.return_value.execute_ultimate_confirmation.call_args_list,
            [call(BattleState.ULTIMATE_TAP_CONFIRMATION)] * 2 + [call(BattleState.ULTIMATE_SWIPE_CONFIRMATION)],
        )
        executor.return_value.execute_once.assert_not_called()
        preview.return_value.recognize_board.assert_not_called()
