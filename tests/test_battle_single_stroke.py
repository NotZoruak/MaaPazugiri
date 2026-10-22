"""验证普通连线整笔完成后只在松手前截图一次，不读取连接数量。"""

from __future__ import annotations

import unittest

import numpy as np
import test_battle_execution as execution_fixture
from test_battle_execution import MarkerReplay, ReplayClock, ReplayController, ReplayJob
from vision.model import ChainEndpointMarker, ChainEndpointResult


class SingleStrokeTests(unittest.TestCase):
    """复用纯图片输入边界，实际运行普通连线执行器。"""

    def setUp(self) -> None:
        self.fixture = execution_fixture.BattleExecutionTests()
        self.fixture.setUp()

    def run_link(self, controller, previewer, *, stopping=lambda: False):
        """注入离线设备边界，不替换轨迹执行和手数维护。"""
        from battle.execution import MaaBattleExecutor

        clock = ReplayClock()
        executor = MaaBattleExecutor(
            controller,
            previewer,
            lambda: self.fixture.confirmations.append(True),
            lambda: True,
            stopping,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        executor.execute_once(self.fixture.observation, self.fixture.resolution, self.fixture.decision)
        return clock

    def test_finishes_all_moves_then_captures_once_before_release(self) -> None:
        """截图与识别必须在七段移动后、松手前，不能在中途插入。"""
        controller, replay = ReplayController(), MarkerReplay(0)
        capture = controller.post_screencap
        calls = []

        def capture_once():
            """检查实际输入时序，回放一张新图。"""
            self.assertEqual(controller.moves, 7)
            self.assertEqual(controller.actions[-1], ("move", (150, 160)))
            calls.append("capture")
            return capture()

        def recognize(image, board, *, move=None):
            """菱形在当前整盘内独立定位，不要求落在最后目标。"""
            self.assertIsNone(move)
            calls.append("recognize")
            marker = ChainEndpointMarker("普通", (70, 80), (65, 75, 10, 10), 0.99)
            return ChainEndpointResult("松手前新帧", "detected", [marker])

        controller.post_screencap = capture_once
        replay.recognize_chain_endpoint = recognize
        self.run_link(controller, replay)
        self.assertEqual(calls, ["capture", "recognize"])
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(self.fixture.confirmations, [True])

    def test_missing_marker_does_not_retry_or_increment(self) -> None:
        """无菱形时松手，整笔不回退重滑，也不累计手数。"""
        for markers, status in (([], "not_detected"), ([], "uncertain")):
            with self.subTest(status=status):
                controller, replay, captures = ReplayController(), MarkerReplay(None), []
                capture = controller.post_screencap

                def capture_once(captures=captures, capture=capture):
                    """只记录次数，不发送设备操作。"""
                    captures.append(True)
                    return capture()

                controller.post_screencap = capture_once
                replay.recognize_chain_endpoint = lambda *args, status=status, markers=markers, **kwargs: (
                    ChainEndpointResult("当前帧", status, markers)
                )
                self.run_link(controller, replay)
                self.assertEqual(controller.moves, 7)
                self.assertEqual(len(captures), 1)
                self.assertEqual(self.fixture.confirmations, [])
                self.assertEqual(controller.actions[-1], ("up", ()))

    def test_start_and_end_diamonds_only_increment_one_operation(self) -> None:
        """起点终点各有菱形时不要求唯一，本次仍只加一手。"""
        controller, replay = ReplayController(), MarkerReplay(0)
        markers = [
            ChainEndpointMarker("普通", (70, 80), (65, 75, 10, 10), 0.99),
            ChainEndpointMarker("普通", (150, 160), (145, 155, 10, 10), 0.99),
        ]
        replay.recognize_chain_endpoint = lambda *args, **kwargs: ChainEndpointResult("当前帧", "uncertain", markers)
        self.run_link(controller, replay)
        self.assertEqual(self.fixture.confirmations, [True])
        self.assertEqual(controller.moves, 7)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_capture_failure_releases_without_retrying(self) -> None:
        """末尾截图失败时安全松手，不重截、不用旧图累计手数。"""
        controller, replay, captures = ReplayController(), MarkerReplay(0), []

        def failed_capture():
            """回放截图失败，get不应调用。"""
            captures.append(True)
            return ReplayJob(False)

        controller.post_screencap = failed_capture
        self.run_link(controller, replay)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(len(captures), 1)
        self.assertEqual(replay.checks, 0)
        self.assertEqual(self.fixture.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_stop_during_capture_prevents_recognition_and_increment(self) -> None:
        """截图返回前收到停止，仍松手但不再识别和累计手数。"""
        controller, replay, stopped = ReplayController(), MarkerReplay(0), [False]
        capture = controller.post_screencap

        def capture_then_stop():
            """在截图边界回放用户停止。"""
            stopped[0] = True
            return capture()

        controller.post_screencap = capture_then_stop
        self.run_link(controller, replay, stopping=lambda: stopped[0])
        self.assertEqual(replay.checks, 0)
        self.assertEqual(self.fixture.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_recognition_error_still_releases_and_does_not_increment(self) -> None:
        """单次菱形识别异常不能留下按住状态。"""
        controller, replay = ReplayController(), MarkerReplay(0)

        def broken_recognition(*args, **kwargs):
            """回放识别器异常。"""
            raise RuntimeError("识别异常")

        replay.recognize_chain_endpoint = broken_recognition
        with self.assertRaisesRegex(RuntimeError, "识别异常"):
            self.run_link(controller, replay)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(self.fixture.confirmations, [])

    def test_empty_capture_does_not_recognize_or_increment(self) -> None:
        """空图不能触发识别，也不能退回使用规划图。"""
        controller, replay = ReplayController(), MarkerReplay(0)
        job = ReplayJob()
        job.get = lambda: np.zeros((0, 0, 3), dtype=np.uint8)
        controller.post_screencap = lambda: job
        self.run_link(controller, replay)
        self.assertEqual(replay.checks, 0)
        self.assertEqual(self.fixture.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_stop_during_recognition_prevents_increment(self) -> None:
        """识别返回前停止，已有菱形也不再累计本次手数。"""
        controller, replay, stopped = ReplayController(), MarkerReplay(0), [False]

        def recognize_then_stop(*args, **kwargs):
            """在当前帧识别边界回放停止。"""
            stopped[0] = True
            marker = ChainEndpointMarker("普通", (70, 80), (65, 75, 10, 10), 0.99)
            return ChainEndpointResult("当前帧", "detected", [marker])

        replay.recognize_chain_endpoint = recognize_then_stop
        self.run_link(controller, replay, stopping=lambda: stopped[0])
        self.assertEqual(self.fixture.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_release_failure_does_not_increment(self) -> None:
        """即使已识别到菱形，释放失败仍上报错误，不先加手数。"""
        controller, replay = ReplayController(), MarkerReplay(0)
        controller.post_touch_up = lambda: ReplayJob(False)
        with self.assertRaisesRegex(RuntimeError, "释放连线触点失败"):
            self.run_link(controller, replay)
        self.assertEqual(self.fixture.confirmations, [])
