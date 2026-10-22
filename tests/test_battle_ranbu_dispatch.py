"""验证完整乱舞轨迹不重复查询原生状态，等待后仍响应停止。"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import test_battle_execution as fixtures  # noqa: E402
from battle.execution import MaaBattleExecutor  # noqa: E402
from battle.ranbu import ranbu_sweep_points  # noqa: E402
from battle.strategy import StrategyDecision  # noqa: E402


class RanbuDispatchTests(unittest.TestCase):
    """只优化状态查询次数，不丢触点或省略等待后的停止检查。"""

    def setUp(self) -> None:
        """提供真实自由布局轨迹及独立回放时钟。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        self.observation = fixture.upper_yellow_observation()
        self.points = ((326, 388), *ranbu_sweep_points(self.observation.board))
        self.clock = fixtures.ReplayClock()

    def executor(self, controller, *, stop=lambda: False, sleep=None) -> MaaBattleExecutor:
        """替换设备与时钟，保留真实逐点派发代码。"""
        return MaaBattleExecutor(
            controller,
            fixtures.MarkerReplay(0),
            lambda: None,
            lambda: True,
            stop,
            clock=self.clock.clock,
            sleep=self.clock.sleep if sleep is None else sleep,
        )

    def test_each_move_queries_native_result_once(self) -> None:
        """保存诊断和处理失败复用同一个原生结果，不查询两遍。"""
        queries = []

        class NativeResultJob:
            """记录每个作业的原生状态读取次数。"""

            def wait(self):
                """回放作业已完成。"""
                return self

            @property
            def succeeded(self):
                """一次属性读取对应一次原生状态查询。"""
                queries.append(True)
                return True

        class CountingController(fixtures.ReplayController):
            """每个移动返回独立作业，触点仍由既有控制器记录。"""

            def post_touch_move(self, *point):
                """记录完整轨迹后返回可计数作业。"""
                super().post_touch_move(*point)
                return NativeResultJob()

        controller = CountingController()
        events = []
        self.assertTrue(
            self.executor(controller)._move_timed_path(self.points, self.observation, 0.0, 0.4, events=events)
        )
        self.assertGreater(controller.moves, 200)
        self.assertEqual(len(queries), controller.moves)
        self.assertEqual(len(events), controller.moves)
        self.assertTrue(all(event["succeeded"] for event in events))

    def test_late_schedule_checks_stop_once_per_point(self) -> None:
        """已经落后于时间表时，不在同一触点间重复读取停止状态。"""
        checks = []

        def stop():
            """模拟原生查询有耗时，但不触发停止。"""
            checks.append(True)
            self.clock.now += 0.002
            return False

        controller = fixtures.ReplayController()
        executor = self.executor(controller, stop=stop)
        self.assertTrue(executor._move_timed_path(self.points, self.observation, 0.0, 0.001))
        self.assertEqual(len(checks), controller.moves)

    def test_stop_during_scheduled_wait_prevents_next_move(self) -> None:
        """确实发生等待时，等待后必须再次查询停止，不能再发触点。"""
        stopped = False

        def sleep(seconds):
            """回放等待期间用户点击停止。"""
            nonlocal stopped
            self.clock.sleep(seconds)
            stopped = True

        controller = fixtures.ReplayController()
        executor = self.executor(controller, stop=lambda: stopped, sleep=sleep)
        self.assertFalse(executor._move_timed_path(self.points, self.observation, 0.0, 0.4))
        self.assertEqual(controller.moves, 0)

    def test_point_only_confirmation_keeps_dense_movement_using_current_size(self) -> None:
        """确认帧对象为空时仍按本帧确认尺寸插值，不退回六十四像素跳跃。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        observation = replace(self.observation, board=replace(self.observation.board, objects=[]))
        controller = fixtures.ReplayController()
        decision = StrategyDecision(
            "ranbu_link",
            "阶段",
            reserved_color="red",
            ranbu_start=(326, 388),
            ranbu_color="yellow",
            ranbu_frame_id=observation.frame_id,
            ranbu_body_size=(96, 100),
        )
        self.executor(controller).execute_once(observation, fixture.resolution, decision)
        positions = [point for kind, point in controller.actions if kind in ("down", "move")]
        self.assertGreater(controller.moves, 200)
        self.assertLessEqual(max(math.dist(a, b) for a, b in pairwise(positions)), 96 * 0.2 + math.sqrt(2))
        self.assertTrue(all(("move", point) in controller.actions for point in ranbu_sweep_points(observation.board)))
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_dense_step_scales_with_the_confirmed_body(self) -> None:
        """不同确认尺寸分别确定触点密度，不把饿鬼尺寸或位置固定下来。"""
        observation = replace(self.observation, board=replace(self.observation.board, objects=[]))
        counts = []
        for size in ((48, 50), (96, 100), (144, 150)):
            with self.subTest(size=size):
                controller = fixtures.ReplayController()
                self.assertTrue(
                    self.executor(controller)._move_timed_path(
                        self.points, observation, self.clock.now, 0.4, body_size=size
                    )
                )
                positions = [self.points[0], *(point for _, point in controller.actions)]
                expected = min(32, min(size) * 0.2) + math.sqrt(2)
                self.assertLessEqual(max(math.dist(a, b) for a, b in pairwise(positions)), expected)
                counts.append(controller.moves)
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])

    def test_invalid_confirmed_size_releases_without_moving(self) -> None:
        """无效密度依据不能退回大步跳跃，也不能让触点留在按下状态。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        controller = fixtures.ReplayController()
        decision = StrategyDecision("ranbu_link", "阶段", ranbu_body_size=(float("nan"), 100))
        with self.assertRaisesRegex(ValueError, "采样尺寸"):
            self.executor(controller).execute_once(self.observation, fixture.resolution, decision)
        self.assertEqual(controller.moves, 0)
        self.assertEqual(controller.actions[-1], ("up", ()))


if __name__ == "__main__":
    unittest.main()
