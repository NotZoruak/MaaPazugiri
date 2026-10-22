"""验证普通整笔连线与乱舞扫动的触控、停止及手数边界。"""

from __future__ import annotations

import math
import sys
import unittest
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.candidates import CandidateRoute  # noqa: E402
from battle.execution import MaaBattleExecutor  # noqa: E402
from battle.observation import BattleObservation  # noqa: E402
from battle.ranbu import ranbu_sweep_points  # noqa: E402
from battle.strategy import BattlePhase, StageResolution, StrategyDecision, StrategyDocument, StrategyMetrics  # noqa: E402
from vision.model import (  # noqa: E402
    BoardObject,
    BoardResult,
    ChainEndpointMarker,
    ChainEndpointResult,
    ChainMovePositions,
)

POINTS = ((10, 20), (30, 40), (50, 60), (70, 80), (90, 100), (110, 120), (130, 140), (150, 160))
MARKER = ChainEndpointMarker("普通", (70, 80), (65, 75, 10, 10), 0.99)


class ReplayJob:
    """提供已完成的离线任务，不发送设备命令。"""

    def __init__(self, succeeded: bool = True) -> None:
        self.succeeded = succeeded

    def wait(self) -> ReplayJob:
        """直接返回已完成状态。"""
        return self

    def get(self) -> np.ndarray:
        """返回用于标志识别回放的非空图。"""
        return np.zeros((180, 180, 3), dtype=np.uint8)


class ReplayController:
    """记录实际执行器发出的完整触控序列。"""

    def __init__(self, *, fail_move: int | None = None) -> None:
        self.actions: list[tuple[str, tuple[int, ...]]] = []
        self.moves = 0
        self.fail_move = fail_move

    def post_touch_down(self, *point: int) -> ReplayJob:
        """记录按下位置。"""
        self.actions.append(("down", point))
        return ReplayJob()

    def post_touch_move(self, *point: int) -> ReplayJob:
        """记录移动位置，并可回放指定移动失败。"""
        self.moves += 1
        self.actions.append(("move", point))
        return ReplayJob(self.moves != self.fail_move)

    def post_touch_up(self) -> ReplayJob:
        """记录释放触控。"""
        self.actions.append(("up", ()))
        return ReplayJob()

    def post_screencap(self) -> ReplayJob:
        """返回离线截图任务。"""
        return ReplayJob()


class ReplayClock:
    """用假时钟检查确认期限，不让测试依赖真实等待。"""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        """返回已推进的测试时间。"""
        return self.now

    def sleep(self, seconds: float) -> None:
        """模拟等待经过。"""
        self.now += seconds


class MarkerReplay:
    """按检查次数回放新菱形出现时机。"""

    def __init__(self, appears_at: int | None) -> None:
        self.appears_at = appears_at
        self.checks = 0
        self.checked_moves: list[ChainMovePositions] = []
        self.failures: list[dict[str, object]] = []
        self.ranbu_reports: list[dict[str, object]] = []

    def recognize_chain_endpoint(
        self, image: np.ndarray, board: BoardResult, *, move: ChainMovePositions | None = None
    ) -> ChainEndpointResult:
        """只回放当前帧标志，不连接设备。"""
        seen = self.appears_at is not None and self.checks >= self.appears_at
        self.checks += 1
        if move is not None:
            self.checked_moves.append(move)
        return ChainEndpointResult(
            board.frame_id,
            "detected" if seen else "not_detected",
            [MARKER] if seen else [],
            move_outcome="target_reached" if seen else "unknown",
        )

    def save_link_failure(self, image, board, move, result, details) -> None:
        """记录失败诊断内容，测试中不产生图片文件。"""
        self.failures.append(details)

    def save_ranbu_report(self, image, board, start, color, previous_color, points) -> None:
        """执行器回放不写磁盘，只保留原有输入记录。"""

    def save_ranbu_execution_report(self, before, held, report, *, pass_images=(), grab_images=()) -> None:
        """保存真实执行器生成的输入记录，不写入图片。"""
        self.ranbu_reports.append({**report, "held_image_available": held is not None})


class BattleExecutionTests(unittest.TestCase):
    """核对完整路线、计数与触控释放的业务边界。"""

    def setUp(self) -> None:
        self.confirmations: list[bool] = []
        self.board = BoardResult("当前帧", (180, 180), (0, 0, 180, 180), "测试", {})
        self.observation = BattleObservation(
            "当前帧", BattlePhase.OPERABLE, np.zeros((180, 180, 3), dtype=np.uint8), self.board, StrategyMetrics()
        )
        self.resolution = StageResolution("阶段", StrategyDocument.builtin_default().stages[0])
        self.decision = StrategyDecision(
            "normal_link", "阶段", route=CandidateRoute("red", tuple(range(1, 9)), POINTS, 0.5, 200.0)
        )

    def test_down_submission_wait_and_negative_result_all_attempt_release(self) -> None:
        """按下可能已到达设备时，无论提交、等待还是结果失败都尝试释放。"""
        for mode in ("submit_exception", "wait_exception", "failed"):
            with self.subTest(mode=mode):

                class DownFailureController(ReplayController):
                    """回放按下阶段异常，实际不连接设备。"""

                    def __init__(self, failure_mode: str) -> None:
                        super().__init__()
                        self.failure_mode = failure_mode

                    def post_touch_down(self, *point: int) -> ReplayJob:
                        """记录可能已经生效的按下，并在指定边界失败。"""
                        self.actions.append(("down", point))
                        if self.failure_mode == "submit_exception":
                            raise RuntimeError("测试按下提交异常")
                        if self.failure_mode == "failed":
                            return ReplayJob(False)

                        class FailingWait(ReplayJob):
                            """设备可能收到按下，但等待返回时连接异常。"""

                            def wait(self) -> ReplayJob:
                                """回放无法取得按下结果。"""
                                raise RuntimeError("测试按下等待异常")

                        return FailingWait()

                controller, preview = DownFailureController(mode), MarkerReplay(0)
                with self.assertRaises(RuntimeError):
                    self.executor(controller, preview)._hold_and_move(
                        POINTS,
                        self.observation,
                        check_marker=False,
                        hold_start_seconds=0.0,
                        gesture_duration=0.8,
                        capture_ranbu=True,
                    )
                self.assertEqual(controller.actions, [("down", POINTS[0]), ("up", ())])
                self.assertFalse(preview.ranbu_reports[0]["dispatch_completed"])
                self.assertEqual(preview.ranbu_reports[0]["stop_reason"], "input_failed")

    def executor(
        self, controller: ReplayController, replay: MarkerReplay, *, stop_after: int = 100, can_start: bool = True
    ) -> MaaBattleExecutor:
        """使用实际执行器，仅替换设备与图像输入。"""
        clock = ReplayClock()
        return MaaBattleExecutor(
            controller,
            replay,  # type: ignore[arg-type]
            lambda: self.confirmations.append(True),
            lambda: can_start,
            lambda: controller.moves >= stop_after,
            clock=clock.clock,
            sleep=clock.sleep,
        )

    def test_normal_link_counts_success_once_after_completing_the_route(self) -> None:
        """成功证据可早于末点，计数只增加一次且完整执行后释放。"""
        controller = ReplayController()
        self.executor(controller, MarkerReplay(0)).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(self.confirmations, [True])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_unconfirmed_gesture_uses_the_opposite_end_on_the_next_fresh_route(self) -> None:
        """整笔未确认后只保留换向状态，下一次使用新帧路线的另一端起步。"""
        controller = ReplayController()
        executor = self.executor(controller, MarkerReplay(None))
        executor.execute_once(self.observation, self.resolution, self.decision)
        next_points = tuple((x + 3, y + 2) for x, y in POINTS)
        next_decision = replace(self.decision, route=replace(self.decision.route, points=next_points))
        executor.execute_once(self.observation, self.resolution, next_decision)
        starts = [point for kind, point in controller.actions if kind == "down"]
        self.assertEqual(starts, [POINTS[0], next_points[-1]])

    def test_slow_ranbu_controller_keeps_intermediate_touch_points(self) -> None:
        """输入较慢时宁可超过目标时长，也不能跳过中间触点。"""
        clock = ReplayClock()
        controller = ReplayController()
        original_move = controller.post_touch_move

        def slow_move(*point: int) -> ReplayJob:
            """模拟移动命令实际需要四毫秒。"""
            clock.now += 0.004
            return original_move(*point)

        controller.post_touch_move = slow_move
        executor = MaaBattleExecutor(
            controller,
            MarkerReplay(0),
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        executor.execute_once(self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertGreater(controller.moves, 18)
        points = [point for kind, point in controller.actions if kind in ("down", "move")]
        self.assertTrue(all(math.dist(a, b) <= 80 for a, b in pairwise(points)))
        self.assertLessEqual(clock.now, max(0.4, controller.moves * 0.004) + 0.01)
        self.assertTrue(
            all(
                ("move", point) in controller.actions
                for point in ranbu_sweep_points(self.upper_yellow_observation().board)
            )
        )

        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_faster_ranbu_keeps_the_entire_original_touch_sequence(self) -> None:
        """四百毫秒只压缩等待，与八百毫秒逐点经过相同位置。"""
        observation = self.upper_yellow_observation()
        points = ((326, 388), *ranbu_sweep_points(observation.board))
        sequences = []
        for duration in (0.8, 0.4):
            clock, controller = ReplayClock(), ReplayController()
            executor = MaaBattleExecutor(
                controller,
                MarkerReplay(0),
                lambda: None,
                lambda: True,
                lambda: False,
                clock=clock.clock,
                sleep=clock.sleep,
            )
            self.assertTrue(executor._move_timed_path(points, observation, 0.0, duration))
            self.assertAlmostEqual(clock.now, duration)
            sequences.append(controller.actions)
        self.assertGreater(len(sequences[0]), 200)
        self.assertEqual(sequences[0], sequences[1])

    def test_ranbu_executes_the_confirmed_point_without_selecting_a_new_one(self) -> None:
        """预选点通过颜色检查后，执行器不能因编号或中心漂移而换点。"""
        from test_battle_candidates import ghoul

        current_board = BoardResult(
            "本帧",
            (720, 1280),
            (0, 210, 720, 780),
            "离线",
            {},
            objects=[ghoul(0, 500, 600, "yellow"), ghoul(99, 110, 400, "blue")],
        )
        observation = replace(self.upper_yellow_observation(), board=current_board)
        controller = ReplayController()
        self.executor(controller, MarkerReplay(0), stop_after=10000).execute_once(
            observation,
            self.resolution,
            StrategyDecision("ranbu_link", "阶段", reserved_color="red", ranbu_start=(100, 400)),
        )
        self.assertEqual(controller.actions[0], ("down", (100, 400)))
        self.assertTrue(all(("move", point) in controller.actions for point in ranbu_sweep_points(current_board)))

    def test_confirmed_point_becoming_reserved_is_not_replaced_or_clicked(self) -> None:
        """预选点当前证据不再允许时，执行器不能自行挑另一个未经确认的点。"""
        from test_battle_candidates import ghoul

        current_board = replace(self.board, objects=[ghoul(1, 100, 100, "red"), ghoul(2, 50, 50, "blue")])
        observation = replace(self.observation, board=current_board)
        controller = ReplayController()
        self.executor(controller, MarkerReplay(0)).execute_once(
            observation,
            self.resolution,
            StrategyDecision("ranbu_link", "阶段", reserved_color="red", ranbu_start=(100, 100)),
        )
        self.assertEqual(controller.actions, [])

    def test_consecutive_ranbu_sweeps_rotate_colors_using_fresh_frame_positions(self) -> None:
        """跨帧仅记上次颜色；三轮分别选蓝、绿、黄，触点必须来自各自新帧。"""
        from test_battle_candidates import ghoul

        controller = ReplayController()
        executor = self.executor(controller, MarkerReplay(0), stop_after=10000)
        for offset in (0, 10, 20):
            objects = [
                ghoul(1, 100 + offset, 400, "blue"),
                ghoul(2, 300 + offset, 400, "yellow"),
                ghoul(3, 500 + offset, 400, "green"),
                ghoul(0, 400, 700, "red"),
            ]
            current_board = BoardResult(str(offset), (720, 1280), (0, 210, 720, 780), "离线", {}, objects=objects)
            observation = BattleObservation(
                str(offset), BattlePhase.RANBU, np.zeros((1280, 720, 3), np.uint8), current_board, StrategyMetrics()
            )
            executor.execute_once(
                observation, self.resolution, StrategyDecision("ranbu_link", "阶段", reserved_color="red")
            )
        self.assertEqual(
            [point for kind, point in controller.actions if kind == "down"], [(100, 400), (510, 400), (320, 400)]
        )

    def test_skipped_ranbu_gesture_does_not_advance_color_rotation(self) -> None:
        """未获准按下的轮次不能当作已消除一种颜色。"""
        from test_battle_candidates import ghoul

        allowed = [False]
        clock, controller = ReplayClock(), ReplayController()
        executor = MaaBattleExecutor(
            controller,
            MarkerReplay(0),
            lambda: None,
            lambda: allowed[0],
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        current_board = BoardResult(
            "当前帧",
            (720, 1280),
            (0, 210, 720, 780),
            "离线",
            {},
            objects=[ghoul(1, 100, 400, "blue"), ghoul(2, 300, 400, "green")],
        )
        observation = BattleObservation(
            "当前帧", BattlePhase.RANBU, np.zeros((1280, 720, 3), np.uint8), current_board, StrategyMetrics()
        )
        decision = StrategyDecision("ranbu_link", "阶段", reserved_color="red")
        executor.execute_once(observation, self.resolution, decision)
        self.assertEqual(controller.actions, [])
        allowed[0] = True
        executor.execute_once(observation, self.resolution, decision)
        self.assertEqual(controller.actions[0], ("down", (100, 400)))

    def upper_yellow_observation(self) -> BattleObservation:
        """用本次失败现场的真实尺寸与触点构造离线输入。"""
        points = ((326, 388), (335, 310))
        objects = [
            BoardObject(
                index,
                "ghoul",
                "yellow",
                None,
                "ghoul_yellow",
                "黄色饿鬼",
                center,
                (center[0] - 48, center[1] - 50, 96, 100),
                {"width": 96, "height": 100},
                [],
                point,
                "identified",
            )
            for index, center, point in ((1, (326.16, 407.06), points[0]), (2, (335.13, 329.27), points[1]))
        ]
        board = BoardResult("当前帧", (720, 1280), (0, 210, 720, 780), "失败现场回放", {}, objects=objects)
        return replace(self.observation, board=board, image=np.zeros((1280, 720, 3), dtype=np.uint8))

    def test_normal_link_moves_continuously_through_the_target_body(self) -> None:
        """不能一步跳到眼睛，必须以有时间间隔的短段经过目标身体。"""
        observation = self.upper_yellow_observation()
        clock = ReplayClock()
        controller = ReplayController()
        timestamps: list[float] = []
        move = controller.post_touch_move

        def record_move(*point: int) -> ReplayJob:
            """记录每个真实移动命令的派发时间。"""
            timestamps.append(clock.clock())
            return move(*point)

        controller.post_touch_move = record_move
        executor = MaaBattleExecutor(
            controller,
            MarkerReplay(0),
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        self.assertTrue(executor._hold_and_move(((326, 388), (335, 310)), observation, hold_start_seconds=0))
        points = [point for action, point in controller.actions if action == "move"]
        self.assertGreater(len(points), 1)
        self.assertEqual(points[-1], (335, 310))
        self.assertTrue(all(math.dist(a, b) <= 20 for a, b in pairwise([(326, 388), *points])))
        self.assertLess(min(math.dist(point, (335.13, 329.27)) for point in points[:-1]), 8)
        self.assertGreaterEqual(timestamps[-1] - timestamps[0], 0.05)
        self.assertLessEqual(timestamps[-1] - timestamps[0], 0.2)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_stop_during_continuous_movement_releases_before_the_target(self) -> None:
        """用户停止能中断段内移动，不再截图确认或继续移动。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        executor = self.executor(controller, replay, stop_after=2)
        succeeded = executor._hold_and_move(
            ((326, 388), (335, 310)), self.upper_yellow_observation(), hold_start_seconds=0
        )
        self.assertFalse(succeeded)
        self.assertEqual(controller.moves, 2)
        self.assertNotIn(("move", (335, 310)), controller.actions)
        self.assertEqual(replay.checks, 0)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_continuous_movement_failure_releases_the_touch(self) -> None:
        """任一中间移动失败都释放触控并上报，不继续后面的移动。"""
        controller = ReplayController(fail_move=2)
        executor = self.executor(controller, MarkerReplay(0))
        with self.assertRaisesRegex(RuntimeError, "移动连线触点失败"):
            executor._hold_and_move(((326, 388), (335, 310)), self.upper_yellow_observation(), hold_start_seconds=0)
        self.assertEqual(controller.moves, 2)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_no_marker_does_not_increment_the_counter(self) -> None:
        """整笔完成后未检到菱形，不重试和虚增手数。"""
        controller = ReplayController()
        replay = MarkerReplay(None)
        self.executor(controller, replay).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(replay.checks, 1)
        self.assertEqual(self.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(len(replay.failures), 1)
        self.assertEqual(replay.failures[0]["confirmation_mode"], "after_full_gesture")

    def test_user_stop_interrupts_remaining_moves_and_releases(self) -> None:
        """用户停止优先于剩余候选路线。"""
        controller = ReplayController()
        executor = self.executor(controller, MarkerReplay(0), stop_after=2)
        executor._hold_and_move(POINTS, self.observation, hold_start_seconds=0.0)
        self.assertEqual(controller.moves, 2)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_move_failure_releases_and_does_not_report_success(self) -> None:
        """后续移动失败时，即使已有菱形也要释放并上报异常。"""
        controller = ReplayController(fail_move=4)
        with self.assertRaisesRegex(RuntimeError, "移动连线触点失败"):
            self.executor(controller, MarkerReplay(0)).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 4)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(self.confirmations, [])

    def test_ranbu_moves_without_checking_normal_link_markers(self) -> None:
        """关闭普通标志检查时保持乱舞完整移动行为。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        succeeded = self.executor(controller, replay)._hold_and_move(
            POINTS, self.observation, check_marker=False, hold_start_seconds=0.0
        )
        self.assertFalse(succeeded)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(replay.checks, 0)

    def test_ranbu_sweep_is_a_continuous_gesture_without_the_normal_start_hold(self) -> None:
        """全盘扫动跨越多个画面帧，零起点长按，也不调用普通菱形确认。"""
        clock = ReplayClock()
        controller = ReplayController()
        replay = MarkerReplay(0)
        first_move_times: list[float] = []
        original_move = controller.post_touch_move

        def record_move(*point: int) -> ReplayJob:
            """记录第一处移动时间，避免重新加入普通连线长按。"""
            first_move_times.append(clock.clock())
            return original_move(*point)

        controller.post_touch_move = record_move
        executor = MaaBattleExecutor(
            controller,
            replay,
            lambda: self.confirmations.append(True),
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        executor.execute_once(self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertGreater(controller.moves, 18)
        self.assertGreaterEqual(clock.now, 0.4)
        self.assertLessEqual(clock.now, 0.42)
        self.assertLess(first_move_times[0], 0.05)
        points = [point for kind, point in controller.actions if kind in ("down", "move")]
        self.assertTrue(all(math.dist(a, b) <= 80 for a, b in pairwise(points)))
        self.assertEqual(replay.checks, 0)
        self.assertEqual(self.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_ranbu_records_actual_moves_and_releases_without_capture(self) -> None:
        """记录派发时间，整盘扫完直接松手，不再追加按住截图。"""
        clock, replay = ReplayClock(), MarkerReplay(0)

        class CaptureController(ReplayController):
            """截图需要四十毫秒，验证其发生在移动末尾和松手之前。"""

            def post_screencap(self) -> ReplayJob:
                """用假时钟回放截图耗时。"""
                self.actions.append(("capture", ()))
                clock.now += 0.04
                return ReplayJob()

        controller = CaptureController()
        executor = MaaBattleExecutor(
            controller, replay, lambda: None, lambda: True, lambda: False, clock=clock.clock, sleep=clock.sleep
        )
        executor.execute_once(self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertNotIn(("capture", ()), controller.actions)
        self.assertEqual(len(replay.ranbu_reports), 1)
        report = replay.ranbu_reports[0]
        self.assertTrue(report["dispatch_completed"])
        self.assertFalse(report["game_connection_verified"])
        self.assertFalse(report["held_image_available"])
        self.assertAlmostEqual(report["movement_elapsed_ms"], 400, places=2)
        self.assertAlmostEqual(report["capture_elapsed_ms"], 0, places=2)
        events = report["events"]
        self.assertEqual(
            [entry["kind"] for entry in events], [kind for kind, _ in controller.actions if kind != "capture"]
        )
        self.assertEqual(
            [entry["point"] for entry in events if entry["kind"] == "move"],
            [list(point) for kind, point in controller.actions if kind == "move"],
        )
        self.assertTrue(all(entry["succeeded"] for entry in events))

    def test_interrupted_ranbu_reports_incomplete_dispatch(self) -> None:
        """停止后的报告必须标为未完成，不能把少量触点称为整盘派发成功。"""
        controller, replay = ReplayController(), MarkerReplay(0)
        executor = self.executor(controller, replay, stop_after=3)
        executor.execute_once(self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(len(replay.ranbu_reports), 1)
        self.assertFalse(replay.ranbu_reports[0]["dispatch_completed"])
        self.assertFalse(replay.ranbu_reports[0]["held_image_available"])

    def test_ranbu_never_calls_optional_capture_even_if_it_would_fail(self) -> None:
        """即使截图接口会失败或抛错，单遍扫动也不会调用它。"""
        for mode in ("failed", "exception"):
            with self.subTest(mode=mode):

                class CaptureFailureController(ReplayController):
                    """仅让诊断截图失败，不改变触控接口。"""

                    def __init__(self, capture_mode: str) -> None:
                        super().__init__()
                        self.capture_mode = capture_mode

                    def post_screencap(self) -> ReplayJob:
                        """回放两种截图失败边界。"""
                        if self.capture_mode == "exception":
                            raise RuntimeError("测试截图异常")
                        return ReplayJob(False)

                controller, replay = CaptureFailureController(mode), MarkerReplay(0)
                executor = self.executor(controller, replay, stop_after=10000)
                executor.execute_once(
                    self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段")
                )
                self.assertEqual(controller.actions[-1], ("up", ()))
                self.assertTrue(replay.ranbu_reports[0]["dispatch_completed"])
                self.assertEqual(replay.ranbu_reports[0]["capture_status"], "not_attempted")
                self.assertFalse(replay.ranbu_reports[0]["held_image_available"])

    def test_ranbu_move_failure_releases_and_reports_failed_input(self) -> None:
        """移动失败仍上报实际失败触点，不能写成完整派发。"""
        controller, replay = ReplayController(fail_move=3), MarkerReplay(0)
        executor = self.executor(controller, replay)
        with self.assertRaisesRegex(RuntimeError, "移动连线触点失败"):
            executor.execute_once(
                self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段")
            )
        self.assertEqual(controller.actions[-1], ("up", ()))
        report = replay.ranbu_reports[0]
        self.assertFalse(report["dispatch_completed"])
        self.assertEqual(report["stop_reason"], "input_failed")
        self.assertFalse(report["events"][-2]["succeeded"])
        self.assertEqual(report["events"][-1]["kind"], "up")

    def test_ranbu_deadline_interrupts_a_sweep_inside_a_segment(self) -> None:
        """乱舞输入期限结束时立即停止后续段内移动并松手。"""
        clock = ReplayClock()
        controller = ReplayController()
        executor = MaaBattleExecutor(
            controller,
            MarkerReplay(0),
            lambda: None,
            lambda: clock.now < 0.08,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        executor.execute_once(self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertGreater(clock.now, 0)
        self.assertLessEqual(clock.now, 0.1)
        self.assertNotIn(("move", (719, 989)), controller.actions)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_user_stop_interrupts_ranbu_before_the_first_sweep_corner(self) -> None:
        """停止能打断较长的第一段，不能先跳到角点再检查。"""
        controller = ReplayController()
        executor = self.executor(controller, MarkerReplay(0), stop_after=3)
        executor.execute_once(self.upper_yellow_observation(), self.resolution, StrategyDecision("ranbu_link", "阶段"))
        self.assertEqual(controller.moves, 3)
        self.assertNotIn(("move", (0, 210)), controller.actions)
        self.assertEqual(controller.actions[-1], ("up", ()))

    def reserved_board(self) -> None:
        """回放本帧中全部为保留色的三只普通饿鬼。"""
        self.board.objects = [
            BoardObject(
                identifier,
                "ghoul",
                "red",
                None,
                "ghoul_red",
                "红色饿鬼",
                point,
                (point[0] - 5, point[1] - 5, 10, 10),
                {},
                [],
                point,
                "identified",
            )
            for identifier, point in ((1, (20, 30)), (2, (50, 60)), (3, (120, 150)))
        ]

    def test_all_reserved_ranbu_only_moves_to_one_other_ghoul_then_releases(self) -> None:
        """全为保留色时仅做两点短连，不扫全屏、不累计普通手数。"""
        self.reserved_board()
        controller = ReplayController()
        replay = MarkerReplay(0)
        decision = StrategyDecision("ranbu_short_link", "阶段", reserved_color="red")
        self.executor(controller, replay).execute_once(self.observation, self.resolution, decision)
        self.assertEqual(controller.actions[0], ("down", (20, 30)))
        self.assertEqual(controller.actions[-2:], [("move", (50, 60)), ("up", ())])
        points = [point for kind, point in controller.actions if kind == "move"]
        self.assertGreater(len(points), 1)
        self.assertTrue(all(20 <= x <= 50 and 30 <= y <= 60 for x, y in points))
        self.assertEqual(replay.checks, 0)
        self.assertEqual(self.confirmations, [])

    def test_ranbu_short_link_honors_stop_and_start_deadline(self) -> None:
        """乱舞短连同样遵守用户停止和新动作启动期限。"""
        self.reserved_board()
        decision = StrategyDecision("ranbu_short_link", "阶段", reserved_color="red")
        for stop_after, can_start in ((0, True), (100, False)):
            with self.subTest(stop_after=stop_after, can_start=can_start):
                controller = ReplayController()
                self.executor(controller, MarkerReplay(None), stop_after=stop_after, can_start=can_start).execute_once(
                    self.observation, self.resolution, decision
                )
                self.assertEqual(controller.actions, [])

    def test_ranbu_short_link_releases_on_move_failure(self) -> None:
        """两点短连移动失败时仍释放触控，不静默吞掉错误。"""
        self.reserved_board()
        controller = ReplayController(fail_move=1)
        decision = StrategyDecision("ranbu_short_link", "阶段", reserved_color="red")
        with self.assertRaisesRegex(RuntimeError, "移动连线触点失败"):
            self.executor(controller, MarkerReplay(None)).execute_once(self.observation, self.resolution, decision)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(self.confirmations, [])

    def test_ranbu_short_link_requires_two_distinct_trusted_reserved_ghouls(self) -> None:
        """对象不足、位置异常、颜色不一致或证据不足时不启用兜底。"""
        decision = StrategyDecision("ranbu_short_link", "阶段", reserved_color="red")
        for invalid in ("single", "color", "quality", "position", "same_point", "special"):
            with self.subTest(invalid=invalid):
                self.reserved_board()
                if invalid == "single":
                    self.board.objects = self.board.objects[:1]
                elif invalid == "color":
                    self.board.objects[0].color = "blue"
                elif invalid == "quality":
                    self.board.objects[0].quality = "uncertain"
                elif invalid == "position":
                    self.board.objects[0].candidate_point = (float("nan"), 30)
                elif invalid == "same_point":
                    for item in self.board.objects:
                        item.candidate_point = (20, 30)
                else:
                    self.board.objects[0].kind = "special_ghoul"
                    self.board.objects[0].subtype = "black"
                controller = ReplayController()
                self.executor(controller, MarkerReplay(None)).execute_once(self.observation, self.resolution, decision)
                self.assertEqual(controller.actions, [])


if __name__ == "__main__":
    unittest.main()
