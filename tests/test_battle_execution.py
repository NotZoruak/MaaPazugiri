"""验证普通连线不会因早期成功证据而丢掉剩余路线。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.candidates import CandidateRoute  # noqa: E402
from battle.execution import MaaBattleExecutor  # noqa: E402
from battle.observation import BattleObservation  # noqa: E402
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

    def recognize_chain_endpoint(
        self, image: np.ndarray, board: BoardResult, *, move: ChainMovePositions | None = None
    ) -> ChainEndpointResult:
        """只回放当前帧标志，不连接设备。"""
        seen = self.appears_at is not None and self.checks >= self.appears_at
        self.checks += 1
        if move is None:
            return ChainEndpointResult(board.frame_id, "not_detected")
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

    def test_early_marker_still_visits_the_entire_planned_route(self) -> None:
        """菱形即时或稍晚到达当前目标，都继续逐段完成八点路线。"""
        for appears_at in (0, 3):
            with self.subTest(appears_at=appears_at):
                self.confirmations.clear()
                controller = ReplayController()
                executor = self.executor(controller, MarkerReplay(appears_at))
                executor._hold_and_move(POINTS, self.observation, hold_start_seconds=0.0)
                self.assertEqual(
                    controller.actions,
                    [("down", POINTS[0]), *[("move", point) for point in POINTS[1:]], ("up", ())],
                )

    def test_normal_link_counts_success_once_after_completing_the_route(self) -> None:
        """成功证据可早于末点，计数只增加一次且完整执行后释放。"""
        controller = ReplayController()
        self.executor(controller, MarkerReplay(3)).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(self.confirmations, [True])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_no_marker_does_not_increment_the_counter(self) -> None:
        """首段始终没有接入证据时停止后续移动，不虚增普通手数。"""
        controller = ReplayController()
        replay = MarkerReplay(None)
        self.executor(controller, replay).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 1)
        self.assertEqual(self.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(len(replay.failures), 1)
        self.assertLessEqual(replay.failures[0]["attempts"], 31)

    def test_each_target_must_be_confirmed_before_the_next_move(self) -> None:
        """标志还停在前一只时复查当前目标，确认后才继续下一段。"""
        controller = ReplayController()
        replay = MarkerReplay(2)
        executor = self.executor(controller, replay)
        executor.execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(replay.checked_moves[:3], [ChainMovePositions(POINTS[0], POINTS[1])] * 3)
        self.assertEqual(replay.checked_moves[3], ChainMovePositions(POINTS[1], POINTS[2]))
        self.assertEqual(self.confirmations, [True])

    def test_marker_remaining_at_previous_target_does_not_confirm_the_next_target(self) -> None:
        """始终检测到旧端点也不能把第三只误认为已经接上。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        original = replay.recognize_chain_endpoint

        def recognize(image, board, *, move=None):
            """第一段成功，之后菱形始终留在上一目标。"""
            result = original(image, board, move=move)
            result.move_outcome = "target_reached" if controller.moves == 1 else "still_at_start"
            return result

        replay.recognize_chain_endpoint = recognize
        self.executor(controller, replay).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 2)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(self.confirmations, [True])
        self.assertEqual(replay.failures[0]["confirmed_targets"], 2)
        self.assertEqual(replay.failures[0]["target_index"], 3)

    def test_confirmation_never_starts_another_capture_at_the_deadline(self) -> None:
        """340毫秒首轮加10毫秒等待耗尽预算后，不再启动第二轮截图。"""
        clock = ReplayClock()
        controller = ReplayController()
        replay = MarkerReplay(1)

        def capture():
            """模拟每次截图与识别链路的长耗时。"""
            clock.now += 0.34
            return ReplayJob()

        controller.post_screencap = capture
        executor = MaaBattleExecutor(
            controller,
            replay,
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
            confirmation_timeout=0.35,
        )
        feedback = executor._confirm_move(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertFalse(feedback.reached)
        self.assertEqual(replay.checks, 1)

    def test_late_recognition_is_not_accepted_after_the_confirmation_deadline(self) -> None:
        """即使菱形正确，识别返回时已经超时也不能放行下一段。"""
        clock = ReplayClock()
        controller = ReplayController()
        replay = MarkerReplay(0)
        original = replay.recognize_chain_endpoint

        def recognize(image, board, *, move=None):
            """模拟识别返回晚于预算。"""
            clock.now += 0.4
            return original(image, board, move=move)

        replay.recognize_chain_endpoint = recognize
        executor = MaaBattleExecutor(
            controller,
            replay,
            lambda: None,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
            confirmation_timeout=0.35,
        )
        feedback = executor._confirm_move(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertFalse(feedback.reached)

    def test_expired_capture_does_not_start_another_recognition(self) -> None:
        """截图已耗尽预算时不再启动识别，避免继续增加无效等待。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        clock = ReplayClock()

        def capture():
            """模拟截图返回已经超过1.5秒。"""
            clock.now += 1.6
            return ReplayJob()

        controller.post_screencap = capture
        executor = MaaBattleExecutor(
            controller, replay, lambda: None, lambda: True, lambda: False, clock=clock.clock, sleep=clock.sleep
        )
        feedback = executor._confirm_move(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertFalse(feedback.reached)
        self.assertEqual(replay.checks, 0)

    def test_stop_during_capture_prevents_further_recognition(self) -> None:
        """截图期间收到停止后，不再启动识别或接受该目标。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        stopped = False

        def capture():
            """模拟等待截图时用户停止。"""
            nonlocal stopped
            stopped = True
            return ReplayJob()

        controller.post_screencap = capture
        executor = MaaBattleExecutor(controller, replay, lambda: None, lambda: True, lambda: stopped)
        feedback = executor._confirm_move(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertFalse(feedback.reached)
        self.assertEqual(replay.checks, 0)

    def test_default_budget_accepts_a_slow_ui_response(self) -> None:
        """默认预算容纳超过350毫秒的界面响应，确认后立即继续。"""
        clock = ReplayClock()
        controller = ReplayController()
        replay = MarkerReplay(1)

        def capture():
            """回放响应较慢的截图链路。"""
            clock.now += 0.4
            return ReplayJob()

        controller.post_screencap = capture
        executor = MaaBattleExecutor(
            controller, replay, lambda: None, lambda: True, lambda: False, clock=clock.clock, sleep=clock.sleep
        )
        feedback = executor._confirm_move(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertTrue(feedback.reached)
        self.assertEqual(replay.checks, 2)
        self.assertLess(feedback.elapsed_ms, 1500)

    def test_confirmation_can_continue_after_eight_checks(self) -> None:
        """第九次以后仍在总期限内时，允许继续确认目标。"""
        clock = ReplayClock()
        replay = MarkerReplay(10)
        executor = MaaBattleExecutor(
            ReplayController(), replay, lambda: None, lambda: True, lambda: False, clock=clock.clock, sleep=clock.sleep
        )
        feedback = executor._confirm_move(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertTrue(feedback.reached)
        self.assertEqual(replay.checks, 11)

    def test_failed_target_returns_to_a_fresh_marker_before_retrying(self) -> None:
        """重滑起点取自新截图中的实际菱形，不能沿用上个目标坐标。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        live_point = (37, 49)

        def recognize(image, board, *, move=None):
            """首段成功，下一段重滑后才成功；全盘搜索给出新位置。"""
            if move is None:
                marker = ChainEndpointMarker("普通", live_point, (32, 44, 10, 10), 0.99)
                return ChainEndpointResult("重查帧", "detected", [marker])
            reached = controller.moves == 1 or controller.moves >= 4
            return ChainEndpointResult(
                "确认帧", "detected", [MARKER], move_outcome="target_reached" if reached else "still_at_start"
            )

        replay.recognize_chain_endpoint = recognize
        self.executor(controller, replay)._hold_and_move(POINTS[:3], self.observation, hold_start_seconds=0.0)
        self.assertEqual(
            controller.actions,
            [
                ("down", POINTS[0]),
                ("move", POINTS[1]),
                ("move", POINTS[2]),
                ("move", live_point),
                ("move", POINTS[2]),
                ("up", ()),
            ],
        )

    def test_a_target_can_only_be_retried_twice(self) -> None:
        """首次失败后最多两次回退重滑，每次都重新定位菱形。"""
        controller = ReplayController()
        replay = MarkerReplay(0)
        live_points = iter(((37, 49), (39, 51)))
        reads: list[tuple[int, int]] = []

        def recognize(image, board, *, move=None):
            """第三只始终未接上，实际菱形位置在每次重查时变化。"""
            if move is None:
                point = next(live_points)
                reads.append(point)
                marker = ChainEndpointMarker("普通", point, (0, 0, 10, 10), 0.99)
                return ChainEndpointResult("重查帧", "detected", [marker])
            return ChainEndpointResult(
                "确认帧",
                "detected",
                [MARKER],
                move_outcome="target_reached" if controller.moves == 1 else "still_at_start",
            )

        replay.recognize_chain_endpoint = recognize
        self.executor(controller, replay)._hold_and_move(POINTS[:3], self.observation, hold_start_seconds=0.0)
        self.assertEqual(reads, [(37, 49), (39, 51)])
        self.assertEqual(controller.moves, 6)
        self.assertEqual(controller.actions[-1], ("up", ()))
        self.assertEqual(replay.failures[0]["retries"], 2)

    def test_first_attempt_and_two_retries_share_one_waiting_budget(self) -> None:
        """首次确认与两次重滑共享1.5秒，不能累计到四秒以上。"""
        controller = ReplayController()
        replay = MarkerReplay(None)
        clock = ReplayClock()

        def recognize(image, board, *, move=None):
            """菱形始终停在起点，允许回退但目标始终未接入。"""
            if move is None:
                return ChainEndpointResult("重查帧", "detected", [MARKER])
            return ChainEndpointResult("确认帧", "detected", [MARKER], move_outcome="still_at_start")

        replay.recognize_chain_endpoint = recognize
        executor = MaaBattleExecutor(
            controller, replay, lambda: None, lambda: True, lambda: False, clock=clock.clock, sleep=clock.sleep
        )
        feedback = executor._confirm_move_with_retries(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertFalse(feedback.reached)
        self.assertEqual(feedback.retries, 2)
        self.assertLessEqual(feedback.elapsed_ms, 1500)

    def test_first_attempt_still_allows_the_slowest_success_seen_in_logs(self) -> None:
        """首次确认保留约0.9秒，覆盖日志中约0.8秒的正常响应。"""
        controller = ReplayController()
        replay = MarkerReplay(1)
        clock = ReplayClock()

        def capture():
            """模拟两次各耗时400毫秒的截图和处理。"""
            clock.now += 0.4
            return ReplayJob()

        controller.post_screencap = capture
        executor = MaaBattleExecutor(
            controller, replay, lambda: None, lambda: True, lambda: False, clock=clock.clock, sleep=clock.sleep
        )
        feedback = executor._confirm_move_with_retries(self.observation, ChainMovePositions(POINTS[0], POINTS[1]))
        self.assertTrue(feedback.reached)
        self.assertEqual(feedback.retries, 0)
        self.assertEqual(controller.actions, [])

    def test_ambiguous_current_markers_do_not_trigger_rollback(self) -> None:
        """多个菱形无法唯一定位时，不猜测回退点。"""
        controller = ReplayController()
        replay = MarkerReplay(None)

        def recognize(image, board, *, move=None):
            """移动未确认，全盘重查给出两个冲突菱形。"""
            if move is None:
                return ChainEndpointResult("重查帧", "uncertain", [MARKER, MARKER])
            return ChainEndpointResult("确认帧", "not_detected")

        replay.recognize_chain_endpoint = recognize
        self.executor(controller, replay)._hold_and_move(POINTS[:2], self.observation, hold_start_seconds=0.0)
        self.assertEqual(controller.actions, [("down", POINTS[0]), ("move", POINTS[1]), ("up", ())])
        self.assertEqual(replay.failures[0]["stop_reason"], "current_marker_not_unique")

    def test_stop_after_recovery_capture_prevents_rollback(self) -> None:
        """重查菱形期间用户停止时，不再派发回退或重滑。"""
        stopped = False
        controller = ReplayController()
        replay = MarkerReplay(None)
        clock = ReplayClock()

        def recognize(image, board, *, move=None):
            """在全盘重查返回时模拟停止请求。"""
            nonlocal stopped
            if move is None:
                stopped = True
                return ChainEndpointResult("重查帧", "detected", [MARKER])
            return ChainEndpointResult("确认帧", "not_detected")

        replay.recognize_chain_endpoint = recognize
        executor = MaaBattleExecutor(
            controller, replay, lambda: None, lambda: True, lambda: stopped, clock=clock.clock, sleep=clock.sleep
        )
        executor._hold_and_move(POINTS[:2], self.observation, hold_start_seconds=0.0)
        self.assertEqual(controller.actions, [("down", POINTS[0]), ("move", POINTS[1]), ("up", ())])
        self.assertEqual(replay.failures, [])

    def test_failed_rollback_still_releases_the_touch(self) -> None:
        """回退命令失败必须释放触控并上报错误。"""
        controller = ReplayController(fail_move=2)
        replay = MarkerReplay(None)

        def recognize(image, board, *, move=None):
            """目标未接入，但能独立定位当前菱形。"""
            if move is None:
                return ChainEndpointResult("重查帧", "detected", [MARKER])
            return ChainEndpointResult("确认帧", "not_detected")

        replay.recognize_chain_endpoint = recognize
        with self.assertRaisesRegex(RuntimeError, "回退到当前大菱形失败"):
            self.executor(controller, replay)._hold_and_move(POINTS[:2], self.observation, hold_start_seconds=0.0)
        self.assertEqual(controller.moves, 2)
        self.assertEqual(controller.actions[-1], ("up", ()))

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
        self.assertEqual(controller.actions, [("down", (20, 30)), ("move", (50, 60)), ("up", ())])
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
