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
from vision.model import BoardObject, BoardResult, ChainEndpointMarker, ChainEndpointResult  # noqa: E402

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


class MarkerReplay:
    """按检查次数回放新菱形出现时机。"""

    def __init__(self, appears_at: int | None) -> None:
        self.appears_at = appears_at
        self.checks = 0

    def recognize_chain_endpoint(self, image: np.ndarray, board: BoardResult) -> ChainEndpointResult:
        """只回放当前帧标志，不连接设备。"""
        seen = self.appears_at is not None and self.checks >= self.appears_at
        self.checks += 1
        return ChainEndpointResult(board.frame_id, "detected" if seen else "not_found", [MARKER] if seen else [])


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
        return MaaBattleExecutor(
            controller,
            replay,  # type: ignore[arg-type]
            lambda: self.confirmations.append(True),
            lambda: can_start,
            lambda: controller.moves >= stop_after,
        )

    def test_early_marker_still_visits_the_entire_planned_route(self) -> None:
        """起点或第四点出现菱形，都不能提前结束八点路线。"""
        for appears_at in (0, 3):
            with self.subTest(appears_at=appears_at):
                self.confirmations.clear()
                controller = ReplayController()
                executor = self.executor(controller, MarkerReplay(appears_at))
                executor._hold_and_move(POINTS, self.observation, (), hold_start_seconds=0.0)
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
        """只有完整移动而无成功证据时，不虚增普通手数。"""
        controller = ReplayController()
        self.executor(controller, MarkerReplay(None)).execute_once(self.observation, self.resolution, self.decision)
        self.assertEqual(controller.moves, 7)
        self.assertEqual(self.confirmations, [])
        self.assertEqual(controller.actions[-1], ("up", ()))

    def test_user_stop_interrupts_remaining_moves_and_releases(self) -> None:
        """用户停止优先于剩余候选路线。"""
        controller = ReplayController()
        executor = self.executor(controller, MarkerReplay(0), stop_after=2)
        executor._hold_and_move(POINTS, self.observation, (), hold_start_seconds=0.0)
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
            POINTS, self.observation, (), check_marker=False, hold_start_seconds=0.0
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
