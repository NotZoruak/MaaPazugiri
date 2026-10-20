"""回放普通连线前的连续棋盘观测，验证何时允许发送操作。"""

from __future__ import annotations

import sys
import unittest
from contextlib import ExitStack
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.candidates import build_route_preview  # noqa: E402
from battle.session import BattleOutcome, BattleState  # noqa: E402
from battle.stability import NormalLinkStability, boards_are_similar  # noqa: E402
from battle.strategy import StrategyDecision, StrategyDocument  # noqa: E402
from custom.action.battle import BattleAction  # noqa: E402
from test_battle_candidates import board, ghoul  # noqa: E402
from vision.model import BoardResult  # noqa: E402


class BattleStabilityActionTests(unittest.TestCase):
    """保留实际观测、策略与路线搜索，只替换截图和设备输入边界。"""

    def run_replay(
        self,
        boards: list[BoardResult],
        *,
        states: list[BattleState] | None = None,
        decision_kind: str | None = None,
    ) -> tuple[list[tuple[int, StrategyDecision]], list[BoardResult]]:
        """逐帧回放，记录实际获准执行的操作及路线使用的当前帧。"""
        states = states or [*[BattleState.OPERABLE] * len(boards), BattleState.VICTORY]
        clock = SimpleNamespace(now=0.0)
        observed: list[BoardResult] = []
        planned: list[BoardResult] = []
        operations: list[tuple[int, StrategyDecision]] = []
        image = np.zeros((1280, 720, 3), dtype=np.uint8)

        def advance(seconds: float) -> None:
            """推进回放时钟，不实际等待。"""
            clock.now += seconds

        job = SimpleNamespace(succeeded=True, get=lambda: image)
        job.wait = lambda: job
        controller = SimpleNamespace(post_screencap=lambda: job)
        context = SimpleNamespace(
            tasker=SimpleNamespace(controller=controller, stopping=False),
            get_node_data=lambda name: None,
            run_recognition_direct=lambda *args, **kwargs: None,
        )

        def recognize(current, frame_id: str, **kwargs) -> BoardResult:
            """每次返回独立识别结果，编号来自本次观测。"""
            result = replace(boards[len(observed)], frame_id=frame_id)
            observed.append(result)
            return result

        def routes(current, result: BoardResult):
            """使用实际候选算法规划当前帧，记录路线来源。"""
            planned.append(result)
            return build_route_preview(result)

        def execute(observation, resolution, decision: StrategyDecision) -> None:
            """仅记录获准的实际行动，等待不发送任何输入。"""
            if decision.kind != "wait":
                operations.append((len(observed), decision))

        def run(observe, *, stopped, on_operable, on_ranbu):
            """逐次经过真实截图入口与状态回调，直到终局。"""
            for expected_state in states:
                advance(0.25)
                state = observe()
                self.assertEqual(state, expected_state)
                if state == BattleState.VICTORY:
                    return BattleOutcome.VICTORY
                if state == BattleState.OPERABLE:
                    on_operable()
                elif state == BattleState.RANBU:
                    on_ranbu()
            self.fail("回放必须包含终局")

        previewer = SimpleNamespace(recognize_board=recognize, preview_routes=routes)
        patches = [
            patch("custom.action.battle.time", SimpleNamespace(monotonic=lambda: clock.now, sleep=advance)),
            patch("custom.action.battle.BattleSession", return_value=SimpleNamespace(run=run)),
            patch("custom.action.battle.BoardPreviewer", return_value=previewer),
            patch("custom.action.battle.MaaBattleExecutor", return_value=SimpleNamespace(execute_once=execute)),
            patch("custom.action.battle.BattleFrameRecognizer.analyze", side_effect=states),
            patch(
                "custom.action.battle.UltimateChargeRecognizer",
                return_value=SimpleNamespace(analyze=lambda *args, **kwargs: SimpleNamespace(slots=[])),
            ),
        ]
        if decision_kind is not None:
            patches.append(
                patch(
                    "custom.action.battle.decide_action",
                    return_value=StrategyDecision(decision_kind, StrategyDocument.builtin_default().initial_stage_id),
                )
            )
        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            result = BattleAction().run(context, SimpleNamespace(custom_action_param="{}"))
        self.assertTrue(result.success)
        self.assertEqual(len(observed), len(boards))
        return operations, planned

    def test_a_single_board_never_starts_a_normal_link(self) -> None:
        """只有一帧棋盘不能证明已经稳定。"""
        operations, planned = self.run_replay([board(ghoul(1, 100), ghoul(2, 190))])
        self.assertEqual(operations, [])
        self.assertEqual(planned, [])

    def test_small_jitter_allows_a_link_planned_from_the_latest_board(self) -> None:
        """两帧轻微抖动时放行，路线必须使用新坐标而非上一帧。"""
        operations, planned = self.run_replay(
            [board(ghoul(1, 100), ghoul(2, 190)), board(ghoul(20, 192), ghoul(10, 102))]
        )
        self.assertEqual([index for index, _ in operations], [2])
        self.assertEqual(len(planned), 1)
        route = operations[0][1].route
        self.assertIsNotNone(route)
        assert route is not None
        self.assertEqual(set(route.points), {(102, 400), (192, 400)})

    def test_moving_board_waits_for_two_adjacent_similar_results(self) -> None:
        """棋盘明显移动时继续观察，最新两帧接近后才执行。"""
        operations, planned = self.run_replay(
            [
                board(ghoul(1, 100), ghoul(2, 190)),
                board(ghoul(1, 140), ghoul(2, 230)),
                board(ghoul(1, 142), ghoul(2, 232)),
            ]
        )
        self.assertEqual([index for index, _ in operations], [3])
        self.assertEqual(len(planned), 1)

    def test_each_operation_requires_a_new_pair_of_board_results(self) -> None:
        """动作后的第一帧不能沿用动作前的稳定结论。"""
        operations, _ = self.run_replay([board(ghoul(1, 100), ghoul(2, 190)) for _ in range(4)])
        self.assertEqual([index for index, _ in operations], [2, 4])

    def test_nonoperable_state_breaks_consecutive_board_results(self) -> None:
        """未知或不可操作画面会打断连续棋盘证据。"""
        for interrupted in (BattleState.UNKNOWN, BattleState.INOPERABLE):
            with self.subTest(state=interrupted):
                operations, _ = self.run_replay(
                    [board(ghoul(1, 100), ghoul(2, 190)), board(ghoul(1, 100), ghoul(2, 190))],
                    states=[BattleState.OPERABLE, interrupted, BattleState.OPERABLE, BattleState.VICTORY],
                )
                self.assertEqual(operations, [])

    def test_ultimate_and_bomb_do_not_require_normal_link_stability(self) -> None:
        """约束只影响普通连线，不推迟策略优先选择的奥义或炸弹。"""
        for kind in ("ultimate", "bomb"):
            with self.subTest(kind=kind):
                operations, _ = self.run_replay([board(ghoul(1, 100))], decision_kind=kind)
                self.assertEqual([decision.kind for _, decision in operations], [kind])

    def test_ranbu_breaks_the_normal_link_stability_sequence(self) -> None:
        """进入乱舞再回到普通状态后，重新累计两次普通棋盘结果。"""
        operations, _ = self.run_replay(
            [board(ghoul(1, 100), ghoul(2, 190)) for _ in range(4)],
            states=[
                BattleState.OPERABLE,
                BattleState.RANBU,
                BattleState.OPERABLE,
                BattleState.OPERABLE,
                BattleState.VICTORY,
            ],
        )
        self.assertEqual(
            [(index, decision.kind) for index, decision in operations], [(2, "ranbu_link"), (4, "normal_link")]
        )

    def test_ranbu_can_act_on_its_first_board(self) -> None:
        """乱舞维持原有即时扫动规则。"""
        operations, _ = self.run_replay(
            [board(ghoul(1, 100), ghoul(2, 190))], states=[BattleState.RANBU, BattleState.VICTORY]
        )
        self.assertEqual([decision.kind for _, decision in operations], ["ranbu_link"])


class BoardSimilarityTests(unittest.TestCase):
    """直接核对独立棋盘信息的比较，不借助设备或识别替身。"""

    def test_numbering_and_evidence_scores_are_not_cross_frame_identity(self) -> None:
        """编号、顺序、模板分数变化不妨碍几何与类别相同的棋盘匹配。"""
        previous = board(ghoul(1, 100), ghoul(2, 190), frame_id="第一帧")
        current = board(
            replace(ghoul(12, 192), evidence=[{"score": 0.81}], quality="single_feature"),
            ghoul(11, 102),
            frame_id="第二帧",
        )
        before = asdict(previous), asdict(current)
        self.assertTrue(boards_are_similar(previous, current))
        self.assertEqual((asdict(previous), asdict(current)), before)

    def test_opposite_movements_do_not_cancel_out(self) -> None:
        """对象数和整体平均位置相同，仍不能掩盖个别对象的明显移动。"""
        previous = board(ghoul(1, 100), ghoul(2, 190), frame_id="第一帧")
        current = board(ghoul(1, 120), ghoul(2, 170), frame_id="第二帧")
        self.assertFalse(boards_are_similar(previous, current))

    def test_object_count_and_semantics_must_stay_the_same(self) -> None:
        """数量、颜色、强化、炸弹等级或障碍物种类改变时，不能放行。"""
        first, second = ghoul(1, 100), ghoul(2, 190)
        previous = board(first, second, frame_id="第一帧")
        for objects in (
            [first],
            [first, second, ghoul(3, 280)],
            [first, replace(second, color="blue")],
            [first, replace(second, enhanced=True)],
            [first, replace(second, kind="bomb", bomb_level=2)],
        ):
            with self.subTest(objects=objects):
                self.assertFalse(boards_are_similar(previous, board(*objects, frame_id="第二帧")))
        for kind, attribute, old, new in (
            ("bomb", "bomb_level", 1, 2),
            ("obstacle", "subtype", "shell", "rock"),
        ):
            with self.subTest(kind=kind):
                a = replace(first, kind=kind, **{attribute: old})
                b = replace(a, **{attribute: new})
                self.assertFalse(boards_are_similar(board(a, frame_id="第一帧"), board(b, frame_id="第二帧")))

    def test_size_and_candidate_touch_point_changes_are_checked(self) -> None:
        """中心相同但对象尺度或触点明显变化时仍继续等待。"""
        original = ghoul(1, 100)
        previous = board(original, frame_id="第一帧")
        for changed in (
            replace(original, body_box=(50, 350, 125, 100)),
            replace(original, candidate_point=(110, 400)),
            replace(original, candidate_point=None),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(boards_are_similar(previous, board(changed, frame_id="第二帧")))

    def test_object_matching_is_one_to_one(self) -> None:
        """匹配存在歧义时可重新配对，但不能把同一个目标用两次。"""
        previous = board(ghoul(1, 100), ghoul(2, 104), frame_id="第一帧")
        self.assertTrue(boards_are_similar(previous, board(ghoul(1, 103), ghoul(2, 99), frame_id="第二帧")))
        self.assertFalse(boards_are_similar(previous, board(ghoul(1, 101), ghoul(2, 114), frame_id="第二帧")))

    def test_tolerance_scales_with_resolution_and_object_size(self) -> None:
        """截图等比放大后，相同相对抖动的判断保持一致；小对象容差更小。"""
        for scale in (0.5, 1.0, 2.0):
            with self.subTest(scale=scale):
                original = ghoul(1, 100)
                scaled = replace(
                    original,
                    center=(100 * scale, 400 * scale),
                    body_box=tuple(value * scale for value in original.body_box),
                    candidate_point=(100 * scale, 400 * scale),
                )
                previous = replace(
                    board(scaled, frame_id="第一帧"),
                    source_size=(round(720 * scale), round(1280 * scale)),
                    board_region=(0, 210 * scale, 720 * scale, 780 * scale),
                )
                for offset, expected in ((2 * scale, True), (8 * scale, False)):
                    moved = replace(
                        scaled,
                        center=(100 * scale + offset, 400 * scale),
                        candidate_point=(100 * scale + offset, 400 * scale),
                    )
                    self.assertEqual(
                        boards_are_similar(previous, replace(previous, frame_id="第二帧", objects=[moved])), expected
                    )
        tiny = replace(ghoul(1, 100), body_box=(90, 390, 20, 20))
        moved = replace(tiny, center=(103, 400), candidate_point=(103, 400))
        self.assertFalse(boards_are_similar(board(tiny, frame_id="第一帧"), board(moved, frame_id="第二帧")))

    def test_incompatible_or_invalid_board_information_is_rejected(self) -> None:
        """空棋盘、范围切换、重复帧和无效几何都不能证明稳定。"""
        previous = board(ghoul(1, 100), frame_id="第一帧")
        current = replace(previous, frame_id="第二帧")
        for changed in (
            previous,
            replace(current, source_size=(1440, 2560)),
            replace(current, board_region=(0, 200, 720, 780)),
            replace(current, coverage={"recognition_mode": "lightweight", "skipped_kinds": ["obstacle"]}),
            replace(current, objects=[]),
            replace(current, objects=[replace(ghoul(1, 100), center=(float("nan"), 400))]),
            replace(current, objects=[replace(ghoul(1, 100), candidate_point=(100, float("inf")))]),
            replace(current, objects=[replace(ghoul(1, 100), body_box=(50, 350, 0, 100))]),
            replace(current, objects=[replace(ghoul(1, 100), center=(-1, 400))]),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(boards_are_similar(previous, changed))
        duplicated = board(ghoul(1, 100), ghoul(1, 190), frame_id="第一帧")
        self.assertFalse(boards_are_similar(duplicated, replace(duplicated, frame_id="第二帧")))

    def test_invalid_result_breaks_the_stability_sequence(self) -> None:
        """识别缺失后需要重新拿到两份有效结果，不补回上一帧对象。"""
        tracker = NormalLinkStability()
        first = board(ghoul(1, 100), frame_id="第一帧")
        self.assertFalse(tracker.observe(first))
        self.assertFalse(tracker.observe(None))
        self.assertFalse(tracker.observe(replace(first, frame_id="第二帧")))
        self.assertTrue(tracker.observe(replace(first, frame_id="第三帧")))


if __name__ == "__main__":
    unittest.main()
