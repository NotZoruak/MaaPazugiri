"""回放普通连线前的连续棋盘观测，验证何时允许发送操作。"""

from __future__ import annotations

import json
import sys
import tempfile
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
from battle.observation import BattleObserver  # noqa: E402
from battle.preview import BoardPreviewer  # noqa: E402
from battle.session import BattleOutcome, BattleState  # noqa: E402
from battle.stability import NormalLinkStability, boards_are_similar  # noqa: E402
from battle.strategy import StrategyDecision, StrategyDocument  # noqa: E402
from custom.action.battle import BattleAction  # noqa: E402
from test_battle_candidates import board, ghoul  # noqa: E402
from vision.model import BoardResult, Hit  # noqa: E402
from vision.recognizer import BoardRecognizer  # noqa: E402


class BattleStabilityActionTests(unittest.TestCase):
    """保留实际观测、策略与路线搜索，只替换截图和设备输入边界。"""

    def test_enhanced_adjacent_template_scales_do_not_prevent_stability(self) -> None:
        """同色强化对象的小抖动及相邻模板尺度切换不等于棋盘下落。"""
        for first_scale, second_scale in ((0.9, 1.0), (1.0, 1.1), (1.1, 1.0)):
            with self.subTest(scales=(first_scale, second_scale)):
                first, second = ghoul(1, 200), ghoul(9, 203)
                first.enhanced = second.enhanced = True
                first.body_box = (150, 350, round(96 * first_scale, 2), round(104 * first_scale, 2))
                second.body_box = (153, 350, round(96 * second_scale, 2), round(104 * second_scale, 2))
                self.assertTrue(boards_are_similar(board(first, frame_id="上帧"), board(second, frame_id="本帧")))
        second.body_box = (153, 350, 115.2, 124.8)
        first.body_box = (150, 350, 86.4, 93.6)
        self.assertFalse(boards_are_similar(board(first, frame_id="上帧"), board(second, frame_id="本帧")))

    def run_replay(
        self,
        boards: list[BoardResult],
        *,
        states: list[BattleState] | None = None,
        decision_kind: str | None = None,
        diagnostic_directory: Path | None = None,
        reserved_color: str | None = None,
        maximum_capture_gap: float | None = None,
        minimum_capture_gap_after_action: float | None = None,
        diagnostic_delay_after_release: float = 0.0,
    ) -> tuple[list[tuple[int, StrategyDecision]], list[BoardResult]]:
        """逐帧回放，记录实际获准执行的操作及路线使用的当前帧。"""
        states = states or [*[BattleState.OPERABLE] * len(boards), BattleState.VICTORY]
        clock = SimpleNamespace(now=0.0)
        observed: list[BoardResult] = []
        planned: list[BoardResult] = []
        operations: list[tuple[int, StrategyDecision]] = []
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        capture_times: list[float] = []
        action_times: list[tuple[int, float]] = []
        executor_stub = SimpleNamespace(last_gesture_released_at=None)

        def advance(seconds: float) -> None:
            """推进回放时钟，不实际等待。"""
            clock.now += seconds

        job = SimpleNamespace(succeeded=True, get=lambda: image)
        job.wait = lambda: job

        def capture():
            """记录每次截图时间，核对松手后的固定响应间隔。"""
            capture_times.append(clock.now)
            image[:] = 0
            if len(observed) < len(boards):
                colors = {"red": (0, 0, 200), "yellow": (0, 200, 220), "blue": (220, 100, 0), "green": (0, 180, 0)}
                for item in boards[len(observed)].objects:
                    x, y, w, h = (round(v) for v in item.body_box)
                    if item.color in colors:
                        image[max(0, y) : y + h, max(0, x) : x + w] = colors[item.color]
            return job

        controller = SimpleNamespace(post_screencap=capture)
        context = SimpleNamespace(
            tasker=SimpleNamespace(controller=controller, stopping=False),
            get_node_data=lambda name: None,
            run_recognition_direct=lambda *args, **kwargs: None,
        )

        def recognize(current, frame_id: str, **kwargs) -> BoardResult:
            """每次返回独立识别结果，编号来自本次观测。"""
            result = replace(boards[len(observed)], frame_id=frame_id)
            return result

        original_observe = BattleObserver.observe

        def observe_board(instance, current, phase, **kwargs):
            """记录每个处理帧；局部确认帧不再调用整盘识别。"""
            result = original_observe(instance, current, phase, **kwargs)
            observed.append(result.board)
            return result

        def routes(current, result: BoardResult):
            """使用实际候选算法规划当前帧，记录路线来源。"""
            planned.append(result)
            return build_route_preview(result)

        def execute(observation, resolution, decision: StrategyDecision) -> None:
            """仅记录获准的实际行动，等待不发送任何输入。"""
            if decision.kind != "wait":
                operations.append((len(observed), decision))
                action_times.append((len(capture_times), clock.now))
                executor_stub.last_gesture_released_at = clock.now
                advance(diagnostic_delay_after_release)

        executor_stub.execute_once = execute

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

        if diagnostic_directory is None:
            previewer = SimpleNamespace(
                recognize_board=recognize, preview_routes=routes, save_stability_report=lambda report: None
            )
        else:
            previewer = BoardPreviewer(Path("离线测试模板"), diagnostic_directory)
            previewer.recognize_board = recognize
            previewer.preview_routes = routes
        patches = [
            patch("custom.action.battle.BattleObserver.observe", new=observe_board),
            patch("custom.action.battle.time", SimpleNamespace(monotonic=lambda: clock.now, sleep=advance)),
            patch("custom.action.battle.BattleSession", return_value=SimpleNamespace(run=run)),
            patch("custom.action.battle.BoardPreviewer", return_value=previewer),
            patch("custom.action.battle.MaaBattleExecutor", return_value=executor_stub),
            patch("custom.action.battle.BattleFrameRecognizer.analyze", side_effect=states),
            patch(
                "custom.action.battle.UltimateChargeRecognizer",
                return_value=SimpleNamespace(analyze=lambda *args, **kwargs: SimpleNamespace(slots=[])),
            ),
        ]
        if reserved_color is not None:
            strategy = StrategyDocument.builtin_default()
            stage = strategy.stages[0]
            strategy = replace(
                strategy, stages=(replace(stage, frenzy=replace(stage.frenzy, reserved_color=reserved_color)),)
            )
            patches.append(patch("custom.action.battle.load_strategy", return_value=strategy))
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
        if maximum_capture_gap is not None:
            self.assertTrue(
                all(b - a <= maximum_capture_gap for a, b in zip(capture_times, capture_times[1:], strict=False))
            )
        if minimum_capture_gap_after_action is not None:
            self.assertTrue(
                all(
                    capture_times[index] - stamp >= minimum_capture_gap_after_action
                    for index, stamp in action_times
                    if index < len(capture_times)
                )
            )
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

    def test_switching_valid_touch_features_does_not_delay_a_stable_board(self) -> None:
        """眼睛和嘴巴之间切换合法触点时，仍从第二帧身体中心规划。"""
        first = [replace(ghoul(index, x), candidate_point=(x, 380)) for index, x in ((1, 100), (2, 190))]
        second = [replace(item, candidate_point=(item.center[0], 410)) for item in first]
        operations, _ = self.run_replay([board(*first), board(*second)])
        self.assertEqual([index for index, _ in operations], [2])
        route = operations[0][1].route
        assert route is not None
        self.assertEqual(set(route.points), {(100, 400), (190, 400)})

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
        self.assertEqual([(index, decision.kind) for index, decision in operations], [(4, "normal_link")])

    def test_ranbu_cannot_act_on_its_first_board(self) -> None:
        """乱舞标志不代表棋盘已落稳，单帧不能直接起手。"""
        operations, _ = self.run_replay(
            [board(ghoul(1, 100), ghoul(2, 190))], states=[BattleState.RANBU, BattleState.VICTORY]
        )
        self.assertEqual(operations, [])

    def test_ranbu_waits_for_stability_and_uses_latest_color_positions(self) -> None:
        """下落及颜色位置变化后，只有最新两帧稳定才发起乱舞。"""
        frames = [
            board(ghoul(1, 100, color="yellow"), ghoul(2, 190, color="red")),
            board(ghoul(3, 100, color="red"), ghoul(4, 190, color="yellow")),
            board(ghoul(8, 102, color="red"), ghoul(9, 192, color="yellow")),
        ]
        operations, planned = self.run_replay(frames, states=[*[BattleState.RANBU] * 3, BattleState.VICTORY])
        self.assertEqual([(index, decision.kind) for index, decision in operations], [(3, "ranbu_link")])
        self.assertEqual(planned, [])

    def test_each_ranbu_sweep_requires_fresh_stability_evidence(self) -> None:
        """每次扫动后重置判断，不能沿用操作前的稳定结论。"""
        frames = [board(ghoul(1, 100), ghoul(2, 190)) for _ in range(4)]
        operations, _ = self.run_replay(frames, states=[*[BattleState.RANBU] * 4, BattleState.VICTORY])
        self.assertEqual(
            [(index, decision.kind) for index, decision in operations], [(2, "ranbu_link"), (4, "ranbu_link")]
        )

    def test_ranbu_only_confirms_the_preselected_point_and_preserves_its_coordinates(self) -> None:
        """其他对象下落或数量变化时，只要同一预选点颜色不变就放行。"""
        frames = [
            board(ghoul(1, 100, color="blue"), ghoul(2, 300)),
            board(ghoul(99, 110, color="blue"), ghoul(0, 500, color="yellow"), ghoul(3, 600)),
        ]
        operations, _ = self.run_replay(
            frames, states=[BattleState.RANBU, BattleState.RANBU, BattleState.VICTORY], reserved_color="red"
        )
        self.assertEqual([(index, decision.kind) for index, decision in operations], [(2, "ranbu_link")])
        self.assertEqual(operations[0][1].ranbu_start, (100, 400))

    def test_ranbu_waits_half_a_second_then_requires_a_fresh_pair_at_the_selected_point(self) -> None:
        """按用户观察等待半秒消除动画，之后仍须两张新图确认起点。"""
        frames = [board(ghoul(1, 100, color="blue"), ghoul(2, 300)) for _ in range(4)]
        operations, _ = self.run_replay(
            frames,
            states=[*[BattleState.RANBU] * 4, BattleState.VICTORY],
            maximum_capture_gap=0.501,
            minimum_capture_gap_after_action=0.49,
        )
        self.assertEqual([index for index, _ in operations], [2, 4])

    def test_ranbu_diagnostic_writing_overlaps_the_half_second_pause(self) -> None:
        """诊断耗时计入半秒等待，不能保存后再重新计时。"""
        frames = [board(ghoul(1, 100, color="blue"), ghoul(2, 300)) for _ in range(4)]
        operations, _ = self.run_replay(
            frames,
            states=[*[BattleState.RANBU] * 4, BattleState.VICTORY],
            maximum_capture_gap=0.501,
            minimum_capture_gap_after_action=0.49,
            diagnostic_delay_after_release=0.2,
        )
        self.assertEqual([index for index, _ in operations], [2, 4])

    def test_all_reserved_colors_wait_instead_of_starting_a_short_link(self) -> None:
        """最新起手规则要求非保留色，全为保留色不能触发旧短连例外。"""
        frames = [board(ghoul(1, 100), ghoul(2, 300)) for _ in range(2)]
        operations, _ = self.run_replay(
            frames, states=[BattleState.RANBU, BattleState.RANBU, BattleState.VICTORY], reserved_color="red"
        )
        self.assertEqual(operations, [])

    def test_interruption_breaks_ranbu_stability_evidence(self) -> None:
        """未知或不可操作状态会使乱舞重新累计两帧稳定证据。"""
        for interrupted in (BattleState.UNKNOWN, BattleState.INOPERABLE):
            with self.subTest(state=interrupted):
                frames = [board(ghoul(1, 100), ghoul(2, 190)) for _ in range(3)]
                operations, _ = self.run_replay(
                    frames,
                    states=[BattleState.RANBU, interrupted, BattleState.RANBU, BattleState.RANBU, BattleState.VICTORY],
                )
                self.assertEqual([(index, decision.kind) for index, decision in operations], [(3, "ranbu_link")])

    def test_ranbu_short_link_also_requires_stability(self) -> None:
        """保留色短连也需要落稳，不能在过渡帧误触。"""
        operations, _ = self.run_replay(
            [board(ghoul(1, 100), ghoul(2, 190))],
            states=[BattleState.RANBU, BattleState.VICTORY],
            decision_kind="ranbu_short_link",
        )
        self.assertEqual(operations, [])

    def test_latest_stability_report_records_the_rejected_pair_and_reason(self) -> None:
        """未放行时保存两帧精简信息，后续通过不能覆盖最近一次拒绝证据。"""
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            first = board(ghoul(1, 100), ghoul(2, 190))
            changed = board(ghoul(1, 100), ghoul(2, 190), ghoul(3, 280))
            self.run_replay([first, changed], diagnostic_directory=destination)
            report_path = destination / "latest-stability.json"
            self.assertTrue(report_path.is_file())
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertFalse(report["stable"])
            self.assertIn("数量", report["reason"])
            self.assertEqual(len(report["previous"]["objects"]), 2)
            self.assertEqual(len(report["current"]["objects"]), 3)
            self.assertNotEqual(report["previous"]["frame_id"], report["current"]["frame_id"])
            self.assertNotIn("evidence", report["current"]["objects"][0])
            rejected_path = destination / "latest-stability-rejected.json"
            rejected = rejected_path.read_text(encoding="utf-8")
            self.run_replay([first, first], diagnostic_directory=destination)
            self.assertEqual(set(destination.iterdir()), {report_path, rejected_path})
            self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["stable"])
            self.assertEqual(rejected_path.read_text(encoding="utf-8"), rejected)


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

    def test_size_changes_and_touch_points_outside_the_object_are_rejected(self) -> None:
        """中心相同但尺度变化或触点脱离对象时仍继续等待。"""
        original = ghoul(1, 100)
        previous = board(original, frame_id="第一帧")
        for changed in (
            replace(original, body_box=(50, 350, 125, 100)),
            replace(original, candidate_point=(200, 400)),
            replace(original, candidate_point=None),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(boards_are_similar(previous, board(changed, frame_id="第二帧")))

    def test_score_order_can_switch_eyes_and_mouth_without_moving_the_object(self) -> None:
        """以运行样本的眼嘴位置复现真实对象整理器的触点切换。"""
        recognizer = BoardRecognizer.__new__(BoardRecognizer)
        recognizer.profile = SimpleNamespace(
            board_roi=(0, 210, 720, 780), merge_fraction=0.35, variable_ghoul_sizes=False
        )
        mouth = Hit("red_mouth", "ghoul", "red", None, "mouth", (204, 111), (96, 100), (205.5, 121.5), 0.95668)
        eye = Hit("red_eye", "ghoul", "red", None, "eye", (204, 112), (96, 100), (203, 94.5), 0.9266)
        first = recognizer._object(recognizer._group([mouth, eye])[0], 1, 1, 1, (720, 1280), [])
        second = recognizer._object(
            recognizer._group([replace(mouth, score=0.9266), replace(eye, score=0.95668)])[0],
            8,
            1,
            1,
            (720, 1280),
            [],
        )
        self.assertEqual(first.candidate_point, (205.5, 331.5))
        self.assertEqual(second.candidate_point, (203, 304.5))
        self.assertLess(abs(first.center[1] - second.center[1]), 1)
        self.assertTrue(boards_are_similar(board(first, frame_id="第一帧"), board(second, frame_id="第二帧")))

    def test_switching_from_an_invalid_touch_point_is_not_stable(self) -> None:
        """上一帧触点已经脱离对象，下一帧恢复合法时仍需再确认一帧。"""
        valid = ghoul(1, 100)
        invalid = replace(valid, candidate_point=(200, 400))
        self.assertFalse(boards_are_similar(board(invalid, frame_id="第一帧"), board(valid, frame_id="第二帧")))

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
                for offset, expected in ((2 * scale, True), (6 * scale, True), (8 * scale, True), (12 * scale, False)):
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
