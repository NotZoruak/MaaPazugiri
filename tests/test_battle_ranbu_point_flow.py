"""验证乱舞第二帧只检查预选点，且当前帧证据传递到执行器。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

import test_battle_execution as fixtures  # noqa: E402
from battle.observation import BattleObserver  # noqa: E402
from battle.ranbu import RanbuStartConfirmation  # noqa: E402
from battle.strategy import BattlePhase, StrategyDecision  # noqa: E402
from test_battle_candidates import board, ghoul  # noqa: E402


class RanbuPointFlowTests(unittest.TestCase):
    """对象框移动不再干扰预选点颜色，但未知与保留色仍不得起手。"""

    def test_seed_prefers_same_color_point_with_more_current_frame_color_margin(self) -> None:
        """同色内优先周边颜色更稳的点，第二帧仍只确认选定原点。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        tracker = RanbuStartConfirmation()
        first = board(ghoul(1, 100, color="blue"), ghoul(2, 300, color="blue"), frame_id="一")
        with (
            patch("battle.ranbu.identify_ranbu_point_color", return_value="blue"),
            patch("battle.ranbu.ranbu_point_color_margin", side_effect=[1, 4]) as margin,
        ):
            self.assertIsNone(tracker.observe(first, "red", image=image))
            self.assertEqual(tracker.last_report["point"], (300, 400))
            self.assertEqual(margin.call_count, 2)
            self.assertEqual(
                tracker.observe(replace(first, frame_id="二", objects=[]), "red", image=image), ((300, 400), "blue")
            )
            self.assertEqual(margin.call_count, 2)

    def test_color_rotation_is_kept_when_another_color_has_more_margin(self) -> None:
        """颜色轮换顺序保留，不让余量评分独占其他颜色的机会。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        tracker = RanbuStartConfirmation()
        first = board(ghoul(1, 100, color="blue"), ghoul(2, 300, color="yellow"), frame_id="一")
        with (
            patch("battle.ranbu.identify_ranbu_point_color", return_value="blue"),
            patch("battle.ranbu.ranbu_point_color_margin", return_value=1) as margin,
        ):
            self.assertIsNone(tracker.observe(first, "red", image=image))
            self.assertEqual(tracker.last_report["point"], (100, 400))
            self.assertEqual(margin.call_count, 1)

    def test_an_unreadable_first_candidate_does_not_starve_other_visible_starts(self) -> None:
        """首个候选被特效遮挡时，预选当前帧内其他颜色可信的对象。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        image[350:450, 250:350] = (220, 100, 0)
        tracker = RanbuStartConfirmation()
        first = board(ghoul(1, 100, color="yellow"), ghoul(2, 300, color="blue"), frame_id="一")
        self.assertIsNone(tracker.observe(first, "red", image=image))
        self.assertIsNotNone(tracker.pending_layout_for(image))
        self.assertEqual(
            tracker.observe(replace(first, frame_id="二", objects=[]), "red", image=image), ((300, 400), "blue")
        )

    def test_two_pixel_checks_confirm_without_second_frame_objects(self) -> None:
        """第二帧没有整盘检测对象，仍可凭同点局部颜色证据确认。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        tracker = RanbuStartConfirmation()
        first = board(ghoul(1, 100, color="blue"), frame_id="一")
        second = replace(first, frame_id="二", objects=[])
        with patch("battle.ranbu.identify_ranbu_point_color", return_value="blue") as local:
            self.assertIsNone(tracker.observe(first, "red", image=image))
            self.assertEqual(tracker.pending_layout_for(image), (first.board_region, first.profile))
            self.assertEqual(tracker.observe(second, "red", image=image), ((100, 400), "blue"))
            self.assertEqual(tracker.confirmed_body_size, first.objects[0].body_box[2:])
            self.assertEqual(local.call_count, 2)
        self.assertIsNone(tracker.pending_layout_for(image))

    def test_unknown_reserved_or_changed_point_color_never_uses_old_evidence(self) -> None:
        """局部证据未知、变保留色或变成其他色，均清除这次确认。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        for observed in (None, "red"):
            tracker = RanbuStartConfirmation()
            first = board(ghoul(1, 100, color="blue"), frame_id="一")
            with patch("battle.ranbu.identify_ranbu_point_color", side_effect=["blue", observed]):
                self.assertIsNone(tracker.observe(first, "red", image=image))
                self.assertIsNone(tracker.observe(replace(first, frame_id="二", objects=[]), "red", image=image))
            self.assertIsNone(tracker.pending_layout_for(image))

    def test_changed_nonreserved_color_starts_a_new_pair_at_the_same_point(self) -> None:
        """同点变成其他可消颜色时，将本帧作为新基线，不再整盘寻找落点。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        tracker = RanbuStartConfirmation()
        first = board(ghoul(1, 100, color="blue"), frame_id="一")
        with patch("battle.ranbu.identify_ranbu_point_color", side_effect=["blue", "yellow", "yellow"]):
            self.assertIsNone(tracker.observe(first, "red", image=image))
            self.assertIsNone(tracker.observe(replace(first, frame_id="二", objects=[]), "red", image=image))
            self.assertIsNotNone(tracker.pending_layout_for(image))
            self.assertEqual(
                tracker.observe(replace(first, frame_id="三", objects=[]), "red", image=image), ((100, 400), "yellow")
            )

    def test_observer_records_skipped_objects_in_point_only_frame(self) -> None:
        """局部确认帧有新帧编号与空对象表，不复制旧对象或声称不存在炸弹。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        preview = Mock()
        ultimate = SimpleNamespace(analyze=lambda *a, **kw: SimpleNamespace(slots=[]))
        observer = BattleObserver(lambda *a: None, preview, ultimate, "阶段")
        observed = observer.observe(image, BattlePhase.RANBU, ranbu_layout=((0, 210, 720, 780), "已核对布局"))
        preview.recognize_board.assert_not_called()
        self.assertEqual(observed.board.objects, [])
        self.assertEqual(observed.board.coverage["mode"], "ranbu_start_point_only")
        self.assertIn("bomb", observed.board.coverage["skipped_kinds"])

    def test_executor_accepts_only_same_frame_pixel_confirmation(self) -> None:
        """局部帧没有对象框，执行器使用带帧编号的颜色证据，拒绝过期证据。"""
        fixture = fixtures.BattleExecutionTests()
        fixture.setUp()
        observed = fixture.upper_yellow_observation()
        observed = replace(observed, board=replace(observed.board, objects=[]))
        for frame_id, expected in ((observed.frame_id, True), ("旧帧", False)):
            controller = fixtures.ReplayController()
            preview = fixtures.MarkerReplay(0)
            fixture.executor(controller, preview, stop_after=3).execute_once(
                observed,
                fixture.resolution,
                StrategyDecision(
                    "ranbu_link",
                    "阶段",
                    reserved_color="red",
                    ranbu_start=(326, 388),
                    ranbu_color="yellow",
                    ranbu_frame_id=frame_id,
                ),
            )
            self.assertEqual(bool(controller.actions), expected)
            if expected:
                self.assertEqual(preview.ranbu_reports[0]["start_color"], "yellow")


if __name__ == "__main__":
    unittest.main()
