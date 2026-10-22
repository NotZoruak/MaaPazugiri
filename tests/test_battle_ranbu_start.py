"""验证乱舞只核对同一预选落点连续两帧的颜色。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.ranbu import RanbuStartConfirmation, ghoul_color_at_point  # noqa: E402
from test_battle_candidates import board, ghoul  # noqa: E402


class RanbuStartConfirmationTests(unittest.TestCase):
    """对象编号、其他对象的移动和数量变化不构成起点拒绝条件。"""

    def test_same_color_at_the_selected_point_allows_other_objects_to_change(self) -> None:
        """只要预选点仍为蓝色，其他颜色移动、数量变化和编号变化均可接受。"""
        tracker = RanbuStartConfirmation()
        first = board(ghoul(1, 100, color="blue"), ghoul(2, 300), frame_id="第一帧")
        second = board(ghoul(99, 110, color="blue"), ghoul(0, 500, color="yellow"), ghoul(3, 600), frame_id="第二帧")
        self.assertIsNone(tracker.observe(first, "red"))
        self.assertEqual(tracker.observe(second, "red"), ((100, 400), "blue"))

    def test_changed_color_needs_two_new_matching_observations(self) -> None:
        """同一点变色后不能起手；当前帧可作为新颜色的第一次证据。"""
        tracker = RanbuStartConfirmation()
        self.assertIsNone(tracker.observe(board(ghoul(1, 100, color="blue"), frame_id="一"), "red"))
        self.assertIsNone(tracker.observe(board(ghoul(2, 100, color="yellow"), frame_id="二"), "red"))
        self.assertEqual(
            tracker.observe(board(ghoul(3, 100, color="yellow"), frame_id="三"), "red"), ((100, 400), "yellow")
        )

    def test_reserved_color_never_confirms(self) -> None:
        """预选点变为保留色或全盘只剩保留色时不得放行。"""
        tracker = RanbuStartConfirmation()
        self.assertIsNone(tracker.observe(board(ghoul(1, 100, color="blue"), frame_id="一"), "red"))
        for frame_id in ("二", "三"):
            self.assertIsNone(tracker.observe(board(ghoul(2, 100), ghoul(3, 300), frame_id=frame_id), "red"))

    def test_reserved_color_change_and_missing_observation_invalidate_confirmation(self) -> None:
        """保留色改变或识别结果缺失时，不能沿用之前的非保留色结论。"""
        tracker = RanbuStartConfirmation()
        self.assertIsNone(tracker.observe(board(ghoul(1, 100, color="blue"), frame_id="一"), None))
        self.assertIsNone(tracker.observe(board(ghoul(2, 100, color="blue"), frame_id="二"), "blue"))
        self.assertIsNone(tracker.observe(board(ghoul(3, 100, color="yellow"), frame_id="三"), "red"))
        self.assertIsNone(tracker.observe(None, "red"))
        self.assertIsNone(tracker.observe(board(ghoul(4, 100, color="yellow"), frame_id="四"), "red"))

    def test_missing_point_does_not_reuse_the_previous_color(self) -> None:
        """饿鬼移走后不能把附近对象或旧坐标补回预选点。"""
        tracker = RanbuStartConfirmation()
        self.assertIsNone(tracker.observe(board(ghoul(1, 100, color="blue"), frame_id="一"), "red"))
        self.assertIsNone(tracker.observe(board(ghoul(2, 300, color="blue"), frame_id="二"), "red"))
        self.assertEqual(
            tracker.observe(board(ghoul(3, 302, color="blue"), frame_id="三"), "red"), ((300, 400), "blue")
        )

    def test_same_frame_reset_and_layout_change_do_not_confirm(self) -> None:
        """重复帧、状态中断或布局变化后均需要重新取得两次颜色证据。"""
        for interruption in ("重复帧", "中断", "布局"):
            with self.subTest(interruption=interruption):
                tracker = RanbuStartConfirmation()
                first = board(ghoul(1, 100, color="blue"), frame_id="一")
                self.assertIsNone(tracker.observe(first, "red"))
                second = replace(first, frame_id="二")
                if interruption == "重复帧":
                    second = first
                elif interruption == "中断":
                    tracker.reset()
                else:
                    second = replace(second, board_region=(0, 211, 720, 779))
                self.assertIsNone(tracker.observe(second, "red"))

    def test_color_rotation_is_preserved_after_confirmation(self) -> None:
        """确认蓝色后，下一轮优先绿再黄，跳过保留红色。"""
        tracker = RanbuStartConfirmation()
        objects = [ghoul(1, 100, color="blue"), ghoul(2, 300, color="yellow"), ghoul(3, 500, color="green")]
        for index, expected in enumerate((((100, 400), "blue"), ((500, 400), "green"), ((300, 400), "yellow"))):
            self.assertIsNone(tracker.observe(board(*objects, frame_id=f"{index}一"), "red"))
            self.assertEqual(tracker.observe(board(*objects, frame_id=f"{index}二"), "red"), expected)
            tracker.reset()

    def test_ambiguous_special_and_noninteractive_objects_are_not_ordinary_colors(self) -> None:
        """异色重叠、特殊黑色、炸弹和缺少可信触点都不能确认普通起点。"""
        blue = ghoul(1, 100, color="blue")
        red = ghoul(2, 100)
        self.assertIsNone(ghoul_color_at_point(board(blue, red), (100, 400)))
        for kind in ("special_ghoul", "bomb"):
            self.assertIsNone(ghoul_color_at_point(board(replace(blue, kind=kind, color=None)), (100, 400)))
        self.assertIsNone(ghoul_color_at_point(board(replace(blue, candidate_point=None)), (100, 400)))
        unrelated = replace(ghoul(9, 500), candidate_point=(float("nan"), 400))
        self.assertEqual(ghoul_color_at_point(board(blue, unrelated), (100, 400)), "blue")


if __name__ == "__main__":
    unittest.main()
