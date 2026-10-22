"""验证整体稳定证据与少量局部识别差异分离，并避让不确定区域。"""

from __future__ import annotations

import unittest
from dataclasses import replace

import test_battle_stability as fixtures
from battle.candidates import build_route_preview
from battle.stability import NormalLinkStability, boards_are_similar
from test_battle_candidates import board, ghoul


def anchors():
    """生成覆盖棋盘不同方向的稳定对象，不依赖固定格子。"""
    points = (
        (70, 300),
        (190, 310),
        (330, 340),
        (460, 300),
        (620, 320),
        (90, 470),
        (240, 490),
        (380, 460),
        (520, 500),
        (650, 480),
        (70, 660),
        (200, 640),
        (350, 680),
        (460, 650),
        (630, 670),
        (100, 910),
        (220, 900),
        (360, 920),
        (490, 910),
        (640, 900),
    )
    return [ghoul(i, x, y, "blue") for i, (x, y) in enumerate(points, 1)]


class LocalStabilityTests(unittest.TestCase):
    """少量局部差异只限制对应区域，实际移动与危险对象变化仍等待。"""

    def test_shell_and_green_dropout_can_use_other_stable_objects(self) -> None:
        """复现20:10的贝壳与绿色漏检，其余对象匹配时不再阻塞全盘。"""
        known = anchors()
        shell = replace(ghoul(90, 197, 352), kind="obstacle", color=None, subtype="shell", candidate_point=None)
        green = ghoul(91, 522, 827, "green")
        previous, current = board(*known, shell, green, frame_id="上帧"), board(*known, frame_id="本帧")
        tracker = NormalLinkStability()
        self.assertFalse(tracker.observe(previous))
        self.assertTrue(tracker.observe(current))
        planning = tracker.route_board
        assert planning is not None
        self.assertEqual(planning.frame_id, current.frame_id)
        self.assertEqual(planning.board_region, current.board_region)
        self.assertTrue(planning.route_exclusion_regions)
        self.assertFalse(current.route_exclusion_regions)
        self.assertTrue(all(item in current.objects for item in planning.objects))
        self.assertIn("局部", tracker.last_report["reason"])

    def test_partial_changes_do_not_hide_board_falling(self) -> None:
        """整体下落、变化过多、锚点只集中一角均不能容差放行。"""
        known = anchors()
        for current_objects in (
            [
                replace(
                    item,
                    center=(item.center[0], item.center[1] + 20),
                    candidate_point=(item.center[0], item.center[1] + 20),
                )
                for item in known
            ],
            known[:15],
        ):
            with self.subTest(current=current_objects):
                self.assertFalse(boards_are_similar(board(*known, frame_id="上帧"), board(*current_objects)))
        crowded = [ghoul(i, 100 + i, 400 + i) for i in range(1, 21)]
        self.assertFalse(boards_are_similar(board(*crowded, ghoul(90, 600), frame_id="上帧"), board(*crowded)))

    def test_obvious_local_movement_is_not_treated_as_detection_flicker(self) -> None:
        """其他位置稳定也不能掩盖一只对象明显移动。"""
        known = anchors()
        self.assertFalse(
            boards_are_similar(
                board(*known, ghoul(90, 300, 550), frame_id="上帧"),
                board(*known, ghoul(91, 340, 550)),
            )
        )

    def test_black_or_bomb_changes_never_use_local_tolerance(self) -> None:
        """黑色可能影响最高优先级，炸弹可能影响触控，均不能当作普通漏检。"""
        known = anchors()
        for special in (
            replace(ghoul(90, 300, 550), kind="special_ghoul", color="black", subtype="black"),
            replace(ghoul(90, 300, 550), kind="bomb", color=None, bomb_level=1),
        ):
            with self.subTest(kind=special.kind):
                self.assertFalse(boards_are_similar(board(*known, special, frame_id="上帧"), board(*known)))
                self.assertFalse(boards_are_similar(board(*known, frame_id="上帧"), board(*known, special)))

    def test_current_unmatched_objects_are_not_planned_and_masks_do_not_leak(self) -> None:
        """不确定对象只形成避让区，完整稳定后恢复，重置后不沿用许可。"""
        known = anchors()
        first = board(*known, frame_id="上帧")
        current = board(*known, ghoul(90, 300, 550, "yellow"), frame_id="本帧")
        tracker = NormalLinkStability()
        tracker.observe(first)
        self.assertTrue(tracker.observe(current))
        planning = tracker.route_board
        assert planning is not None
        self.assertNotIn(90, [item.id for item in planning.objects])
        self.assertTrue(planning.route_exclusion_regions)
        next_frame = replace(current, frame_id="下一帧")
        self.assertTrue(tracker.observe(next_frame))
        self.assertIs(tracker.route_board, next_frame)
        self.assertFalse(tracker.route_board.route_exclusion_regions)
        tracker.reset()
        self.assertFalse(tracker.observe(replace(current, frame_id="重置后")))

    def test_route_cannot_cross_or_touch_an_uncertain_region(self) -> None:
        """两个端点都在区域外，也必须阻止中间线段穿入或擦到避让区。"""
        for y in (380, 400):
            original = board(ghoul(1, 100, y), ghoul(2, 220, y))
            self.assertIsNotNone(build_route_preview(original).selected_route)
            masked = replace(original, route_exclusion_regions=[(150, 380, 20, 40)])
            self.assertIsNone(build_route_preview(masked).selected_route)
        safe = board(ghoul(1, 100, 350), ghoul(2, 220, 350))
        self.assertIsNotNone(
            build_route_preview(replace(safe, route_exclusion_regions=[(150, 380, 20, 40)])).selected_route
        )

    def test_invalid_exclusion_regions_do_not_allow_routes(self) -> None:
        """无效避让区不能被当成空列表跳过。"""
        original = board(ghoul(1, 100), ghoul(2, 220))
        for region in ((150, 380, 0, 40), (150, 380, -20, 40), (float("nan"), 380, 20, 40)):
            with self.subTest(region=region):
                result = build_route_preview(replace(original, route_exclusion_regions=[region]))
                self.assertEqual(result.status, "invalid_board")
                self.assertFalse(result.routes)

    def test_diagonal_crossing_is_blocked_in_both_directions(self) -> None:
        """斜线与反向斜线都不能从避让框中间穿过。"""
        for first, second in (((100, 350), (170, 420)), ((170, 420), (100, 350))):
            original = board(ghoul(1, *first), ghoul(2, *second))
            self.assertIsNotNone(build_route_preview(original).selected_route)
            masked = replace(original, route_exclusion_regions=[(130, 380, 10, 10)])
            self.assertIsNone(build_route_preview(masked).selected_route)

    def test_uncertain_endpoint_is_not_a_route_candidate(self) -> None:
        """触点落在避让区内时，不因另一端安全而允许连线。"""
        original = board(ghoul(1, 100), ghoul(2, 160))
        self.assertIsNotNone(build_route_preview(original).selected_route)
        masked = replace(original, route_exclusion_regions=[(150, 380, 20, 40)])
        self.assertIsNone(build_route_preview(masked).selected_route)

    def test_black_priority_waits_when_only_black_path_crosses_uncertainty(self) -> None:
        """即使彩色仍有路线，黑色路径受阻时也不能让彩色抢先。"""
        from test_battle_black_priority import black

        known = anchors()
        blacks = [black(90, 100, 550), black(91, 220, 550)]
        shell = replace(
            ghoul(92, 160, 550),
            kind="obstacle",
            color=None,
            subtype="shell",
            candidate_point=None,
            body_box=(150, 540, 20, 20),
        )
        replay = fixtures.BattleStabilityActionTests()
        operations, _ = replay.run_replay(
            [board(*known, *blacks, shell), board(*known, *blacks), board(*known, *blacks)]
        )
        self.assertEqual([(index, decision.kind) for index, decision in operations], [(3, "black_link")])

    def test_actual_action_can_continue_in_the_safe_current_region(self) -> None:
        """真实入口在局部漏检后使用本帧规划视图，不补回旧对象。"""
        known = anchors()
        shell = replace(ghoul(90, 197, 352), kind="obstacle", color=None, subtype="shell", candidate_point=None)
        replay = fixtures.BattleStabilityActionTests()
        executed = []
        operations, planned = replay.run_replay([board(*known, shell), board(*known)], executed_boards=executed)
        self.assertEqual([index for index, _ in operations], [2])
        self.assertTrue(planned[0].route_exclusion_regions)
        self.assertNotIn(90, [item.id for item in planned[0].objects])
        self.assertIs(executed[0], planned[0])
