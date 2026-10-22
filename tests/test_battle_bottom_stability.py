"""验证底部动画遮挡容差及普通连线的当前帧避让。"""

from __future__ import annotations

import unittest
from dataclasses import replace

import test_battle_stability as stability_tests
from battle.stability import NormalLinkStability, boards_are_similar
from test_battle_candidates import board, ghoul


class BottomStabilityTests(unittest.TestCase):
    """底部最多一只普通饿鬼误差，其余对象仍须严格对应。"""

    def test_one_bottom_ghoul_can_disappear_appear_or_jitter(self) -> None:
        """动画造成单只漏检、重现、颜色或几何变化时不再阻塞上方路线。"""
        upper = [ghoul(1, 100), ghoul(2, 190)]
        first = board(*upper, ghoul(3, 500, 910), frame_id="上帧")
        for changed in ([], [ghoul(8, 515, 910)], [ghoul(8, 500, 910, "blue")]):
            second = board(*upper, *changed, frame_id="本帧")
            with self.subTest(changed=changed):
                self.assertTrue(boards_are_similar(first, second))
                self.assertTrue(boards_are_similar(second, first))

    def test_other_changes_still_require_stability(self) -> None:
        """中上部、多个底部对象或底部道具变化均不得套用单只饿鬼容差。"""
        upper = [ghoul(1, 100), ghoul(2, 190)]
        cases = [
            (board(*upper, ghoul(3, 500, 700)), board(*upper)),
            (board(*upper, ghoul(3, 400, 910), ghoul(4, 550, 910)), board(*upper)),
            (
                board(*upper, ghoul(3, 400, 910), ghoul(4, 550, 910)),
                board(*upper, ghoul(3, 420, 910), ghoul(4, 570, 910)),
            ),
            (board(*upper, replace(ghoul(3, 500, 910), kind="bomb", color=None)), board(*upper)),
            (board(*upper, replace(ghoul(3, 500, 910), color="black")), board(*upper)),
            (board(*upper, ghoul(3, 500, 910)), board(ghoul(1, 120), upper[1])),
            (board(*upper, ghoul(3, 100, 910)), board(*upper, ghoul(4, 600, 910))),
        ]
        for first, second in cases:
            with self.subTest(first=first.objects, second=second.objects):
                self.assertFalse(boards_are_similar(replace(first, frame_id="上帧"), second))

    def test_bottom_band_scales_with_board_and_current_body_size(self) -> None:
        """整体缩放画面时保留同样语义，不使用固定像素边界。"""
        for scale in (0.5, 1.0, 2.0):
            original = board(ghoul(1, 100), ghoul(2, 190), ghoul(3, 500, 910))
            scaled = replace(
                original,
                frame_id="上帧",
                source_size=tuple(round(value * scale) for value in original.source_size),
                board_region=tuple(value * scale for value in original.board_region),
                objects=[
                    replace(
                        item,
                        center=tuple(value * scale for value in item.center),
                        body_box=tuple(value * scale for value in item.body_box),
                        candidate_point=tuple(value * scale for value in item.center),
                    )
                    for item in original.objects
                ],
            )
            with self.subTest(scale=scale):
                self.assertTrue(
                    boards_are_similar(scaled, replace(scaled, frame_id="本帧", objects=scaled.objects[:2]))
                )

    def test_one_remaining_object_is_not_enough_stability_evidence(self) -> None:
        """缺少其余对象的稳定证据时，不能仅靠底部例外放行。"""
        self.assertFalse(
            boards_are_similar(board(ghoul(1, 100), ghoul(2, 500, 910), frame_id="上帧"), board(ghoul(1, 100)))
        )

    def test_tolerated_frame_limits_routes_then_restores_full_board(self) -> None:
        """容差许可仅属于本帧，完全稳定后恢复底部候选，重置后不能沿用。"""
        first = board(ghoul(1, 100), ghoul(2, 190), ghoul(3, 500, 910), frame_id="上帧")
        second = board(ghoul(10, 102), ghoul(20, 192), ghoul(30, 515, 910), frame_id="本帧")
        stability = NormalLinkStability()
        self.assertFalse(stability.observe(first))
        self.assertTrue(stability.observe(second))
        route_board = stability.route_board
        assert route_board is not None
        self.assertEqual(route_board.frame_id, second.frame_id)
        self.assertEqual([item.id for item in route_board.objects], [10, 20])
        self.assertLess(route_board.board_region[3], second.board_region[3])
        self.assertEqual(len(second.objects), 3)
        self.assertIn("底部", str(stability.last_report["reason"]))
        third = replace(second, frame_id="下一帧")
        self.assertTrue(stability.observe(third))
        self.assertIs(stability.route_board, third)
        stability.reset()
        self.assertIsNone(stability.route_board)
        self.assertFalse(stability.observe(replace(third, frame_id="重置后")))
        self.assertIsNone(stability.route_board)

    def test_actual_action_uses_current_upper_objects_after_bottom_dropout(self) -> None:
        """实际策略入口不能拿容差许可在底部规划，也不能补回旧帧漏检项。"""
        replay = stability_tests.BattleStabilityActionTests()
        operations, planned = replay.run_replay(
            [
                board(ghoul(1, 100), ghoul(2, 190), ghoul(3, 400, 910), ghoul(4, 490, 910)),
                board(ghoul(10, 102), ghoul(20, 192), ghoul(40, 492, 910)),
            ]
        )
        self.assertEqual([index for index, _ in operations], [2])
        self.assertEqual([item.id for item in planned[0].objects], [10, 20])
        route = operations[0][1].route
        assert route is not None
        self.assertEqual(set(route.points), {(102, 400), (192, 400)})


if __name__ == "__main__":
    unittest.main()
