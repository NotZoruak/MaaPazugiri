"""验证自由布局同帧候选路线，不执行游戏操作。"""

from __future__ import annotations

import sys
import unittest
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.candidates import build_route_preview  # noqa: E402
from vision.connectability import ConnectabilityResult, ConnectabilityStatus  # noqa: E402
from vision.model import BoardObject, BoardResult  # noqa: E402


def ghoul(identifier: int, x: float, y: float = 400.0, color: str = "red") -> BoardObject:
    """生成不同位置的当前帧普通饿鬼，不使用固定网格。"""
    return BoardObject(
        id=identifier,
        kind="ghoul",
        color=color,
        bomb_level=None,
        label=f"ghoul_{color}",
        name="测试饿鬼",
        center=(x, y),
        body_box=(x - 50, y - 50, 100, 100),
        size={"width": 100, "height": 100},
        evidence=[],
        candidate_point=(x, y),
        quality="identified",
    )


def board(*objects: BoardObject, frame_id: str = "测试当前帧") -> BoardResult:
    """构造当前帧对象输入。"""
    return BoardResult(frame_id, (720, 1280), (0, 210, 720, 780), "测试", {}, objects=list(objects))


class BattleCandidateTests(unittest.TestCase):
    """确保路线只来自当前帧可靠候选边。"""

    def test_touch_points_use_current_body_centers_instead_of_eye_features(self) -> None:
        """同一对象眼部触点切换不改变落点，滑动经过当前识别出的身体中心。"""
        first, second = ghoul(1, 148, 386), ghoul(2, 234, 382)
        first.candidate_point = (148, 368)
        second.candidate_point = (234, 364)
        result = build_route_preview(board(first, second))
        self.assertIsNotNone(result.selected_route)
        self.assertEqual(result.selected_route.points, ((148, 386), (234, 382)))

    def test_body_center_covered_by_a_bomb_keeps_the_verified_visible_touch_point(self) -> None:
        """身体中心被炸弹覆盖时，不能绕过原候选触点的炸弹排除。"""
        first, second = ghoul(1, 100), ghoul(2, 190)
        first.candidate_point = (100, 380)
        bomb = ghoul(3, 100, 435)
        bomb.kind = "bomb"
        bomb.color = None
        result = build_route_preview(board(first, second, bomb))
        self.assertIsNotNone(result.selected_route)
        self.assertEqual(result.selected_route.points, ((100, 380), (190, 400)))

    def test_rounded_body_center_outside_the_image_keeps_the_visible_point(self) -> None:
        """普通连线与乱舞共用身体落点时，也必须保证取整后仍在截图内。"""
        edge = ghoul(1, 719.8)
        edge.candidate_point = (700, 380)
        result = build_route_preview(board(edge, ghoul(2, 630)))
        self.assertIsNotNone(result.selected_route)
        self.assertEqual(result.selected_route.points[0], (700, 380))

    def test_branching_route_keeps_constrained_endpoints(self) -> None:
        """局部余量更大的三角形不能挤掉两端受限的完整路线。"""
        links = {(1, 2): 0.1, (2, 3): 0.1, (3, 4): 0.9, (3, 5): 0.8, (4, 5): 0.8, (4, 6): 0.1}

        def assess(frame: BoardResult, first: int, second: int) -> ConnectabilityResult:
            """固定图拓扑以隔离搜索行为，其余用例核对真实几何。"""
            margin = links.get((first, second))
            status = ConnectabilityStatus.CANDIDATE if margin is not None else ConnectabilityStatus.REJECTED
            return ConnectabilityResult(frame.frame_id, first, second, status, "测试图", edge_clearances=(margin or 0,))

        objects = [ghoul(identifier, 100 + identifier * 5) for identifier in range(1, 7)]
        with patch("battle.candidates.assess_ghoul_connectability", side_effect=assess):
            result = build_route_preview(board(*objects), beam_width=1)
        assert result.selected_route is not None
        self.assertEqual(set(result.selected_route.object_ids), set(range(1, 7)))

    def test_picture_two_route_keeps_missing_objects(self) -> None:
        """真实图2的3、6、17已有候选边，不能被搜索漏掉。"""
        from vision.image_io import read_image
        from vision.recognizer import BoardRecognizer

        source = ROOT / "training/board-samples/test/图2.png"
        if not source.is_file():
            self.skipTest("需要用户图2本地回归样本")
        frame = BoardRecognizer(ROOT / "resource/base/image/board/development/profile.json").analyze(
            read_image(source), frame_id="图2路径回归"
        )
        result = build_route_preview(frame)
        assert result.selected_route is not None
        self.assertGreaterEqual(len(result.selected_route.object_ids), 33)
        self.assertTrue({3, 6, 17}.issubset(result.selected_route.object_ids))

    def test_chain_uses_only_candidate_edges_and_current_touch_points(self) -> None:
        """相邻三只形成一条路线，不能跨过中间对象生成捷径。"""
        result = build_route_preview(board(ghoul(1, 100), ghoul(2, 190), ghoul(3, 280)))
        self.assertEqual({(edge.first_id, edge.second_id) for edge in result.candidate_edges}, {(1, 2), (2, 3)})
        self.assertIsNotNone(result.selected_route)
        assert result.selected_route is not None
        self.assertEqual(result.selected_route.object_ids, (1, 2, 3))
        self.assertEqual(result.selected_route.points, ((100, 400), (190, 400), (280, 400)))
        self.assertFalse(result.automatic_action_allowed)

    def test_uncertain_boundary_is_reported_but_not_used_in_route(self) -> None:
        """边界附近仅报告不确定关系。"""
        result = build_route_preview(board(ghoul(1, 100), ghoul(2, 244)))
        self.assertFalse(result.candidate_edges)
        self.assertEqual(len(result.uncertain_edges), 1)
        self.assertIsNone(result.selected_route)

    def test_colors_and_special_objects_are_not_mixed(self) -> None:
        """异色、炸弹和特殊黑色饿鬼不进入普通连线。"""
        special = ghoul(3, 140)
        special.kind = "special_ghoul"
        bomb = ghoul(4, 160)
        bomb.kind = "bomb"
        result = build_route_preview(board(ghoul(1, 100), ghoul(2, 190, color="blue"), special, bomb))
        self.assertFalse(result.candidate_edges)
        self.assertIsNone(result.selected_route)
        self.assertEqual({item["object_id"] for item in result.excluded_objects}, {3, 4})

    def test_missing_or_invalid_touch_points_are_excluded(self) -> None:
        """几何位置存在也不能给没有有效候选触点的对象生成路线。"""
        first, second = ghoul(1, 100), ghoul(2, 190)
        second.candidate_point = None
        self.assertIsNone(build_route_preview(board(first, second)).selected_route)
        second.candidate_point = (float("nan"), 400)
        self.assertIsNone(build_route_preview(board(first, second)).selected_route)

    def test_duplicate_object_ids_invalidate_preview(self) -> None:
        """编号冲突时不输出可能关联到错误对象的路线。"""
        result = build_route_preview(board(ghoul(1, 100), ghoul(1, 190)))
        self.assertEqual(result.status, "invalid_board")
        self.assertIsNone(result.selected_route)

    def test_input_order_does_not_change_route(self) -> None:
        """同帧对象换序不改变候选边或拟选路线。"""
        objects = [ghoul(1, 100), ghoul(2, 190), ghoul(3, 280)]
        first = build_route_preview(board(*objects))
        second = build_route_preview(board(*reversed(objects)))
        self.assertEqual(first.selected_route, second.selected_route)
        self.assertEqual(first.candidate_edges, second.candidate_edges)

    def test_new_frame_does_not_reuse_old_positions(self) -> None:
        """编号相同但新帧对象已移动时重新判定。"""
        first = build_route_preview(board(ghoul(1, 100), ghoul(2, 190), frame_id="前一帧"))
        second = build_route_preview(board(ghoul(1, 100), ghoul(2, 600), frame_id="当前帧"))
        self.assertIsNotNone(first.selected_route)
        self.assertIsNone(second.selected_route)
        self.assertEqual(second.frame_id, "当前帧")

    def test_search_budget_is_bounded_and_route_never_repeats_objects(self) -> None:
        """有限预算下仍只输出合法候选路径，并说明搜索截断。"""
        objects = [ghoul(index + 1, x, y) for index, (x, y) in enumerate(((100, 400), (190, 400), (145, 475)))]
        result = build_route_preview(board(*objects), beam_width=1, max_expansions=1)
        self.assertLessEqual(result.search_expansions, 1)
        self.assertTrue(result.search_truncated)
        assert result.selected_route is not None
        self.assertEqual(len(result.selected_route.object_ids), len(set(result.selected_route.object_ids)))
        edges = {frozenset((edge.first_id, edge.second_id)) for edge in result.candidate_edges}
        for first, second in pairwise(result.selected_route.object_ids):
            self.assertIn(frozenset((first, second)), edges)

    def test_enhanced_ghouls_keep_their_color(self) -> None:
        """强化属性不改写已确认的普通饿鬼颜色。"""
        first = ghoul(1, 100)
        first.enhanced = True
        self.assertIsNotNone(build_route_preview(board(first, ghoul(2, 190))).selected_route)


if __name__ == "__main__":
    unittest.main()
