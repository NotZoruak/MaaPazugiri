"""核对终点大菱形、起点小星星与当前帧对象关联。"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import cv2
import numpy as np

from agent.vision.chain_endpoint import ChainEndpointRecognizer
from agent.vision.image_io import Image, read_image
from agent.vision.model import BoardObject, BoardResult, ChainMovePositions
from agent.vision.recognizer import BoardRecognizer

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "resource/base/image/board/chain_endpoint/profile.json"
SAMPLES = ROOT / "training/chain_samples"
EXAMPLES = (
    ("小于7.png", "under_7", (488.0, 357.0)),
    ("达到7.png", "from_7", (393.0, 478.0)),
    ("达到14.png", "from_14", (215.0, 677.0)),
    ("达到25.png", "from_25", (595.0, 429.0)),
)
HAS_SAMPLES = all((SAMPLES / name).is_file() for name, _, _ in EXAMPLES)


def board_for(image: Image, frame_id: str = "当前帧") -> BoardResult:
    """构造不带历史位置的当前帧输入。"""
    height, width = image.shape[:2]
    return BoardResult(frame_id, (width, height), (0, height * 0.164, width, height * 0.61), "样例", {})


def ghoul(identifier: int, center: tuple[float, float]) -> BoardObject:
    """构造当前帧中独立识别出的普通饿鬼。"""
    x, y = center
    return BoardObject(
        identifier,
        "ghoul",
        "red",
        None,
        "ghoul_red",
        "红色饿鬼",
        center,
        (x - 50, y - 50, 100, 100),
        {"width": 100, "height": 100},
        [],
        center,
        "identified",
    )


@unittest.skipUnless(HAS_SAMPLES, "需要本地连线开发截图，未分发原图时跳过")
class ChainEndpointTests(unittest.TestCase):
    """通过实际连线画面核对视觉标记，不读取连接数量。"""

    @classmethod
    def setUpClass(cls) -> None:
        """各帧共享静态模板，动态候选始终重新生成。"""
        cls.recognizer = ChainEndpointRecognizer(PROFILE)

    def test_four_confirmed_appearances(self) -> None:
        """四档外观均定位到终点，而非起点的小星星。"""
        for filename, variant, expected in EXAMPLES:
            with self.subTest(filename=filename):
                image = read_image(SAMPLES / filename)
                result = self.recognizer.analyze(image, board_for(image, filename))
                self.assertEqual(result.status, "detected")
                self.assertEqual(len(result.markers), 1)
                self.assertEqual(result.markers[0].variant, variant)
                self.assertLess(math.dist(result.markers[0].center, expected), 7)
                self.assertIsNone(result.object_id)

    def test_runtime_pink_glow_does_not_hide_the_white_diamond(self) -> None:
        """18:10现场的粉色动效改变外缘色相，但当前帧白色菱形仍在真实终点。"""
        sample = SAMPLES / "运行_20261021_181031.png"
        if not sample.is_file():
            self.skipTest("需要本地18:10运行现场")
        image = read_image(sample)
        result = self.recognizer.analyze(image, board_for(image), move=ChainMovePositions((393, 830), (424, 924)))
        self.assertEqual(result.status, "detected")
        self.assertEqual(result.move_outcome, "target_reached")
        self.assertEqual(len(result.markers), 1)
        self.assertLess(math.dist(result.markers[0].center, (424, 919)), 5)

    def test_start_markers_are_not_endpoints(self) -> None:
        """单独保留小星星和其连线时不能误报终点。"""
        starts = (
            ("小于7.png", (178, 480, 80, 90)),
            ("达到7.png", (437, 865, 80, 90)),
            ("达到14.png", (617, 750, 80, 90)),
            ("达到25.png", (333, 365, 80, 90)),
        )
        for filename, (x, y, width, height) in starts:
            with self.subTest(filename=filename):
                source = read_image(SAMPLES / filename)
                image = np.zeros_like(source)
                image[y : y + height, x : x + width] = source[y : y + height, x : x + width]
                result = self.recognizer.analyze(image, board_for(image))
                self.assertEqual(result.status, "not_detected")
                self.assertFalse(result.markers)

    def test_translation_and_resolution_do_not_reuse_positions(self) -> None:
        """同一识别器每帧重新搜索，并还原同布局截图坐标。"""
        source = read_image(SAMPLES / "达到7.png")
        self.recognizer.analyze(source, board_for(source, "上一帧"))
        transform = np.array(((1, 0, 40), (0, 1, -65)), dtype=np.float32)
        moved = cv2.warpAffine(source, transform, (720, 1280))
        image = cast(Image, cv2.resize(moved, (540, 960)))
        result = self.recognizer.analyze(image, board_for(image, "新帧"))
        self.assertEqual(result.status, "detected")
        self.assertEqual(result.frame_id, "新帧")
        self.assertLess(math.dist(result.markers[0].center, ((393 + 40) * 0.75, (478 - 65) * 0.75)), 7)

    def test_two_markers_remain_ambiguous(self) -> None:
        """两个独立大菱形同时出现时不能仅按最高分选终点。"""
        source = read_image(SAMPLES / "达到7.png")
        image = np.zeros_like(source)
        crop = source[432:522, 350:435]
        image[432:522, 350:435] = crop
        image[650:740, 100:185] = crop
        result = self.recognizer.analyze(image, board_for(image))
        self.assertEqual(result.status, "uncertain")
        self.assertEqual(len(result.markers), 2)
        self.assertIsNone(result.object_id)

    def test_current_frame_object_association(self) -> None:
        """只关联当前帧唯一近邻饿鬼，不生成缺失对象。"""
        image = read_image(SAMPLES / "达到7.png")
        board = board_for(image)
        board.objects = [ghoul(8, (393, 480)), ghoul(9, (470, 530))]
        result = self.recognizer.analyze(image, board)
        self.assertEqual(result.object_id, 8)
        self.assertEqual(result.association_status, "matched")
        board.objects = [ghoul(8, (390, 480)), ghoul(9, (396, 480))]
        ambiguous = self.recognizer.analyze(image, board)
        self.assertIsNone(ambiguous.object_id)
        self.assertEqual(ambiguous.association_status, "ambiguous")

    def test_no_chain_board_has_no_endpoint(self) -> None:
        """普通眼部、炸弹闪光和界面图案不能被当作终点。"""
        paths = list((ROOT / "training/board-samples/test").glob("MuMu-20261018-*.png"))
        paths = [path for path in paths if any(time in path.name for time in ("095352", "100950", "102410"))]
        if not paths:
            self.skipTest("需要本地未连线截图")
        for path in paths:
            with self.subTest(filename=path.name):
                image = read_image(path)
                result = self.recognizer.analyze(image, board_for(image))
                self.assertEqual(result.status, "not_detected")

    def test_board_recognizer_exports_endpoint_in_both_modes(self) -> None:
        """完整与轻量模式均输出终点结果，仍禁止自动操作。"""
        recognizer = BoardRecognizer(ROOT / "resource/base/image/board/development/profile.json", workers=2)
        image = read_image(SAMPLES / "达到25.png")
        for lightweight in (False, True):
            with self.subTest(lightweight=lightweight):
                board = recognizer.analyze(
                    image, "同帧", lightweight=lightweight, chain_move=ChainMovePositions((100, 900), (595, 429))
                )
                self.assertIsNotNone(board.chain_endpoint)
                assert board.chain_endpoint is not None
                self.assertEqual(board.chain_endpoint.status, "detected")
                self.assertEqual(board.chain_endpoint.move_outcome, "target_reached")
                self.assertEqual(board.to_dict()["chain_endpoint"]["frame_id"], "同帧")
                self.assertFalse(board.automatic_action_allowed)

    def test_projection_bound_preserves_marker_results(self) -> None:
        """上界筛选和全部窗口精查在四档开发图上保留相同终点结果。"""
        for filename, _, _ in EXAMPLES:
            with self.subTest(filename=filename):
                image = read_image(SAMPLES / filename)
                board = board_for(image)
                filtered = self.recognizer.analyze(image, board)
                with patch("agent.vision.chain_endpoint.projection_may_reach", return_value=True):
                    unfiltered = self.recognizer.analyze(image, board)
                self.assertEqual(filtered.status, unfiltered.status)
                self.assertEqual(filtered.markers, unfiltered.markers)

    def test_move_checks_both_positions_in_four_variants(self) -> None:
        """四档均按本段移动起止位置确认，无须主体编号。"""
        for filename, _, endpoint in EXAMPLES:
            image = read_image(SAMPLES / filename)
            other = (100.0, 900.0)
            for move, outcome in (
                (ChainMovePositions(other, endpoint), "target_reached"),
                (ChainMovePositions(endpoint, other), "still_at_start"),
            ):
                with self.subTest(filename=filename, outcome=outcome):
                    result = self.recognizer.analyze(image, board_for(image), move=move)
                    self.assertEqual(result.move_outcome, outcome)
                    self.assertEqual(result.coverage["search_scope"], "move_start_and_end")
                    self.assertIsNone(result.object_id)

    def test_move_never_falls_back_to_board_search(self) -> None:
        """两处未命中时不能用第三处的大菱形补回。"""
        image = read_image(SAMPLES / "达到7.png")
        move = ChainMovePositions((100, 800), (250, 900))
        result = self.recognizer.analyze(image, board_for(image), move=move)
        self.assertEqual(result.status, "not_detected")
        self.assertEqual(result.move_outcome, "unknown")
        self.assertFalse(result.markers)

    def test_overlapping_move_regions_are_deduplicated(self) -> None:
        """区域重叠时同一菱形只计一次，角色无法区分则未知。"""
        image = read_image(SAMPLES / "达到7.png")
        result = self.recognizer.analyze(image, board_for(image), move=ChainMovePositions((388, 478), (398, 478)))
        self.assertEqual(len(result.markers), 1)
        self.assertEqual(result.move_outcome, "unknown")

    def test_move_conflicts_and_invalid_coordinates_are_unknown(self) -> None:
        """两处均有大菱形或动作坐标无效时不可放行。"""
        source = read_image(SAMPLES / "达到7.png")
        image = np.zeros_like(source)
        crop = source[432:522, 350:435]
        image[432:522, 350:435] = crop
        image[650:740, 100:185] = crop
        result = self.recognizer.analyze(image, board_for(image), move=ChainMovePositions((393, 478), (143, 696)))
        self.assertEqual(len(result.markers), 2)
        self.assertEqual(result.move_outcome, "unknown")
        for start in ((float("nan"), 800), (-20, 800), (True, 800)):
            with self.subTest(start=start):
                invalid = self.recognizer.analyze(source, board_for(source), move=ChainMovePositions(start, (393, 478)))
                self.assertEqual(invalid.move_outcome, "unknown")
                self.assertFalse(invalid.markers)

    def test_move_uses_source_resolution_coordinates(self) -> None:
        """动作与标记均使用输入截图坐标，容许停留点略偏离核心。"""
        source = read_image(SAMPLES / "达到7.png")
        image = cast(Image, cv2.resize(source, (540, 960)))
        move = ChainMovePositions((75, 600), ((393 + 20) * 0.75, (478 - 25) * 0.75))
        result = self.recognizer.analyze(image, board_for(image), move=move)
        self.assertEqual(result.move_outcome, "target_reached")
        self.assertLess(math.dist(result.markers[0].center, (393 * 0.75, 478 * 0.75)), 7)

    def test_nearest_endpoint_must_cover_marker(self) -> None:
        """只在起点方框内的标记不能因欧氏距离更近而误判目标接入。"""
        source = read_image(SAMPLES / "达到7.png")
        transform = np.array(((1, 0, -38), (0, 1, 177)), dtype=np.float32)
        image = cast(Image, cv2.warpAffine(source, transform, (720, 1280)))
        result = self.recognizer.analyze(image, board_for(image), move=ChainMovePositions((300, 600), (412, 655)))
        self.assertEqual(result.status, "detected")
        self.assertEqual(result.move_outcome, "unknown")


class ChainEndpointInputTests(unittest.TestCase):
    """不依赖本地训练截图的输入和提前返回回归。"""

    def test_unsupported_layout_keeps_unknown_move_feedback(self) -> None:
        """棋盘布局不支持时仍输出完整的未知动作反馈。"""
        recognizer = BoardRecognizer(ROOT / "resource/base/image/board/development/profile.json", workers=1)
        image = np.zeros((300, 300, 3), dtype=np.uint8)
        board = recognizer.analyze(image, "无效布局", chain_move=ChainMovePositions((50, 100), (100, 200)))
        self.assertEqual(board.scene_state, "unknown_ui")
        self.assertIsNotNone(board.chain_endpoint)
        assert board.chain_endpoint is not None
        self.assertEqual(board.chain_endpoint.move_outcome, "unknown")
        self.assertEqual(board.chain_endpoint.status, "uncertain")

    def test_malformed_move_coordinates_do_not_trigger_search(self) -> None:
        """畸形坐标必须返回未知，不能抛出下标错误或回退搜索。"""
        recognizer = ChainEndpointRecognizer(PROFILE)
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        for malformed in ((100,), (100, 800, 20), (), ("100", 800)):
            with self.subTest(malformed=malformed):
                move = ChainMovePositions(cast(tuple[float, float], malformed), (393, 478))
                result = recognizer.analyze(image, board_for(image), move=move)
                self.assertEqual(result.move_outcome, "unknown")
                self.assertEqual(result.status, "uncertain")
                self.assertFalse(result.markers)


if __name__ == "__main__":
    unittest.main()
