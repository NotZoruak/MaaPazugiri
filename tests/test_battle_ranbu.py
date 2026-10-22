"""验证乱舞颜色门禁与普通可操作状态互斥，不执行任何滑动。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.ranbu import (  # noqa: E402
    RANBU_LOWER_BGR,
    RANBU_ROI,
    choose_ranbu_start,
    matches_ranbu_marker,
    ranbu_sweep_points,
)
from battle.recognition import RECOGNITIONS, BattleFrameRecognizer  # noqa: E402
from battle.session import BattleOutcome, BattleSession, BattleState  # noqa: E402
from test_battle_candidates import ghoul  # noqa: E402
from vision.image_io import read_image  # noqa: E402
from vision.model import BoardResult  # noqa: E402


class RanbuRecognitionTests(unittest.TestCase):
    """颜色标记只确认乱舞阶段，不读取5.0数字或决定滑动时长。"""

    def test_available_colors_rotate_without_repeating_or_starving_yellow(self) -> None:
        """有其他颜色时轮换，不能因蓝色编号总靠前而漏掉黄色。"""
        board = BoardResult(
            "当前帧",
            (720, 1280),
            (0, 210, 720, 780),
            "离线",
            {},
            objects=[
                ghoul(1, 100, 400, "blue"),
                ghoul(2, 300, 400, "yellow"),
                ghoul(3, 500, 400, "green"),
                ghoul(0, 400, 700, "red"),
            ],
        )
        self.assertEqual(choose_ranbu_start(board, "red", previous_color="blue"), (500, 400))
        self.assertEqual(choose_ranbu_start(board, "red", previous_color="green"), (300, 400))
        self.assertEqual(choose_ranbu_start(board, "red", previous_color="yellow"), (100, 400))

    def test_previous_color_is_used_when_it_is_the_only_available_color(self) -> None:
        """仅剩上一种非保留色时继续使用，不因轮换规则空等。"""
        board = BoardResult(
            "当前帧",
            (720, 1280),
            (0, 210, 720, 780),
            "离线",
            {},
            objects=[ghoul(1, 100, 400, "blue"), ghoul(2, 300, 400, "red")],
        )
        self.assertEqual(choose_ranbu_start(board, "red", previous_color="blue"), (100, 400))

    def test_start_uses_visible_anchor_when_body_center_overlaps_reserved_color(self) -> None:
        """估计身体中心落在保留色中央时，优先自身可见部位，避免按错颜色。"""
        blue = ghoul(1, 100, 430, "blue")
        blue.candidate_point = (100, 390)
        reserved = ghoul(2, 100, 450, "red")
        board = BoardResult("当前帧", (720, 1280), (0, 210, 720, 780), "离线", {}, objects=[blue, reserved])
        self.assertEqual(choose_ranbu_start(board, "red"), (100, 390))
        blue.candidate_point = None
        self.assertIsNone(choose_ranbu_start(board, "red"))

    def test_full_board_sweep_does_not_depend_on_ghoul_positions_or_colors(self) -> None:
        """完整折返只由棋盘边界决定，覆盖顶部到底部且不跟随同色身体位置。"""
        top = ghoul(1, 100, 305, "blue")
        bottom = ghoul(2, 160, 965, "blue")
        reserved = ghoul(3, 300, 600, "red")
        top.candidate_point = (100, 285)
        bottom.candidate_point = (160, 945)
        board = BoardResult("当前帧", (720, 1280), (0, 210, 720, 780), "离线", {}, objects=[top, bottom, reserved])
        start = choose_ranbu_start(board, "red")
        self.assertEqual(start, (100, 305))
        points = ranbu_sweep_points(board)
        empty = BoardResult("其他帧", (720, 1280), (0, 210, 720, 780), "离线", {})
        self.assertEqual(points, ranbu_sweep_points(empty))
        self.assertEqual(points[0], (0, 210))
        self.assertEqual(points[-1], (719, 989))
        self.assertEqual({x for x, _ in points}, {0, 719})
        heights = sorted({y for _, y in points})
        self.assertTrue(all(b - a <= 36 for a, b in zip(heights, heights[1:], strict=False)))

    def test_invalid_current_body_positions_do_not_enter_the_sweep(self) -> None:
        """当前帧的非有限坐标必须跳过，不能在取整时崩溃或补造落点。"""
        valid = ghoul(1, 100, 305, "blue")
        invalid = ghoul(2, float("nan"), 600, "blue")
        board = BoardResult("当前帧", (720, 1280), (0, 210, 720, 780), "离线", {}, objects=[valid, invalid])
        self.assertEqual(
            ranbu_sweep_points(board),
            ranbu_sweep_points(BoardResult("空", (720, 1280), (0, 210, 720, 780), "离线", {})),
        )

    def test_rounded_body_center_outside_the_image_falls_back_to_the_visible_point(self) -> None:
        """719.8像素的中心取整会到720，不能把有效眼部触点替换成越界按下点。"""
        edge = ghoul(1, 719.8, 400, "blue")
        edge.candidate_point = (700, 380)
        board = BoardResult("当前帧", (720, 1280), (0, 210, 720, 780), "离线", {}, objects=[edge])
        start = choose_ranbu_start(board, "red")
        self.assertEqual(start, (700, 380))
        self.assertTrue(all(0 <= x < 720 and 210 <= y < 990 for x, y in ranbu_sweep_points(board)))

    def test_special_black_start_does_not_inherit_an_overlapping_ordinary_color(self) -> None:
        """黑色起点与普通对象坐标重合时，也不能推断黑色使用普通颜色规则。"""
        black = ghoul(0, 100, 305)
        black.kind, black.color, black.subtype = "special_ghoul", None, "black"
        board = BoardResult(
            "当前帧",
            (720, 1280),
            (0, 210, 720, 780),
            "离线",
            {},
            objects=[black, ghoul(1, 100, 305, "blue"), ghoul(2, 160, 965, "blue")],
        )
        points = ranbu_sweep_points(board)
        self.assertEqual(points, ranbu_sweep_points(BoardResult("空", (720, 1280), (0, 210, 720, 780), "离线", {})))

    def test_special_black_does_not_starve_available_ordinary_colors(self) -> None:
        """黑色规则未知，普通可用颜色优先轮换；只有黑色可用时沿用其独立扫动。"""
        black = ghoul(0, 100, 700)
        black.kind, black.color, black.subtype = "special_ghoul", None, "black"
        board = BoardResult(
            "当前帧",
            (720, 1280),
            (0, 210, 720, 780),
            "离线",
            {},
            objects=[black, ghoul(1, 100, 400, "blue"), ghoul(2, 300, 400, "green")],
        )
        self.assertEqual(choose_ranbu_start(board, "red"), (100, 400))
        self.assertEqual(choose_ranbu_start(board, "red", previous_color="blue"), (300, 400))
        board.objects = [black]
        self.assertEqual(choose_ranbu_start(board, "red"), (100, 700))

    def marker_frame(self) -> np.ndarray:
        """生成仅含当前已核对固定颜色区域的测试帧。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        x, y, width, height = RANBU_ROI
        image[y : y + height, x : x + width] = RANBU_LOWER_BGR
        return image

    def recognizer(self, **hits: bool) -> BattleFrameRecognizer:
        """固定原生状态证据，单独核对颜色门禁与互斥处理。"""

        def recognize(method, parameters, image):
            """只返回传入条件的命中结果。"""
            if method.value == "OCR":
                return SimpleNamespace(all_results=[])
            name = next(name for name, (_, param) in RECOGNITIONS.items() if param is parameters)
            return SimpleNamespace(hit=hits.get(name, False))

        return BattleFrameRecognizer(recognize)

    def test_color_requires_every_pixel_and_supported_layout(self) -> None:
        """全部颜色命中才确认，单像素不符或尺寸未知时拒绝。"""
        image = self.marker_frame()
        self.assertTrue(matches_ranbu_marker(image))
        image[200:250, 565:650] = 255
        self.assertTrue(matches_ranbu_marker(image))
        x, y, _, _ = RANBU_ROI
        image[y, x] = 0
        self.assertFalse(matches_ranbu_marker(image))
        self.assertFalse(matches_ranbu_marker(np.zeros((640, 360, 3), np.uint8)))

    def test_ranbu_is_a_separate_phase_and_conflicting_evidence_is_unknown(self) -> None:
        """乱舞需有战斗界面证据，与普通可操作证据同时命中时保持未知。"""
        image = self.marker_frame()
        self.assertEqual(self.recognizer(gear=True).analyze(image), BattleState.RANBU)
        self.assertEqual(self.recognizer().analyze(image), BattleState.UNKNOWN)
        self.assertEqual(self.recognizer(gear=True, brightness=True).analyze(image), BattleState.UNKNOWN)
        self.assertEqual(self.recognizer(victory=True).analyze(image), BattleState.VICTORY)
        ordinary = np.zeros_like(image)
        self.assertEqual(self.recognizer(gear=True, brightness=True).analyze(ordinary), BattleState.OPERABLE)

    def test_ranbu_confirms_activity_without_starting_ordinary_preview(self) -> None:
        """乱舞确认重置连续未确认计时，但不调用普通棋盘预演。"""
        now = [0.0]
        frames = iter(((119.0, BattleState.RANBU), (238.0, BattleState.RANBU), (240.0, BattleState.VICTORY)))

        def observe():
            """模拟本轮新截图所确认的时刻与状态。"""
            now[0], state = next(frames)
            return state

        preview = Mock()
        session = BattleSession(clock=lambda: now[0], sleep=lambda _: None)
        self.assertEqual(session.run(observe, stopped=lambda: False, on_operable=preview), BattleOutcome.VICTORY)
        preview.assert_not_called()

    def test_local_markers_do_not_match_other_reviewed_frames(self) -> None:
        """现有两张乱舞图颜色命中，其余41张开发图不命中。"""
        manifest = ROOT / "debug/battle-ultimate/sources.json"
        if not manifest.is_file():
            self.skipTest("需要本地训练图清单")
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            with self.subTest(source=item["source"]):
                self.assertEqual(matches_ranbu_marker(read_image(Path(item["source"]))), item["index"] in (10, 17))
