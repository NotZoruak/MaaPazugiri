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

from battle.ranbu import RANBU_LOWER_BGR, RANBU_ROI, matches_ranbu_marker  # noqa: E402
from battle.recognition import RECOGNITIONS, BattleFrameRecognizer  # noqa: E402
from battle.session import BattleOutcome, BattleSession, BattleState  # noqa: E402
from vision.image_io import read_image  # noqa: E402


class RanbuRecognitionTests(unittest.TestCase):
    """颜色标记只确认乱舞阶段，不读取5.0数字或决定滑动时长。"""

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
