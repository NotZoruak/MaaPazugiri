"""验证倒计时图案的同帧组合证据，避免末尾0.9秒漏判。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.ranbu_state import RANBU_COUNTDOWN_ROI, matches_ranbu_countdown  # noqa: E402
from vision.image_io import read_image  # noqa: E402


class RanbuCountdownTests(unittest.TestCase):
    """数字、低位小数点与发光描边必须一起出现，单独颜色不能确认。"""

    def countdown(self) -> np.ndarray:
        """构造满足实测笔画部件位置的图案，单独控制每一条必要证据。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        x, y, _, _ = RANBU_COUNTDOWN_ROI
        mask = np.zeros((49, 83), np.uint8)
        mask[11:39, 13:24] = 255
        mask[10:39, 48:60] = 255
        mask[32:37, 38:42] = 255
        patch = image[y : y + 49, x : x + 83]
        patch[cv2.dilate(mask, np.ones((5, 5), np.uint8)) != 0] = (255, 30, 255)
        patch[mask != 0] = 255
        return image

    def test_requires_two_digits_low_decimal_point_and_magenta_border(self) -> None:
        """缺小数点、缺一位数字、缺描边时都不能作为乱舞正面证据。"""
        image = self.countdown()
        self.assertTrue(matches_ranbu_countdown(image))
        x, y, _, _ = RANBU_COUNTDOWN_ROI
        without_dot = image.copy()
        without_dot[y + 29 : y + 40, x + 33 : x + 45] = 0
        self.assertFalse(matches_ranbu_countdown(without_dot))
        without_digit = image.copy()
        without_digit[y + 5 : y + 43, x + 45 : x + 64] = 0
        self.assertFalse(matches_ranbu_countdown(without_digit))
        without_border = image.copy()
        without_border[np.min(without_border, axis=2) <= 220] = 0
        self.assertFalse(matches_ranbu_countdown(without_border))

    def test_rejects_unknown_layouts_and_unrelated_white_shapes(self) -> None:
        """未知尺寸、颜色类型和无小数点的大块文字一律拒绝。"""
        self.assertFalse(matches_ranbu_countdown(np.zeros((640, 360, 3), np.uint8)))
        self.assertFalse(matches_ranbu_countdown(self.countdown().astype(np.float32)))
        image = np.zeros((1280, 720, 3), np.uint8)
        image[200:240, 565:640] = 255
        self.assertFalse(matches_ranbu_countdown(image))

    def test_reviewed_training_frames_do_not_introduce_false_positives(self) -> None:
        """开发集两张乱舞为正，其余41张必须保持阴性。"""
        manifest = ROOT / "debug/battle-ultimate/sources.json"
        if not manifest.is_file():
            self.skipTest("需要本地训练图清单")
        for item in json.loads(manifest.read_text(encoding="utf-8")):
            with self.subTest(source=item["source"]):
                self.assertEqual(matches_ranbu_countdown(read_image(Path(item["source"]))), item["index"] in (10, 17))

    def test_all_five_live_before_and_held_frames_include_the_last_point_nine(self) -> None:
        """现场五轮按下前与按住帧全部确认，包括原颜色标记失效的0.9秒。"""
        directory = ROOT / "debug/ranbu-regression-20261021"
        paths = [
            directory / f"ranbu-execution-{index:02d}.{kind}.png"
            for index in range(1, 6)
            for kind in ("before", "held")
        ]
        if not all(path.is_file() for path in paths):
            self.skipTest("需要本地五轮乱舞诊断截图")
        for path in paths:
            with self.subTest(source=path.name):
                self.assertTrue(matches_ranbu_countdown(read_image(path)))


if __name__ == "__main__":
    unittest.main()
