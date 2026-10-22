"""验证乱舞预选点使用当前帧局部像素，并回归五次实机起点。"""

from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from vision.image_io import Image, read_image  # noqa: E402
from vision.ranbu_point import identify_ranbu_point_color, ranbu_point_color_margin  # noqa: E402


class RanbuPointPixelTests(unittest.TestCase):
    """双色、空白和尺寸边界不能作为颜色确认依据。"""

    def test_margin_uses_current_pixels_without_relaxing_the_original_color_check(self) -> None:
        """颜色余量只评估当前画面，不改变颜色判定阈值。"""
        blue = cv2.cvtColor(np.full((200, 220, 3), (100, 220, 200), dtype=np.uint8), cv2.COLOR_HSV2BGR)
        self.assertEqual(ranbu_point_color_margin(blue, (110, 100), (96, 100), "blue"), 4)
        self.assertEqual(ranbu_point_color_margin(blue, (110, 100), (96, 100), "red"), 0)
        blue[:, :] = 0
        self.assertEqual(ranbu_point_color_margin(blue, (110, 100), (96, 100), "blue"), 0)
        self.assertIsNone(identify_ranbu_point_color(blue, (110, 100), (96, 100)))

    def test_four_colors_and_dynamic_sizes(self) -> None:
        """位置和身体尺寸由调用方传入，不假定固定大小或棋盘网格。"""
        for color, hsv in (
            ("red", (0, 220, 200)),
            ("yellow", (23, 220, 200)),
            ("blue", (100, 220, 200)),
            ("green", (60, 220, 200)),
        ):
            for size in ((28.0, 32.0), (96.0, 100.0), (140.0, 146.0)):
                with self.subTest(color=color, size=size):
                    image = cv2.cvtColor(np.full((200, 220, 3), hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
                    self.assertEqual(identify_ranbu_point_color(image, (110, 100), size), color)

    def test_blank_black_and_low_saturation_are_unknown(self) -> None:
        """黑色与白色不自动归入普通四色，低饱和特效同样拒绝。"""
        for fill in ((0, 0, 0), (255, 255, 255), (70, 70, 70), (150, 160, 150)):
            with self.subTest(fill=fill):
                image = np.full((200, 220, 3), fill, dtype=np.uint8)
                self.assertIsNone(identify_ranbu_point_color(image, (110, 100), (96, 100)))

    def test_competing_colors_and_unilateral_body_support_are_unknown(self) -> None:
        """触点落在异色边界或只有身体一侧有色时，不用多数色强行放行。"""
        red = cv2.cvtColor(np.full((200, 220, 3), (0, 220, 200), dtype=np.uint8), cv2.COLOR_HSV2BGR)
        blue = cv2.cvtColor(np.full((200, 220, 3), (100, 220, 200), dtype=np.uint8), cv2.COLOR_HSV2BGR)
        for boundary in (100, 110, 120):
            mixed = red.copy()
            mixed[:, boundary:] = blue[:, boundary:]
            self.assertIsNone(identify_ranbu_point_color(mixed, (110, 100), (96, 100)))
        red[:, :105] = 0
        self.assertIsNone(identify_ranbu_point_color(red, (110, 100), (96, 100)))

    def test_current_pixels_can_change_color_without_any_object_results(self) -> None:
        """同一点的新帧变色时直接读取新颜色，不沿用任何历史对象结论。"""
        for color, hsv in (("blue", (100, 220, 200)), ("red", (0, 220, 200))):
            image = cv2.cvtColor(np.full((200, 220, 3), hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)
            self.assertEqual(identify_ranbu_point_color(image, (110, 100), (96, 100)), color)

    def test_invalid_geometry_partial_patch_and_image_are_unknown(self) -> None:
        """不完整采样与无效尺寸均不作为稳定颜色证据。"""
        image = np.full((200, 220, 3), (0, 0, 200), dtype=np.uint8)
        for point, size in (
            ((0, 0), (96, 100)),
            ((110, 100), (0, 100)),
            ((110, 100), (float("nan"), 100)),
            ((110, 100), (300, 100)),
            ((110, 100), (1, 1)),
            ((-1, 100), (96, 100)),
        ):
            self.assertIsNone(identify_ranbu_point_color(image, point, size))
        for invalid in (
            np.zeros((10, 10), dtype=np.uint8),
            np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((10, 10, 3), dtype=np.float32),
        ):
            self.assertIsNone(identify_ranbu_point_color(invalid, (5, 5), (4, 4)))  # type: ignore[arg-type]

    def test_real_selected_points_and_rescaled_images(self) -> None:
        """五次实机图含脸部空白、动画与位置偏移；初始检测身体均为96×100。"""
        source = Path(
            os.environ.get(
                "RANBU_REGRESSION_DIR",
                str(Path(__file__).resolve().parents[1] / "debug/ranbu-regression-20261021"),
            )
        )
        if not all((source / f"ranbu-execution-{index:02}.before.png").is_file() for index in range(1, 6)):
            self.skipTest("本机未保留五次乱舞现场图片，可设置RANBU_REGRESSION_DIR指定样本目录")
        for index in range(1, 6):
            report = json.loads((source / f"ranbu-execution-{index:02}.json").read_text(encoding="utf-8"))
            image = read_image(source / f"ranbu-execution-{index:02}.before.png")
            point = tuple(report["start"])
            with self.subTest(sequence=index):
                self.assertEqual(identify_ranbu_point_color(image, point, (96, 100)), report["start_color"])
                smaller: Image = cv2.resize(image, (360, 640), interpolation=cv2.INTER_AREA)
                self.assertEqual(
                    identify_ranbu_point_color(smaller, (round(point[0] / 2), round(point[1] / 2)), (48, 50)),
                    report["start_color"],
                )


if __name__ == "__main__":
    unittest.main()
