"""验证残留检查只返回当前帧强阳性，不从阴性宣称全盘已接入。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from vision.image_io import read_image  # noqa: E402
from vision.ranbu_coverage import inspect_ranbu_coverage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class RanbuCoverageTests(unittest.TestCase):
    """实机三次漏连应触发补扫，对照两次全连只返回未知并允许松手。"""

    def test_few_remaining_targets_are_not_diluted_by_the_whole_board(self) -> None:
        """最新黄绿黄仅剩少量白边，也应触发有界补扫。"""
        source = ROOT / "debug/ranbu-red-miss-review-20261022"
        if not all((source / f"ranbu-execution-{slot:02}.held.png").is_file() for slot in (4, 6, 1)):
            self.skipTest("未保留本机少量漏连现场")
        for slot in (4, 6, 1):
            report = json.loads((source / f"ranbu-execution-{slot:02}.json").read_text(encoding="utf-8"))
            original = read_image(source / f"ranbu-execution-{slot:02}.held.png")
            for scale in (0.5, 1.0, 1.5):
                with self.subTest(slot=slot, scale=scale):
                    image = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    result = inspect_ranbu_coverage(
                        image, (0, 210 * scale, 720 * scale, 780 * scale), report["start_color"]
                    )
                    self.assertEqual(result.status, "present")

    def test_confirmed_full_green_supplement_does_not_trigger_another_sweep(self) -> None:
        """先前从11连补齐14连的现场，不能被新的局部白边规则误报。"""
        path = ROOT / "debug/ranbu-latency-review-20261022/ranbu-execution-04.pass-02.png"
        if not path.is_file():
            self.skipTest("未保留本机补齐14连的对照")
        original = read_image(path)
        for scale in (0.5, 1.0, 1.5):
            image = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self.assertEqual(
                inspect_ranbu_coverage(image, (0, 210 * scale, 720 * scale, 780 * scale), "green").status, "unknown"
            )

    def test_real_held_images_and_rescaling(self) -> None:
        """用已观察的五张按住截图及不同缩放验证，不用对象识别器。"""
        samples = (
            ("ranbu-regression-20261021", 3, "present"),
            ("ranbu-regression-20261021", 4, "present"),
            ("ranbu-regression-20261021", 5, "present"),
            ("ranbu-review-20261022", 1, "unknown"),
            ("ranbu-review-20261022", 2, "unknown"),
        )
        if not all(
            (ROOT / "debug" / folder / f"ranbu-execution-{index:02}.held.png").is_file() for folder, index, _ in samples
        ):
            self.skipTest("未保留本机五张乱舞残留回归图片")
        for folder, index, expected in samples:
            source = ROOT / "debug" / folder
            report = json.loads((source / f"ranbu-execution-{index:02}.json").read_text(encoding="utf-8"))
            original = read_image(source / f"ranbu-execution-{index:02}.held.png")
            for scale in (0.5, 1.0, 1.5):
                image = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                region = (0.0, 210.0 * scale, 720.0 * scale, 780.0 * scale)
                with self.subTest(folder=folder, index=index, scale=scale):
                    result = inspect_ranbu_coverage(image, region, report["start_color"])
                    self.assertEqual(result.status, expected)
                    self.assertLessEqual(result.sampled_pixels, 360 * 390)
                    self.assertGreater(result.cyan_pixels, 0)

    def test_no_positive_claim_from_blank_color_or_missing_held_glow(self) -> None:
        """全白、全黑与纯色均不代表残留，也不代表已经全连。"""
        for fill in ((0, 0, 0), (255, 255, 255), (200, 130, 0), (80, 190, 80)):
            image = np.full((300, 220, 3), fill, dtype=np.uint8)
            for color in ("blue", "green", "yellow"):
                self.assertEqual(inspect_ranbu_coverage(image, (0, 0, 220, 300), color).status, "unknown")

    def test_invalid_geometry_and_unverified_colors_are_unknown(self) -> None:
        """无效输入和没有实机阳性样本的红色不得猜测。"""
        image = np.zeros((300, 220, 3), dtype=np.uint8)
        for region in ((0, 0, -1, 10), (300, 300, 10, 10), (0, float("nan"), 10, 10)):
            self.assertEqual(inspect_ranbu_coverage(image, region, "blue").status, "unknown")
        for color in (None, "red", "black"):
            self.assertEqual(inspect_ranbu_coverage(image, (0, 0, 220, 300), color).status, "unknown")


if __name__ == "__main__":
    unittest.main()
