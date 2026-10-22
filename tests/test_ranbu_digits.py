"""在七张真实按住截图上验证数字过滤和Maa原生OCR。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from vision.image_io import read_image  # noqa: E402
from vision.model import ChainEndpointMarker  # noqa: E402
from vision.ranbu_digits import parse_ranbu_digit_text, prepare_ranbu_digits  # noqa: E402

CASES = (
    ("ranbu-regression-20261021", 1, (97, 818, 72, 87), 12),
    ("ranbu-regression-20261021", 2, (271, 782, 72, 87), 13),
    ("ranbu-regression-20261021", 3, (635, 669, 72, 87), 7),
    ("ranbu-regression-20261021", 4, (20, 586, 72, 87), 4),
    ("ranbu-regression-20261021", 5, (550, 644, 72, 87), 4),
    ("ranbu-review-20261022", 1, (26, 504, 72, 87), 12),
    ("ranbu-review-20261022", 2, (137, 841, 72, 87), 14),
)


class RanbuDigitTests(unittest.TestCase):
    """图片部件与OCR必须相互约束，不能凭较低分数接受额外位数。"""

    def prepared_cases(self):
        """只使用固定回归截图，本测试控制器不会连接或操作游戏。"""
        for directory, index, box, expected in CASES:
            path = ROOT / "debug" / directory / f"ranbu-execution-{index:02d}.held.png"
            if not path.is_file():
                self.skipTest("需要七张本地乱舞诊断截图")
            x, y, width, height = box
            marker = ChainEndpointMarker("普通", (x + width * 0.5, y + height * 0.5), box, 0.9)
            yield prepare_ranbu_digits(read_image(path), marker), expected

    def test_real_components_require_matching_digit_count(self) -> None:
        """七张实机图分别有2、2、1、1、1、2、2个独立数字。"""
        for prepared, expected in self.prepared_cases():
            with self.subTest(expected=expected):
                self.assertIsNotNone(prepared.filtered, prepared.reason)
                self.assertEqual(prepared.digit_count, len(str(expected)))
                self.assertEqual(parse_ranbu_digit_text(prepared, str(expected)), expected)
                self.assertIsNone(parse_ranbu_digit_text(prepared, str(expected) + "1"))
                self.assertIsNone(parse_ranbu_digit_text(prepared, "HIT"))
        prepared, expected = list(self.prepared_cases())[-1]
        self.assertEqual(expected, 14)
        self.assertIsNone(parse_ranbu_digit_text(prepared, "141"))
        self.assertEqual(parse_ranbu_digit_text(prepared, "１４"), 14)

    def test_missing_or_invalid_pixels_are_unknown(self) -> None:
        """没有双重颜色证据、框越界或几何无效时，不造数字图。"""
        marker = ChainEndpointMarker("普通", (180, 850), (144, 806, 72, 87), 0.9)
        for image in (np.zeros((1280, 720, 3), np.uint8), np.full((1280, 720, 3), 255, np.uint8)):
            prepared = prepare_ranbu_digits(image, marker)
            self.assertIsNone(prepared.filtered)
            self.assertIsNone(prepared.digit_count)
        marker.box = (144, float("nan"), 72, 87)
        self.assertIsNone(prepare_ranbu_digits(np.zeros((1280, 720, 3), np.uint8), marker).filtered)

    def test_native_ocr_recognizes_all_seven_filtered_real_images(self) -> None:
        """原生识别仅接收过滤图，回放结果必须是12、13、7、4、4、12、14。"""
        import maa
        from maa.controller import CustomController
        from maa.custom_action import CustomAction
        from maa.library import Library
        from maa.pipeline import JOCR, JRecognitionType
        from maa.resource import Resource
        from maa.tasker import Tasker
        from maa.toolkit import Toolkit

        cases = list(self.prepared_cases())
        results = []

        class Replay(CustomController):
            """只返回离线图片，不实现任何设备输入。"""

            def connect(self):
                """建立本地图片回放连接。"""
                return True

            def request_uuid(self):
                """返回独立回放标识。"""
                return "ranbu-digit-pixels-only"

            def screencap(self):
                """提供图片控制器所需的离线占位帧。"""
                return np.zeros((1280, 720, 3), np.uint8)

        class Probe(CustomAction):
            """在真实Maa上下文中逐张识别过滤图。"""

            def run(self, context, argv):
                """保存原生识别原文和分数，供外层逐项断言。"""
                for prepared, _ in cases:
                    detail = context.run_recognition_direct(JRecognitionType.OCR, JOCR(), prepared.filtered)
                    results.append([(item.text, item.score) for item in detail.all_results] if detail else [])
                return CustomAction.RunResult(success=True)

        Library.open(Path(maa.__file__).parent / "bin", agent_server=False)
        Toolkit.init_option(ROOT / "debug/ranbu-digit-native-test")
        resource, controller = Resource(), Replay()
        self.assertTrue(resource.post_bundle(ROOT / "resource/base").wait().succeeded)
        self.assertTrue(controller.post_connection().wait().succeeded)
        tasker = Tasker()
        self.assertTrue(tasker.bind(resource, controller))
        self.assertTrue(resource.register_custom_action("RanbuDigitProbe", Probe()))
        self.assertTrue(
            tasker.post_task(
                "RanbuDigitProbe", {"RanbuDigitProbe": {"action": "Custom", "custom_action": "RanbuDigitProbe"}}
            )
            .wait()
            .succeeded
        )
        self.assertEqual(len(results), len(CASES))
        for (prepared, expected), recognized in zip(cases, results, strict=True):
            with self.subTest(expected=expected, recognized=recognized):
                self.assertEqual(len(recognized), 1)
                text, score = recognized[0]
                self.assertGreaterEqual(score, 0.9)
                self.assertEqual(parse_ranbu_digit_text(prepared, text), expected)


if __name__ == "__main__":
    unittest.main()
