"""验证乱舞读取游戏连线数量，而非用触控派发数量代替。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.ranbu_feedback import recognize_ranbu_chain_count  # noqa: E402
from vision.model import ChainEndpointMarker, ChainEndpointResult  # noqa: E402
from vision.ranbu_digits import RanbuDigits  # noqa: E402


class RanbuFeedbackTests(unittest.TestCase):
    """仅允许当前菱形上方明确的数字作为数量证据。"""

    def test_offline_count_only_ocr_reads_the_filtered_current_digits(self) -> None:
        """离线OCR只接收本帧验证后的数字图，不读取邻近身体和连线。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        marker = ChainEndpointMarker("普通", (55, 629), (20, 586, 72, 87), 0.92)
        calls = []

        def recognize(method, params, current):
            """回放真实第四轮显示的数字。"""
            calls.append((params.roi, current))
            return SimpleNamespace(all_results=[SimpleNamespace(text="4", score=0.999)])

        result = ChainEndpointResult("当前", "detected", [marker])
        filtered = np.zeros((90, 50, 3), np.uint8)
        prepared = RanbuDigits((0, 521, 158, 70), filtered, 1, (), "已核对")
        with patch("vision.ranbu_digits.prepare_ranbu_digits", return_value=prepared) as prepare:
            self.assertEqual(recognize_ranbu_chain_count(recognize, image, result), 4)
            self.assertIs(prepare.call_args.args[0], image)
            self.assertIs(prepare.call_args.args[1], marker)
        self.assertEqual(calls[0][0], (0, 0, 0, 0))
        self.assertIs(calls[0][1], filtered)

    def test_uncertain_or_conflicting_text_is_unknown(self) -> None:
        """背景数字、低置信度和多个不一致的数量不能变成进度。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        marker = ChainEndpointMarker("普通", (300, 825), (265, 782, 72, 87), 0.95)
        result = ChainEndpointResult("当前", "detected", [marker])
        prepared = RanbuDigits((0, 0, 50, 50), np.zeros((50, 50, 3), np.uint8), 1, (), "已核对")
        for texts in ((("4", 0.3),), (("4HIT", 0.99),), (("4", 0.99), ("9", 0.99)), (("141", 0.99),)):
            with self.subTest(texts=texts):
                detail = SimpleNamespace(all_results=[SimpleNamespace(text=t, score=s) for t, s in texts])
                with patch("vision.ranbu_digits.prepare_ranbu_digits", return_value=prepared):
                    self.assertIsNone(recognize_ranbu_chain_count(lambda *args, value=detail: value, image, result))
        self.assertIsNone(recognize_ranbu_chain_count(lambda *args: self.fail("没有菱形不应调用OCR"), image, None))


if __name__ == "__main__":
    unittest.main()
