"""验证奥义确认提示不依赖被特效遮挡的设置齿轮。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from maa.pipeline import JRecognitionType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.recognition import BattleFrameRecognizer  # noqa: E402
from battle.session import BattleState  # noqa: E402


class BattleConfirmationTests(unittest.TestCase):
    """验证确认提示文字先于齿轮门槛完成识别。"""

    def test_swipe_prompt_is_detected_when_ultimate_overlay_hides_gear(self) -> None:
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        ocr_calls = 0

        def recognize(method, parameters, _image):
            nonlocal ocr_calls
            if method == JRecognitionType.OCR:
                ocr_calls += 1
                return SimpleNamespace(
                    hit=True,
                    all_results=[SimpleNamespace(text="スワイプで位置を決めよう")],
                )
            return SimpleNamespace(
                hit=False,
                all_results=[],
            )

        state = BattleFrameRecognizer(recognize).analyze(image)

        self.assertEqual(state, BattleState.ULTIMATE_SWIPE_CONFIRMATION)
        self.assertEqual(ocr_calls, 1)


if __name__ == "__main__":
    unittest.main()
