"""用现场图核对起点是否出现新增选中描边，不把预选颜色当作游戏结果。"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from vision.image_io import read_image  # noqa: E402
from vision.ranbu_grab import inspect_ranbu_grab  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "training/ranbu-touch-samples"


class RanbuGrabSafetyTests(unittest.TestCase):
    """无效图像、单像素采样和纯青色特效不能作为起点接入证据。"""

    def test_invalid_geometry_and_plain_glow_remain_unknown(self) -> None:
        """没有身体颜色证据时，即使新增青色也不放行。"""
        before = np.zeros((200, 220, 3), np.uint8)
        after = np.full_like(before, (255, 255, 0))
        for size in ((1, 1), (96, 100), (1000, 100), (float("nan"), 100)):
            self.assertEqual(inspect_ranbu_grab(before, after, (110, 100), size, "blue").status, "unknown")
        self.assertEqual(inspect_ranbu_grab(before, after[:100], (110, 100), (96, 100), "blue").status, "unknown")


class RanbuGrabPixelTests(unittest.TestCase):
    """已选黄绿蓝与错色现场必须分开，原图不变不能作为接入证据。"""

    def test_real_correct_and_wrong_color_pairs(self) -> None:
        """同点颜色与新增描边同时成立才允许继续，错色两例必须拒绝。"""
        manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            with self.subTest(sequence=sample["source_sequence"]):
                before = read_image(ROOT / f"{sample['stem']}.before.png")
                current = read_image(ROOT / f"{sample['stem']}.held.png")
                result = inspect_ranbu_grab(
                    before, current, tuple(sample["point"]), tuple(sample["body_size"]), sample["color"]
                )
                self.assertEqual(result.status, sample["expected"])
                json.dumps(asdict(result))
                unchanged = inspect_ranbu_grab(
                    before, before, tuple(sample["point"]), tuple(sample["body_size"]), sample["color"]
                )
                self.assertEqual(unchanged.status, "unknown")


if __name__ == "__main__":
    unittest.main()
