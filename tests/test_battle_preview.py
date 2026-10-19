"""核对预演报告的同帧关联、原图保护和逐次轻量识别。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.candidates import build_route_preview  # noqa: E402
from battle.preview import BoardPreviewer, write_preview_report  # noqa: E402
from test_battle_candidates import board, ghoul  # noqa: E402


class BattlePreviewTests(unittest.TestCase):
    """输出只作为诊断，不改变识别结果或触控许可。"""

    def test_report_preserves_input_and_records_same_frame(self) -> None:
        """图像和结构化输入保持不变，JSON保留完整识别覆盖信息。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        result = board(ghoul(1, 100), ghoul(2, 190))
        result.coverage = {"recognition_mode": "lightweight", "skipped_kinds": ["obstacle"]}
        original = asdict(result)
        preview = build_route_preview(result)
        with tempfile.TemporaryDirectory() as directory:
            write_preview_report(image, result, preview, Path(directory), "当前帧")
            saved = json.loads((Path(directory) / "当前帧.preview.json").read_text(encoding="utf-8"))
            self.assertTrue((Path(directory) / "当前帧.preview.png").is_file())
        self.assertEqual(saved["frame_id"], result.frame_id)
        self.assertEqual(saved["preview"]["frame_id"], result.frame_id)
        self.assertEqual(saved["board"]["coverage"], result.coverage)
        self.assertFalse(saved["automatic_action_allowed"])
        self.assertEqual(asdict(result), original)
        self.assertFalse(np.any(image))

    def test_mismatched_frames_are_rejected(self) -> None:
        """禁止把旧帧的候选路线画到新帧诊断图。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        result = board(ghoul(1, 100), ghoul(2, 190))
        preview = build_route_preview(result)
        preview.frame_id = "其他帧"
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            write_preview_report(image, result, preview, Path(directory), "无效")

    def test_static_recognizer_is_lazy_and_lightweight_is_passed_each_time(self) -> None:
        """缓存识别器静态模板，但每次传入新帧与当前轻量开关。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, patch("battle.preview.BoardRecognizer") as factory:
            previewer = BoardPreviewer(Path("测试模板.json"), Path(directory))
            factory.assert_not_called()
            for frame_id, lightweight in (("第一帧", False), ("第二帧", True)):
                factory.return_value.analyze.return_value = board(ghoul(1, 100), ghoul(2, 190), frame_id=frame_id)
                previewer.analyze(image, frame_id, lightweight=lightweight)
                factory.return_value.analyze.assert_called_with(image, frame_id=frame_id, lightweight=lightweight)
            factory.assert_called_once()


if __name__ == "__main__":
    unittest.main()
