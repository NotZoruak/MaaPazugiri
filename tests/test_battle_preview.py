"""核对预演报告的同帧关联、原图保护和逐次轻量识别。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.candidates import build_route_preview  # noqa: E402
from battle.preview import BoardPreviewer, write_preview_report  # noqa: E402
from test_battle_candidates import board, ghoul  # noqa: E402
from vision.chain_endpoint import ChainEndpointRecognizer  # noqa: E402
from vision.image_io import read_image  # noqa: E402
from vision.model import ChainEndpointResult, ChainMovePositions  # noqa: E402


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

    def test_move_feedback_uses_the_new_frame_without_old_object_association(self) -> None:
        """真实菱形识别确认移动终点，但不得关联规划帧的旧对象。"""
        source = ROOT / "training/chain_samples/小于7.png"
        if not source.is_file():
            self.skipTest("需要本地终点标志样本")
        image = read_image(source)
        old = replace(ghoul(1, 488), center=(488, 357), candidate_point=(488, 357))
        planned = board(old, frame_id="规划帧")
        original = asdict(planned)
        move = ChainMovePositions((100, 900), (488, 357))
        with tempfile.TemporaryDirectory() as directory, patch("battle.preview.BoardRecognizer") as factory:
            factory.return_value.endpoint_recognizer = ChainEndpointRecognizer(
                ROOT / "resource/base/image/board/chain_endpoint/profile.json"
            )
            result = BoardPreviewer(Path("测试模板.json"), Path(directory)).recognize_chain_endpoint(
                image, planned, move=move
            )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.move_outcome, "target_reached")
        self.assertNotEqual(result.frame_id, "规划帧")
        self.assertIsNone(result.object_id)
        self.assertEqual(result.coverage["search_scope"], "move_start_and_end")
        self.assertEqual(asdict(planned), original)

    def test_failure_report_preserves_last_screenshot_and_confirmation_details(self) -> None:
        """失败现场带有目标序号与确认数量，并覆盖旧报告以限制文件总量。"""
        image = np.full((1280, 720, 3), 42, dtype=np.uint8)
        planned = board(ghoul(1, 100), ghoul(2, 190))
        move = ChainMovePositions((100, 600), (190, 600))
        result = ChainEndpointResult("失败帧", "detected", move_outcome="still_at_start")
        details = {"target_index": 3, "planned_targets": 14, "confirmed_targets": 2, "attempts": 8}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            previewer = BoardPreviewer(Path("测试模板.json"), destination)
            for _ in range(3):
                previewer.save_link_failure(image, planned, move, result, details)
            saved = json.loads((destination / "latest-link-failure.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["confirmed_targets"], 2)
            self.assertEqual(saved["target_index"], 3)
            self.assertEqual(saved["recognition"]["move_outcome"], "still_at_start")
            self.assertEqual(saved["planned_frame_id"], planned.frame_id)
            self.assertTrue(np.array_equal(read_image(destination / "latest-link-failure.png"), image))
            self.assertEqual(len(list(destination.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
