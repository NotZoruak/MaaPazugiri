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

    def test_grab_frames_are_preserved_and_stale_frames_are_removed(self) -> None:
        """未扫动的接入现场也要保存，槽位复用不能残留旧的第二张接入图。"""
        before = np.zeros((32, 32, 3), np.uint8)
        first, second = np.full_like(before, 42), np.full_like(before, 84)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            previewer = BoardPreviewer(Path("离线模板"), destination)
            previewer.save_ranbu_execution_report(before, second, {}, grab_images=(first, second))
            saved = json.loads((destination / "latest-ranbu-execution.json").read_text(encoding="utf-8"))
            self.assertEqual(len(saved["grab_images"]), 2)
            self.assertTrue(np.array_equal(read_image(destination / saved["grab_images"][0]), first))
            for _ in range(6):
                previewer.save_ranbu_execution_report(before, first, {}, grab_images=(first,))
            self.assertFalse((destination / "ranbu-execution-01.grab-02.png").exists())

    def test_ranbu_preserves_each_pass_and_removes_stale_second_pass(self) -> None:
        """首遍和补扫都保留原图，槽位复用时删除上一轮多余补扫图。"""
        image = np.zeros((1280, 720, 3), np.uint8)
        first = np.full_like(image, 42)
        second = np.full_like(image, 84)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            previewer = BoardPreviewer(Path("离线模板"), destination)
            previewer.save_ranbu_execution_report(image, second, {}, pass_images=(first, second))
            saved = json.loads((destination / "latest-ranbu-execution.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["pass_images"], ["ranbu-execution-01.pass-01.png", "ranbu-execution-01.pass-02.png"])
            self.assertTrue(np.array_equal(read_image(destination / saved["pass_images"][0]), first))
            for _ in range(6):
                previewer.save_ranbu_execution_report(image, first, {}, pass_images=(first,))
            self.assertFalse((destination / "ranbu-execution-01.pass-02.png").exists())

    def test_ranbu_execution_reports_keep_six_rounds_and_remove_stale_held_images(self) -> None:
        """轮换保留六轮证据，截图失败时不能残留该槽位上一轮的松手前图片。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        details = {"dispatch_completed": True, "game_connection_verified": False, "events": []}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            previewer = BoardPreviewer(Path("离线模板"), destination)
            self.assertTrue(callable(getattr(previewer, "save_ranbu_execution_report", None)))
            for sequence in range(1, 9):
                previewer.save_ranbu_execution_report(image, image if sequence < 8 else None, details)
            latest = json.loads((destination / "latest-ranbu-execution.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["sequence"], 8)
            self.assertIsNone(latest["held_image"])
            self.assertFalse((destination / "ranbu-execution-02.held.png").exists())
            reports = [
                json.loads(path.read_text(encoding="utf-8")) for path in destination.glob("ranbu-execution-*.json")
            ]
            self.assertEqual({report["sequence"] for report in reports}, set(range(3, 9)))
            self.assertEqual(len({report["run_id"] for report in reports}), 1)
            self.assertEqual(len(list(destination.glob("*.before.png"))), 6)
            self.assertEqual(len(list(destination.glob("*.held.png"))), 5)
            self.assertEqual(details, {"dispatch_completed": True, "game_connection_verified": False, "events": []})

    def test_ranbu_diagnostic_handles_rejected_nonfinite_objects(self) -> None:
        """其他对象坐标无效时仍可保存当前有效起点，诊断不能使乱舞崩溃。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        current = board(ghoul(1, 100, 400, "blue"), ghoul(2, float("nan"), 600, "blue"))
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            previewer = BoardPreviewer(Path("离线模板"), destination)
            previewer.save_ranbu_report(image, current, (100, 400), "blue", "green", ((100, 400), (200, 600)))
            report = json.loads((destination / "latest-ranbu.json").read_text(encoding="utf-8"))
            self.assertEqual(report["color"], "blue")
            self.assertIsNone(report["board"]["objects"][1]["center"][0])
            self.assertTrue((destination / "latest-ranbu.png").is_file())

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
        """整笔未确认现场保留计划与单次检查结果，覆盖旧报告以限制文件总量。"""
        image = np.full((1280, 720, 3), 42, dtype=np.uint8)
        planned = board(ghoul(1, 100), ghoul(2, 190))
        move = ChainMovePositions((100, 600), (190, 600))
        result = ChainEndpointResult("失败帧", "detected", move_outcome="still_at_start")
        details = {
            "planned_targets": 14,
            "confirmation_mode": "after_full_gesture",
            "stop_reason": "marker_not_detected",
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            previewer = BoardPreviewer(Path("测试模板.json"), destination)
            for _ in range(3):
                previewer.save_link_failure(image, planned, move, result, details)
            saved = json.loads((destination / "latest-link-failure.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["planned_targets"], 14)
            self.assertEqual(saved["confirmation_mode"], "after_full_gesture")
            self.assertEqual(saved["recognition"]["move_outcome"], "still_at_start")
            self.assertEqual(saved["planned_frame_id"], planned.frame_id)
            self.assertTrue(np.array_equal(read_image(destination / "latest-link-failure.png"), image))
            self.assertEqual(len(list(destination.iterdir())), 2)


if __name__ == "__main__":
    unittest.main()
