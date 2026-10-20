"""验证奥义花标三态识别、颜色变化和原图保护，不执行角色操作。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.session import BattleState  # noqa: E402
from battle.ultimate import UltimateChargeRecognizer, UltimateChargeState  # noqa: E402
from battle.ultimate_report import write_ultimate_report  # noqa: E402
from vision.image_io import read_image  # noqa: E402


class UltimateChargeTests(unittest.TestCase):
    """只验证充能观测，满充能不表示当前允许点击。"""

    def setUp(self) -> None:
        self.directory = ROOT / "resource/base/image/ultimate"
        self.recognizer = UltimateChargeRecognizer(self.directory)

    def observe(self, image: np.ndarray, *, frame_id: str):
        """花标特征单测显式提供已确认状态，正式流程由前置状态检测提供。"""
        return self.recognizer.analyze(image, frame_id=frame_id, phase=BattleState.OPERABLE)

    def canvas(self, template_name: str) -> np.ndarray:
        """把真实模板放在当前已核对布局的三个不同搜索区中。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        template = read_image(self.directory / template_name)
        height, width = template.shape[:2]
        for x, y in ((40, 1084), (260, 1124), (480, 1084)):
            image[y : y + height, x : x + width] = template
        return image

    def test_ready_flower_supports_four_hues(self) -> None:
        """同一花形改变为红黄绿蓝后仍是满充能，不按单一花色过滤。"""
        original = self.canvas("ready_flower.png")
        for hue in (0, 30, 60, 105):
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0][hsv[:, :, 1] > 60] = hue
            image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            result = self.observe(image, frame_id=f"色相{hue}")
            self.assertEqual([slot.state for slot in result.slots], [UltimateChargeState.READY] * 3)
            self.assertFalse(result.automatic_action_allowed)

    def test_charging_has_positive_text_evidence_in_four_hues(self) -> None:
        """看到あと文字才认定充能中，文字可来自不同花色。"""
        original = self.canvas("charging_text.png")
        for hue in (0, 30, 60, 105):
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            hsv[:, :, 0][hsv[:, :, 1] > 60] = hue
            result = self.observe(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), frame_id="文字")
            self.assertEqual([slot.state for slot in result.slots], [UltimateChargeState.CHARGING] * 3)

    def test_absent_or_conflicting_evidence_is_unknown(self) -> None:
        """没有标记和相互冲突的证据都不能伪报满充能或充能中。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        empty = self.observe(image, frame_id="空白")
        self.assertEqual([slot.state for slot in empty.slots], [UltimateChargeState.UNKNOWN] * 3)
        image = self.canvas("ready_flower.png")
        text = read_image(self.directory / "charging_text.png")
        height, width = text.shape[:2]
        image[1070 : 1070 + height, 20 : 20 + width] = text
        conflict = self.observe(image, frame_id="冲突")
        self.assertEqual(conflict.slots[0].state, UltimateChargeState.UNKNOWN)

    def test_new_frame_does_not_reuse_previous_result(self) -> None:
        """同一识别器只缓存模板，不复用前一帧满充能结果。"""
        first = self.observe(self.canvas("ready_flower.png"), frame_id="前一帧")
        second = self.observe(np.zeros((1280, 720, 3), dtype=np.uint8), frame_id="新帧")
        self.assertEqual(first.slots[0].state, UltimateChargeState.READY)
        self.assertEqual(second.slots[0].state, UltimateChargeState.UNKNOWN)
        self.assertEqual(second.frame_id, "新帧")

    def test_unsupported_layout_is_unknown(self) -> None:
        """未核对的尺寸和通道不能套用固定界面区域。"""
        for image in (np.zeros((640, 360, 3), np.uint8), np.zeros((1280, 720), np.uint8)):
            result = self.observe(image, frame_id="布局未核对")
            self.assertFalse(result.layout_supported)
            self.assertEqual([slot.state for slot in result.slots], [UltimateChargeState.UNKNOWN] * 3)
            skipped = self.recognizer.analyze(image, frame_id="布局与状态均未知", phase=BattleState.UNKNOWN)
            self.assertFalse(skipped.layout_supported)
            self.assertFalse(skipped.checked)

    def test_report_preserves_input_and_frame(self) -> None:
        """同帧报告保留三态与证据，诊断标记不修改原图。"""
        image = self.canvas("ready_flower.png")
        original = image.copy()
        result = self.observe(image, frame_id="报告当前帧")
        with tempfile.TemporaryDirectory() as directory:
            write_ultimate_report(image, result, Path(directory), "测试")
            report = json.loads((Path(directory) / "测试.ultimate.json").read_text(encoding="utf-8"))
        self.assertEqual(report["frame_id"], result.frame_id)
        self.assertFalse(report["automatic_action_allowed"])
        self.assertEqual([slot["state"] for slot in report["slots"]], ["ready"] * 3)
        self.assertTrue(np.array_equal(image, original))

    def test_local_reviewed_samples(self) -> None:
        """按本轮逐图核对标签回归，文件名中的旧标签不参与判断。"""
        manifest = ROOT / "training/battle-state/ultimate_samples.json"
        if not manifest.is_file():
            self.skipTest("需要本地已核对训练样本清单")
        for sample in json.loads(manifest.read_text(encoding="utf-8"))["samples"]:
            with self.subTest(source=sample["source"]):
                result = self.observe(read_image(ROOT / sample["source"]), frame_id="开发样本")
                self.assertEqual([slot.state.value for slot in result.slots], sample["expected_states"])

    def test_phase_gate_prevents_feature_checks(self) -> None:
        """未知、不可操作和终局即使有满充能花标，也不启动奥义检查。"""
        image = self.canvas("ready_flower.png")
        for phase in (BattleState.UNKNOWN, BattleState.INOPERABLE, BattleState.VICTORY, BattleState.DEFEAT):
            with patch.object(self.recognizer, "_feature") as feature:
                result = self.recognizer.analyze(image, frame_id="先判断状态", phase=phase)
                feature.assert_not_called()
            self.assertFalse(result.checked)
            self.assertEqual([slot.state for slot in result.slots], [UltimateChargeState.UNKNOWN] * 3)
        result = self.recognizer.analyze(image, frame_id="乱舞已确认", phase=BattleState.RANBU)
        self.assertTrue(result.checked)
