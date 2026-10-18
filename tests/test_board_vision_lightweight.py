"""核对轻量模式的保留对象、切换行为及固定区域边界。"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from agent.vision.config import Profile
from agent.vision.image_io import read_image
from agent.vision.recognizer import BoardRecognizer

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "resource/base/image/board/development/profile.json"
SAMPLE = ROOT / "training/board-samples/raw/030_enhanced_ghouls_greenx2_bluex12_bronze_bell.png"


class LightweightRecognitionTests(unittest.TestCase):
    """用含强化饿鬼和钟的开发图核对模式切换不会污染缓存。"""

    @unittest.skipUnless(SAMPLE.is_file(), "需要本地开发截图，未分发原图时跳过")
    def test_modes_preserve_retained_objects_and_reset_on_same_instance(self) -> None:
        """轻量模式保留完整证据，恢复完整模式后应再次出现钟。"""
        recognizer = BoardRecognizer(PROFILE, workers=2)
        image = read_image(SAMPLE)
        full = recognizer.analyze(image, "同一帧").to_dict()
        self.assertTrue(any(obj["kind"] == "obstacle" and obj["subtype"] == "bronze_bell" for obj in full["objects"]))
        light = recognizer.analyze(image, "同一帧", lightweight=True).to_dict()
        restored = recognizer.analyze(image, "同一帧").to_dict()
        expected = [
            {key: value for key, value in obj.items() if key != "id"}
            for obj in full["objects"]
            if obj["kind"] != "obstacle"
        ]
        actual = [{key: value for key, value in obj.items() if key != "id"} for obj in light["objects"]]
        self.assertEqual(actual, expected)
        self.assertTrue(any(obj["enhanced"] for obj in light["objects"]))
        self.assertTrue(any(obj["kind"] == "bomb" for obj in light["objects"]))
        self.assertEqual(full["objects"], restored["objects"])
        self.assertEqual(light["coverage"]["skipped_kinds"], ["obstacle"])
        self.assertFalse(light["automatic_action_allowed"])

    def test_fixed_search_area_is_only_allowed_for_bronze_bell(self) -> None:
        """钟的固定区域不能扩大为任意障碍物的固定位置假设。"""
        profile = json.loads(PROFILE.read_text(encoding="utf-8"))
        bell = next(item for item in profile["templates"] if item.get("subtype") == "bronze_bell")
        bell["subtype"] = "rock"
        with self.assertRaises(ValueError):
            Profile.from_dict(profile)


if __name__ == "__main__":
    unittest.main()
