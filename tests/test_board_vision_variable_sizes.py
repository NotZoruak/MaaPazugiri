"""核对活动尺寸搜索、类别排除和同一部位跨尺度去重。"""

from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from agent.vision.config import Profile
from agent.vision.image_io import read_image, write_image
from agent.vision.model import Hit
from agent.vision.recognizer import BoardRecognizer

ROOT = Path(__file__).resolve().parents[1]
NORMAL = ROOT / "resource/base/image/board/development/profile.json"


class VariableGhoulSizeTests(unittest.TestCase):
    """通过实际图像匹配核对配置边界，不连接设备。"""

    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="活动尺寸-")
        self.addCleanup(self.directory.cleanup)
        self.base = Path(self.directory.name)

    def recognizer(self, **options: object) -> BoardRecognizer:
        """在临时目录构建含眼睛和嘴巴的可控模板。"""
        rng = np.random.default_rng(41)
        patterns = {part: rng.integers(25, 230, (20, 24, 3), dtype=np.uint8) for part in ("eye", "mouth")}
        templates = []
        for part, pattern in patterns.items():
            write_image(self.base / f"{part}.png", pattern)
            write_image(self.base / f"{part}.mask.png", np.full((20, 24), 255, dtype=np.uint8))
            templates.append(
                {
                    "id": part,
                    "kind": "ghoul",
                    "color": "blue",
                    "part": part,
                    "image": f"{part}.png",
                    "mask": f"{part}.mask.png",
                    "center_offset": [12, 30 if part == "eye" else -10],
                    "body_size": [80, 80],
                    "threshold": 0.99,
                    "min_color_fraction": 0,
                }
            )
        data = {
            "reference_size": [420, 240],
            "board_roi": [0, 0, 420, 240],
            "scales": [1, 2],
            "templates": templates,
            **options,
        }
        self.path = self.base / "profile.json"
        self.path.write_text(json.dumps(data), encoding="utf-8")
        return BoardRecognizer(self.path, full_search=True, workers=1)

    def image(self, placements: list[tuple[int, int, int]]) -> np.ndarray:
        """按已知中心放入不同尺寸的同色五官。"""
        image = np.zeros((240, 420, 3), dtype=np.uint8)
        for cx, cy, scale in placements:
            for part, dy in (("eye", 30), ("mouth", -10)):
                pattern = read_image(self.base / f"{part}.png")
                pattern = cv2.resize(pattern, (24 * scale, 20 * scale), interpolation=cv2.INTER_LINEAR)
                x, y = cx - 12 * scale, cy - dy * scale
                image[y : y + 20 * scale, x : x + 24 * scale] = pattern
        return image

    def test_adjacent_small_and_large_ghouls_keep_their_own_sizes(self) -> None:
        """把全棋盘套用同一尺寸或合并相邻同色对象时应失败。"""
        recognizer = self.recognizer(variable_ghoul_sizes=True)
        result = recognizer.analyze(self.image([(90, 130, 1), (210, 130, 2)]), "混合尺寸")
        objects = sorted(result.objects, key=lambda obj: obj.center[0])
        self.assertEqual([obj.center for obj in objects], [(90, 130), (210, 130)])
        self.assertEqual([obj.size["width"] for obj in objects], [80, 160])
        moved = recognizer.analyze(self.image([(140, 130, 2), (310, 130, 1)]), "下一帧")
        self.assertEqual(sorted(obj.center for obj in moved.objects), [(140, 130), (310, 130)])

    def test_per_template_scales_do_not_enlarge_bombs(self) -> None:
        """扩大饿鬼搜索范围时，不应把炸弹也当成相同尺寸分布。"""
        self.recognizer()
        data = json.loads(self.path.read_text(encoding="utf-8"))
        bomb = copy.deepcopy(data["templates"][0])
        bomb.update(id="flame", kind="bomb", bomb_level=1, scales=[1])
        bomb.pop("color")
        data["templates"] = [bomb]
        self.path.write_text(json.dumps(data), encoding="utf-8")
        recognizer = BoardRecognizer(self.path, full_search=True, workers=1)
        result = recognizer.analyze(self.image([(90, 130, 1), (250, 130, 2)]), "炸弹尺度")
        self.assertEqual([obj.center for obj in result.objects], [(90, 130)])

    def test_shared_mouth_is_merged_without_swallowing_an_adjacent_small_ghoul(self) -> None:
        """同一嘴巴的错误小尺度命中归并，相邻真实小饿鬼仍独立。"""
        recognizer = self.recognizer(variable_ghoul_sizes=True)
        hits = [
            Hit("eye", "ghoul", "blue", None, "eye", (200, 150), (160, 160), (200, 110), 0.99),
            Hit("mouth", "ghoul", "blue", None, "mouth", (200, 150), (160, 160), (200, 170), 0.95),
            Hit("mouth", "ghoul", "blue", None, "mouth", (225, 155), (80, 80), (201, 171), 0.90),
            Hit("eye", "ghoul", "blue", None, "eye", (260, 150), (80, 80), (260, 130), 0.98),
        ]
        groups = recognizer._group(hits)
        objects = [recognizer._object(group, index, 1, 1, (420, 240), []) for index, group in enumerate(groups)]
        self.assertEqual([obj.center for obj in objects], [(200, 150), (260, 150)])
        self.assertEqual([obj.size["width"] for obj in objects], [160, 80])

    def test_small_unmatched_color_region_is_still_reported(self) -> None:
        """残留区域门槛要考虑最小搜索尺度，不能漏掉小对象的不确定提示。"""
        recognizer = self.recognizer(variable_ghoul_sizes=True, scales=[0.5, 1, 2])
        image = np.zeros((240, 420, 3), dtype=np.uint8)
        image[100:120, 100:150] = (200, 100, 0)
        result = recognizer.analyze(image, "未匹配小色块")
        self.assertFalse(result.objects)
        self.assertTrue(any(issue["code"] == "unresolved_color_region" for issue in result.issues))

    def test_excluded_kinds_are_explicit_and_not_reported_missing(self) -> None:
        """主动排除的类别不能重新搜索或误报成缺失模板。"""
        data = json.loads(NORMAL.read_text(encoding="utf-8"))
        for spec in data["templates"]:
            spec["image"] = str(NORMAL.parent / spec["image"])
            spec["mask"] = str(NORMAL.parent / spec["mask"])
        data.update(excluded_kinds=["obstacle", "special_ghoul"], exclude_enhanced=True)
        self.path = self.base / "profile.json"
        self.path.write_text(json.dumps(data), encoding="utf-8")
        recognizer = BoardRecognizer(self.path, workers=1)
        result = recognizer.analyze(np.zeros((1280, 720, 3), dtype=np.uint8), "空画面")
        self.assertEqual(result.coverage["skipped_kinds"], ["obstacle", "special_ghoul"])
        self.assertEqual(result.coverage["missing_templates"], [])
        self.assertEqual(result.coverage["skipped_variants"], ["enhanced_ghoul"])
        self.assertTrue(all(spec["kind"] in ("ghoul", "bomb") for spec, _, _ in recognizer.scaled_patterns))
        self.assertFalse(any(spec.get("enhanced") for spec, _, _ in recognizer.scaled_patterns))

    def test_invalid_event_parameters_are_rejected(self) -> None:
        """错误类别、非布尔开关与非法模板尺度不能静默生效。"""
        data = json.loads(NORMAL.read_text(encoding="utf-8"))
        for bad in (
            {"excluded_kinds": ["unknown"]},
            {"excluded_kinds": "obstacle"},
            {"exclude_enhanced": "false"},
            {"variable_ghoul_sizes": 1},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Profile.from_dict({**data, **bad})
        data["templates"][0]["scales"] = [True, 0]
        with self.assertRaises(ValueError):
            Profile.from_dict(data)

    @unittest.skipUnless(
        (ROOT / "training/board-samples/events/big-ghouls/raw/001_mixed_sizes.png").is_file(),
        "需要本地活动实机截图，未分发原图时跳过",
    )
    def test_large_yellow_ghoul_is_not_counted_again_as_a_small_mouth(self) -> None:
        """大黄色饿鬼嘴巴内的小尺度命中不能成为第二只饿鬼。"""
        data = json.loads(NORMAL.read_text(encoding="utf-8"))
        data.update(
            scales=[round(0.8 + index * 0.1, 2) for index in range(15)],
            board_roi=[0, 280, 720, 710],
            standard_object_roi=[0, 280, 720, 710],
            variable_ghoul_sizes=True,
            excluded_kinds=["obstacle", "special_ghoul"],
            exclude_enhanced=True,
        )
        data["templates"] = [
            spec for spec in data["templates"] if spec["kind"] == "ghoul" and not spec.get("enhanced", False)
        ]
        for spec in data["templates"]:
            spec["image"] = str(NORMAL.parent / spec["image"])
            spec["mask"] = str(NORMAL.parent / spec["mask"])
        path = self.base / "profile.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        recognizer = BoardRecognizer(path, workers=2)
        image = read_image(ROOT / "training/board-samples/events/big-ghouls/raw/001_mixed_sizes.png")
        result = recognizer.analyze(image, "大黄色饿鬼")
        self.assertEqual(len(result.objects), 29)
        big_yellow = [
            obj for obj in result.objects if obj.color == "yellow" and obj.center[0] > 600 and 700 < obj.center[1] < 810
        ]
        self.assertEqual(len(big_yellow), 1)
        self.assertGreater(big_yellow[0].size["width"], 140)

    @unittest.skipUnless(
        (ROOT / "training/board-vision/events/big-ghouls/samples.json").is_file(),
        "需要本地活动开发参考，未分发原图时跳过",
    )
    def test_event_profile_matches_both_development_scenes(self) -> None:
        """逐个关联近似人工参考，防止数量相同但漏检与误检相互抵消。"""
        profile = ROOT / "resource/base/image/board/big_ghouls/profile.json"
        recognizer = BoardRecognizer(profile, workers=2)
        references = json.loads(
            (ROOT / "training/board-vision/events/big-ghouls/samples.json").read_text(encoding="utf-8")
        )
        for sample in references["samples"]:
            with self.subTest(sample=sample["file"]):
                image = read_image(ROOT / "training/board-samples/events/big-ghouls/raw" / sample["file"])
                result = recognizer.analyze(image, sample["file"])
                remaining = [obj for obj in result.objects if obj.kind == "ghoul"]
                self.assertEqual(len(remaining), 29)
                for reference in sample["ghouls"]:
                    candidates = [obj for obj in remaining if obj.color == reference["color"]]
                    self.assertTrue(candidates, reference)
                    matched = min(candidates, key=lambda obj: math.dist(obj.center, reference["center"]))
                    self.assertLessEqual(
                        math.dist(matched.center, reference["center"]), references["center_tolerance_pixels"], reference
                    )
                    remaining.remove(matched)
                self.assertFalse(remaining)
                self.assertEqual(sum(obj.kind == "bomb" for obj in result.objects), sample["bomb_count"])
                self.assertFalse(any(obj.enhanced or obj.kind not in ("ghoul", "bomb") for obj in result.objects))
                self.assertFalse(result.automatic_action_allowed)
                self.assertFalse(result.coverage["missing_templates"])
                light = recognizer.analyze(image, sample["file"], lightweight=True)
                self.assertEqual(result.objects, light.objects)


if __name__ == "__main__":
    unittest.main()
