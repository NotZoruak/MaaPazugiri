"""核对离线视觉原型的数据正确性与原图保护。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "board_vision.py"


class BoardVisionTests(unittest.TestCase):
    """通过真实命令行核对可观察输出，不连接模拟器。"""

    def setUp(self) -> None:
        self.workspace = tempfile.TemporaryDirectory(prefix="棋盘原型-")
        self.addCleanup(self.workspace.cleanup)
        self.base = Path(self.workspace.name)
        self.sources = self.base / "原始截图"
        self.sources.mkdir()

    def run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CLI), *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )

    def write_image(self, path: Path, image: np.ndarray) -> None:
        success, encoded = cv2.imencode(".png", image)
        self.assertTrue(success)
        path.write_bytes(encoded.tobytes())

    def prepare(self, image: np.ndarray, templates: list[dict]) -> Path:
        source = self.sources / "模板来源.png"
        self.write_image(source, image)
        recipe = {
            "name": "测试配置",
            "reference_size": [200, 100],
            "board_roi": [20, 10, 160, 80],
            "scales": [1.0],
            "sample_role": "development",
            "templates": templates,
        }
        catalog = self.sources / "裁切清单.json"
        catalog.write_text(json.dumps(recipe, ensure_ascii=False), encoding="utf-8")
        result = self.run_cli("prepare", "--catalog", str(catalog), "--output", str(self.base / "模板"))
        self.assertEqual(result.returncode, 0, result.stderr)
        return self.base / "模板" / "profile.json"

    def analyze(self, profile: Path, image: np.ndarray, output: Path | None = None) -> dict:
        source = self.sources / "分析输入.png"
        self.write_image(source, image)
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        destination = output or self.base / "识别结果"
        result = self.run_cli(
            "analyze", "--profile", str(profile), "--input", str(source), "--output", str(destination)
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)
        self.assertTrue((destination / "分析输入.annotated.png").is_file())
        return json.loads((destination / "分析输入.json").read_text(encoding="utf-8"))

    def test_mask_ignores_character_changes_inside_bomb(self) -> None:
        """外焰未变时，改变掩码排除的角色内容不应导致漏检。"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        pattern = np.random.default_rng(4).integers(30, 230, (30, 30, 3), dtype=np.uint8)
        image[30:60, 50:80] = pattern
        template = {
            "id": "bomb_flame",
            "kind": "bomb",
            "bomb_level": 1,
            "source": "模板来源.png",
            "crop": [50, 30, 30, 30],
            "ignore_rects": [[6, 6, 18, 18]],
            "center_offset": [15, 15],
            "body_size": [30, 30],
            "part": "flame",
            "threshold": 0.99,
        }
        profile = self.prepare(image, [template])
        image[36:54, 56:74] = [240, 10, 180]
        report = self.analyze(profile, image)
        bombs = [item for item in report["objects"] if item["kind"] == "bomb"]
        self.assertEqual(len(bombs), 1)
        self.assertEqual(bombs[0]["center"], [65.0, 45.0])
        self.assertEqual(bombs[0]["bomb_level"], 1)
        self.assertEqual(report["scene_state"], "unverified")

    def test_scaled_coordinates_preserve_adjacent_same_color_objects(self) -> None:
        """区域偏移和二倍输入不能导致坐标偏移或吞掉相邻对象。"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        pattern = np.random.default_rng(8).integers(20, 230, (12, 16, 3), dtype=np.uint8)
        image[44:56, 62:78] = pattern
        template = {
            "id": "blue_eye",
            "kind": "ghoul",
            "color": "blue",
            "source": "模板来源.png",
            "crop": [62, 44, 16, 12],
            "center_offset": [8, 6],
            "body_size": [32, 32],
            "part": "eye",
            "threshold": 0.99,
            "min_color_fraction": 0.0,
        }
        profile = self.prepare(image, [template])
        image[44:56, 94:110] = pattern
        enlarged = cv2.resize(image, (400, 200), interpolation=cv2.INTER_NEAREST)
        report = self.analyze(profile, enlarged)
        centers = sorted(item["center"] for item in report["objects"] if item["kind"] == "ghoul")
        self.assertEqual(centers, [[140.0, 100.0], [204.0, 100.0]])

    def test_ambiguous_bomb_level_stays_unknown(self) -> None:
        """两个等级提供同样证据时不能任意选择其中一个。"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[30:50, 50:70] = np.random.default_rng(9).integers(20, 230, (20, 20, 3), dtype=np.uint8)
        templates = [
            {
                "id": f"bomb_{level}",
                "kind": "bomb",
                "bomb_level": level,
                "source": "模板来源.png",
                "crop": [50, 30, 20, 20],
                "center_offset": [10, 10],
                "body_size": [24, 24],
                "part": "flame",
                "threshold": 0.99,
            }
            for level in (1, 2)
        ]
        report = self.analyze(self.prepare(image, templates), image)
        self.assertEqual(len(report["objects"]), 1)
        self.assertIsNone(report["objects"][0]["bomb_level"])
        self.assertIn("bomb_3", report["coverage"]["missing_templates"])
        self.assertTrue(any(issue["code"] == "ambiguous_bomb_level" for issue in report["issues"]))

    def test_incompatible_layout_returns_unknown_instead_of_detections(self) -> None:
        """长宽比变化未经确认时，不能套用开发配置给出有效对象。"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        profile = self.prepare(image, [])
        report = self.analyze(profile, np.zeros((150, 200, 3), dtype=np.uint8))
        self.assertEqual(report["scene_state"], "unknown_ui")
        self.assertEqual(report["objects"], [])
        self.assertTrue(any(issue["code"] == "unsupported_layout" for issue in report["issues"]))

    def test_plain_background_does_not_match_textured_flame(self) -> None:
        """高亮平坦背景不能因未去均值的相关系数而被识别为炸弹。"""
        source = np.full((100, 200, 3), 220, dtype=np.uint8)
        source[30:60, 50:80] = np.random.default_rng(13).integers(180, 241, (30, 30, 3), dtype=np.uint8)
        template = {
            "id": "textured_flame",
            "kind": "bomb",
            "bomb_level": 1,
            "source": "模板来源.png",
            "crop": [50, 30, 30, 30],
            "ignore_rects": [[6, 6, 18, 18]],
            "center_offset": [15, 15],
            "body_size": [30, 30],
            "part": "flame",
            "threshold": 0.95,
        }
        profile = self.prepare(source, [template])
        report = self.analyze(profile, np.full((100, 200, 3), 220, dtype=np.uint8))
        self.assertEqual(report["objects"], [])

    def test_analysis_refuses_output_inside_original_image_directory(self) -> None:
        """输出目录放进原始截图目录时，命令应在写入前拒绝。"""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        profile = self.prepare(image, [])
        source = self.sources / "原始输入.png"
        self.write_image(source, image)
        result = self.run_cli(
            "analyze",
            "--profile",
            str(profile),
            "--input",
            str(source),
            "--output",
            str(self.sources / "错误输出"),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.sources / "错误输出").exists())


if __name__ == "__main__":
    unittest.main()
