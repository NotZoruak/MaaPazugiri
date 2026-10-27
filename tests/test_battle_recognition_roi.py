"""验证战斗内置识别只传当前帧的静态界面区域。"""

from __future__ import annotations

import sys
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from battle.recognition import (  # noqa: E402
    HP_REGIONS,
    RECOGNITIONS,
    ULTIMATE_CONFIRMATION_ROI,
    BattleFrameRecognizer,
    _recognize_region,
    recognize_hp_percent,
)
from battle.session import BattleState  # noqa: E402


class BattleRecognitionRoiTests(unittest.TestCase):
    """传输体积缩小时，像素、文字分流及共享参数必须保持正确。"""

    def setUp(self) -> None:
        """使用非均匀图案检查裁剪偏移，避免空图掩盖区域错误。"""
        self.image = np.random.default_rng(42).integers(0, 256, (1280, 720, 3), dtype=np.uint8)

    def assert_region(self, parameters, current, roi) -> None:
        """确认传给框架的就是对应区域，并使用区域内的局部坐标。"""
        x, y, width, height = roi
        self.assertEqual(parameters.roi, (0, 0, width, height))
        self.assertEqual(current.shape, (height, width, 3))
        self.assertTrue(current.flags.c_contiguous)
        np.testing.assert_array_equal(current, self.image[y : y + height, x : x + width])

    def test_state_checks_send_only_their_roi_and_keep_shared_parameters(self) -> None:
        """状态检查传输局部像素，共享识别参数不能被改成局部坐标。"""
        for name, (_, parameters) in RECOGNITIONS.items():
            with self.subTest(name=name):
                before = asdict(parameters)

                def recognize(method, local, current, roi=parameters.roi):
                    self.assert_region(local, current, roi)
                    return SimpleNamespace(hit=True)

                self.assertTrue(BattleFrameRecognizer(recognize)._hit(name, self.image))
                self.assertEqual(asdict(parameters), before)

    def test_animated_prompt_keeps_ocr_text_dispatch_on_its_current_region(self) -> None:
        """只缩小奥义提示 OCR 输入，点击与滑动仍由当前帧文字决定。"""
        for text, expected in (
            ("タップで位置を決めよう", BattleState.ULTIMATE_TAP_CONFIRMATION),
            ("スワイプで位置を決めよう", BattleState.ULTIMATE_SWIPE_CONFIRMATION),
            ("位置を決めよう", BattleState.UNKNOWN),
            ("", None),
        ):
            with self.subTest(text=text):

                def recognize(method, parameters, current, text=text):
                    self.assert_region(parameters, current, ULTIMATE_CONFIRMATION_ROI)
                    return SimpleNamespace(all_results=[SimpleNamespace(text=text)])

                self.assertEqual(BattleFrameRecognizer(recognize)._ultimate_confirmation(self.image), expected)

    def test_hp_crops_each_side_and_reads_the_supplied_frame(self) -> None:
        """敌我血量独立读取当前区域，不依赖返回框的全屏坐标。"""
        for side, roi in HP_REGIONS.items():
            with self.subTest(side=side):

                def recognize(method, parameters, current, roi=roi):
                    self.assert_region(parameters, current, roi)
                    return SimpleNamespace(all_results=[SimpleNamespace(text="25/100")])

                self.assertEqual(recognize_hp_percent(recognize, self.image, side), 25.0)


class NativeRoiEquivalenceTests(unittest.TestCase):
    """在同一套 Maa 资源上比较全图 ROI 与先裁剪的小图结果。"""

    @classmethod
    def setUpClass(cls) -> None:
        """创建纯图片识别上下文，不连接设备或实现输入接口。"""
        import maa
        from maa.controller import CustomController
        from maa.library import Library
        from maa.resource import Resource
        from maa.tasker import Tasker
        from maa.toolkit import Toolkit

        cls.root = Path(__file__).resolve().parents[1]
        directory = cls.root / "training/board-samples/raw"
        cls.sources = [
            source for source in sorted(directory.glob("[0-9][0-9][0-9]_*.png")) if int(source.name[:3]) <= 10
        ]
        if not cls.sources:
            raise unittest.SkipTest("需要本机保留的战斗实机截图")
        negative = cls.root / "training/battle-state/negative_enemy_action.png"
        if negative.is_file():
            cls.sources.append(negative)
        Library.open(Path(maa.__file__).resolve().parent / "bin", agent_server=False)
        Toolkit.init_option(cls.root / "debug/battle-roi-tests")
        cls.resource = Resource()
        if not cls.resource.use_cpu() or not cls.resource.post_bundle(cls.root / "resource/base").wait().succeeded:
            raise RuntimeError("无法加载离线 Maa 识别资源")

        class ImageOnlyController(CustomController):
            """仅提供离线标识，所有输入操作均未实现。"""

            def request_uuid(self) -> str:
                """返回固定的纯识别标识。"""
                return "battle-roi-equivalence"

        cls.controller = ImageOnlyController()
        cls.tasker = Tasker()
        if not cls.tasker.bind(cls.resource, cls.controller):
            raise RuntimeError("无法绑定离线 Maa 识别器")

    def recognize(self, method, parameters, image):
        """直接执行本地原生识别，不请求控制器截图。"""
        result = self.tasker.post_recognition(method, parameters, image).wait().get()
        self.assertIsNotNone(result)
        self.assertEqual(len(result.nodes), 1)
        self.assertIsNotNone(result.nodes[0].recognition)
        return result.nodes[0].recognition

    def test_native_hit_text_score_and_boxes_match_on_reviewed_screenshots(self) -> None:
        """命中、文字、计数与分数一致，局部框恢复偏移后也与全图相同。"""
        from maa.pipeline import JOCR, JRecognitionType
        from vision.image_io import read_image

        specifications = [
            *RECOGNITIONS.values(),
            (JRecognitionType.OCR, JOCR(roi=ULTIMATE_CONFIRMATION_ROI, order_by="Horizontal")),
            *((JRecognitionType.OCR, JOCR(roi=roi, order_by="Horizontal")) for roi in HP_REGIONS.values()),
        ]
        for source in self.sources:
            image = read_image(source)
            for method, parameters in specifications:
                with self.subTest(source=source.name, method=method, roi=parameters.roi):
                    full = self.recognize(method, parameters, image)
                    local = _recognize_region(self.recognize, method, parameters, image)
                    self.assertEqual(local.hit, full.hit)
                    self.assertEqual(len(local.all_results), len(full.all_results))
                    for actual, expected in zip(local.all_results, full.all_results, strict=True):
                        x, y, width, height = actual.box
                        self.assertEqual(
                            (x + parameters.roi[0], y + parameters.roi[1], width, height), tuple(expected.box)
                        )
                        self.assertEqual(getattr(actual, "text", None), getattr(expected, "text", None))
                        self.assertEqual(getattr(actual, "count", None), getattr(expected, "count", None))
                        self.assertAlmostEqual(getattr(actual, "score", 0.0), getattr(expected, "score", 0.0), places=5)


if __name__ == "__main__":
    unittest.main()
