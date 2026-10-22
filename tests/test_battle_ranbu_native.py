"""用本地截图和原生 Maa OCR 回归乱舞反馈，不连接或操作游戏。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import maa

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.preview import BoardPreviewer  # noqa: E402
from battle.ranbu_feedback import recognize_ranbu_chain_count  # noqa: E402
from battle.recognition import BattleFrameRecognizer  # noqa: E402
from battle.session import BattleState  # noqa: E402
from maa.controller import CustomController  # noqa: E402
from maa.custom_action import CustomAction  # noqa: E402
from maa.library import Library  # noqa: E402
from maa.resource import Resource  # noqa: E402
from maa.tasker import Tasker  # noqa: E402
from maa.toolkit import Toolkit  # noqa: E402
from vision.image_io import Image, read_image  # noqa: E402
from vision.model import BoardResult  # noqa: E402


class ScreenshotController(CustomController):
    """只回放保留图片，不实现任何触控接口。"""

    def __init__(self, image: Image) -> None:
        self.image = image
        super().__init__()

    def connect(self) -> bool:
        """初始化本地回放。"""
        return True

    def request_uuid(self) -> str:
        """标明测试控制器身份。"""
        return "ranbu-feedback-offline"

    def screencap(self) -> Image:
        """返回显式传入的测试图。"""
        return self.image.copy()


class RanbuNativeTests(unittest.TestCase):
    """现场五轮数值与末尾状态必须由实际识别接口得到。"""

    def test_five_live_rounds_read_counts_and_keep_the_point_nine_state(self) -> None:
        """五轮均识别乱舞，连线数字分别为12、13、7、4、4。"""
        directory = ROOT / "debug/ranbu-regression-20261021"
        paths = [directory / f"ranbu-execution-{n:02d}.held.png" for n in range(1, 6)]
        if not all(path.is_file() for path in paths):
            self.skipTest("需要本机已保留的五轮现场截图")
        Library.open(Path(maa.__file__).resolve().parent / "bin", agent_server=False)
        Toolkit.init_option(ROOT / "debug/ranbu-native-tests")
        resource = Resource()
        self.assertTrue(resource.post_bundle(ROOT / "resource/base").wait().succeeded)
        controller = ScreenshotController(read_image(paths[0]))
        self.assertTrue(controller.post_connection().wait().succeeded)
        tasker = Tasker()
        self.assertTrue(tasker.bind(resource, controller))
        self.addCleanup(lambda: tasker.post_stop().wait())
        found: list[tuple[BattleState, int | None]] = []
        previewer = BoardPreviewer(ROOT / "resource/base/image/board/development/profile.json", directory)

        class Probe(CustomAction):
            """只读取实机截图中的状态和数字，不派发设备输入。"""

            def run(self, context, argv):
                """在真实原生上下文运行相同的反馈函数。"""
                recognize = context.run_recognition_direct
                frame = BattleFrameRecognizer(recognize)
                for index, path in enumerate(paths):
                    image = read_image(path)
                    board = BoardResult(str(index), (720, 1280), (0, 210, 720, 780), "现场回归", {})
                    endpoint = previewer.recognize_chain_endpoint(image, board)
                    found.append((frame.analyze(image), recognize_ranbu_chain_count(recognize, image, endpoint)))
                return CustomAction.RunResult(success=True)

        self.assertTrue(resource.register_custom_action("RanbuFeedbackProbe", Probe()))
        job = tasker.post_task(
            "RanbuFeedbackProbe",
            {"RanbuFeedbackProbe": {"action": "Custom", "custom_action": "RanbuFeedbackProbe"}},
        ).wait()
        self.assertTrue(job.succeeded)
        self.assertEqual(found, [(BattleState.RANBU, n) for n in (12, 13, 7, 4, 4)])


if __name__ == "__main__":
    unittest.main()
