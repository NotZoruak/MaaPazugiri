"""使用只提供图片的控制器验证真实战斗动作的next和on_error分流。"""

from __future__ import annotations

import sys
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import maa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.candidates import CandidateRoute, RoutePreview  # noqa: E402
from battle.session import BattleSession, BattleState  # noqa: E402
from battle.strategy import StrategyDecision, StrategyDocument  # noqa: E402
from custom.action.battle import BattleAction  # noqa: E402
from maa.controller import CustomController  # noqa: E402
from maa.library import Library  # noqa: E402
from maa.resource import Resource  # noqa: E402
from maa.tasker import Tasker  # noqa: E402
from maa.toolkit import Toolkit  # noqa: E402
from test_battle_candidates import ghoul  # noqa: E402
from utils import logger  # noqa: E402
from vision.image_io import Image, read_image  # noqa: E402
from vision.model import BoardObject, BoardResult  # noqa: E402


class ImageOnlyController(CustomController):
    """只回放测试图，不连接设备；所有输入接口均未实现。"""

    def __init__(self, image: Image) -> None:
        self.image = image
        self.capture_times: list[float] = []
        super().__init__()

    def connect(self) -> bool:
        """仅初始化图片回放，不连接设备。"""
        return True

    def request_uuid(self) -> str:
        """返回固定离线测试标识。"""
        return "battle-action-image-only"

    def screencap(self) -> Image:
        """向Maa提供本次测试图。"""
        self.capture_times.append(time.monotonic())
        return self.image.copy()


class BattleActionNativeTests(unittest.TestCase):
    """在真实Maa任务中检查CustomAction返回后的分支。"""

    def run_case(
        self,
        image: Image,
        expected: str,
        *,
        next_image: Image | None = None,
        expected_state: str = "可操作",
        expected_operation: str | None = None,
        board_objects: list[BoardObject] | None = None,
        minimum_after_action_delay: float | None = None,
        board_frames: tuple[list[BoardObject], ...] | None = None,
        expected_messages: int = 1,
    ) -> None:
        """保留真实 Maa 调度和页面识别，仅回放棋盘识别与设备输入边界。"""
        # 导入动作会初始化AgentServer；本进程离线测试明确切回框架接口。
        Library.open(Path(maa.__file__).resolve().parent / "bin", agent_server=False)
        Toolkit.init_option(ROOT / "debug/battle-action-tests")
        resource = Resource()
        self.assertTrue(resource.post_bundle(ROOT / "resource/base").wait().succeeded)
        self.assertTrue(resource.register_custom_action("BattleAction", BattleAction()))
        controller = ImageOnlyController(image)
        self.assertTrue(controller.post_connection().wait().succeeded)
        tasker = Tasker()
        self.assertTrue(tasker.bind(resource, controller))
        self.addCleanup(lambda: tasker.post_stop().wait())
        override = {
            "RB_RunBattle": {"next": ["Battle_TestNext"], "on_error": ["Battle_TestError"]},
            "Battle_TestNext": {},
            "Battle_TestError": {},
        }
        session = BattleSession(timeout_seconds=5.0 if next_image is not None else 0.05, poll_interval=0.005)
        observed_images: list[Image] = []
        gui_messages: list[str] = []
        action_finished: list[float] = []
        frames = board_frames or (board_objects or [],)
        sink = logger.add(lambda message: gui_messages.append(message.record["message"]), level="INFO")
        self.addCleanup(logger.remove, sink)

        def recognize_current(current: Image, frame_id: str, **kwargs: object) -> BoardResult:
            """记录交给当前棋盘识别入口的图像，再让测试控制器展示终局。"""
            observed_images.append(current.copy())
            if next_image is not None and len(observed_images) == len(frames):
                controller.image = next_image
            return BoardResult(
                frame_id,
                (current.shape[1], current.shape[0]),
                (0, 0, 720, 1280),
                "离线测试",
                {},
                objects=frames[len(observed_images) - 1],
            )

        def record_action(observation, resolution, decision: StrategyDecision) -> None:
            """记录有输入的动作完成时间，等待不视为执行。"""
            if decision.kind != "wait":
                action_finished.append(time.monotonic())

        with (
            patch("custom.action.battle.BattleSession", return_value=session),
            patch("custom.action.battle.BoardPreviewer") as previewer,
            patch("custom.action.battle.MaaBattleExecutor") as executor,
        ):
            previewer.return_value.recognize_board.side_effect = recognize_current
            previewer.return_value.preview_routes.return_value = RoutePreview("离线测试")
            executor.return_value.execute_once.side_effect = record_action
            detail = tasker.post_task("RB_RunBattle", override).wait().get()
        self.assertIsNotNone(detail)
        assert detail is not None
        completed = {node.name for node in detail.nodes if node.completed}
        self.assertIn(expected, completed)
        other = "Battle_TestError" if expected == "Battle_TestNext" else "Battle_TestNext"
        self.assertNotIn(other, completed)
        if minimum_after_action_delay is not None:
            self.assertEqual(len(action_finished), 1)
            subsequent = [stamp for stamp in controller.capture_times if stamp > action_finished[0]]
            self.assertTrue(subsequent)
            self.assertGreaterEqual(subsequent[0] - action_finished[0], minimum_after_action_delay)
        if next_image is not None:
            self.assertEqual(len(observed_images), len(frames))
            self.assertTrue(np.array_equal(observed_images[0], image))
            self.assertEqual(len(gui_messages), expected_messages)
            self.assertIn(f"当前状态：{expected_state}", gui_messages[-1])
            self.assertIn("所属阶段：全程作战", gui_messages[-1])
            self.assertIn("进行操作：", gui_messages[-1])
            self.assertNotIn("StrategyDecision", gui_messages[-1])
            if expected_operation is not None:
                self.assertIn(f"进行操作：{expected_operation}", gui_messages[-1])
        else:
            self.assertEqual(observed_images, [])
            self.assertEqual(gui_messages, [])

    def test_victory_goes_to_next(self) -> None:
        """识别胜利后正常交还next，不点击结算页面。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        image[294:435, 88:285] = read_image(ROOT / "resource/base/image/region/战斗结算评级.png")
        self.run_case(image, "Battle_TestNext")

    def test_game_defeat_also_goes_to_next(self) -> None:
        """游戏失败正常交还next，不进入续关。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        image[432:574, 64:665] = read_image(ROOT / "resource/base/image/region/战斗失败确认.png")
        self.run_case(image, "Battle_TestNext")

    def test_watchdog_expiration_goes_to_on_error(self) -> None:
        """持续不能确认可操作时交还on_error。"""
        self.run_case(np.zeros((1280, 720, 3), dtype=np.uint8), "Battle_TestError")

    def test_conflicting_endings_do_not_report_completion(self) -> None:
        """矛盾的胜负证据保持未知，不误报正常结束。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        image[294:435, 88:285] = read_image(ROOT / "resource/base/image/region/战斗结算评级.png")
        image[432:574, 64:665] = read_image(ROOT / "resource/base/image/region/战斗失败确认.png")
        self.run_case(image, "Battle_TestError")

    def test_operable_frame_is_passed_to_preview_before_next_observation(self) -> None:
        """确认可操作后把同一帧送入预演，再正常识别下一帧终局。"""
        sources = list((ROOT / "training/board-samples/raw").glob("001_*.png"))
        if not sources:
            self.skipTest("需要本地已确认的可操作截图")
        victory = np.zeros((1280, 720, 3), dtype=np.uint8)
        victory[294:435, 88:285] = read_image(ROOT / "resource/base/image/region/战斗结算评级.png")
        self.run_case(read_image(sources[0]), "Battle_TestNext", next_image=victory)

    def test_inoperable_frame_does_not_start_board_preview(self) -> None:
        """齿轮存在但灰度不符时不调用对象识别或候选预演。"""
        sources = list((ROOT / "training/board-samples/raw").glob("006_*.png"))
        if not sources:
            self.skipTest("需要本地已确认的不可操作截图")
        self.run_case(read_image(sources[0]), "Battle_TestError")

    def test_ranbu_without_a_trusted_start_reports_wait(self) -> None:
        """没有可信起点的乱舞帧应显示等待，不能声称实际开始扫动。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        with patch(
            "custom.action.battle.BattleFrameRecognizer.analyze", side_effect=[BattleState.RANBU, BattleState.VICTORY]
        ):
            self.run_case(image, "Battle_TestNext", next_image=image, expected_state="乱舞", expected_operation="等待")

    def test_normal_link_waits_one_second_before_the_next_screenshot(self) -> None:
        """普通滑动交还主循环后，至少等待一秒再取下一帧。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        stage_id = StrategyDocument.builtin_default().initial_stage_id
        with (
            patch(
                "custom.action.battle.BattleFrameRecognizer.analyze",
                side_effect=[BattleState.OPERABLE, BattleState.OPERABLE, BattleState.VICTORY],
            ),
            patch(
                "custom.action.battle.decide_action",
                return_value=StrategyDecision(
                    "normal_link",
                    stage_id,
                    route=CandidateRoute("red", (1, 2), ((10, 20), (30, 40)), 0.5, 28.0),
                ),
            ),
        ):
            objects = [ghoul(1, 100), ghoul(2, 190)]
            self.run_case(
                image,
                "Battle_TestNext",
                next_image=image,
                board_frames=(objects, objects),
                expected_messages=2,
                expected_operation="普通连线",
                minimum_after_action_delay=0.99,
            )

    def test_ranbu_all_reserved_uses_the_short_link_action(self) -> None:
        """真实战斗入口在全为保留色时改派两点短连，不误报等待或全屏扫动。"""
        image = np.zeros((1280, 720, 3), dtype=np.uint8)
        strategy = StrategyDocument.builtin_default()
        stage = strategy.stages[0]
        strategy = replace(strategy, stages=(replace(stage, frenzy=replace(stage.frenzy, reserved_color="red")),))
        objects = [
            BoardObject(
                identifier,
                "ghoul",
                "red",
                None,
                "ghoul_red",
                "红色饿鬼",
                point,
                (point[0] - 5, point[1] - 5, 10, 10),
                {},
                [],
                point,
                "identified",
            )
            for identifier, point in ((1, (20, 30)), (2, (50, 60)))
        ]
        with (
            patch("custom.action.battle.load_strategy", return_value=strategy),
            patch(
                "custom.action.battle.BattleFrameRecognizer.analyze",
                side_effect=[BattleState.RANBU, BattleState.VICTORY],
            ),
        ):
            self.run_case(
                image,
                "Battle_TestNext",
                next_image=image,
                expected_state="乱舞",
                expected_operation="乱舞短连",
                board_objects=objects,
            )


if __name__ == "__main__":
    unittest.main()
