"""验证启动任务强制先关闭游戏，且成功提示交由标题识别节点输出。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from custom.action.restart_game import RestartGameAction  # noqa: E402


class StartGameTests(unittest.TestCase):
    """用控制器替身检查应用关闭与启动的先后及失败处理。"""

    def run_action(self, stop_result: bool | Exception, startup: bool = True) -> tuple[bool, Mock, Mock]:
        """读取真实任务参数执行动作，不连接设备。"""
        pipeline = json.loads((ROOT / "resource/base/pipeline/Normal/StartGame.json").read_text(encoding="utf-8"))
        params = pipeline["StartGame"]["action"]["param"]["custom_action_param"]
        if not startup:
            params = {"package_name": params["package_name"]}
        controller = Mock()
        if isinstance(stop_result, Exception):
            controller.post_stop_app.side_effect = stop_result
        else:
            controller.post_stop_app.return_value.wait.return_value = SimpleNamespace(
                succeeded=stop_result, failed=not stop_result
            )
        controller.post_start_app.return_value.wait.return_value = SimpleNamespace(succeeded=True)
        context = SimpleNamespace(tasker=SimpleNamespace(controller=controller))
        argv = SimpleNamespace(custom_action_param=json.dumps(params))
        with patch("custom.action.restart_game.logger") as logger:
            result = RestartGameAction().run(context, argv)  # type: ignore[arg-type]
        return result.success, controller, logger

    def test_startup_always_stops_before_start(self) -> None:
        """关闭命令成功时继续启动，未运行的应用也走此路径，不提前输出成功。"""
        success, controller, logger = self.run_action(True)
        self.assertTrue(success)
        self.assertEqual(
            [call[0] for call in controller.mock_calls],
            ["post_stop_app", "post_stop_app().wait", "post_start_app", "post_start_app().wait"],
        )
        logger.info.assert_not_called()
        logger.warning.assert_not_called()

    def test_failed_stop_prevents_start(self) -> None:
        """关闭返回失败时任务失败，不继续启动。"""
        success, controller, _ = self.run_action(False)
        self.assertFalse(success)
        controller.post_start_app.assert_not_called()

    def test_stop_exception_prevents_start(self) -> None:
        """关闭抛出异常时任务失败，不继续启动。"""
        success, controller, _ = self.run_action(RuntimeError("关闭失败"))
        self.assertFalse(success)
        controller.post_start_app.assert_not_called()

    def test_existing_recovery_behavior_is_preserved(self) -> None:
        """通用重启调用继续保留已有的关闭失败后尝试启动行为。"""
        success, controller, logger = self.run_action(False, startup=False)
        self.assertTrue(success)
        controller.post_start_app.assert_called_once()
        logger.info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
