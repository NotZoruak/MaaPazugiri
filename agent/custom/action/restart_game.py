from __future__ import annotations

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger
from utils.params import parse_params


@AgentServer.custom_action("RestartGameAction")
class RestartGameAction(CustomAction):
    """通过当前 Maa 控制器停止并重新启动指定 Android 应用。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        try:
            params = parse_params(argv.custom_action_param, "package_name")
            package_name = params["package_name"]
            if not isinstance(package_name, str) or not package_name.strip():
                raise ValueError("package_name 必须是非空字符串")
            package_name = package_name.strip()
        except ValueError as error:
            logger.error(f"RestartGameAction 参数无效：{error}")
            return CustomAction.RunResult(success=False)

        controller = context.tasker.controller
        try:
            stop_job = controller.post_stop_app(package_name).wait()
            if stop_job.failed:
                logger.warning(f"停止游戏未成功，继续尝试启动：{package_name}")
        except Exception as error:
            logger.warning(f"停止游戏时发生异常，继续尝试启动：{error}")

        try:
            start_job = controller.post_start_app(package_name).wait()
        except Exception as error:
            logger.error(f"启动游戏时发生异常：{error}")
            return CustomAction.RunResult(success=False)

        if not start_job.succeeded:
            logger.error(f"启动游戏失败：{package_name}")
            return CustomAction.RunResult(success=False)

        logger.info(f"游戏已重新启动：{package_name}")
        return CustomAction.RunResult(success=True)
