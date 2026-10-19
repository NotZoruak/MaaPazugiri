"""提供整场战斗的唯一Maa自定义动作入口。"""

from __future__ import annotations

import time
from uuid import uuid4

from battle.preview import BoardPreviewer
from battle.recognition import BattleFrameRecognizer, Image
from battle.session import BattleSession, BattleState
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger
from utils.runtime_paths import get_runtime_paths


@AgentServer.custom_action("BattleAction")
class BattleAction(CustomAction):
    """在内部观察至胜负或连续未确认超时，外部只接收动作返回结果。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        """胜利或游戏失败正常返回，超时与运行错误返回动作失败。"""
        try:
            controller = context.tasker.controller
            recognizer = BattleFrameRecognizer(context.run_recognition_direct)
            paths = get_runtime_paths()
            previewer = BoardPreviewer(
                paths.resource_dir / "base/image/board/development/profile.json",
                paths.debug_dir / "battle-preview",
            )
            last_state: BattleState | None = None
            current_image: Image | None = None
            last_preview_at = float("-inf")

            def observe() -> BattleState:
                """重新截图并核对状态，不在本阶段发送棋盘输入。"""
                nonlocal last_state, current_image
                current_image = None
                job = controller.post_screencap().wait()
                if not job.succeeded:
                    return BattleState.UNKNOWN
                image = job.get()
                state = BattleState.UNKNOWN if not image.size else recognizer.analyze(image)
                if state == BattleState.OPERABLE:
                    current_image = image
                if state != last_state:
                    logger.info("战斗内部状态：{}", state.value)
                    last_state = state
                return state

            def preview_current_frame() -> None:
                """限频预演同一张可操作截图，处理后不保留动态对象或图像。"""
                nonlocal current_image, last_preview_at
                try:
                    if current_image is None or time.monotonic() - last_preview_at < 1.0:
                        return
                    configuration = context.get_node_data("Flag_LightweightRecognition")
                    lightweight = configuration.get("enabled", False) if configuration is not None else False
                    if not isinstance(lightweight, bool):
                        raise ValueError("轻量识别配置的enabled必须为布尔值")
                    preview = previewer.analyze(current_image, uuid4().hex, lightweight=lightweight)
                    route = preview.selected_route
                    logger.info(
                        "候选预演：候选边={}，不确定边={}，拟选编号={}，未执行触控。",
                        len(preview.candidate_edges),
                        len(preview.uncertain_edges),
                        route.object_ids if route is not None else (),
                    )
                    last_preview_at = time.monotonic()
                finally:
                    current_image = None

            logger.info("进入战斗动作：当前仅进行候选路线预演，不执行触控。")
            outcome = BattleSession().run(
                observe, stopped=lambda: context.tasker.stopping, on_operable=preview_current_frame
            )
        except Exception:
            logger.exception("战斗自定义动作发生运行错误")
            return CustomAction.RunResult(success=False)
        logger.info("战斗动作结束：{}，正常完成={}", outcome.value, outcome.completed)
        return CustomAction.RunResult(success=outcome.completed)
