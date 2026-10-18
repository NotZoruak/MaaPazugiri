from __future__ import annotations

from typing import cast
from uuid import uuid4

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.define import RectType
from utils import logger
from utils.runtime_paths import get_runtime_paths
from vision.recognizer import BoardRecognizer


@AgentServer.custom_recognition("BoardVision")
class BoardVisionRecognition(CustomRecognition):
    """将离线棋盘识别器接入 Maa 自定义识别。"""

    def __init__(self) -> None:
        super().__init__()
        self._recognizer: BoardRecognizer | None = None

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        """识别当前帧，并把结构化棋盘状态写入 Maa 识别详情。"""
        try:
            configuration = context.get_node_data("Flag_LightweightRecognition")
            lightweight = configuration.get("enabled", False) if configuration is not None else False
            if not isinstance(lightweight, bool):
                raise ValueError("轻量识别配置的 enabled 必须为布尔值")
            recognizer = self._get_recognizer()
            result = recognizer.analyze(argv.image, frame_id=uuid4().hex, lightweight=lightweight)
        except Exception:
            logger.exception("棋盘自定义识别失败")
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"scene_state": "recognition_error", "objects": [], "automatic_action_allowed": False},
            )

        detail = result.to_dict()
        if result.scene_state == "unknown_ui":
            return CustomRecognition.AnalyzeResult(box=None, detail=detail)

        box = cast(RectType, tuple(round(value) for value in result.board_region))
        logger.debug(
            "棋盘识别结果：场景状态={}，对象数={}，问题数={}，耗时={}毫秒",
            result.scene_state,
            len(result.objects),
            len(result.issues),
            result.elapsed_ms,
        )
        return CustomRecognition.AnalyzeResult(box=box, detail=detail)

    def _get_recognizer(self) -> BoardRecognizer:
        """首次调用时加载静态模板，后续帧复用识别器实例。"""
        if self._recognizer is None:
            profile_path = (
                get_runtime_paths().resource_dir / "base" / "image" / "board" / "development" / "profile.json"
            )
            if not profile_path.is_file():
                raise FileNotFoundError(f"找不到棋盘识别配置：{profile_path}")
            self._recognizer = BoardRecognizer(profile_path)
        return self._recognizer
