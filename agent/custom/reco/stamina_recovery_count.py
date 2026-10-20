from __future__ import annotations

from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from utils import logger
from utils.params import parse_params

SUCCESS_NODE_NAME = "BattleSettlement_BrokenJadeStaminaRecoverySucceeded"


@AgentServer.custom_recognition("BrokenJadeStaminaRecoveryCountRecognition")
class BrokenJadeStaminaRecoveryCountRecognition(CustomRecognition):
    """按本次任务成功节点记录统计碎玉换体次数。"""

    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        try:
            params = parse_params(argv.custom_recognition_param, "maximum")
            maximum = params["maximum"]
            if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < -1:
                raise ValueError("maximum 必须是 -1 或非负整数")

            if maximum == -1:
                return CustomRecognition.AnalyzeResult(box=None, detail={"maximum": maximum, "unlimited": True})

            count = sum(
                1
                for node in argv.task_detail.nodes
                if node.name == SUCCESS_NODE_NAME and node.recognition is not None and node.recognition.hit
            )
        except (ValueError, RuntimeError) as error:
            logger.error("碎玉换体次数识别失败：{}", error)
            return CustomRecognition.AnalyzeResult(box=None, detail={"error": str(error)})
        except Exception:
            logger.exception("读取本次任务中的碎玉换体记录失败")
            return CustomRecognition.AnalyzeResult(box=None, detail={"error": "无法读取本次任务记录"})

        detail: dict[str, Any] = {"count": count, "maximum": maximum}
        if count < maximum:
            return CustomRecognition.AnalyzeResult(box=None, detail=detail)

        logger.info("本次任务已成功碎玉换体 {} 次，达到最大次数 {}", count, maximum)
        box = (0, 0, max(1, argv.image.shape[1]), max(1, argv.image.shape[0]))
        return CustomRecognition.AnalyzeResult(box=box, detail=detail)
