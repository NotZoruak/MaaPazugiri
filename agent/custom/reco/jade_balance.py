from __future__ import annotations

import re
import unicodedata
from typing import Any

from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_recognition import CustomRecognition
from maa.define import OCRResult
from utils import logger
from utils.params import parse_params


@AgentServer.custom_recognition("JadeBalanceThresholdRecognition")
class JadeBalanceThresholdRecognition(CustomRecognition):
    def analyze(
        self,
        context: Context,
        argv: CustomRecognition.AnalyzeArg,
    ) -> CustomRecognition.AnalyzeResult:
        try:
            params = parse_params(
                argv.custom_recognition_param,
                "template_node",
                "ocr_node",
                "roi_offset",
                "threshold",
            )
            template_node = self._node_name(params["template_node"], "template_node")
            ocr_node = self._node_name(params["ocr_node"], "ocr_node")
            roi_offset = self._roi_offset(params["roi_offset"])
            threshold = params["threshold"]
            if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
                raise ValueError("threshold 必须是非负整数")
        except ValueError as error:
            logger.error("宝玉余额识别参数无效：{}", error)
            return CustomRecognition.AnalyzeResult(box=None, detail={"error": str(error)})

        template_detail = context.run_recognition(template_node, argv.image)
        if template_detail is None or not template_detail.hit or template_detail.box is None:
            return CustomRecognition.AnalyzeResult(box=None, detail={"reason": "未匹配到碎玉回体模板"})

        match_box = template_detail.box
        image_height, image_width = argv.image.shape[:2]
        x = match_box.x + roi_offset[0]
        y = match_box.y + roi_offset[1]
        right = min(image_width, x + match_box.w + roi_offset[2])
        bottom = min(image_height, y + match_box.h + roi_offset[3])
        left = max(0, x)
        top = max(0, y)
        if right <= left or bottom <= top:
            logger.warning("宝玉余额 OCR 区域无效：{}", [left, top, right - left, bottom - top])
            return CustomRecognition.AnalyzeResult(box=None, detail={"reason": "宝玉余额 OCR 区域无效"})

        ocr_roi = [left, top, right - left, bottom - top]
        ocr_detail = context.run_recognition(
            ocr_node,
            argv.image,
            {ocr_node: {"recognition": {"param": {"roi": ocr_roi}}}},
        )
        if ocr_detail is None or not ocr_detail.hit:
            logger.warning("未能 OCR 识别剩余宝玉数量，区域：{}", ocr_roi)
            return CustomRecognition.AnalyzeResult(box=None, detail={"reason": "未识别到剩余宝玉数量", "roi": ocr_roi})

        ocr_text = "".join(item.text for item in ocr_detail.all_results if isinstance(item, OCRResult))
        normalized_text = unicodedata.normalize("NFKC", ocr_text)
        digits = "".join(re.findall(r"\d+", normalized_text))
        if not digits:
            logger.warning("OCR 结果中没有数字：{}", ocr_text)
            return CustomRecognition.AnalyzeResult(
                box=None,
                detail={"reason": "OCR 结果中没有宝玉数量", "ocr_text": ocr_text, "roi": ocr_roi},
            )

        jade_count = int(digits)
        detail: dict[str, Any] = {
            "jade_count": jade_count,
            "threshold": threshold,
            "ocr_text": ocr_text,
            "ocr_roi": ocr_roi,
        }
        if jade_count <= threshold:
            logger.info("剩余宝玉数量 {} 小于等于宝玉保留量 {}，停止碎玉换体", jade_count, threshold)
            return CustomRecognition.AnalyzeResult(box=match_box, detail=detail)

        logger.info("剩余宝玉数量 {} 高于宝玉保留量 {}，继续检查碎玉换体", jade_count, threshold)
        return CustomRecognition.AnalyzeResult(box=None, detail=detail)

    @staticmethod
    def _node_name(value: object, name: str) -> str:
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} 必须是非空字符串")
        return value

    @staticmethod
    def _roi_offset(value: object) -> tuple[int, int, int, int]:
        if not isinstance(value, list) or len(value) != 4 or any(type(item) is not int for item in value):
            raise ValueError("roi_offset 必须是四个整数")
        return value[0], value[1], value[2], value[3]
