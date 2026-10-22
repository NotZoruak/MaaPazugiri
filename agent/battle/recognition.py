"""在战斗动作内部使用原生直接识别，不暴露额外流程入口。"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from maa.define import RecognitionDetail
from maa.pipeline import JOCR, JColorMatch, JRecognitionParam, JRecognitionType, JTemplateMatch
from numpy.typing import NDArray

from .ranbu import matches_ranbu_marker
from .ranbu_state import matches_ranbu_countdown
from .session import BattleState
from .strategy import parse_hp_percent

Image = NDArray[np.uint8]
RECOGNITIONS: dict[str, tuple[JRecognitionType, JRecognitionParam]] = {
    "gear": (
        JRecognitionType.TemplateMatch,
        JTemplateMatch(template=["region/战斗中_设置齿轮.png"], roi=(651, 24, 58, 59), threshold=[0.9]),
    ),
    "brightness": (
        JRecognitionType.ColorMatch,
        JColorMatch(lower=[[140]], upper=[[255]], roi=(677, 49, 6, 6), method=6, count=36),
    ),
    "victory": (
        JRecognitionType.TemplateMatch,
        JTemplateMatch(template=["region/战斗结算评级.png"], roi=(88, 294, 197, 141), threshold=[0.9]),
    ),
    "defeat": (
        JRecognitionType.TemplateMatch,
        JTemplateMatch(template=["region/战斗失败确认.png"], roi=(64, 432, 601, 142), threshold=[0.9]),
    ),
}
NativeRecognition = Callable[[JRecognitionType, JRecognitionParam, Image], RecognitionDetail | None]

HP_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "enemy": (219, 10, 285, 27),
    "ally": (206, 1193, 382, 33),
}

ULTIMATE_CONFIRMATION_ROI = (72, 168, 576, 104)


def recognize_hp_percent(recognize: NativeRecognition, image: Image, side: str) -> float | None:
    """在同一张核对过的战斗截图中 OCR 指定一方的 A/B 血量并换算百分比。"""
    roi = HP_REGIONS.get(side)
    if roi is None or image.shape != (1280, 720, 3) or image.dtype != np.uint8:
        return None
    detail = recognize(JRecognitionType.OCR, JOCR(roi=roi, order_by="Horizontal"), image)
    if detail is None:
        return None
    texts = [getattr(result, "text", "") for result in detail.all_results]
    parsed = [value for text in texts if (value := parse_hp_percent(text)) is not None]
    if parsed:
        return parsed[0]
    return parse_hp_percent("".join(texts))


class BattleFrameRecognizer:
    """只分析显式传入的当前帧；不执行动作或复用历史对象位置。"""

    def __init__(self, recognize: NativeRecognition) -> None:
        self.recognize = recognize

    def _hit(self, name: str, image: Image) -> bool:
        """只把原生详情中的命中标志作为识别成功。"""
        method, parameters = RECOGNITIONS[name]
        detail = self.recognize(method, parameters, image)
        if detail is None:
            raise RuntimeError(f"无法启动战斗内部识别：{name}")
        return detail.hit

    def analyze(self, image: Image) -> BattleState:
        """先检查胜负与奥义选点提示，再确认齿轮和可操作证据。"""
        if image.shape != (1280, 720, 3) or image.dtype != np.uint8:
            return BattleState.UNKNOWN
        victory = self._hit("victory", image)
        defeat = self._hit("defeat", image)
        if victory and defeat:
            return BattleState.UNKNOWN
        if victory:
            return BattleState.VICTORY
        if defeat:
            return BattleState.DEFEAT
        confirmation = self._ultimate_confirmation(image)
        if confirmation is not None:
            return confirmation
        if not self._hit("gear", image):
            return BattleState.UNKNOWN
        operable = self._hit("brightness", image)
        ranbu = matches_ranbu_marker(image) or matches_ranbu_countdown(image)
        if operable and ranbu:
            return BattleState.UNKNOWN
        if ranbu:
            return BattleState.RANBU
        return BattleState.OPERABLE if operable else BattleState.INOPERABLE

    def _ultimate_confirmation(self, image: Image) -> BattleState | None:
        """只用提示文案区分点击和滑动选点，不读取动画箭头。"""
        detail = self.recognize(
            JRecognitionType.OCR,
            JOCR(roi=ULTIMATE_CONFIRMATION_ROI, order_by="Horizontal"),
            image,
        )
        if detail is None:
            raise RuntimeError("无法启动奥义位置确认提示OCR")
        text = "".join(getattr(result, "text", "") for result in detail.all_results)
        is_tap = "タップ" in text
        is_swipe = "スワイプ" in text
        if is_tap == is_swipe:
            return BattleState.UNKNOWN if is_tap or "位置" in text or "決め" in text else None
        return BattleState.ULTIMATE_TAP_CONFIRMATION if is_tap else BattleState.ULTIMATE_SWIPE_CONFIRMATION
