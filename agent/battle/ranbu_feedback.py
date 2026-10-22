"""从当前按住截图读取乱舞连线反馈，不参与棋盘路线选择。"""

from __future__ import annotations

from dataclasses import dataclass, field

from maa.pipeline import JOCR, JRecognitionType
from vision.image_io import Image
from vision.model import ChainEndpointResult

from .recognition import NativeRecognition
from .session import BattleState


@dataclass(frozen=True)
class RanbuHoldFeedback:
    """保存同帧状态与残留像素；实时流程不读取连线数字。"""

    state: BattleState
    count: int | None
    residual_status: str = "unknown"
    residual_details: dict[str, object] = field(default_factory=dict)
    state_checked: bool = True


def recognize_ranbu_chain_count(
    recognize: NativeRecognition, image: Image, endpoint: ChainEndpointResult | None
) -> int | None:
    """仅供离线诊断读取数字，实时乱舞扫动不调用此函数。"""
    from vision.ranbu_digits import parse_ranbu_digit_text, prepare_ranbu_digits

    if endpoint is None or endpoint.status not in ("detected", "uncertain") or image.ndim != 3:
        return None
    counts: set[int] = set()
    for marker in endpoint.markers:
        prepared = prepare_ranbu_digits(image, marker)
        if prepared.filtered is None:
            continue
        detail = recognize(JRecognitionType.OCR, JOCR(), prepared.filtered)
        if detail is None:
            continue
        for result in detail.all_results:
            text = getattr(result, "text", "")
            score = getattr(result, "score", 0.0)
            if score >= 0.85 and (value := parse_ranbu_digit_text(prepared, text)) is not None:
                counts.add(value)
    return next(iter(counts)) if len(counts) == 1 else None
