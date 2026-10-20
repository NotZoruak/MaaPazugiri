"""独立识别当前帧三个角色的奥义充能状态，不推断当前能否操作。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from vision.image_io import Image, read_image

from .session import BattleState

Box = tuple[int, int, int, int]
Feature = NDArray[np.float32]
# 以下仅为已核对720×1280布局中的固定界面搜索区，不用于定位动态棋盘对象。
SLOT_REGIONS: tuple[Box, ...] = ((20, 1070, 120, 125), (240, 1110, 120, 105), (460, 1070, 120, 125))
READY_THRESHOLD = 0.75
CHARGING_THRESHOLD = 0.78


class UltimateChargeState(StrEnum):
    """区分满充能、有剩余次数文字以及证据不足。"""

    READY = "ready"
    CHARGING = "charging"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class UltimateSlot:
    """保存本帧单个位置的充能结论与可核对证据，不提供点击点。"""

    slot: int
    state: UltimateChargeState = UltimateChargeState.UNKNOWN
    ready_score: float = 0.0
    charging_score: float = 0.0
    ready_box: Box | None = None
    charging_box: Box | None = None
    reason: str = "未检查"


@dataclass
class UltimateObservation:
    """保存同帧充能观测；满充能与页面当前可操作是两个独立条件。"""

    frame_id: str
    source_size: tuple[int, int]
    slots: list[UltimateSlot]
    layout_supported: bool = True
    phase: BattleState = BattleState.UNKNOWN
    checked: bool = False
    automatic_action_allowed: bool = False
    issues: list[str] = field(default_factory=list)


class UltimateChargeRecognizer:
    """只缓存静态模板特征，每次在显式传入图像中重新搜索三个花标。"""

    def __init__(self, template_directory: Path) -> None:
        self._ready = self._feature(read_image(template_directory / "ready_flower.png"))
        self._charging = self._feature(read_image(template_directory / "charging_text.png"))
        self._mask = read_image(template_directory / "ready_mask.png")[:, :, 0].copy()
        if self._mask.shape != self._ready.shape or not np.any(self._mask):
            raise ValueError("奥义满充能模板掩码为空或尺寸不一致")
        for template in (self._ready, self._charging):
            if not template.size or min(template.shape) <= 1 or float(template.std()) < 1:
                raise ValueError("奥义模板为空或缺少可比较的形状特征")
            if any(template.shape[0] > region[3] or template.shape[1] > region[2] for region in SLOT_REGIONS):
                raise ValueError("奥义模板大于已核对的界面搜索区")
            template.setflags(write=False)
        self._mask.setflags(write=False)

    @staticmethod
    def _feature(image: Image) -> Feature:
        """取三通道最小值突出白色轮廓和文字，不限定红黄蓝绿色相。"""
        return image.min(axis=2).astype(np.float32)

    @staticmethod
    def _match(feature: Feature, template: Feature, region: Box, mask: Image | None = None) -> tuple[float, Box]:
        """只在本帧当前区域搜索，低对比或无效分数不能作为命中。"""
        x, y, _, _ = region
        matches = cv2.matchTemplate(feature, template, cv2.TM_CCOEFF_NORMED, mask=mask)
        matches = np.nan_to_num(matches, nan=0, posinf=0, neginf=0)
        _, score, _, point = cv2.minMaxLoc(matches)
        return max(0.0, min(1.0, float(score))), (x + point[0], y + point[1], template.shape[1], template.shape[0])

    def analyze(self, image: Image, *, frame_id: str, phase: BattleState) -> UltimateObservation:
        """同时比较无字花形和あと文字，均命中或均缺失时保持未知。"""
        size = (image.shape[1], image.shape[0]) if image.ndim >= 2 else (0, 0)
        supported = image.shape == (1280, 720, 3) and image.dtype == np.uint8
        if phase not in (BattleState.OPERABLE, BattleState.RANBU):
            return UltimateObservation(
                frame_id,
                size,
                [UltimateSlot(slot) for slot in (1, 2, 3)],
                layout_supported=supported,
                phase=phase,
                issues=["当前未确认普通可操作棋盘或乱舞时间，跳过奥义检查"],
            )
        if not supported:
            return UltimateObservation(
                frame_id,
                size,
                [UltimateSlot(slot) for slot in (1, 2, 3)],
                layout_supported=False,
                phase=phase,
                issues=["未核对的布局或像素格式，不套用固定界面区域"],
            )
        result = UltimateObservation(frame_id, size, [], phase=phase, checked=True)
        for slot, region in enumerate(SLOT_REGIONS, 1):
            x, y, width, height = region
            feature = self._feature(image[y : y + height, x : x + width])
            ready_score, ready_box = self._match(feature, self._ready, region, self._mask)
            charging_score, charging_box = self._match(feature, self._charging, region)
            ready = ready_score >= READY_THRESHOLD
            charging = charging_score >= CHARGING_THRESHOLD
            state = UltimateChargeState.UNKNOWN
            reason = "没有足够花形或文字证据"
            if ready and not charging:
                state, reason = UltimateChargeState.READY, "命中无字花形，未命中剩余次数文字"
            elif charging and not ready:
                state, reason = UltimateChargeState.CHARGING, "命中あと文字，未命中无字花形"
            elif ready and charging:
                reason = "无字花形与剩余次数文字证据冲突"
            result.slots.append(
                UltimateSlot(
                    slot,
                    state,
                    round(ready_score, 6),
                    round(charging_score, 6),
                    ready_box if ready else None,
                    charging_box if charging else None,
                    reason,
                )
            )
        return result
