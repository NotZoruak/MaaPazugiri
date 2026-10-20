"""输出奥义充能的同帧诊断，不修改原图或开放自动输入。"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2
from vision.image_io import Image, write_image, write_text_atomic

from .ultimate import CHARGING_THRESHOLD, READY_THRESHOLD, UltimateChargeState, UltimateObservation


def write_ultimate_report(image: Image, observation: UltimateObservation, destination: Path, stem: str) -> None:
    """标记本帧命中的花形或文字区域，并保存状态、分数和未知原因。"""
    if image.shape[:2] != tuple(reversed(observation.source_size)):
        raise ValueError("奥义观测与输入图像尺寸不一致")
    annotated = image.copy()
    for slot in observation.slots:
        color = (80, 200, 80) if slot.state == UltimateChargeState.READY else (0, 160, 255)
        for box in (slot.ready_box, slot.charging_box):
            if box is None:
                continue
            x, y, width, height = box
            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
            label = f"U{slot.slot}:{slot.state.value}"
            cv2.putText(annotated, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
            cv2.putText(annotated, label, (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    report = {
        **asdict(observation),
        "mode": "charge_observation_only",
        "thresholds": {"ready": READY_THRESHOLD, "charging": CHARGING_THRESHOLD},
        "legend": {"ready": "已满充能", "charging": "仍在充能", "unknown": "无法确认"},
        "notice": "仅记录充能，不表示当前界面允许释放；绿色真实满充能样本尚待补充验证",
    }
    write_image(destination / f"{stem}.ultimate.png", annotated)
    write_text_atomic(destination / f"{stem}.ultimate.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
