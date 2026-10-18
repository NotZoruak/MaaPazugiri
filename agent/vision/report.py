"""输出编号诊断图与中文说明，不把匹配分数当成准确率。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2

from .image_io import Image, write_image, write_text_atomic
from .model import BoardResult

DRAW_COLORS = {
    "red": (65, 65, 235),
    "yellow": (25, 220, 240),
    "blue": (245, 185, 30),
    "green": (70, 200, 45),
    "black": (180, 180, 245),
}
OBSTACLE_COLORS = {
    "shell": (180, 210, 255),
    "rock": (210, 190, 180),
    "box": (50, 160, 240),
    "branch": (70, 150, 190),
    "bronze_bell": (220, 80, 230),
}


def write_report(image: Image, result: BoardResult, destination: Path, stem: str) -> dict[str, int]:
    """保存原图副本上的编号标记和对应 JSON。"""
    annotated = image.copy()
    for item in result.objects:
        color = DRAW_COLORS.get(item.color or "", (220, 80, 230))
        if item.kind == "obstacle":
            color = OBSTACLE_COLORS.get(item.subtype or "", (210, 210, 210))
        x, y, width, height = map(round, item.body_box)
        cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
        center = tuple(map(round, item.center))
        cv2.drawMarker(annotated, center, color, cv2.MARKER_CROSS, 10, 2)
        label = str(item.id) + ("?" if item.quality in ("single_feature", "level_unknown") else "")
        if item.kind == "obstacle":
            label += ":" + (item.subtype or "obstacle")
        elif item.kind == "special_ghoul":
            label += ":black"
        if item.enhanced:
            label += ":enhanced"
        cv2.putText(annotated, label, (x, max(16, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
        cv2.putText(annotated, label, (x, max(16, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        if item.candidate_point:
            cv2.circle(annotated, tuple(map(round, item.candidate_point)), 3, (255, 255, 255), -1)
    for issue in result.issues:
        if "box" in issue and issue.get("display_box", True):
            x, y, width, height = map(round, issue["box"])
            cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 140, 255), 1)
            cv2.putText(annotated, "?", (x, max(12, y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
    if result.chain_endpoint is not None:
        for marker in result.chain_endpoint.markers:
            x, y, width, height = map(round, marker.box)
            color = (255, 80, 220)
            cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 2)
            center = tuple(map(round, marker.center))
            cv2.drawMarker(annotated, center, color, cv2.MARKER_DIAMOND, 18, 2)
            label = "END" if result.chain_endpoint.status == "detected" else "END?"
            if result.chain_endpoint.object_id is not None:
                label += f":{result.chain_endpoint.object_id}"
            cv2.putText(annotated, label, (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(annotated, label, (x, max(16, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    destination.mkdir(parents=True, exist_ok=True)
    write_image(destination / f"{stem}.annotated.png", annotated)
    write_text_atomic(
        destination / f"{stem}.json", json.dumps(result.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    return dict(Counter(item.name for item in result.objects))
