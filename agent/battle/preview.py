"""复用当前帧棋盘识别，输出无触控的候选路线预演。"""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import cv2
from vision.image_io import Image, write_image, write_text_atomic
from vision.model import BoardResult
from vision.recognizer import BoardRecognizer
from vision.report import DRAW_COLORS, annotate_board

from .candidates import RoutePreview, build_route_preview


def write_preview_report(image: Image, board: BoardResult, preview: RoutePreview, destination: Path, stem: str) -> None:
    """在同帧编号图上标记候选边、边界不确定边和拟选路线。"""
    if board.frame_id != preview.frame_id or image.shape[:2] != tuple(reversed(board.source_size)):
        raise ValueError("棋盘、路线和输入图像必须来自同一帧且尺寸一致")
    annotated = annotate_board(image, board)
    objects = {item.id: item for item in board.objects}
    for edges, color in ((preview.candidate_edges, (80, 160, 80)), (preview.uncertain_edges, (0, 170, 255))):
        for edge in edges:
            first = tuple(map(round, objects[edge.first_id].center))
            second = tuple(map(round, objects[edge.second_id].center))
            cv2.line(annotated, first, second, color, 1, cv2.LINE_AA)
    route = preview.selected_route
    if route is not None:
        color = DRAW_COLORS[route.color]
        for first, second in pairwise(route.points):
            begin, end = tuple(map(round, first)), tuple(map(round, second))
            cv2.line(annotated, begin, end, (0, 0, 0), 7, cv2.LINE_AA)
            cv2.arrowedLine(annotated, begin, end, color, 3, cv2.LINE_AA, tipLength=0.12)
        for step, point in enumerate(route.points, 1):
            center = tuple(map(round, point))
            cv2.circle(annotated, center, 11, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(
                annotated,
                str(step),
                (center[0] - 5, center[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    report = {
        "frame_id": board.frame_id,
        "mode": "preview_only",
        "automatic_action_allowed": False,
        "board": board.to_dict(),
        "preview": asdict(preview),
        "legend": {"绿色细线": "连接候选", "橙色细线": "边界不确定，不参与路线", "粗箭头与数字": "拟选顺序"},
    }
    write_image(destination / f"{stem}.preview.png", annotated)
    write_text_atomic(
        destination / f"{stem}.preview.json",
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
    )


class BoardPreviewer:
    """只缓存静态识别器，每次独立识别显式传入的当前帧。"""

    def __init__(self, profile_path: Path, destination: Path) -> None:
        self.profile_path = profile_path
        self.destination = destination
        self._recognizer: BoardRecognizer | None = None

    def analyze(self, image: Image, frame_id: str, *, lightweight: bool = False, stem: str = "latest") -> RoutePreview:
        """识别本帧并保存预演，原图和棋盘识别的触控许可均不修改。"""
        if self._recognizer is None:
            self._recognizer = BoardRecognizer(self.profile_path)
        board = self._recognizer.analyze(image, frame_id=frame_id, lightweight=lightweight)
        preview = build_route_preview(board)
        write_preview_report(image, board, preview, self.destination, stem)
        return preview
