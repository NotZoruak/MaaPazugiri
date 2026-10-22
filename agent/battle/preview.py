"""复用当前帧棋盘识别，输出无触控的候选路线预演。"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, replace
from itertools import pairwise
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
from vision.image_io import Image, write_image, write_text_atomic
from vision.model import BoardResult, ChainEndpointResult, ChainMovePositions
from vision.recognizer import BoardRecognizer
from vision.report import DRAW_COLORS, annotate_board

from .candidates import RoutePreview, build_route_preview


def _finite_diagnostic(value: Any) -> Any:
    """诊断中把无效数值写成空值，不修改本帧识别对象。"""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_diagnostic(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_diagnostic(item) for item in value]
    return value


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
        self._ranbu_execution_sequence = 0
        self._ranbu_execution_run_id = uuid4().hex

    def recognize_board(self, image: Image, frame_id: str, *, lightweight: bool = False) -> BoardResult:
        """只识别当前帧棋盘对象，不生成路线或修改触控许可。"""
        if self._recognizer is None:
            self._recognizer = BoardRecognizer(self.profile_path)
        return self._recognizer.analyze(image, frame_id=frame_id, lightweight=lightweight)

    def preview_routes(self, image: Image, board: BoardResult, *, stem: str = "latest") -> RoutePreview:
        """根据已经识别的当前帧棋盘单独生成候选路线预演。"""
        preview = build_route_preview(board)
        write_preview_report(image, board, preview, self.destination, stem)
        return preview

    def save_stability_report(self, report: dict[str, object]) -> None:
        """覆盖保存最近两帧稳定性比较，即使尚未允许规划路线也能排查。"""
        write_text_atomic(
            self.destination / "latest-stability.json",
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )
        if report.get("stable") is False and report.get("previous") is not None:
            write_text_atomic(
                self.destination / "latest-stability-rejected.json",
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            )

    def save_ranbu_report(
        self,
        image: Image,
        board: BoardResult,
        start: tuple[int, int],
        color: str | None,
        previous_color: str | None,
        points: tuple[tuple[int, int], ...],
    ) -> None:
        """覆盖保留最近一轮乱舞起点与原图，便于核对留色及落点，不积累图片。"""
        report = (
            json.dumps(
                _finite_diagnostic(
                    {
                        "board": board.to_dict(),
                        "start": start,
                        "color": color,
                        "previous_color": previous_color,
                        "points": points,
                    }
                ),
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )
        write_image(self.destination / "latest-ranbu.png", image)
        write_text_atomic(self.destination / "latest-ranbu.json", report)

    def save_ranbu_execution_report(
        self,
        before: Image,
        held: Image | None,
        details: dict[str, object],
        *,
        pass_images: Sequence[Image] = (),
        grab_images: Sequence[Image] = (),
    ) -> None:
        """轮换保存最近六轮实际输入及松手前现场，不把派发成功视为游戏接入成功。"""
        self._ranbu_execution_sequence += 1
        slot = (self._ranbu_execution_sequence - 1) % 6 + 1
        stem = f"ranbu-execution-{slot:02d}"
        before_name, held_name = f"{stem}.before.png", f"{stem}.held.png"
        write_image(self.destination / before_name, before)
        if held is not None:
            write_image(self.destination / held_name, held)
        else:
            (self.destination / held_name).unlink(missing_ok=True)
        pass_names: list[str] = []
        for index in range(1, 3):
            name = f"{stem}.pass-{index:02d}.png"
            if index <= len(pass_images):
                write_image(self.destination / name, pass_images[index - 1])
                pass_names.append(name)
            else:
                (self.destination / name).unlink(missing_ok=True)
        grab_names: list[str] = []
        for index in range(1, 3):
            name = f"{stem}.grab-{index:02d}.png"
            if index <= len(grab_images):
                write_image(self.destination / name, grab_images[index - 1])
                grab_names.append(name)
            else:
                (self.destination / name).unlink(missing_ok=True)
        report = {
            **details,
            "run_id": self._ranbu_execution_run_id,
            "sequence": self._ranbu_execution_sequence,
            "before_image": before_name,
            "held_image": held_name if held is not None else None,
            "pass_images": pass_names,
            "grab_images": grab_names,
        }
        serialized = json.dumps(_finite_diagnostic(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        write_text_atomic(self.destination / f"{stem}.json", serialized)
        write_text_atomic(self.destination / "latest-ranbu-execution.json", serialized)

    def recognize_chain_endpoint(
        self, image: Image, board: BoardResult, *, move: ChainMovePositions | None = None
    ) -> ChainEndpointResult | None:
        """在新截图中确认菱形落点，不把规划帧对象关联到新截图。"""
        if self._recognizer is None:
            self._recognizer = BoardRecognizer(self.profile_path)
        endpoint_recognizer = self._recognizer.endpoint_recognizer
        if endpoint_recognizer is None:
            return None
        current_frame = replace(board, frame_id=uuid4().hex, objects=[], chain_endpoint=None)
        return endpoint_recognizer.analyze(image, current_frame, move=move)

    def save_link_failure(
        self,
        image: Image | None,
        board: BoardResult,
        move: ChainMovePositions,
        result: ChainEndpointResult | None,
        details: dict[str, object],
    ) -> None:
        """保留最近一次未确认普通连线的现场，不无限积累截图。"""
        report = {
            "planned_frame_id": board.frame_id,
            "move": asdict(move),
            "recognition": asdict(result) if result is not None else None,
            **details,
        }
        if image is not None:
            write_image(self.destination / "latest-link-failure.png", image)
        report["screenshot_available"] = image is not None
        write_text_atomic(
            self.destination / "latest-link-failure.json",
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        )

    def analyze(self, image: Image, frame_id: str, *, lightweight: bool = False, stem: str = "latest") -> RoutePreview:
        """兼容旧接口，依次识别棋盘并生成路线预演。"""
        board = self.recognize_board(image, frame_id, lightweight=lightweight)
        return self.preview_routes(image, board, stem=stem)
