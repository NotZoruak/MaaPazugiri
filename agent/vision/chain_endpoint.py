"""独立定位当前帧的大菱形终点，不读取头顶数字或复用旧对象位置。"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .image_io import Image, read_image
from .matching import PreparedTemplate, prepare_template, projection_may_reach
from .model import COLORS, BoardObject, BoardResult, ChainEndpointMarker, ChainEndpointResult, ChainMovePositions

EXPECTED_VARIANTS = ("under_7", "from_7", "from_14", "from_25")


def _valid_coordinate(value: object) -> bool:
    """核对外部传入的坐标数值，排除布尔值和非有限值。"""
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _valid_point(value: object) -> bool:
    """核对外部坐标必须恰好包含两个有效分量。"""
    return isinstance(value, (tuple, list)) and len(value) == 2 and all(_valid_coordinate(item) for item in value)


@dataclass(frozen=True)
class _Template:
    """保存允许跨帧缓存的静态模板派生数据。"""

    variant: str
    pattern: PreparedTemplate
    anchor: tuple[float, float]
    threshold: float


class ChainEndpointRecognizer:
    """在本次动作两端附近找白色核心；无动作参数时支持离线全盘诊断。"""

    def __init__(self, profile_path: Path) -> None:
        specification: dict[str, Any] = json.loads(profile_path.read_text(encoding="utf-8"))
        if specification.get("version") != 1:
            raise ValueError("不支持的终点菱形配置版本")
        self.reference_size = tuple(int(value) for value in specification["reference_size"])
        if len(self.reference_size) != 2 or min(self.reference_size) <= 0:
            raise ValueError("终点菱形参考尺寸无效")
        self.aspect_tolerance = float(specification["aspect_tolerance"])
        self.brightness_floor = int(specification["brightness_floor"])
        self.minimum_core_radius = float(specification["minimum_core_radius"])
        self.maximum_core_radius = float(specification["maximum_core_radius"])
        self.search_radius = int(specification["local_search_radius"])
        self.max_candidates = int(specification["max_candidates"])
        self.scales = tuple(float(value) for value in specification["scales"])
        self.supported_variants = tuple(dict.fromkeys(item["id"] for item in specification["templates"]))
        self.templates: list[_Template] = []
        for item in specification["templates"]:
            original = read_image(profile_path.parent / item["image"])
            mask = cast(Image, cv2.cvtColor(read_image(profile_path.parent / item["mask"]), cv2.COLOR_BGR2GRAY))
            if mask.shape != original.shape[:2] or not np.any(mask):
                raise ValueError("终点模板与掩码尺寸不匹配或掩码为空")
            threshold = float(item["threshold"])
            if not 0 < threshold <= 1:
                raise ValueError("终点模板阈值必须在零至一之间")
            for scale in self.scales:
                if not math.isfinite(scale) or not 0.8 <= scale <= 1.2:
                    raise ValueError("终点模板尺度必须在0.8至1.2之间")
                width, height = round(original.shape[1] * scale), round(original.shape[0] * scale)
                resized = cast(Image, cv2.resize(original, (width, height), interpolation=cv2.INTER_LINEAR))
                resized_mask = cast(Image, cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST))
                self.templates.append(
                    _Template(
                        item["id"],
                        prepare_template(resized, resized_mask),
                        (item["anchor"][0] * width / original.shape[1], item["anchor"][1] * height / original.shape[0]),
                        threshold,
                    )
                )

    def analyze(
        self, image: Image, board: BoardResult, *, move: ChainMovePositions | None = None
    ) -> ChainEndpointResult:
        """独立搜索本帧标记；多个终点或对象关联不唯一时保留不确定信息。"""
        started = time.perf_counter()
        result = ChainEndpointResult(
            board.frame_id,
            "not_detected",
            coverage={
                "supported_variants": list(self.supported_variants),
                "missing_variants": [
                    variant for variant in EXPECTED_VARIANTS if variant not in self.supported_variants
                ],
                "positions_reused_from_previous_frame": False,
                "search_scope": "move_start_and_end" if move is not None else "current_frame_board_region",
                "full_board_fallback": False,
                "number_ocr": False,
                "independent_evaluation_completed": False,
            },
        )
        if move is not None:
            result.move_outcome = "unknown"
        height, width = image.shape[:2]
        reference_width, reference_height = self.reference_size
        if (width, height) != board.source_size or abs(
            (width / height) / (reference_width / reference_height) - 1
        ) > self.aspect_tolerance:
            result.status = "uncertain"
            result.issues.append("截图尺寸或布局与本帧棋盘结果不一致")
            return self._finish(result, started)
        normalized = cast(Image, cv2.resize(image, (reference_width, reference_height), interpolation=cv2.INTER_AREA))
        sx, sy = width / reference_width, height / reference_height
        bx, by, bw, bh = board.board_region
        left, top = max(0, round(bx / sx)), max(0, round(by / sy))
        right, bottom = min(reference_width, round((bx + bw) / sx)), min(reference_height, round((by + bh) / sy))
        if right <= left or bottom <= top:
            result.status = "uncertain"
            result.issues.append("当前帧棋盘区域无效")
            return self._finish(result, started)
        regions = [(left, top, right, bottom)]
        move_points: tuple[tuple[float, float], tuple[float, float]] | None = None
        if move is not None:
            if not all(_valid_point(point) for point in (move.start, move.end)) or not all(
                left <= point[0] / sx < right and top <= point[1] / sy < bottom for point in (move.start, move.end)
            ):
                result.status = "uncertain"
                result.issues.append("本次移动起止位置无效或不在已核对的棋盘区域内")
                return self._finish(result, started)
            move_points = ((move.start[0] / sx, move.start[1] / sy), (move.end[0] / sx, move.end[1] / sy))
            # 半径是参考分辨率中的开发参数，覆盖触点与身体菱形的常见偏移。
            radius = 55
            regions = [
                (
                    max(left, round(x - radius)),
                    max(top, round(y - radius)),
                    min(right, round(x + radius + 1)),
                    min(bottom, round(y + radius + 1)),
                )
                for x, y in move_points
            ]
            result.coverage["move_positions"] = {"start": list(move.start), "end": list(move.end)}
            result.coverage["search_radius_reference_pixels"] = radius
        result.coverage["search_regions"] = [
            [round(x1 * sx, 2), round(y1 * sy, 2), round((x2 - x1) * sx, 2), round((y2 - y1) * sy, 2)]
            for x1, y1, x2, y2 in regions
        ]
        centers: list[tuple[float, float]] = []
        for bounds in regions:
            for center in self._core_centers(normalized, bounds):
                if not any(math.dist(center, other) < 3 for other in centers):
                    centers.append(center)
        result.coverage["bright_core_candidates"] = len(centers)
        if len(centers) > self.max_candidates:
            result.status = "uncertain"
            result.issues.append("当前帧明亮核心过多，无法可靠排除强光干扰")
            return self._finish(result, started)
        hits: list[ChainEndpointMarker] = []
        for center in centers:
            hits.extend(
                hit
                for hit in self._match(normalized, center)
                if any(x1 <= hit.center[0] < x2 and y1 <= hit.center[1] < y2 for x1, y1, x2, y2 in regions)
            )
        # 同一标记的多档模板或尺度命中在本帧合并，独立标记必须保留。
        accepted: list[ChainEndpointMarker] = []
        for hit in sorted(hits, key=lambda item: (-item.score, item.variant, item.center[1], item.center[0])):
            if not any(math.dist(hit.center, other.center) < 14 for other in accepted):
                accepted.append(hit)
        for hit in sorted(accepted, key=lambda item: (item.center[1], item.center[0])):
            x, y, tw, th = hit.box
            result.markers.append(
                ChainEndpointMarker(
                    hit.variant,
                    (round(hit.center[0] * sx, 2), round(hit.center[1] * sy, 2)),
                    (round(x * sx, 2), round(y * sy, 2), round(tw * sx, 2), round(th * sy, 2)),
                    round(hit.score, 6),
                )
            )
        if len(result.markers) == 1:
            result.status = "detected"
            self._associate(result, board.objects)
            if move_points is not None:
                marker = accepted[0].center
                start_distance, end_distance = (math.dist(marker, point) for point in move_points)
                # 两端过近或标记距两端接近时，无法可靠判断归属。
                if math.dist(*move_points) >= 24 and abs(start_distance - end_distance) >= 12:
                    nearest = 1 if end_distance < start_distance else 0
                    x1, y1, x2, y2 = regions[nearest]
                    if x1 <= marker[0] < x2 and y1 <= marker[1] < y2:
                        result.move_outcome = "target_reached" if nearest == 1 else "still_at_start"
                    else:
                        result.issues.append("大菱形未落在较近动作端的检查区域内，归属证据矛盾")
                else:
                    result.issues.append("大菱形距本次移动两端过于接近，无法区分归属")
        elif result.markers:
            result.status = "uncertain"
            result.issues.append("当前帧存在多个大菱形候选，不能确认唯一终点")
        return self._finish(result, started)

    def _core_centers(self, image: Image, bounds: tuple[int, int, int, int]) -> list[tuple[float, float]]:
        """只在本次指定区域内提取白色核心，不补搜其他区域。"""
        left, top, right, bottom = bounds
        region = image[top:bottom, left:right]
        white = np.where(np.min(region, axis=2) >= self.brightness_floor, 255, 0).astype(np.uint8)
        distance = cv2.distanceTransform(white, cv2.DIST_L2, 5)
        local_maximum = cv2.dilate(distance, np.ones((17, 17), dtype=np.uint8))
        peaks = np.where(
            (distance >= self.minimum_core_radius)
            & (distance <= self.maximum_core_radius)
            & (distance >= local_maximum - 1e-6),
            255,
            0,
        ).astype(np.uint8)
        count, _, _, centers = cv2.connectedComponentsWithStats(peaks)
        return [(float(centers[index, 0]) + left, float(centers[index, 1]) + top) for index in range(1, count)]

    def _match(self, image: Image, center: tuple[float, float]) -> list[ChainEndpointMarker]:
        """只在当前帧明亮核心附近核对静态模板，掩码排除数字与大部分来线。"""
        hits = []
        for template in self.templates:
            th, tw = template.pattern.image.shape[:2]
            expected_x, expected_y = center[0] - template.anchor[0], center[1] - template.anchor[1]
            left, top = max(0, round(expected_x) - self.search_radius), max(0, round(expected_y) - self.search_radius)
            right = min(image.shape[1], round(expected_x) + tw + self.search_radius + 1)
            bottom = min(image.shape[0], round(expected_y) + th + self.search_radius + 1)
            patch = image[top:bottom, left:right]
            if patch.shape[0] < th or patch.shape[1] < tw:
                continue
            if not projection_may_reach(patch, template.pattern, template.threshold):
                continue
            response = cv2.matchTemplate(
                patch, template.pattern.image, cv2.TM_CCOEFF_NORMED, mask=template.pattern.mask
            )
            response = np.where(np.isfinite(response), response, -1).astype(np.float32)
            _, score, _, position = cv2.minMaxLoc(response)
            if score < template.threshold:
                continue
            x, y = left + position[0], top + position[1]
            hits.append(
                ChainEndpointMarker(
                    template.variant,
                    (x + template.anchor[0], y + template.anchor[1]),
                    (float(x), float(y), float(tw), float(th)),
                    min(1.0, score),
                )
            )
        return hits

    @staticmethod
    def _associate(result: ChainEndpointResult, objects: list[BoardObject]) -> None:
        """仅把视觉标记关联到当前帧唯一近邻，禁止补造被遮挡的饿鬼。"""
        marker = result.markers[0]
        candidates: list[tuple[float, int]] = []
        for obj in objects:
            if obj.kind != "ghoul" or obj.color not in COLORS:
                continue
            width, height = obj.size.get("width"), obj.size.get("height")
            if (
                isinstance(width, bool)
                or isinstance(height, bool)
                or not isinstance(width, (float, int))
                or not isinstance(height, (float, int))
            ):
                continue
            if not math.isfinite(width) or not math.isfinite(height) or width <= 0 or height <= 0:
                continue
            if not all(math.isfinite(value) for value in obj.center):
                continue
            distance = math.hypot(
                (marker.center[0] - obj.center[0]) / width, (marker.center[1] - obj.center[1]) / height
            )
            if distance <= 0.4:
                candidates.append((distance, obj.id))
        candidates.sort()
        if not candidates:
            result.association_status = "unmatched"
            result.issues.append("已定位终点菱形，但当前帧未识别出对应饿鬼")
        elif len(candidates) > 1 and candidates[1][0] - candidates[0][0] < 0.15:
            result.association_status = "ambiguous"
            result.issues.append("终点菱形附近存在多个位置接近的饿鬼")
        else:
            result.object_id = candidates[0][1]
            result.association_status = "matched"

    @staticmethod
    def _finish(result: ChainEndpointResult, started: float) -> ChainEndpointResult:
        """登记本帧检测耗时。"""
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result
