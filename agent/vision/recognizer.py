"""组合局部模板、颜色和几何证据识别自由位置对象。"""

from __future__ import annotations

import hashlib
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .config import Profile, load_json
from .image_io import Image, read_image
from .matching import (
    ColorGate,
    ColorIntegral,
    PreparedTemplate,
    color_integral,
    local_response,
    prepare_board,
    prepare_template,
    rectangle_count,
)
from .model import COLORS, EXPECTED_LABELS, LABEL_NAMES, BoardObject, BoardResult, Hit
from .templates import check_pattern

HSV_RANGES = {
    "red": (((0, 115, 65), (12, 255, 255)), ((165, 115, 65), (179, 255, 255))),
    "yellow": (((13, 100, 80), (37, 255, 255)),),
    "blue": (((80, 85, 55), (115, 255, 255)),),
    "green": (((35, 70, 45), (79, 255, 255)),),
    "black": (((0, 0, 25), (179, 150, 170)),),
}


def _checked_workers(value: object) -> int:
    """核对显式并行数，避免把布尔值当作整数。"""
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 16:
        raise ValueError("模板并行数必须为1至16的整数")
    return value


class BoardRecognizer:
    """仅输出静态诊断结果，不提供点击或拖动接口。"""

    def __init__(
        self, profile_path: Path, *, full_search: bool = False, color_prefilter: bool = True, workers: int | None = None
    ) -> None:
        self.workers = min(8, os.process_cpu_count() or 1) if workers is None else _checked_workers(workers)
        self.full_search = full_search
        self.color_prefilter = color_prefilter
        self.profile = Profile.from_dict(load_json(profile_path))
        self.profile_sha256 = hashlib.sha256(profile_path.read_bytes()).hexdigest()
        self.patterns: list[tuple[dict[str, Any], Image, Image]] = []
        self.scaled_patterns: list[tuple[dict[str, Any], float, PreparedTemplate]] = []
        for item in self.profile.templates:
            image = read_image(profile_path.parent / item["image"])
            mask_image = read_image(profile_path.parent / item["mask"])
            mask = cast(Image, cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY))
            if mask.shape != image.shape[:2]:
                raise ValueError(f"模板与掩码大小不一致：{item['id']}")
            check_pattern(image, mask, item["id"])
            self.patterns.append((item, image, mask))
            for scale in self.profile.scales:
                tw, th = max(2, round(image.shape[1] * scale)), max(2, round(image.shape[0] * scale))
                if tw > self.profile.board_roi[2] or th > self.profile.board_roi[3]:
                    continue
                resized = cast(Image, cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR))
                support = cv2.resize((mask > 0).astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
                resized_mask = np.where(support >= 1.0 - 1e-6, 255, 0).astype(np.uint8)
                if np.count_nonzero(resized_mask) >= 8:
                    self.scaled_patterns.append((item, scale, prepare_template(resized, resized_mask)))

    def analyze(self, image: Image, frame_id: str) -> BoardResult:
        """识别一帧，并将全部位置还原为输入截图坐标。"""
        started = time.perf_counter()
        height, width = image.shape[:2]
        ref_width, ref_height = self.profile.reference_size
        sx, sy = width / ref_width, height / ref_height
        rx, ry, rw, rh = self.profile.board_roi
        known = {
            self._label(item["kind"], item.get("color"), item.get("bomb_level"), item.get("subtype"))
            for item in self.profile.templates
        }
        missing = sorted(set(EXPECTED_LABELS) - known)
        result = BoardResult(
            frame_id,
            (width, height),
            (rx * sx, ry * sy, rw * sx, rh * sy),
            self.profile.name,
            {
                "missing_templates": missing,
                "sample_role": self.profile.role,
                "profile_sha256": self.profile_sha256,
                "independent_evaluation_completed": False,
                "search_mode": "full" if self.full_search else "coarse_refine",
                "color_prefilter": self.color_prefilter and not self.full_search,
                "template_variants_cached": len(self.scaled_patterns),
                "frame_preprocessing_shared": True,
                "positions_reused_from_previous_frame": False,
                "template_workers": self.workers,
                "parallel_results_in_template_order": True,
                "standard_object_roi": self.profile.standard_object_roi or self.profile.board_roi,
                "coarse_search_parameters": None
                if self.full_search
                else {
                    "downscale": 0.5,
                    "threshold": 0.55,
                    "max_candidates": 120,
                    "refine_radius": 5,
                },
            },
        )
        if abs((width / height) / (ref_width / ref_height) - 1) > self.profile.aspect_tolerance:
            result.scene_state = "unknown_ui"
            result.issues.append({"code": "unsupported_layout", "message": "输入长宽比与已核对布局不一致"})
            result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return result
        normalized = cast(Image, cv2.resize(image, (ref_width, ref_height), interpolation=cv2.INTER_AREA))
        board = normalized[ry : ry + rh, rx : rx + rw]
        hsv = cast(Image, cv2.cvtColor(board, cv2.COLOR_BGR2HSV))
        color_masks = {color: self._color_mask(hsv, color) for color in HSV_RANGES}
        integrals = {color: color_integral(mask) for color, mask in color_masks.items()}
        coarse_board = None if self.full_search else prepare_board(board)
        regular_board, regular_integrals, regular_coarse = board, integrals, coarse_board
        dx, dy = 0, 0
        if self.profile.standard_object_roi is not None:
            ox, oy, ow, oh = self.profile.standard_object_roi
            dx, dy = ox - rx, oy - ry
            regular_board = board[dy : dy + oh, dx : dx + ow]
            regular_integrals = {
                color: color_integral(mask[dy : dy + oh, dx : dx + ow]) for color, mask in color_masks.items()
            }
            regular_coarse = None if self.full_search else prepare_board(regular_board)

        def regular_hits(kind: str) -> list[Hit]:
            """保留普通对象原有的搜索边界和颜色占比分母。"""
            hits = self._match(regular_board, regular_integrals, kind, regular_coarse, enhanced=False)
            for hit in hits:
                hit.center = (hit.center[0] + dx, hit.center[1] + dy)
                hit.anchor = (hit.anchor[0] + dx, hit.anchor[1] + dy)
            return hits

        bomb_groups = [group for group in self._group(regular_hits("bomb")) if group[0].accepted]
        ghoul_hits = [*regular_hits("ghoul"), *self._match(board, integrals, "ghoul", coarse_board, enhanced=True)]
        accepted = [hit for hit in ghoul_hits if not any(self._in_bomb(hit.anchor, group[0]) for group in bomb_groups)]
        extra_hits = [
            hit
            for kind in ("obstacle", "special_ghoul", "prop")
            for hit in self._match(board, integrals, kind, coarse_board)
            if not any(self._in_bomb(hit.anchor, group[0]) for group in bomb_groups)
        ]
        groups = [*bomb_groups, *self._group(accepted), *self._group(extra_hits)]
        groups.sort(key=lambda group: (group[0].center[1], group[0].center[0]))
        for group in groups:
            obj = self._object(group, len(result.objects) + 1, sx, sy, (width, height), bomb_groups)
            result.objects.append(obj)
            if obj.quality == "level_unknown":
                result.issues.append(
                    {"code": "ambiguous_bomb_level", "object_id": obj.id, "message": "外焰等级证据接近，保留未知等级"}
                )
        self._unresolved({color: color_masks[color] for color in COLORS}, groups, result, sx, sy)
        if missing:
            result.issues.append(
                {"code": "missing_templates", "labels": missing, "message": "缺少类别模板，尚未验证完整覆盖"}
            )
        result.issues.append({"code": "offline_unverified", "message": "单帧未确认稳定性与触控有效性，禁止自动操作"})
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    @staticmethod
    def _color_mask(hsv: Image, color: str) -> Image:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in HSV_RANGES[color]:
            mask |= cast(Image, cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)))
        return mask

    def _match(
        self,
        board: Image,
        integrals: dict[str, ColorIntegral],
        kind: str,
        coarse_board: Image | None,
        *,
        enhanced: bool | None = None,
    ) -> list[Hit]:
        regions: dict[tuple[int, ...], tuple[Image, Image | None]] = {}
        bomb_floor = (
            min(
                (float(spec.get("threshold", 0.84)) for spec, _, _ in self.patterns if spec["kind"] == "bomb"),
                default=1.0,
            )
            - self.profile.bomb_margin
        )
        selected = [
            item
            for item in self.scaled_patterns
            if item[0]["kind"] == kind
            and (kind != "ghoul" or enhanced is None or bool(item[0].get("enhanced", False)) == enhanced)
        ]
        for spec, _, _ in selected:
            if spec.get("search_roi") is not None:
                region = tuple(map(int, spec["search_roi"]))
                if region not in regions:
                    x, y, width, height = region
                    dx, dy = x - self.profile.board_roi[0], y - self.profile.board_roi[1]
                    cropped = board[dy : dy + height, dx : dx + width]
                    regions[region] = (cropped, None if self.full_search else prepare_board(cropped))

        def match_one(item: tuple[dict[str, Any], float, PreparedTemplate]) -> list[Hit]:
            """独立处理一个模板尺度，不保存当前帧外的数据。"""
            hits: list[Hit] = []
            spec, scale, pattern = item
            resized, resized_mask = pattern.image, pattern.mask
            th, tw = resized.shape[:2]
            search_board, search_coarse = board, coarse_board
            offset_x, offset_y = 0, 0
            if spec.get("search_roi") is not None:
                region = tuple(map(int, spec["search_roi"]))
                x, y = region[:2]
                offset_x, offset_y = x - self.profile.board_roi[0], y - self.profile.board_roi[1]
                search_board, search_coarse = regions[region]
            if tw > search_board.shape[1] or th > search_board.shape[0]:
                return hits
            gate = None
            if kind in ("ghoul", "special_ghoul") and self.color_prefilter and not self.full_search:
                gate = ColorGate(
                    integrals[spec["color"]],
                    (float(spec["center_offset"][0]) * scale, float(spec["center_offset"][1]) * scale),
                    (float(spec["body_size"][0]) * scale, float(spec["body_size"][1]) * scale),
                    float(spec.get("min_color_fraction", 0.3)),
                )
            response = local_response(search_board, pattern, self.full_search, search_coarse, gate)
            peaks = cv2.dilate(response, np.ones((5, 5), dtype=np.uint8))
            threshold = float(spec.get("threshold", 0.84))
            search_threshold = bomb_floor if kind == "bomb" else threshold
            ys, xs = np.where((response >= search_threshold) & (response >= peaks - 1e-7))
            ordered = sorted(
                zip(xs.tolist(), ys.tolist(), strict=True), key=lambda point: -float(response[point[1], point[0]])
            )
            kept: list[tuple[int, int]] = []
            for px, py in ordered:
                if len(kept) >= 60:
                    break
                if any(math.hypot(px - ox, py - oy) < min(tw, th) * 0.35 for ox, oy in kept):
                    continue
                bx, by = px + offset_x, py + offset_y
                if (kind == "bomb" or spec.get("check_template_color", False)) and not self._template_color_compatible(
                    board[by : by + th, bx : bx + tw], resized, resized_mask
                ):
                    continue
                if kind == "special_ghoul" and spec.get("part") == "eye":
                    eye = cv2.cvtColor(board[by : by + th, bx : bx + tw], cv2.COLOR_BGR2HSV)
                    red = ((eye[:, :, 0] <= 20) | (eye[:, :, 0] >= 165)) & (eye[:, :, 1] >= 60) & (eye[:, :, 2] >= 60)
                    if np.count_nonzero(red & (resized_mask > 0)) < 0.08 * np.count_nonzero(resized_mask):
                        continue
                cx = bx + float(spec["center_offset"][0]) * scale
                cy = by + float(spec["center_offset"][1]) * scale
                bw, bh = (float(value) * scale for value in spec["body_size"])
                if not (0 <= cx < board.shape[1] and 0 <= cy < board.shape[0]):
                    continue
                color = spec.get("color")
                fraction: float | None = None
                if color is not None:
                    x1, x2 = max(0, round(cx - bw / 2)), min(board.shape[1], round(cx + bw / 2))
                    y1, y2 = max(0, round(cy - bh / 2)), min(board.shape[0], round(cy + bh / 2))
                    count = rectangle_count(integrals[color], x1, y1, x2, y2)
                    fraction = float(count / max(1, (x2 - x1) * (y2 - y1)))
                    if fraction < float(spec.get("min_color_fraction", 0.3)):
                        continue
                if spec.get("enhanced", False) and not self._enhanced_color_compatible(
                    board, (cx, cy), (bw, bh), str(color)
                ):
                    continue
                kept.append((px, py))
                hits.append(
                    Hit(
                        str(spec["id"]),
                        kind,
                        color,
                        spec.get("bomb_level"),
                        str(spec.get("part", "pattern")),
                        (cx, cy),
                        (bw, bh),
                        (bx + tw / 2, by + th / 2),
                        float(response[py, px]),
                        fraction,
                        float(response[py, px]) >= threshold,
                        spec.get("subtype"),
                        bool(spec.get("enhanced", False)),
                        int(spec.get("minimum_parts", 1)),
                    )
                )
            return hits

        if self.workers == 1 or len(selected) <= 1:
            return [hit for item in selected for hit in match_one(item)]
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return [hit for group in executor.map(match_one, selected) for hit in group]

    @classmethod
    def _enhanced_color_compatible(
        cls, board: Image, center: tuple[float, float], body_size: tuple[float, float], color: str
    ) -> bool:
        """核对身体两侧和下部主色，排除红眼、嘴部与邻居造成的异色命中。"""
        cx, cy = center
        bw, bh = body_size
        x1, x2 = max(0, math.floor(cx - bw * 0.41)), min(board.shape[1], math.ceil(cx + bw * 0.41))
        y1, y2 = max(0, math.floor(cy - bh * 0.29)), min(board.shape[0], math.ceil(cy + bh * 0.42))
        hsv = cast(Image, cv2.cvtColor(board[y1:y2, x1:x2], cv2.COLOR_BGR2HSV))
        yy, xx = np.ogrid[y1:y2, x1:x2]
        selected = ((xx - cx) / (bw * 0.4)) ** 2 + ((yy - cy - bh * 0.08) / (bh * 0.33)) ** 2 <= 1
        selected |= (np.abs(xx - cx) < bw * 0.4) & (yy >= cy - bh * 0.28) & (yy < cy + bh * 0.1)
        selected &= ~((np.abs(xx - cx) < bw * 0.21) & (yy < cy + bh * 0.22))
        counts = {}
        for candidate_color in COLORS:
            mask = cls._color_mask(hsv, candidate_color) > 0
            if candidate_color == "red":
                mask &= (hsv[:, :, 0] <= 12) & (hsv[:, :, 2] >= 175)
            counts[candidate_color] = int(np.count_nonzero(mask & selected))
        return counts[color] >= np.count_nonzero(selected) * 0.06 and all(
            counts[color] > count for candidate_color, count in counts.items() if candidate_color != color
        )

    @staticmethod
    def _template_color_compatible(candidate: Image, template: Image, mask: Image) -> bool:
        """在外焰或指定模板的掩码内核对显著颜色，排除异色相似轮廓。"""
        expected = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        actual = cv2.cvtColor(candidate, cv2.COLOR_BGR2HSV)
        selected = (mask > 0) & (expected[:, :, 1] > 90) & (expected[:, :, 2] > 80)
        count = np.count_nonzero(selected)
        if count < 12:
            return True
        delta = np.abs(actual[:, :, 0].astype(np.int16) - expected[:, :, 0].astype(np.int16))
        difference = np.minimum(delta, 180 - delta)
        compatible = selected & (difference <= 15) & (actual[:, :, 1] > 55)
        return bool(np.count_nonzero(compatible) / count >= 0.55)

    def _group(self, hits: list[Hit]) -> list[list[Hit]]:
        groups: list[list[Hit]] = []
        for hit in sorted(hits, key=lambda item: (not item.accepted, -item.score)):
            for group in groups:
                first = group[0]
                limit = self.profile.merge_fraction * min(*hit.body_size, *first.body_size)
                if (
                    hit.kind == first.kind
                    and hit.color == first.color
                    and hit.subtype == first.subtype
                    and math.dist(hit.center, first.center) <= limit
                ):
                    group.append(hit)
                    break
            else:
                groups.append([hit])
        return [
            group
            for group in groups
            if len({hit.part for hit in group if hit.accepted}) >= max(hit.minimum_parts for hit in group)
        ]

    @staticmethod
    def _label(kind: str, color: str | None, level: int | None, subtype: str | None) -> str:
        """统一构造类别标签，障碍物按细分类别扩展。"""
        if kind == "ghoul":
            return f"ghoul_{color}"
        if kind == "bomb":
            return f"bomb_{level if level is not None else 'unknown'}"
        return f"{kind}_{subtype}"

    @staticmethod
    def _in_bomb(point: tuple[float, float], bomb: Hit) -> bool:
        return ((point[0] - bomb.center[0]) / (bomb.body_size[0] * 0.43)) ** 2 + (
            (point[1] - bomb.center[1]) / (bomb.body_size[1] * 0.4)
        ) ** 2 < 1

    def _object(
        self,
        group: list[Hit],
        identifier: int,
        sx: float,
        sy: float,
        source_size: tuple[int, int],
        bombs: list[list[Hit]],
    ) -> BoardObject:
        first = group[0]
        level = first.bomb_level
        quality = "identified" if len({hit.part for hit in group if hit.accepted}) > 1 else "single_feature"
        if first.kind == "bomb":
            competing = [hit for hit in group if hit.bomb_level != level]
            if competing and first.score - max(hit.score for hit in competing) < self.profile.bomb_margin:
                level = None
                quality = "level_unknown"
        position_hits = [hit for hit in group if hit.accepted]
        weights = [max(0.01, hit.score) ** 4 for hit in position_hits]
        cx = sum(hit.center[0] * weight for hit, weight in zip(position_hits, weights, strict=True)) / sum(weights)
        cy = sum(hit.center[1] * weight for hit, weight in zip(position_hits, weights, strict=True)) / sum(weights)
        rx, ry = self.profile.board_roi[:2]
        center = (round((cx + rx) * sx, 2), round((cy + ry) * sy, 2))
        bw, bh = first.body_size
        x1, y1 = max(0.0, center[0] - bw * sx / 2), max(0.0, center[1] - bh * sy / 2)
        x2, y2 = min(source_size[0], center[0] + bw * sx / 2), min(source_size[1], center[1] + bh * sy / 2)
        candidate = None
        for hit in group if first.kind in ("ghoul", "bomb") else []:
            point = hit.center if first.kind == "bomb" else hit.anchor
            if first.kind == "bomb" or not any(self._in_bomb(point, bomb[0]) for bomb in bombs):
                candidate = (round((point[0] + rx) * sx, 2), round((point[1] + ry) * sy, 2))
                break
        label = self._label(first.kind, first.color, level, first.subtype)
        enhanced = None
        if first.kind == "ghoul":
            if any(hit.enhanced and hit.accepted for hit in group):
                enhanced = True
            elif any(hit.part == "eye" and hit.accepted for hit in group):
                enhanced = False
        name = ("强化" if enhanced else "") + LABEL_NAMES[label]
        evidence = [
            {
                "template_id": hit.template_id,
                "part": hit.part,
                "score": round(hit.score, 5),
                "color_fraction": hit.color_fraction,
                "above_threshold": hit.accepted,
                "bomb_level": hit.bomb_level,
                "subtype": hit.subtype,
                "enhanced": hit.enhanced if hit.kind == "ghoul" else None,
                "anchor": [round((hit.anchor[0] + rx) * sx, 2), round((hit.anchor[1] + ry) * sy, 2)],
                "estimated_center": [round((hit.center[0] + rx) * sx, 2), round((hit.center[1] + ry) * sy, 2)],
            }
            for hit in group
        ]
        return BoardObject(
            identifier,
            first.kind,
            first.color,
            level,
            label,
            name,
            center,
            (round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)),
            {"width": round(bw * sx, 2), "height": round(bh * sy, 2), "basis": "局部模板尺度估计"},
            evidence,
            candidate,
            quality,
            first.subtype,
            enhanced,
        )

    def _unresolved(
        self,
        masks: dict[str, Image],
        groups: list[list[Hit]],
        result: BoardResult,
        sx: float,
        sy: float,
    ) -> None:
        sizes = [
            spec["body_size"][0] * spec["body_size"][1]
            for spec in self.profile.templates
            if spec["kind"] == "ghoul" and not spec.get("enhanced", False)
        ]
        if not sizes:
            return
        min_area = float(np.median(sizes)) * 0.32
        for color, mask in masks.items():
            residual = mask.copy()
            for group in groups:
                hit = group[0]
                if hit.kind == "bomb" or hit.color == color:
                    cx, cy = (round(value) for value in hit.center)
                    axes = (max(1, round(hit.body_size[0] * 0.48)), max(1, round(hit.body_size[1] * 0.48)))
                    cv2.ellipse(residual, (cx, cy), axes, 0, 0, 360, 0, -1)
            residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8))
            contours, _ = cv2.findContours(residual, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                x, y, width, height = cv2.boundingRect(contour)
                rx, ry = self.profile.board_roi[:2]
                boundary = x == 0 or y == 0 or x + width == mask.shape[1] or y + height == mask.shape[0]
                result.issues.append(
                    {
                        "code": "unresolved_color_region",
                        "color": color,
                        "touches_boundary": boundary,
                        "display_box": not boundary,
                        "box": [
                            round((x + rx) * sx, 2),
                            round((y + ry) * sy, 2),
                            round(width * sx, 2),
                            round(height * sy, 2),
                        ],
                        "message": (
                            "颜色延伸至区域边界，可能包含背景或截断对象；保留未知，不画对象候选框"
                            if boundary
                            else "仍有明显色块无法可靠解释，未推测对象数量"
                        ),
                    }
                )
