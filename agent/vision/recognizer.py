"""组合局部模板、颜色和几何证据识别自由位置对象。"""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

from .chain_endpoint import ChainEndpointRecognizer
from .color_checks import BodyColorSupport, enhanced_color_compatible, prepare_body_color_support
from .config import Profile, load_json
from .image_io import Image, read_image
from .matching import (
    ColorGate,
    ColorIntegral,
    PreparedBoard,
    PreparedTemplate,
    color_integral,
    local_peaks,
    prepare_board,
    prepare_template,
    rectangle_count,
)
from .model import (
    COLORS,
    EXPECTED_LABELS,
    LABEL_NAMES,
    BoardObject,
    BoardResult,
    ChainEndpointResult,
    ChainMovePositions,
    Hit,
)
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
        endpoint_profile = profile_path.parent.parent / "chain_endpoint" / "profile.json"
        self.endpoint_recognizer = ChainEndpointRecognizer(endpoint_profile) if endpoint_profile.is_file() else None
        self.patterns: list[tuple[dict[str, Any], Image, Image]] = []
        self.scaled_patterns: list[tuple[dict[str, Any], float, PreparedTemplate]] = []
        self.enhanced_supports: dict[int, tuple[BodyColorSupport, int]] = {}
        for item in self.profile.templates:
            if item["kind"] in self.profile.excluded_kinds or (
                self.profile.exclude_enhanced and item.get("enhanced", False)
            ):
                continue
            image = read_image(profile_path.parent / item["image"])
            mask_image = read_image(profile_path.parent / item["mask"])
            mask = cast(Image, cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY))
            if mask.shape != image.shape[:2]:
                raise ValueError(f"模板与掩码大小不一致：{item['id']}")
            check_pattern(image, mask, item["id"])
            self.patterns.append((item, image, mask))
            for scale in item.get("scales", self.profile.scales):
                tw, th = max(2, round(image.shape[1] * scale)), max(2, round(image.shape[0] * scale))
                if tw > self.profile.board_roi[2] or th > self.profile.board_roi[3]:
                    continue
                resized = cast(Image, cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR))
                support = cv2.resize((mask > 0).astype(np.float32), (tw, th), interpolation=cv2.INTER_LINEAR)
                resized_mask = np.where(support >= 1.0 - 1e-6, 255, 0).astype(np.uint8)
                if np.count_nonzero(resized_mask) >= 8:
                    search_roi = item.get("search_roi")
                    if search_roi is None:
                        regular = item["kind"] == "bomb" or (
                            item["kind"] == "ghoul" and not item.get("enhanced", False)
                        )
                        search_roi = (self.profile.standard_object_roi if regular else None) or self.profile.board_roi
                    divisor = 2 if item["kind"] == "obstacle" else 4
                    coarse_shape = (max(2, round(search_roi[3] / divisor)), max(2, round(search_roi[2] / divisor)))
                    prepared = prepare_template(
                        resized,
                        resized_mask,
                        coarse_color=item["kind"] != "obstacle",
                        coarse_shape=None if self.full_search else coarse_shape,
                    )
                    self.scaled_patterns.append((item, scale, prepared))
                    if item.get("enhanced", False):
                        offset = tuple(float(value) * scale for value in item["center_offset"])
                        body_size = tuple(float(value) * scale for value in item["body_size"])
                        self.enhanced_supports[id(prepared)] = (
                            prepare_body_color_support((offset[0], offset[1]), (body_size[0], body_size[1])),
                            COLORS.index(item["color"]),
                        )

    def analyze(
        self, image: Image, frame_id: str, *, lightweight: bool = False, chain_move: ChainMovePositions | None = None
    ) -> BoardResult:
        """识别一帧，并将全部位置还原为输入截图坐标。"""
        started = time.perf_counter()
        height, width = image.shape[:2]
        ref_width, ref_height = self.profile.reference_size
        sx, sy = width / ref_width, height / ref_height
        rx, ry, rw, rh = self.profile.board_roi
        known = {
            self._label(item["kind"], item.get("color"), item.get("bomb_level"), item.get("subtype"))
            for item, _, _ in self.patterns
        }
        skipped_kinds = sorted(set(self.profile.excluded_kinds) | ({"obstacle"} if lightweight else set()))
        skipped_labels = sorted(
            label for label in EXPECTED_LABELS if any(label.startswith(kind + "_") for kind in skipped_kinds)
        )
        missing = sorted(set(EXPECTED_LABELS) - known - set(skipped_labels))
        result = BoardResult(
            frame_id,
            (width, height),
            (rx * sx, ry * sy, rw * sx, rh * sy),
            self.profile.name,
            {
                "missing_templates": missing,
                "recognition_mode": "lightweight" if lightweight else "full",
                "skipped_kinds": skipped_kinds,
                "skipped_labels": skipped_labels,
                "skipped_variants": ["enhanced_ghoul"] if self.profile.exclude_enhanced else [],
                "variable_ghoul_sizes": self.profile.variable_ghoul_sizes,
                "template_variants_enabled": sum(
                    1 for spec, _, _ in self.scaled_patterns if not lightweight or spec["kind"] != "obstacle"
                ),
                "sample_role": self.profile.role,
                "profile_sha256": self.profile_sha256,
                "independent_evaluation_completed": False,
                "search_mode": "full" if self.full_search else "coarse_refine",
                "color_prefilter": self.color_prefilter and not self.full_search,
                "template_variants_cached": len(self.scaled_patterns),
                "full_resolution_fallback_variants": sum(
                    1
                    for spec, _, pattern in self.scaled_patterns
                    if pattern.coarse_image is None and (not lightweight or spec["kind"] != "obstacle")
                ),
                "coarse_fourier_variants_cached": sum(
                    1
                    for spec, _, pattern in self.scaled_patterns
                    if pattern.coarse_spectrum is not None and (not lightweight or spec["kind"] != "obstacle")
                ),
                "frame_preprocessing_shared": True,
                "positions_reused_from_previous_frame": False,
                "chain_endpoint_available": self.endpoint_recognizer is not None,
                "template_workers": self.workers,
                "parallel_results_in_template_order": True,
                "standard_object_roi": self.profile.standard_object_roi or self.profile.board_roi,
                "coarse_search_parameters": None
                if self.full_search
                else {
                    "downscale": 0.25,
                    "channels": "BGR",
                    "threshold": 0.55,
                    "max_candidates": 120,
                    "refine_radius": 5,
                    "response_storage": "sparse_patches",
                    "minimum_template_fidelity": 0.8,
                    "masked_color_backend": "cached_rfft",
                    "masked_refinement_prefilter": "rectangle_projection_bound",
                    "local_peak_batching": True,
                    "obstacles": {"downscale": 0.5, "channels": "GRAY", "masked_backend": "cached_rfft"},
                },
            },
        )
        if abs((width / height) / (ref_width / ref_height) - 1) > self.profile.aspect_tolerance:
            result.scene_state = "unknown_ui"
            result.issues.append({"code": "unsupported_layout", "message": "输入长宽比与已核对布局不一致"})
            result.chain_endpoint = ChainEndpointResult(
                frame_id,
                "uncertain",
                issues=["截图布局不受支持，不能确认终点菱形"],
                move_outcome="unknown" if chain_move is not None else None,
            )
            result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
            return result
        normalized = cast(Image, cv2.resize(image, (ref_width, ref_height), interpolation=cv2.INTER_AREA))
        board = normalized[ry : ry + rh, rx : rx + rw]
        # 当前帧的独立预处理与模板任务复用同一个线程池，输入只读共享。
        executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None
        try:
            coarse_future = obstacle_future = regular_future = None
            if executor is not None and not self.full_search:
                coarse_future = executor.submit(prepare_board, board)
                if "obstacle" not in skipped_kinds:
                    obstacle_future = executor.submit(prepare_board, board, coarse_color=False)
                if self.profile.standard_object_roi is not None:
                    ox, oy, ow, oh = self.profile.standard_object_roi
                    roi = board[oy - ry : oy - ry + oh, ox - rx : ox - rx + ow]
                    regular_future = executor.submit(prepare_board, roi)
            hsv = cast(Image, cv2.cvtColor(board, cv2.COLOR_BGR2HSV))
            color_masks = {color: self._color_mask(hsv, color) for color in HSV_RANGES}
            enhanced_masks = dict(color_masks)
            enhanced_masks["red"] = cast(
                Image, cv2.bitwise_and(color_masks["red"], cv2.inRange(hsv, (0, 0, 175), (12, 255, 255)))
            )
            enhanced_palette = np.stack([enhanced_masks[color] for color in COLORS], axis=2)
            integrals = {color: color_integral(mask) for color, mask in color_masks.items()}
            coarse_board = (
                None
                if self.full_search
                else (coarse_future.result() if coarse_future is not None else prepare_board(board))
            )
            obstacle_coarse = (
                None
                if self.full_search or "obstacle" in skipped_kinds
                else (
                    obstacle_future.result()
                    if obstacle_future is not None
                    else prepare_board(board, coarse_color=False)
                )
            )
            regular_board, regular_integrals, regular_coarse = board, integrals, coarse_board
            regular_hsv = hsv
            dx, dy = 0, 0
            if self.profile.standard_object_roi is not None:
                ox, oy, ow, oh = self.profile.standard_object_roi
                dx, dy = ox - rx, oy - ry
                regular_board = board[dy : dy + oh, dx : dx + ow]
                regular_hsv = hsv[dy : dy + oh, dx : dx + ow]
                regular_integrals = {
                    color: color_integral(mask[dy : dy + oh, dx : dx + ow]) for color, mask in color_masks.items()
                }
                regular_coarse = (
                    None
                    if self.full_search
                    else (regular_future.result() if regular_future is not None else prepare_board(regular_board))
                )

            extra_kinds = tuple(kind for kind in ("obstacle", "special_ghoul") if kind not in skipped_kinds)
            searches = [
                ("bomb", regular_board, regular_integrals, "bomb", regular_coarse, False),
                ("regular", regular_board, regular_integrals, "ghoul", regular_coarse, False),
                ("enhanced", board, integrals, "ghoul", coarse_board, True),
                *(
                    (kind, board, integrals, kind, obstacle_coarse if kind == "obstacle" else coarse_board, None)
                    for kind in extra_kinds
                ),
            ]
            jobs = [
                (tag, job)
                for tag, pixels, colors, kind, coarse, enhanced in searches
                for job in self._match_jobs(
                    pixels,
                    colors,
                    kind,
                    coarse,
                    hsv=regular_hsv if tag in ("bomb", "regular") else hsv,
                    enhanced_palette=enhanced_palette,
                    enhanced=enhanced,
                )
            ]
            collected: dict[str, list[Hit]] = {tag: [] for tag, *_ in searches}

            def execute(entry: tuple[str, Callable[[], list[Hit]]]) -> list[Hit]:
                """模板任务只读取当前帧，结果仍按模板原序汇集。"""
                return entry[1]()

            if executor is None:
                outputs = [execute(job) for job in jobs]
            else:
                outputs = list(executor.map(execute, jobs))
        finally:
            if executor is not None:
                executor.shutdown()
        for (tag, _), hits in zip(jobs, outputs, strict=True):
            if tag in ("bomb", "regular"):
                for hit in hits:
                    hit.center = (hit.center[0] + dx, hit.center[1] + dy)
                    hit.anchor = (hit.anchor[0] + dx, hit.anchor[1] + dy)
            collected[tag].extend(hits)
        bomb_groups = [group for group in self._group(collected["bomb"]) if group[0].accepted]
        ghoul_hits = [*collected["regular"], *collected["enhanced"]]
        accepted = [hit for hit in ghoul_hits if not any(self._in_bomb(hit.anchor, group[0]) for group in bomb_groups)]
        extra_hits = [
            hit
            for kind in extra_kinds
            for hit in collected[kind]
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
        if self.endpoint_recognizer is not None:
            result.chain_endpoint = self.endpoint_recognizer.analyze(image, result, move=chain_move)
        else:
            result.chain_endpoint = ChainEndpointResult(
                frame_id,
                "unavailable",
                issues=["未安装终点菱形模板，不能确认当前连线终点"],
                move_outcome="unknown" if chain_move is not None else None,
            )
        result.elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return result

    @staticmethod
    def _color_mask(hsv: Image, color: str) -> Image:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for low, high in HSV_RANGES[color]:
            mask |= cast(Image, cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8)))
        return mask

    def _match_jobs(
        self,
        board: Image,
        integrals: dict[str, ColorIntegral],
        kind: str,
        coarse_board: PreparedBoard | None,
        *,
        hsv: Image,
        enhanced_palette: Image,
        enhanced: bool | None = None,
    ) -> list[Callable[[], list[Hit]]]:
        regions: dict[tuple[int, ...], tuple[Image, PreparedBoard | None]] = {}
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
                    regions[region] = (
                        cropped,
                        None if self.full_search else prepare_board(cropped, coarse_color=kind != "obstacle"),
                    )

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
            threshold = float(spec.get("threshold", 0.84))
            search_threshold = bomb_floor if kind == "bomb" else threshold
            ordered = local_peaks(search_board, pattern, self.full_search, search_coarse, gate, search_threshold)
            kept: list[tuple[int, int]] = []
            for px, py, score in ordered:
                if len(kept) >= 60:
                    break
                if any(math.hypot(px - ox, py - oy) < min(tw, th) * 0.35 for ox, oy in kept):
                    continue
                bx, by = px + offset_x, py + offset_y
                if (kind == "bomb" or spec.get("check_template_color", False)) and not self._template_color_compatible(
                    hsv[by : by + th, bx : bx + tw], pattern
                ):
                    continue
                if kind == "special_ghoul" and spec.get("part") == "eye":
                    eye = hsv[by : by + th, bx : bx + tw]
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
                if spec.get("enhanced", False):
                    support, color_index = self.enhanced_supports[id(pattern)]
                    if not enhanced_color_compatible(enhanced_palette, support, (bx, by), color_index):
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
                        score,
                        fraction,
                        score >= threshold,
                        spec.get("subtype"),
                        bool(spec.get("enhanced", False)),
                        int(spec.get("minimum_parts", 1)),
                    )
                )
            return hits

        return [partial(match_one, item) for item in selected]

    @staticmethod
    def _template_color_compatible(actual: Image, pattern: PreparedTemplate) -> bool:
        """在外焰或指定模板的掩码内核对显著颜色，排除异色相似轮廓。"""
        selected, count = pattern.colored_pixels, pattern.colored_count
        if count < 12:
            return True
        assert pattern.expected_hue is not None and selected is not None
        delta = np.abs(actual[:, :, 0].astype(np.int16) - pattern.expected_hue)
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
                    and (math.dist(hit.center, first.center) <= limit or self._same_ghoul_feature(hit, group))
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

    def _same_ghoul_feature(self, hit: Hit, group: list[Hit]) -> bool:
        """活动中同一可见部位的多尺度命中只能支持同一个对象。"""
        if not self.profile.variable_ghoul_sizes or hit.kind != "ghoul":
            return False
        first = group[0]
        if math.dist(hit.center, first.center) > self.profile.merge_fraction * max(*hit.body_size, *first.body_size):
            return False
        return any(
            other.template_id == hit.template_id
            and math.dist(hit.anchor, other.anchor) <= 0.2 * min(*hit.body_size, *other.body_size)
            for other in group
        )

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
        if self.profile.variable_ghoul_sizes and first.kind == "ghoul":
            # 小尺度嘴巴可能命中大嘴内部，保留证据但不让它拉偏身体中心。
            position_hits = [
                hit
                for hit in position_hits
                if all(0.8 <= actual / reference <= 1.2 for actual, reference in zip(hit.body_size, first.body_size))
            ]
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
        if self.profile.variable_ghoul_sizes:
            min_area = min(
                spec["body_size"][0] * spec["body_size"][1] * min(spec.get("scales", self.profile.scales)) ** 2 * 0.32
                for spec in self.profile.templates
                if spec["kind"] == "ghoul" and not spec.get("enhanced", False)
            )
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
