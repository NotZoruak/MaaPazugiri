"""校验原型配置，避免把未知布局套进已知坐标。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import COLORS, OBSTACLE_TYPES, SPECIAL_GHOUL_TYPES


def numbers(value: Any, length: int, name: str) -> tuple[float, ...]:
    """要求固定长度的有限数值数组。"""
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} 必须包含 {length} 个数值")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) for item in value):
        raise ValueError(f"{name} 含有无效数值")
    return tuple(float(item) for item in value)


def load_json(path: Path) -> dict[str, Any]:
    """读取对象形式的 UTF-8 JSON。"""
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须为 JSON 对象：{path}")
    return value


def checked_scales(value: Any) -> tuple[float, ...]:
    """统一校验配置和单个模板的有限尺度列表。"""
    if not isinstance(value, list) or not value or len(value) > 15:
        raise ValueError("搜索尺度必须为包含 1 至 15 项的列表")
    scales = numbers(value, len(value), "scales")
    if any(not 0.1 <= item <= 4 for item in scales):
        raise ValueError("搜索尺度必须位于 0.1 至 4 之间")
    return scales


@dataclass(frozen=True)
class Profile:
    """声明适用尺寸、有效区域和有限尺度搜索范围。"""

    name: str
    reference_size: tuple[int, int]
    board_roi: tuple[int, int, int, int]
    scales: tuple[float, ...]
    templates: list[dict[str, Any]]
    role: str
    aspect_tolerance: float
    merge_fraction: float
    bomb_margin: float
    standard_object_roi: tuple[int, int, int, int] | None = None
    excluded_kinds: tuple[str, ...] = ()
    exclude_enhanced: bool = False
    variable_ghoul_sizes: bool = False

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Profile:
        """先校验布局和参数，再允许读取模板。"""
        size = numbers(value.get("reference_size"), 2, "reference_size")
        roi = numbers(value.get("board_roi"), 4, "board_roi")
        if any(item <= 0 or not item.is_integer() for item in size):
            raise ValueError("参考尺寸必须为正整数")
        if any(not item.is_integer() for item in roi) or min(roi[:2]) < 0 or min(roi[2:]) <= 0:
            raise ValueError("有效区域必须使用非负起点和正整数宽高")
        if roi[0] + roi[2] > size[0] or roi[1] + roi[3] > size[1]:
            raise ValueError("有效区域超出参考图片")
        scales = checked_scales(value.get("scales", [0.9, 1.0, 1.1]))
        excluded = value.get("excluded_kinds", [])
        if not isinstance(excluded, list) or any(
            not isinstance(kind, str) or kind not in ("ghoul", "bomb", "obstacle", "special_ghoul") for kind in excluded
        ):
            raise ValueError("排除类别必须为已登记对象大类的列表")
        for key in ("exclude_enhanced", "variable_ghoul_sizes"):
            if not isinstance(value.get(key, False), bool):
                raise ValueError(f"{key} 必须为布尔值")
        templates = value.get("templates", [])
        if not isinstance(templates, list) or any(not isinstance(item, dict) for item in templates):
            raise ValueError("templates 必须为对象列表")
        parameters = [
            float(value.get(key, default))
            for key, default in (("aspect_tolerance", 0.02), ("merge_fraction", 0.28), ("bomb_margin", 0.035))
        ]
        if any(not math.isfinite(item) or not 0 <= item <= 1 for item in parameters):
            raise ValueError("布局容差、合并比例和等级差值必须位于 0 至 1")
        for item in templates:
            validate_template(item)
            if item.get("search_roi") is not None:
                area = numbers(item["search_roi"], 4, "search_roi")
                if (
                    any(not number.is_integer() for number in area)
                    or min(area[2:]) <= 0
                    or area[0] < roi[0]
                    or area[1] < roi[1]
                    or area[0] + area[2] > roi[0] + roi[2]
                    or area[1] + area[3] > roi[1] + roi[3]
                ):
                    raise ValueError("固定道具的搜索区域必须位于完整有效区域内")
        standard_roi = value.get("standard_object_roi")
        checked_roi = None
        if standard_roi is not None:
            checked = numbers(standard_roi, 4, "standard_object_roi")
            if (
                any(not item.is_integer() for item in checked)
                or min(checked[2:]) <= 0
                or checked[0] < roi[0]
                or checked[1] < roi[1]
                or checked[0] + checked[2] > roi[0] + roi[2]
                or checked[1] + checked[3] > roi[1] + roi[3]
            ):
                raise ValueError("普通饿鬼与炸弹的搜索区域必须位于完整有效区域内")
            checked_roi = (int(checked[0]), int(checked[1]), int(checked[2]), int(checked[3]))
        return cls(
            str(value.get("name", "离线开发配置")),
            (int(size[0]), int(size[1])),
            (int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])),
            scales,
            templates,
            str(value.get("sample_role", "development")),
            parameters[0],
            parameters[1],
            parameters[2],
            standard_object_roi=checked_roi,
            excluded_kinds=tuple(sorted(set(excluded))),
            exclude_enhanced=value.get("exclude_enhanced", False),
            variable_ghoul_sizes=value.get("variable_ghoul_sizes", False),
        )


def validate_template(item: dict[str, Any]) -> None:
    """拒绝未定义颜色、等级和不合理的模板参数。"""
    if "scales" in item:
        checked_scales(item["scales"])
    if not isinstance(item.get("check_template_color", False), bool):
        raise ValueError("模板颜色核对开关必须为布尔值")
    if not isinstance(item.get("enhanced", False), bool):
        raise ValueError("强化模板标记必须为布尔值")
    if item.get("enhanced", False) and item.get("kind") != "ghoul":
        raise ValueError("强化标记仅适用于普通四色饿鬼模板")
    minimum_parts = item.get("minimum_parts", 1)
    if isinstance(minimum_parts, bool) or not isinstance(minimum_parts, int) or not 1 <= minimum_parts <= 8:
        raise ValueError("最少可见部位数必须为1至8的整数")
    fixed_bell = item.get("kind") == "obstacle" and item.get("subtype") == "bronze_bell"
    if item.get("search_roi") is not None and not fixed_bell:
        raise ValueError("固定搜索区域仅适用于已确认固定位置的青铜钟")
    if fixed_bell and item.get("search_roi") is None:
        raise ValueError("青铜钟模板必须指定已核对的固定搜索区域")
    if item.get("kind") == "ghoul":
        if item.get("color") not in COLORS:
            raise ValueError("饿鬼模板必须指定四种已确认颜色之一")
    elif item.get("kind") == "bomb":
        if item.get("bomb_level") not in (1, 2, 3):
            raise ValueError("持久炸弹模板只允许一至三级")
    elif item.get("kind") == "obstacle":
        if item.get("subtype") not in OBSTACLE_TYPES:
            raise ValueError("障碍物模板必须指定已登记的细分类别")
        if item.get("color") is not None or item.get("bomb_level") is not None:
            raise ValueError("障碍物模板不能指定饿鬼颜色或炸弹等级")
    elif item.get("kind") == "special_ghoul":
        if item.get("subtype") not in SPECIAL_GHOUL_TYPES or item.get("color") != "black":
            raise ValueError("特殊饿鬼模板必须指定已登记的黑色类别")
        if item.get("bomb_level") is not None:
            raise ValueError("特殊饿鬼模板不能指定炸弹等级")
    else:
        raise ValueError("模板类别只允许饿鬼、炸弹、障碍物或特殊饿鬼")
    numbers(item.get("center_offset"), 2, "center_offset")
    body = numbers(item.get("body_size"), 2, "body_size")
    if min(body) <= 0:
        raise ValueError("对象主体尺寸必须为正数")
    threshold = float(item.get("threshold", 0.84))
    color_fraction = float(item.get("min_color_fraction", 0.3))
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("模板阈值必须大于 0 且不超过 1")
    if not math.isfinite(color_fraction) or not 0 <= color_fraction <= 1:
        raise ValueError("颜色占比必须位于 0 至 1")
