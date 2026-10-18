"""根据可追溯的裁切清单生成模板与角色忽略掩码。"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import Profile, load_json, numbers
from .image_io import Image, protect_sources, read_image, write_image, write_text_atomic


def prepare_templates(catalog_path: Path, output_dir: Path) -> Path:
    """先核对所有裁切，再生成独立于原图目录的模板配置。"""
    catalog = load_json(catalog_path)
    profile = Profile.from_dict(catalog)
    sources = [(catalog_path.parent / item["source"]).resolve() for item in profile.templates]
    protect_sources(output_dir, [catalog_path, *sources])
    prepared: list[tuple[str, Image, Image, dict[str, Any]]] = []
    identifiers: set[str] = set()
    for item, source in zip(profile.templates, sources, strict=True):
        identifier = str(item.get("id", ""))
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier.casefold() in identifiers:
            raise ValueError(f"模板编号无效或重复：{identifier}")
        identifiers.add(identifier.casefold())
        expected_hash = item.get("source_sha256")
        actual_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected_hash is not None and expected_hash != actual_hash:
            raise ValueError(f"模板原图哈希不一致：{source.name}")
        image = read_image(source)
        if (image.shape[1], image.shape[0]) != profile.reference_size:
            raise ValueError(f"模板原图尺寸与参考布局不一致：{source.name}")
        rectangle = numbers(item.get("crop"), 4, "crop")
        if any(not number.is_integer() for number in rectangle):
            raise ValueError(f"模板裁切必须为整数：{identifier}")
        x, y, width, height = map(int, rectangle)
        if min(x, y) < 0 or min(width, height) < 2 or x + width > image.shape[1] or y + height > image.shape[0]:
            raise ValueError(f"模板裁切超出原图：{identifier}")
        crop = image[y : y + height, x : x + width].copy()
        mask = np.full(crop.shape[:2], 255, dtype=np.uint8)
        polygons = item.get("include_polygons", [])
        if polygons:
            mask.fill(0)
            for polygon in polygons:
                points = np.array([numbers(point, 2, "掩码顶点") for point in polygon], dtype=np.int32)
                cv2.fillPoly(mask, [points], 255)
        for ignored in item.get("ignore_rects", []):
            rx, ry, rw, rh = map(int, numbers(ignored, 4, "ignore_rects"))
            if min(rx, ry) < 0 or min(rw, rh) <= 0 or rx + rw > width or ry + rh > height:
                raise ValueError(f"角色忽略区域超出模板：{identifier}")
            mask[ry : ry + rh, rx : rx + rw] = 0
        check_pattern(crop, mask, identifier)
        entry = {
            key: value
            for key, value in item.items()
            if key not in ("source", "source_sha256", "crop", "ignore_rects", "include_polygons")
        }
        entry.update(
            {
                "image": f"{identifier}.png",
                "mask": f"{identifier}.mask.png",
                "provenance": {
                    "source_name": source.name,
                    "source_sha256": actual_hash,
                    "source_crop": list(rectangle),
                    "ignore_rects": item.get("ignore_rects", []),
                    "include_polygons": polygons,
                },
            }
        )
        prepared.append((identifier, crop, mask, entry))
    output_dir.mkdir(parents=True, exist_ok=True)
    for identifier, crop, mask, _ in prepared:
        write_image(output_dir / f"{identifier}.png", crop)
        write_image(output_dir / f"{identifier}.mask.png", mask)
    result = dict(catalog)
    result["templates"] = [entry for _, _, _, entry in prepared]
    result["catalog_sha256"] = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    destination = output_dir / "profile.json"
    write_text_atomic(destination, json.dumps(result, ensure_ascii=False, indent=4) + "\n")
    return destination


def check_pattern(image: Image, mask: Image, name: str) -> None:
    """拒绝零掩码和没有空间变化的模板，防止无效归一化匹配。"""
    pixels = image[mask > 0]
    if len(pixels) < 8 or float(np.var(pixels.astype(np.float32), axis=0).sum()) < 1:
        raise ValueError(f"模板缺少可比较的有效纹理：{name}")
