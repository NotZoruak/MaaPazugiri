"""以字节编解码方式处理 Windows 中文图片路径。"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]


def read_image(path: Path) -> Image:
    """读取三通道 BGR 图片，失败时给出明确异常。"""
    image = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"无法读取图片：{path}")
    return cast(Image, image)


def write_image(path: Path, image: Image) -> None:
    """写入 PNG 派生图，原图保护由调用方预先检查。"""
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"无法编码图片：{path}")
    write_bytes_atomic(path, encoded.tobytes())


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """原子替换派生文件，避免沿已有符号链接或硬链接改写原图。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".board-vision-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    """以 UTF-8 原子替换文本报告。"""
    write_bytes_atomic(path, text.encode("utf-8"))


def protect_sources(output_dir: Path, sources: list[Path]) -> None:
    """禁止把派生输出放入任何原始输入图片目录。"""
    output = output_dir.resolve()
    for source in sources:
        if output.is_relative_to(source.resolve().parent):
            raise ValueError(f"输出目录不能位于原始图片目录内：{output_dir}")
