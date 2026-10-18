"""缓存静态模板，并在当前帧内用保守颜色筛选减少精细匹配。"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from .image_io import Image

ScoreMap = NDArray[np.float32]
ColorIntegral = NDArray[np.int32]
Frequency = NDArray[np.complex64]


@dataclass(frozen=True)
class CoarseSpectrum:
    """缓存静态掩码及去均值模板的频谱，不含截图数据。"""

    shape: tuple[int, int]
    targets: Frequency
    support: Frequency
    count: float
    norm: float


@dataclass(frozen=True)
class PreparedBoard:
    """当前帧缩图与共享频谱；每次识别重新创建。"""

    image: Image
    shape: tuple[int, int] | None = None
    spectra: Frequency | None = None
    squares: Frequency | None = None


@dataclass(frozen=True)
class RectangleProjection:
    """静态掩码内的实心矩形及其保留的去均值能量。"""

    bounds: tuple[int, int, int, int]
    image: Image
    retained_energy: float


@dataclass(frozen=True)
class PreparedTemplate:
    """保存模板派生数据，不保存截图或棋盘对象位置。"""

    image: Image
    mask: Image
    full_mask: bool
    coarse_image: Image | None = None
    coarse_mask: Image | None = None
    coarse_full_mask: bool = False
    expected_hue: NDArray[np.int16] | None = None
    colored_pixels: NDArray[np.bool_] | None = None
    colored_count: int = 0
    coarse_spectrum: CoarseSpectrum | None = None
    projection: RectangleProjection | None = None


def _prepare_projection(template: Image, mask: Image) -> RectangleProjection | None:
    """加载时选择掩码内保留最多去均值能量的矩形，避免只选大片常量。"""
    heights = [0] * mask.shape[1]
    rectangles: list[tuple[int, int, int, int]] = []
    for y, row in enumerate(mask):
        heights = [height + 1 if value else 0 for height, value in zip(heights, row, strict=True)]
        stack: list[tuple[int, int]] = []
        for x, height in enumerate([*heights, 0]):
            start = x
            while stack and stack[-1][1] > height:
                left, block_height = stack.pop()
                start = left
                if min(x - left, block_height) >= 2:
                    rectangles.append((left, y - block_height + 1, x, y + 1))
            if height and (not stack or stack[-1][1] < height):
                stack.append((start, height))
    if not rectangles:
        return None
    lefts, tops, rights, bottoms = np.array(rectangles, dtype=np.intp).T
    counts = (rights - lefts) * (bottoms - tops)
    sums, squares = cv2.integral2(template, sdepth=cv2.CV_64F, sqdepth=cv2.CV_64F)
    values = sums[bottoms, rights] - sums[tops, rights] - sums[bottoms, lefts] + sums[tops, lefts]
    squares = squares.sum(axis=2)
    energies = squares[bottoms, rights] - squares[tops, rights] - squares[bottoms, lefts] + squares[tops, lefts]
    energies -= np.einsum("rc,rc->r", values, values) / counts
    selected = int(np.argmax(energies))
    bounds = rectangles[selected]
    left, top, right, bottom = bounds
    block = template[top:bottom, left:right].copy()
    _, full_std = cv2.meanStdDev(template, mask=mask)
    full_energy = float(np.sum(full_std * full_std)) * cv2.countNonZero(mask)
    block_energy = float(energies[selected])
    if full_energy <= 1e-8 or block_energy <= 1e-8:
        return None
    return RectangleProjection(bounds, block, min(1.0, block_energy / full_energy))


def prepare_template(
    template: Image,
    mask: Image,
    *,
    coarse_color: bool = True,
    coarse_shape: tuple[int, int] | None = None,
) -> PreparedTemplate:
    """加载时生成颜色缩图与有效掩码，提前排除异色相似轮廓。"""
    full_mask = bool(np.all(mask == 255))
    hsv = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
    selected = (mask > 0) & (hsv[:, :, 1] > 90) & (hsv[:, :, 2] > 80)
    prepared = PreparedTemplate(
        template,
        mask,
        full_mask,
        expected_hue=hsv[:, :, 0].astype(np.int16),
        colored_pixels=selected,
        colored_count=int(np.count_nonzero(selected)),
        projection=_prepare_projection(template, mask) if not full_mask else None,
    )
    if min(template.shape[:2]) < 20:
        return prepared
    height, width = template.shape[:2]
    divisor = 4 if coarse_color else 2
    size = (max(2, round(width / divisor)), max(2, round(height / divisor)))
    source = template if coarse_color else cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    small_template = cast(Image, cv2.resize(source, size))
    support = cv2.resize((mask > 0).astype(np.float32), size)
    small_mask = np.where(support >= 1.0 - 1e-6, 255, 0).astype(np.uint8)
    valid = small_template[small_mask > 0]
    if valid.size < 8 or float(np.var(valid.astype(np.float32))) < 1:
        return prepared
    if coarse_color:
        # 细纹理缩图可能随像素相位消失，静态核对后为这类模板保留原图搜索。
        restored = cv2.resize(small_template, (width, height))
        original = template[mask > 0].astype(np.float32)
        reconstructed = restored[mask > 0].astype(np.float32)
        original -= original.mean(axis=0)
        reconstructed -= reconstructed.mean(axis=0)
        energy = float(np.sum(original * original) * np.sum(reconstructed * reconstructed))
        fidelity = float(np.sum(original * reconstructed)) / math.sqrt(energy) if energy > 0 else 0.0
        if fidelity < 0.8:
            return prepared
    prepared = replace(
        prepared,
        coarse_image=small_template,
        coarse_mask=small_mask,
        coarse_full_mask=bool(np.all(small_mask == 255)),
    )
    if (
        not prepared.coarse_full_mask
        and coarse_shape is not None
        and small_template.shape[0] <= coarse_shape[0]
        and small_template.shape[1] <= coarse_shape[1]
    ):
        shape = (cv2.getOptimalDFTSize(coarse_shape[0]), cv2.getOptimalDFTSize(coarse_shape[1]))
        support = (small_mask > 0).astype(np.float32)
        count = float(support.sum())
        values = small_template.astype(np.float32)
        channels = values.transpose(2, 0, 1) if values.ndim == 3 else values[None, ...]
        means = np.sum(channels * support[None, ...], axis=(1, 2)) / count
        centered = (channels - means[:, None, None]) * support[None, ...]
        norm = float(np.sum(centered * centered))
        spectrum = CoarseSpectrum(
            shape,
            cast(Frequency, np.conj(np.fft.rfft2(centered, s=shape))),
            cast(Frequency, np.conj(np.fft.rfft2(support, s=shape))),
            count,
            norm,
        )
        prepared = replace(prepared, coarse_spectrum=spectrum)
    return prepared


def prepare_board(board: Image, *, coarse_color: bool = True) -> PreparedBoard:
    """每帧独立生成一次颜色缩图，仅在该帧的模板间共享。"""
    height, width = board.shape[:2]
    divisor = 4 if coarse_color else 2
    source = board if coarse_color else cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    image = cast(Image, cv2.resize(source, (max(2, round(width / divisor)), max(2, round(height / divisor)))))
    shape = (cv2.getOptimalDFTSize(image.shape[0]), cv2.getOptimalDFTSize(image.shape[1]))
    values = image.astype(np.float32)
    channels = values.transpose(2, 0, 1) if values.ndim == 3 else values[None, ...]
    return PreparedBoard(
        image,
        shape,
        np.fft.rfft2(channels, s=shape),
        np.fft.rfft2(np.sum(channels * channels, axis=0), s=shape),
    )


def coarse_response(board: PreparedBoard, pattern: PreparedTemplate) -> ScoreMap:
    """共享同帧频谱计算掩码相关系数，原图精查仍走原匹配实现。"""
    assert pattern.coarse_image is not None and pattern.coarse_mask is not None
    cached = pattern.coarse_spectrum
    if cached is None or cached.shape != board.shape or board.spectra is None or board.squares is None:
        return exact_response(
            board.image, pattern.coarse_image, pattern.coarse_mask, full_mask=pattern.coarse_full_mask
        )
    numerator = np.einsum("chw,chw->hw", board.spectra, cached.targets)
    products = np.concatenate(
        (
            numerator[None, ...],
            board.spectra * cached.support[None, ...],
            (board.squares * cached.support)[None, ...],
        )
    )
    # 二值掩码下，方差为像素平方和减去各通道均值项；合并逆变换减少调用次数。
    result = np.fft.irfft2(products, s=cached.shape)
    variance = result[-1] - np.einsum("chw,chw->hw", result[1:-1], result[1:-1]) / cached.count
    denominator = np.sqrt(np.maximum(variance, 0) * cached.norm)
    response = np.full(cached.shape, -1.0, dtype=np.float32)
    np.divide(result[0], denominator, out=response, where=denominator > 1e-4)
    height = board.image.shape[0] - pattern.coarse_image.shape[0] + 1
    width = board.image.shape[1] - pattern.coarse_image.shape[1] + 1
    return response[:height, :width]


def color_integral(mask: Image) -> ColorIntegral:
    """为当前帧建立颜色积分图，区域计数只需四次数组读取。"""
    return cast(ColorIntegral, cv2.integral((mask > 0).astype(np.uint8), sdepth=cv2.CV_32S))


def rectangle_count(integral: ColorIntegral, x1: int, y1: int, x2: int, y2: int) -> int:
    """统计左闭右开区域的颜色像素数。"""
    if x1 >= x2 or y1 >= y2:
        return 0
    return int(integral[y2, x2]) - int(integral[y1, x2]) - int(integral[y2, x1]) + int(integral[y1, x1])


@dataclass(frozen=True)
class ColorGate:
    """仅拒绝整个精细搜索窗口均不可能通过颜色条件的候选。"""

    integral: ColorIntegral
    offset: tuple[float, float]
    body_size: tuple[float, float]
    minimum_fraction: float

    def allows(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """以颜色数量上界和主体面积下界保守放行，并覆盖极值邻域。"""
        if self.minimum_fraction <= 0:
            return True
        height, width = self.integral.shape[0] - 1, self.integral.shape[1] - 1
        # 扩展两像素，保留可能抑制邻近有效峰值的响应。
        low_x = max(0.0, x1 - 2 + self.offset[0])
        low_y = max(0.0, y1 - 2 + self.offset[1])
        high_x = min(width - 1e-9, x2 + 1 + self.offset[0])
        high_y = min(height - 1e-9, y2 + 1 + self.offset[1])
        if low_x > high_x or low_y > high_y:
            return False
        bw, bh = self.body_size
        left, top = max(0, math.floor(low_x - bw / 2) - 1), max(0, math.floor(low_y - bh / 2) - 1)
        right, bottom = min(width, math.ceil(high_x + bw / 2) + 1), min(height, math.ceil(high_y + bh / 2) + 1)
        count_upper = rectangle_count(self.integral, left, top, right, bottom)
        # 两端取整与边缘裁切只会进一步放宽这个面积下界。
        min_width = max(1.0, min(bw, width, low_x + bw / 2, width - high_x + bw / 2) - 2)
        min_height = max(1.0, min(bh, height, low_y + bh / 2, height - high_y + bh / 2) - 2)
        return count_upper >= self.minimum_fraction * min_width * min_height


def exact_response(image: Image, template: Image, mask: Image, *, full_mask: bool | None = None) -> ScoreMap:
    """全白掩码走快速分支，其余掩码保持有效像素约束。"""
    complete = bool(np.all(mask == 255)) if full_mask is None else full_mask
    if complete:
        return cast(ScoreMap, cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED))
    response = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    if cv2.checkRange(response, quiet=True, minVal=-1e30, maxVal=1e30)[0]:
        return cast(ScoreMap, response)
    return cast(ScoreMap, np.nan_to_num(response, copy=False, nan=-1.0, posinf=-1.0, neginf=-1.0))


def projection_may_reach(image: Image, pattern: PreparedTemplate, minimum_score: float) -> bool:
    """仅在矩形投影证明整个窗口不可能达标时排除精查。"""
    projection = pattern.projection
    if projection is None or not 0 < minimum_score <= 1:
        return True
    retained = projection.retained_energy
    remaining = 1.0 - retained
    if retained <= 1e-8 or minimum_score * minimum_score <= remaining:
        return True
    height = image.shape[0] - pattern.image.shape[0] + 1
    width = image.shape[1] - pattern.image.shape[1] + 1
    if height < 1 or width < 1:
        return True
    left, top, right, bottom = projection.bounds
    scores = cv2.matchTemplate(
        image[top : bottom + height - 1, left : right + width - 1], projection.image, cv2.TM_CCOEFF_NORMED
    )
    # 矩形内逐通道去均值是正交投影；完整相关系数的平方上界为
    # 1-a+a*max(c,0)^2，其中a为模板保留能量，c为矩形相关系数。
    # 给原生浮点计算留出余量，临界窗口继续使用完整掩码确认。
    limit = math.sqrt((minimum_score * minimum_score - remaining) / retained) - 1e-3
    return float(np.max(scores)) >= limit


def local_response(
    board: Image,
    pattern: PreparedTemplate,
    full_search: bool,
    coarse_board: PreparedBoard | None,
    gate: ColorGate | None = None,
) -> ScoreMap:
    """在当前帧原图中精细匹配，不使用其他帧的位置信息。"""
    template, mask = pattern.image, pattern.mask
    if full_search or pattern.coarse_image is None or pattern.coarse_mask is None:
        return exact_response(board, template, mask, full_mask=pattern.full_mask)
    height, width = template.shape[:2]
    response = np.full((board.shape[0] - height + 1, board.shape[1] - width + 1), -1.0, dtype=np.float32)
    for x1, y1, refined in _response_patches(board, pattern, coarse_board, gate):
        y2, x2 = y1 + refined.shape[0], x1 + refined.shape[1]
        response[y1:y2, x1:x2] = np.maximum(response[y1:y2, x1:x2], refined)
    return response


def _response_patches(
    board: Image,
    pattern: PreparedTemplate,
    coarse_board: PreparedBoard | None,
    gate: ColorGate | None,
    minimum_score: float | None = None,
) -> list[tuple[int, int, ScoreMap]]:
    """只保存当前帧候选窗口的精查分数，避免反复分配整幅响应图。"""
    template, mask = pattern.image, pattern.mask
    assert pattern.coarse_image is not None and pattern.coarse_mask is not None
    small_board = (
        prepare_board(board, coarse_color=pattern.coarse_image.ndim == 3) if coarse_board is None else coarse_board
    )
    coarse = coarse_response(small_board, pattern)
    peaks = cv2.dilate(coarse, np.ones((3, 3), dtype=np.uint8))
    indices = np.flatnonzero((coarse >= 0.55) & (coarse >= peaks - 1e-7))
    indices = indices[np.argsort(-coarse.ravel()[indices], kind="stable")[:120]]
    ys, xs = np.divmod(indices, coarse.shape[1])
    height, width = template.shape[:2]
    board_height, board_width = board.shape[:2]
    response_width, response_height = board_width - width + 1, board_height - height + 1
    patches = []
    sx, sy = board_width / small_board.image.shape[1], board_height / small_board.image.shape[0]
    for px, py in zip(xs.tolist(), ys.tolist(), strict=True):
        cx, cy = round(px * sx), round(py * sy)
        x1, y1 = max(0, cx - 5), max(0, cy - 5)
        x2, y2 = min(response_width, cx + 6), min(response_height, cy + 6)
        if x1 >= x2 or y1 >= y2 or (gate is not None and not gate.allows(x1, y1, x2, y2)):
            continue
        window = board[y1 : y2 + height - 1, x1 : x2 + width - 1]
        if minimum_score is not None and not projection_may_reach(window, pattern, minimum_score):
            continue
        refined = exact_response(window, template, mask, full_mask=pattern.full_mask)
        patches.append((x1, y1, refined))
    return patches


def local_peaks(
    board: Image,
    pattern: PreparedTemplate,
    full_search: bool,
    coarse_board: PreparedBoard | None,
    gate: ColorGate | None,
    minimum_score: float,
) -> list[tuple[int, int, float]]:
    """在稀疏窗口内找峰值，并核对重叠窗口及窗口外邻域。"""
    kernel = np.ones((5, 5), dtype=np.uint8)
    if full_search or pattern.coarse_image is None or pattern.coarse_mask is None:
        response = exact_response(board, pattern.image, pattern.mask, full_mask=pattern.full_mask)
        maximum = cv2.dilate(response, kernel)
        ys, xs = np.where((response >= minimum_score) & (response >= maximum - 1e-7))
        return sorted(
            [(int(x), int(y), float(response[y, x])) for y, x in zip(ys, xs, strict=True)],
            key=lambda peak: (-peak[2], peak[1], peak[0]),
        )
    patches = _response_patches(board, pattern, coarse_board, gate, minimum_score)
    candidates: dict[tuple[int, int], float] = {}
    by_shape: dict[tuple[int, int], list[tuple[int, int, ScoreMap]]] = {}
    for x, y, response in patches:
        by_shape.setdefault(response.shape, []).append((x, y, response))
    for group in by_shape.values():
        responses = np.stack([entry[2] for entry in group], axis=2)
        maximum = cv2.dilate(responses, kernel).reshape(responses.shape)
        ys, xs, layers = np.where((responses >= minimum_score) & (responses >= maximum - 1e-7))
        for py, px, layer in zip(ys.tolist(), xs.tolist(), layers.tolist(), strict=True):
            x, y, response = group[layer]
            point = (px + x, py + y)
            candidates[point] = max(candidates.get(point, -1.0), float(response[py, px]))
    if not candidates:
        return []
    bounds = np.array([(x, y, x + scores.shape[1], y + scores.shape[0]) for x, y, scores in patches])
    peaks = []
    for (px, py), score in candidates.items():
        nearby = np.flatnonzero(
            (bounds[:, 0] <= px + 2) & (bounds[:, 1] <= py + 2) & (bounds[:, 2] > px - 2) & (bounds[:, 3] > py - 2)
        )
        for index in nearby:
            x, y, response = patches[int(index)]
            neighborhood = response[max(0, py - 2 - y) : py + 3 - y, max(0, px - 2 - x) : px + 3 - x]
            if score < float(np.max(neighborhood) - np.float32(1e-7)):
                break
        else:
            peaks.append((px, py, score))
    return sorted(peaks, key=lambda peak: (-peak[2], peak[1], peak[0]))
