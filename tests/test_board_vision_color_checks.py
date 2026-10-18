"""核对静态身体形状缓存与原逐像素颜色规则一致。"""

from __future__ import annotations

import math
import unittest

import numpy as np

from agent.vision.color_checks import enhanced_color_compatible, prepare_body_color_support


class BoardColorChecksTests(unittest.TestCase):
    """覆盖不同缩放、位置相位和棋盘边缘裁切。"""

    def test_cached_shape_preserves_current_frame_color_decisions(self) -> None:
        """静态相对形状可复用，当前帧的颜色与位置必须每次重查。"""
        rng = np.random.default_rng(728)
        height, width = 193, 211
        palette = (rng.random((height, width, 4)) > 0.72).astype(np.uint8) * 255
        for scale in (0.9, 1.0, 1.1):
            for offset in ((38, 47), (19, 35), (32, 49)):
                center_offset = (offset[0] * scale, offset[1] * scale)
                bw, bh = 96 * scale, 104 * scale
                support = prepare_body_color_support(center_offset, (bw, bh))
                origins = [(-60, -70), (-20, -15), (0, 0), (width - 40, height - 50), (width, height)]
                origins.extend(tuple(map(int, point)) for point in rng.integers(-60, 180, size=(30, 2)))
                for bx, by in origins:
                    cx, cy = bx + center_offset[0], by + center_offset[1]
                    x1, x2 = max(0, math.floor(cx - bw * 0.41)), min(width, math.ceil(cx + bw * 0.41))
                    y1, y2 = max(0, math.floor(cy - bh * 0.29)), min(height, math.ceil(cy + bh * 0.42))
                    if x1 >= x2 or y1 >= y2:
                        for color in range(4):
                            self.assertFalse(enhanced_color_compatible(palette, support, (bx, by), color))
                        continue
                    yy, xx = np.ogrid[y1:y2, x1:x2]
                    selected = ((xx - cx) / (bw * 0.4)) ** 2 + ((yy - cy - bh * 0.08) / (bh * 0.33)) ** 2 <= 1
                    selected |= (np.abs(xx - cx) < bw * 0.4) & (yy >= cy - bh * 0.28) & (yy < cy + bh * 0.1)
                    selected &= ~((np.abs(xx - cx) < bw * 0.21) & (yy < cy + bh * 0.22))
                    counts = [
                        int(np.count_nonzero((palette[y1:y2, x1:x2, color] > 0) & selected)) for color in range(4)
                    ]
                    area = int(np.count_nonzero(selected))
                    for color in range(4):
                        expected = counts[color] >= area * 0.06 and all(
                            counts[color] > count for index, count in enumerate(counts) if index != color
                        )
                        self.assertEqual(enhanced_color_compatible(palette, support, (bx, by), color), expected)
