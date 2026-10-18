"""核对稀疏精查边界和细纹理降采样保护。"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from agent.vision.matching import (
    coarse_response,
    exact_response,
    local_peaks,
    local_response,
    prepare_board,
    prepare_template,
    projection_may_reach,
)


class BoardMatchingTests(unittest.TestCase):
    """比较可观察峰值，覆盖边缘、重复对象及掩码。"""

    def test_rectangle_projection_never_discards_a_matching_window(self) -> None:
        """子区域上界不得排除完整掩码匹配，包括通道亮度变化。"""
        rng = np.random.default_rng(921)
        template = rng.integers(40, 190, (42, 48, 3), dtype=np.uint8)
        mask = np.full(template.shape[:2], 255, dtype=np.uint8)
        mask[:12, :9] = 0
        mask[-9:, -12:] = 0
        pattern = prepare_template(template, mask)
        self.assertIsNotNone(pattern.projection)
        self.assertIsNotNone(prepare_template(template, mask, coarse_color=False).projection)
        rejected = 0
        for index in range(24):
            image = rng.integers(20, 230, (52, 58, 3), dtype=np.uint8)
            if index % 3 == 0:
                shifted = (template.astype(np.int16) + np.array([11, -8, 17])).astype(np.uint8)
                image[5:47, 4:52] = shifted
                hidden = image[5:47, 4:52]
                hidden[mask == 0] = rng.integers(0, 256, (np.count_nonzero(mask == 0), 3), dtype=np.uint8)
            scores = exact_response(image, template, mask, full_mask=False)
            for threshold in (0.75, 0.85, 0.99, 1.0):
                allowed = projection_may_reach(image, pattern, threshold)
                if float(scores.max()) >= threshold - 1e-6:
                    self.assertTrue(allowed)
                if not allowed:
                    self.assertLess(float(scores.max()), threshold)
                    rejected += 1
        self.assertGreater(rejected, 0)

    def test_rectangle_projection_falls_back_for_full_or_constant_templates(self) -> None:
        """不适用的静态模板和非正阈值应交回完整匹配。"""
        template = np.full((30, 32, 3), 90, dtype=np.uint8)
        mask = np.full(template.shape[:2], 255, dtype=np.uint8)
        full = prepare_template(template, mask)
        self.assertIsNone(full.projection)
        self.assertTrue(projection_may_reach(template, full, 0.9))
        mask[:5, :5] = 0
        constant = prepare_template(template, mask)
        self.assertIsNone(constant.projection)
        self.assertTrue(projection_may_reach(template, constant, 0.9))

    def test_rectangle_projection_prefers_information_over_constant_area(self) -> None:
        """大片常量区域不能挤掉较小但含有识别信息的有效区域。"""
        rng = np.random.default_rng(502)
        template = np.full((60, 80, 3), 90, dtype=np.uint8)
        mask = np.zeros(template.shape[:2], dtype=np.uint8)
        mask[:55, :45] = 255
        mask[10:50, 55:80] = 255
        template[10:50, 55:80] = rng.integers(40, 180, (40, 25, 3), dtype=np.uint8)
        pattern = prepare_template(template, mask)
        self.assertIsNotNone(pattern.projection)
        assert pattern.projection is not None
        self.assertGreaterEqual(pattern.projection.bounds[0], 55)
        self.assertGreater(pattern.projection.retained_energy, 0.5)

    def test_gray_shared_fourier_matches_native_masked_correlation(self) -> None:
        """障碍物灰度匹配共享频谱，仍须符合原生单通道掩码NCC。"""
        rng = np.random.default_rng(818)
        board = cv2.GaussianBlur(rng.integers(0, 256, (144, 192, 3), dtype=np.uint8), (7, 7), 0)
        template = board[41:101, 63:127].copy()
        mask = np.full(template.shape[:2], 255, dtype=np.uint8)
        mask[18:36, 17:45] = 0
        frame = prepare_board(board, coarse_color=False)
        pattern = prepare_template(template, mask, coarse_color=False, coarse_shape=frame.image.shape[:2])
        self.assertIsNotNone(frame.spectra)
        self.assertIsNotNone(pattern.coarse_spectrum)
        self.assertIsNotNone(pattern.coarse_image)
        self.assertIsNotNone(pattern.coarse_mask)
        assert pattern.coarse_image is not None and pattern.coarse_mask is not None
        expected = exact_response(frame.image, pattern.coarse_image, pattern.coarse_mask, full_mask=False)
        np.testing.assert_allclose(coarse_response(frame, pattern), expected, atol=2e-4, rtol=2e-4)

    def test_sparse_peaks_preserve_dense_neighborhood_suppression(self) -> None:
        """窗口之间的邻域抑制应与完整响应图相同。"""
        rng = np.random.default_rng(42)
        template = cv2.GaussianBlur(rng.integers(0, 255, (60, 64, 3), dtype=np.uint8), (9, 9), 0)
        mask = np.full(template.shape[:2], 255, dtype=np.uint8)
        mask[:10, :10] = 0
        pattern = prepare_template(template, mask)
        self.assertIsNotNone(pattern.coarse_image)
        board = cv2.GaussianBlur(rng.integers(0, 255, (177, 231, 3), dtype=np.uint8), (9, 9), 0)
        for x, y in ((0, 0), (80, 65), (167, 117)):
            board[y : y + 60, x : x + 64] = template
        coarse = prepare_board(board)
        response = local_response(board, pattern, False, coarse)
        maximum = cv2.dilate(response, np.ones((5, 5), dtype=np.uint8))
        ys, xs = np.where((response >= 0.8) & (response >= maximum - 1e-7))
        expected = sorted(
            [(int(x), int(y), float(response[y, x])) for y, x in zip(ys, xs, strict=True)],
            key=lambda peak: (-peak[2], peak[1], peak[0]),
        )
        actual = local_peaks(board, pattern, False, coarse, None, 0.8)
        self.assertEqual(actual, expected)
        self.assertEqual({(x, y) for x, y, score in actual if score > 0.99}, {(0, 0), (80, 65), (167, 117)})

    def test_fine_texture_and_hidden_content_do_not_disappear(self) -> None:
        """缩图丢失的细纹理须保留；掩码内排除区域变化不能影响定位。"""
        rng = np.random.default_rng(17)
        template = rng.integers(20, 230, (24, 24, 3), dtype=np.uint8)
        mask = np.full((24, 24), 255, dtype=np.uint8)
        mask[7:17, 7:17] = 0
        pattern = prepare_template(template, mask)
        for point in ((9, 7), (102, 81)):
            board = np.zeros((120, 160, 3), dtype=np.uint8)
            x, y = point
            board[y : y + 24, x : x + 24] = template
            board[y + 7 : y + 17, x + 7 : x + 17] = (240, 20, 160)
            peaks = local_peaks(board, pattern, False, prepare_board(board), None, 0.99)
            self.assertEqual([(px, py) for px, py, _ in peaks], [point])

    def test_shared_fourier_matches_native_masked_correlation(self) -> None:
        """频谱只共享当前帧；移动对象与尺寸回退均须符合原生掩码计算。"""
        rng = np.random.default_rng(81)
        first = cv2.GaussianBlur(rng.integers(0, 255, (180, 232, 3), dtype=np.uint8), (15, 15), 0)
        template = first[56:116, 72:136].copy()
        mask = np.full((60, 64), 255, dtype=np.uint8)
        mask[16:34, 18:42] = 0
        frame = prepare_board(first)
        pattern = prepare_template(template, mask, coarse_shape=frame.image.shape[:2])
        self.assertIsNotNone(pattern.coarse_spectrum)
        self.assertIsNotNone(pattern.coarse_image)
        self.assertIsNotNone(pattern.coarse_mask)
        for height, width, x, y in ((180, 232, 72, 56), (180, 232, 144, 112), (184, 236, 144, 112)):
            image = cv2.GaussianBlur(rng.integers(0, 255, (height, width, 3), dtype=np.uint8), (15, 15), 0)
            image[y : y + 60, x : x + 64] = template
            current = prepare_board(image)
            actual = coarse_response(current, pattern)
            expected = exact_response(current.image, pattern.coarse_image, pattern.coarse_mask, full_mask=False)
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-4)
            self.assertEqual(cv2.minMaxLoc(actual)[3], (x // 4, y // 4))
        uniform = coarse_response(prepare_board(np.full_like(first, 120)), pattern)
        self.assertTrue(np.all(np.isfinite(uniform)))
        self.assertLess(float(np.max(uniform)), 0.55)


if __name__ == "__main__":
    unittest.main()
