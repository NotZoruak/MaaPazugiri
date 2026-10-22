"""验证选关时先按横向递增关系排除不完整编号。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from maa.define import OCRResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agent"))

from custom.action.find_target_stage import FindTargetStageAction  # noqa: E402


def ocr_result(stage: int, x: int, y: int = 350) -> OCRResult:
    """构造当前截图中一个真实类型的 OCR 结果。"""
    return OCRResult(box=[x, y, 70, 24], score=0.99, text=f"3-{stage}")


class FindTargetStageTests(unittest.TestCase):
    """以无序 OCR 输出回放横向关卡关系与实际选关分支。"""

    def recognize(self, results: list[OCRResult]):
        """仅替代外部 OCR，保留真实解析和过滤流程。"""
        context = SimpleNamespace(run_recognition=lambda *args: SimpleNamespace(all_results=results))
        return FindTargetStageAction()._recognize_stages(context, np.zeros((1280, 720, 3), dtype=np.uint8))

    def test_clipped_right_label_does_not_expand_visible_range(self) -> None:
        """回放 16:13 的编号和位置，右端残缺的 3-3 不参与范围判断。"""
        stages = self.recognize(
            [
                ocr_result(33, 509, 330),
                ocr_result(3, 676, 396),
                ocr_result(30, 8, 396),
                ocr_result(31, 176, 396),
                ocr_result(32, 340, 460),
            ]
        )
        self.assertEqual([stage.stage for stage in stages], [30, 31, 32, 33])

    def test_valid_order_uses_horizontal_position_and_allows_gaps(self) -> None:
        """编号不必连续，纵坐标及 OCR 输出顺序不影响横向排序。"""
        stages = self.recognize([ocr_result(33, 510, 250), ocr_result(29, 5, 460), ocr_result(31, 180, 330)])
        self.assertEqual([stage.stage for stage in stages], [29, 31, 33])

    def test_inconsistent_first_or_middle_label_is_excluded(self) -> None:
        """不允许错误的首项或中间项阻断其余明确递增编号。"""
        for values, expected in (([99, 30, 31, 32], [30, 31, 32]), ([30, 31, 3, 33], [30, 31, 33])):
            with self.subTest(values=values):
                stages = self.recognize([ocr_result(value, index * 100) for index, value in enumerate(values)])
                self.assertEqual([stage.stage for stage in stages], expected)

    def test_equally_plausible_orders_are_not_guessed(self) -> None:
        """两组同样长的递增解释均可能成立时，不提供可点击目标。"""
        stages = self.recognize([ocr_result(value, index * 100) for index, value in enumerate([30, 32, 31, 33])])
        self.assertEqual(stages, [])

    def test_empty_and_single_result_remain_supported(self) -> None:
        """没有结果保持为空，单个编号保持现有行为。"""
        self.assertEqual(self.recognize([]), [])
        self.assertEqual([stage.stage for stage in self.recognize([ocr_result(3, 100)])], [3])

    def test_filter_precedes_range_decision_and_target_click(self) -> None:
        """截断结果既不能引发假范围错误，也不能冒充目标被点击。"""
        for target in (29, 3):
            with self.subTest(target=target):
                frames = iter(
                    [
                        [ocr_result(value, index * 140) for index, value in enumerate([30, 31, 32, 33, 3])],
                        [ocr_result(target, 200)],
                    ]
                )
                events: list[tuple[str, tuple[int, ...]]] = []
                image = np.zeros((1280, 720, 3), dtype=np.uint8)
                job = SimpleNamespace(succeeded=True, get=lambda image=image: image)
                job.wait = lambda job=job: job

                def record(kind: str, values: tuple[int, ...], events=events, job=job):
                    """记录输入边界以核对真实动作的先后及落点。"""
                    events.append((kind, values))
                    return job

                controller = SimpleNamespace(
                    post_screencap=lambda job=job: job,
                    post_swipe=lambda *values: record("滑动", values),
                    post_click=lambda *values: record("点击", values),
                )
                context = SimpleNamespace(
                    tasker=SimpleNamespace(controller=controller),
                    run_recognition=lambda *args, frames=frames: SimpleNamespace(all_results=next(frames)),
                )
                with patch("custom.action.find_target_stage.time.sleep"):
                    result = FindTargetStageAction().run(
                        context, SimpleNamespace(custom_action_param=f'{{"target_stage":"3-{target}"}}')
                    )
                self.assertTrue(result.success)
                self.assertEqual(
                    events,
                    [
                        ("滑动", (50, 400, 650, 400, 500)),
                        ("滑动", (650, 400, 650, 600, 500)),
                        ("点击", (230, 300)),
                        ("点击", (230, 300)),
                    ],
                )


if __name__ == "__main__":
    unittest.main()
