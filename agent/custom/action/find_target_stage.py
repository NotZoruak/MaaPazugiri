from __future__ import annotations

import re
import time
from dataclasses import dataclass

import numpy as np
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from maa.define import OCRResult
from utils import logger
from utils.params import parse_params

STAGE_OCR_ROI = [0, 182, 720, 331]
MAX_CHANGED_SWIPES = 10
MAX_UNCHANGED_SWIPES = 10
STAGE_PATTERN = re.compile(r"\s*(\d+)\s*[-－—−]\s*(\d+)\s*")


@dataclass(frozen=True)
class StageResult:
    chapter: int
    stage: int
    text: str
    x: int
    y: int


@AgentServer.custom_action("FindTargetStageAction")
class FindTargetStageAction(CustomAction):
    """在章节地图中查找目标关卡，并进入其准备页面。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        try:
            params = parse_params(argv.custom_action_param, "target_stage")
            target = self._parse_stage(params["target_stage"])
            if target is None:
                raise ValueError("target_stage 必须采用“章节-关卡”格式")

            logger.info(f"开始查找目标关卡：{target[0]}-{target[1]}")
            current_image = context.tasker.controller.post_screencap().wait().get()
            if not current_image.size:
                raise RuntimeError("无法获取当前页面截图")
            changed_swipes = 0
            unchanged_swipes = 0
            stages = self._recognize_stages(context, current_image)

            while True:
                if not stages:
                    raise RuntimeError("未能从当前页面识别出关卡编号")

                stage_texts = [stage.text for stage in stages]
                logger.info(f"当前页面识别到关卡：{'、'.join(stage_texts)}")
                target_result = next(
                    (stage for stage in stages if (stage.chapter, stage.stage) == target),
                    None,
                )
                if target_result is not None:
                    self._enter_stage(context, target_result)
                    return CustomAction.RunResult(success=True)

                stage_numbers = [(stage.chapter, stage.stage) for stage in stages]
                minimum = min(stage_numbers)
                maximum = max(stage_numbers)
                if minimum < target < maximum:
                    raise RuntimeError(f"目标关卡 {target[0]}-{target[1]} 位于当前识别范围内，但页面未识别到该关卡")
                if changed_swipes >= MAX_CHANGED_SWIPES:
                    raise RuntimeError(f"已达到识别结果变化的滑动上限：{MAX_CHANGED_SWIPES} 次")

                if target < minimum:
                    start, horizontal_end = (50, 400), (650, 400)
                elif target > maximum:
                    start, horizontal_end = (650, 400), (50, 400)
                else:
                    raise RuntimeError("目标关卡与当前识别范围关系不明确")

                self._swipe_l(context, start, horizontal_end)
                updated_image = context.tasker.controller.post_screencap().wait().get()
                if not updated_image.size:
                    raise RuntimeError("滑动后无法获取页面截图")
                updated_stages = self._recognize_stages(context, updated_image)
                if not updated_stages:
                    raise RuntimeError("滑动后未能识别出关卡编号")
                updated_signature = self._signature(updated_stages)
                previous_signature = self._signature(stages)

                if updated_signature == previous_signature:
                    unchanged_swipes += 1
                    logger.info(f"滑动后识别结果未变化，继续尝试（{unchanged_swipes}/{MAX_UNCHANGED_SWIPES}）")
                    if unchanged_swipes >= MAX_UNCHANGED_SWIPES:
                        raise RuntimeError(f"连续 {MAX_UNCHANGED_SWIPES} 次滑动后识别结果未变化")
                else:
                    changed_swipes += 1
                    unchanged_swipes = 0
                    logger.info(f"滑动后识别结果已变化（{changed_swipes}/{MAX_CHANGED_SWIPES}）")
                stages = updated_stages
        except Exception as error:
            logger.error(f"查找目标关卡失败：{error}")
            return CustomAction.RunResult(success=False)

    def _recognize_stages(self, context: Context, image: np.ndarray) -> list[StageResult]:
        detail = context.run_recognition(
            "RB_FindTargetStage",
            image,
            {
                "RB_FindTargetStage": {
                    "recognition": {
                        "type": "OCR",
                        "param": {"roi": STAGE_OCR_ROI},
                    }
                }
            },
        )
        if detail is None:
            raise RuntimeError("Maa OCR 未能启动")

        results: list[StageResult] = []
        for item in detail.all_results:
            if not isinstance(item, OCRResult):
                continue
            match = STAGE_PATTERN.fullmatch(item.text)
            if match is None:
                continue
            position = self._box_origin(item.box)
            if position is None:
                continue
            results.append(
                StageResult(
                    chapter=int(match.group(1)),
                    stage=int(match.group(2)),
                    text=f"{int(match.group(1))}-{int(match.group(2))}",
                    x=position[0],
                    y=position[1],
                )
            )
        return sorted(results, key=lambda stage: (stage.y, stage.x, stage.chapter, stage.stage))

    @staticmethod
    def _box_origin(box: object) -> tuple[int, int] | None:
        if isinstance(box, (list, tuple)):
            if not box:
                return None
            if len(box) >= 2:
                try:
                    return int(box[0]), int(box[1])
                except (TypeError, ValueError):
                    pass
            for value in box:
                position = FindTargetStageAction._box_origin(value)
                if position is not None:
                    return position
            return None

        x = getattr(box, "x", None)
        y = getattr(box, "y", None)
        if isinstance(x, int) and isinstance(y, int):
            return x, y
        return None

    @staticmethod
    def _parse_stage(value: object) -> tuple[int, int] | None:
        if not isinstance(value, str):
            return None
        match = STAGE_PATTERN.fullmatch(value)
        if match is None:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _signature(stages: list[StageResult]) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((stage.chapter, stage.stage) for stage in stages))

    @staticmethod
    def _swipe_l(
        context: Context,
        start: tuple[int, int],
        horizontal_end: tuple[int, int],
    ) -> None:
        controller = context.tasker.controller
        try:
            points = (start, horizontal_end, (horizontal_end[0], horizontal_end[1] + 200))
            for index, point in enumerate(points):
                if index == 0:
                    job = controller.post_touch_down(*point)
                else:
                    job = controller.post_touch_move(*point)
                if not job.wait().succeeded:
                    raise RuntimeError("执行 L 形滑动手势失败")
        finally:
            if not controller.post_touch_up().wait().succeeded:
                raise RuntimeError("释放滑动触点失败")

    @staticmethod
    def _enter_stage(context: Context, stage: StageResult) -> None:
        controller = context.tasker.controller
        target = (stage.x + 30, stage.y - 50)
        for index in range(2):
            if not controller.post_click(*target).wait().succeeded:
                raise RuntimeError("点击目标关卡失败")
            if index == 0:
                time.sleep(0.5)
        logger.info(f"已连续点击目标关卡：{stage.text}")
