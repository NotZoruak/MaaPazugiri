"""执行策略选出的单步战斗动作，并在动作后交还截图主循环。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from utils import logger
from vision.image_io import Image
from vision.model import ChainEndpointResult, ChainMovePositions

from .observation import BattleObservation
from .preview import BoardPreviewer
from .ranbu import choose_ranbu_start, ranbu_short_link_points, ranbu_sweep_points
from .strategy import StageResolution, StrategyDecision


@dataclass
class _MoveFeedback:
    """保存一次移动的有界复查结果和最后现场。"""

    reached: bool = False
    attempts: int = 0
    elapsed_ms: float = 0.0
    result: ChainEndpointResult | None = None
    image: Image | None = None
    stop_reason: str = "unconfirmed"
    retries: int = 0
    move: ChainMovePositions | None = None


class BattleExecutor(Protocol):
    """定义一次行动派发接口；调用返回后由外层重新获取截图。"""

    def execute_once(
        self,
        observation: BattleObservation,
        resolution: StageResolution,
        decision: StrategyDecision,
    ) -> None:
        """派发至多一个行动；不在当前帧上继续评估下一步。"""
        ...


class MaaBattleExecutor:
    """执行单个策略动作；动作返回后由外层主循环重新截图。"""

    def __init__(
        self,
        controller: Any,
        previewer: BoardPreviewer,
        record_normal_link_success: Callable[[], None],
        can_start_ranbu_link: Callable[[], bool],
        is_stopping: Callable[[], bool],
        *,
        confirmation_timeout: float = 1.5,
        confirmation_poll_interval: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (confirmation_timeout, confirmation_poll_interval)):
            raise ValueError("移动确认期限和复查间隔必须是有限正数")
        self._controller = controller
        self._previewer = previewer
        self._record_normal_link_success = record_normal_link_success
        self._can_start_ranbu_link = can_start_ranbu_link
        self._is_stopping = is_stopping
        self._confirmation_timeout = confirmation_timeout
        self._confirmation_poll_interval = confirmation_poll_interval
        self._clock = clock
        self._sleep = sleep

    def execute_once(
        self,
        observation: BattleObservation,
        resolution: StageResolution,
        decision: StrategyDecision,
    ) -> None:
        """按决策执行一次点击、普通连线或乱舞扫动，不在此处规划下一步。"""
        if decision.kind == "ultimate":
            self._click_ultimate(observation, decision)
        elif decision.kind == "bomb":
            self._click_bomb(observation, decision)
        elif decision.kind == "normal_link":
            self._normal_link(observation, decision)
        elif decision.kind == "ranbu_link":
            self._ranbu_link(observation, decision)
        elif decision.kind == "ranbu_short_link":
            self._ranbu_short_link(observation, decision)
        else:
            logger.debug("本轮不派发设备动作：阶段={}，建议={}。", resolution.stage.id, decision)

    def _click_ultimate(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """点击本帧确认就绪的奥义位置。"""
        point = (observation.ultimate_click_points or {}).get(decision.slot or -1)
        if point is None:
            logger.warning("奥义位置{}没有本帧就绪点击点，跳过触控。", decision.slot)
            return
        if not self._controller.post_click(*point).wait().succeeded:
            raise RuntimeError("点击奥义失败")
        logger.debug("已点击{}号位奥义，回到主循环重新截图。", decision.slot)

    def _click_bomb(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """点击当前帧中确认等级且具有有效触点的首个炸弹。"""
        if observation.board is None or decision.level is None:
            logger.warning("当前帧没有可点击炸弹识别结果，跳过触控。")
            return
        candidates = [
            item
            for item in observation.board.objects
            if item.kind == "bomb"
            and item.bomb_level == decision.level
            and item.candidate_point is not None
            and item.quality in ("identified", "single_feature")
        ]
        if not candidates:
            logger.warning("当前帧没有确认的{}级炸弹触点，跳过触控。", decision.level)
            return
        target = min(candidates, key=lambda item: item.id)
        assert target.candidate_point is not None
        point = tuple(round(value) for value in target.candidate_point)
        if not self._controller.post_click(*point).wait().succeeded:
            raise RuntimeError("点击炸弹失败")
        logger.debug("已点击{}级炸弹对象{}，回到主循环重新截图。", decision.level, target.id)

    def _normal_link(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """按当前帧候选路线连线，期间逐帧查找新出现的菱形标志。"""
        if observation.board is None or decision.route is None or len(decision.route.points) < 2:
            logger.warning("当前帧普通连线缺少可信棋盘或路线，跳过触控。")
            return
        points = tuple(tuple(round(value) for value in point) for point in decision.route.points)
        logger.debug("普通连线目标：颜色={}，对象序列={}。", decision.route.color, decision.route.object_ids)
        found_marker = self._hold_and_move(points, observation)
        if found_marker:
            self._record_normal_link_success()
            logger.debug("至少一段已确认菱形到达目标，本次普通连线手数递增一次。")
        else:
            logger.debug("连线期间未确认菱形到达目标，本次不递增普通连线手数。")

    def _ranbu_link(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """按乱舞规则选择起点并沿棋盘边界生成的固定往返轨迹扫动。"""
        if observation.board is None:
            logger.warning("当前帧没有乱舞棋盘识别结果，跳过扫动。")
            return
        start = choose_ranbu_start(observation.board, decision.reserved_color)
        if start is None:
            logger.debug("当前帧没有符合保留颜色规则的乱舞起点，等待新截图。")
            return
        points = (start, *ranbu_sweep_points(observation.board))
        self._hold_and_move(
            points,
            observation,
            check_marker=False,
            check_ranbu_deadline=True,
            hold_start_seconds=0.0,
        )
        logger.debug("乱舞扫动完成，回到主循环重新截图；乱舞连线不计入普通手数。")

    def _ranbu_short_link(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """全为保留色时，从一只普通饿鬼滑到另一只后立即松手。"""
        if observation.board is None:
            return
        points = ranbu_short_link_points(observation.board, decision.reserved_color)
        if points is None:
            logger.debug("当前帧不满足保留色两点短连条件，等待新截图。")
            return
        self._hold_and_move(
            points,
            observation,
            check_marker=False,
            check_ranbu_deadline=True,
            hold_start_seconds=0.0,
        )
        logger.debug("乱舞保留色短连结束，回到主循环重新截图；不计入普通手数。")

    def _hold_and_move(
        self,
        points: tuple[tuple[int, int], ...],
        observation: BattleObservation,
        *,
        check_marker: bool = True,
        check_ranbu_deadline: bool = False,
        hold_start_seconds: float = 0.5,
    ) -> bool:
        """普通连线逐段确认后继续；乱舞保持直接扫动或两点后松手。"""
        if check_ranbu_deadline and not self._can_start_ranbu_link():
            logger.debug("乱舞已超过新一轮启动期限，不再开始扫动。")
            return False
        if self._is_stopping():
            return False
        down = self._controller.post_touch_down(*points[0]).wait()
        if not down.succeeded:
            raise RuntimeError("按住连线起点失败")
        succeeded = False
        confirmed_targets = 0
        failure: tuple[ChainMovePositions, int, _MoveFeedback] | None = None
        try:
            if hold_start_seconds > 0:
                self._sleep(hold_start_seconds)
            for index, point in enumerate(points[1:], 1):
                if self._is_stopping():
                    break
                if not self._controller.post_touch_move(*point).wait().succeeded:
                    raise RuntimeError("移动连线触点失败")
                if check_marker:
                    move = ChainMovePositions(points[index - 1], point)
                    feedback = self._confirm_move_with_retries(observation, move)
                    actual_move = feedback.move or move
                    logger.debug(
                        "连线逐段确认：目标序号={}/{}，起点={}，目标={}，视觉结果={}，判定={}，复查次数={}，耗时={}ms。",
                        index + 1,
                        len(points),
                        actual_move.start,
                        actual_move.end,
                        feedback.result.move_outcome if feedback.result is not None else "unknown",
                        feedback.stop_reason,
                        feedback.attempts,
                        feedback.elapsed_ms,
                    )
                    if not feedback.reached:
                        failure = (actual_move, index + 1, feedback)
                        break
                    confirmed_targets = index + 1
                    succeeded = True
        finally:
            if not self._controller.post_touch_up().wait().succeeded:
                raise RuntimeError("释放连线触点失败")
        if check_marker:
            logger.info("普通连线结果：已确认接上 {} 个 / 计划 {} 个。", confirmed_targets, len(points))
        if failure is not None and not self._is_stopping():
            move, target_index, feedback = failure
            assert observation.board is not None
            self._previewer.save_link_failure(
                feedback.image,
                observation.board,
                move,
                feedback.result,
                {
                    "target_index": target_index,
                    "planned_targets": len(points),
                    "confirmed_targets": confirmed_targets,
                    "attempts": feedback.attempts,
                    "elapsed_ms": feedback.elapsed_ms,
                    "confirmation_timeout": self._confirmation_timeout,
                    "stop_reason": feedback.stop_reason,
                    "retries": feedback.retries,
                },
            )
            logger.debug("第 {} 个目标未能确认，已松手并保存最近一次连线失败现场。", target_index)
        return succeeded

    def _confirm_move(
        self, observation: BattleObservation, move: ChainMovePositions, *, deadline: float | None = None
    ) -> _MoveFeedback:
        """在期限内复查当前目标，只有唯一菱形到达终点才允许下一段。"""
        feedback = _MoveFeedback(move=move)
        if observation.board is None:
            return feedback
        started = self._clock()
        deadline = (
            min(deadline, started + self._confirmation_timeout)
            if deadline is not None
            else started + self._confirmation_timeout
        )

        def can_continue() -> bool:
            """截图或识别返回后再次核对期限，过期时不启动后续工作。"""
            if self._is_stopping():
                feedback.stop_reason = "stopped"
                return False
            if self._clock() >= deadline:
                feedback.stop_reason = "timeout"
                return False
            return True

        while can_continue():
            feedback.attempts += 1
            job = self._controller.post_screencap().wait()
            if not can_continue():
                break
            if job.succeeded:
                image = job.get()
                if not can_continue():
                    break
                if image.size:
                    feedback.image = image
                    feedback.result = self._previewer.recognize_chain_endpoint(image, observation.board, move=move)
            if not can_continue():
                break
            result = feedback.result
            if result is not None and result.status == "detected" and len(result.markers) == 1:
                feedback.reached = result.move_outcome == "target_reached"
            if feedback.reached:
                feedback.stop_reason = "target_reached"
                break
            remaining = deadline - self._clock()
            if remaining <= 0:
                feedback.stop_reason = "timeout"
                break
            self._sleep(min(self._confirmation_poll_interval, remaining))
        feedback.elapsed_ms = round((self._clock() - started) * 1000, 2)
        return feedback

    def _confirm_move_with_retries(self, observation: BattleObservation, move: ChainMovePositions) -> _MoveFeedback:
        """首次确认和最多两次重滑共享总预算，独立定位每次回退点。"""
        started = self._clock()
        deadline = started + self._confirmation_timeout
        # 日志中正常确认最长约0.8秒，首轮保留总预算的六成，剩余留给重试。
        feedback = self._confirm_move(observation, move, deadline=started + self._confirmation_timeout * 0.6)
        attempts = feedback.attempts
        if observation.board is None:
            return feedback

        def can_continue() -> bool:
            """停止或总预算耗尽后不再派发截图和触控。"""
            if self._is_stopping():
                feedback.stop_reason = "stopped"
                return False
            if self._clock() >= deadline:
                feedback.stop_reason = "step_timeout"
                return False
            return True

        for retry in range(1, 3):
            if feedback.reached or not can_continue():
                break
            job = self._controller.post_screencap().wait()
            if not job.succeeded:
                feedback.stop_reason = "recovery_capture_failed"
                break
            image = job.get()
            if not image.size:
                feedback.stop_reason = "recovery_capture_empty"
                break
            feedback.image = image
            if not can_continue():
                break
            # 独立搜索整张当前帧，不以旧对象位置或上次菱形位置限定搜索。
            current = self._previewer.recognize_chain_endpoint(image, observation.board)
            attempts += 1
            feedback.result = current
            if not can_continue():
                break
            if current is None or current.status != "detected" or len(current.markers) != 1:
                feedback.stop_reason = "current_marker_not_unique"
                break
            # 复查时目标可能已经延迟接入；此时直接继续，无需再回退。
            local = self._previewer.recognize_chain_endpoint(image, observation.board, move=feedback.move or move)
            attempts += 1
            if not can_continue():
                break
            if local is not None and local.status == "detected" and len(local.markers) == 1:
                if local.move_outcome == "target_reached":
                    feedback.reached = True
                    feedback.result = local
                    feedback.stop_reason = "target_reached_on_recheck"
                    break
            point = current.markers[0].center
            if not all(math.isfinite(value) for value in point):
                feedback.stop_reason = "current_marker_invalid"
                break
            rollback = (round(point[0]), round(point[1]))
            bx, by, bw, bh = observation.board.board_region
            width, height = observation.board.source_size
            if not (
                0 <= rollback[0] < width
                and 0 <= rollback[1] < height
                and bx <= rollback[0] < bx + bw
                and by <= rollback[1] < by + bh
            ):
                feedback.stop_reason = "current_marker_invalid"
                break
            if not can_continue():
                break
            logger.debug("连线回退重滑：第 {}/2 次，当前帧菱形={}，重试目标={}。", retry, rollback, move.end)
            if not self._controller.post_touch_move(*rollback).wait().succeeded:
                raise RuntimeError("回退到当前大菱形失败")
            # 为回退触点保留短暂响应间隔，重滑仍必须由菱形到位确认。
            if not can_continue():
                break
            remaining = deadline - self._clock()
            if remaining <= 0:
                feedback.stop_reason = "step_timeout"
                break
            self._sleep(min(self._confirmation_poll_interval, remaining))
            if not can_continue():
                break
            if not self._controller.post_touch_move(*move.end).wait().succeeded:
                raise RuntimeError("重新滑向连线目标失败")
            now = self._clock()
            retry_deadline = now + max(0.0, deadline - now) / (3 - retry)
            feedback = self._confirm_move(observation, ChainMovePositions(rollback, move.end), deadline=retry_deadline)
            feedback.retries = retry
            attempts += feedback.attempts
        feedback.attempts = attempts
        feedback.elapsed_ms = round((self._clock() - started) * 1000, 2)
        return feedback
