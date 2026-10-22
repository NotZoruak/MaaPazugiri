"""执行策略选出的单步战斗动作，并在动作后交还截图主循环。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import replace
from typing import Any, Protocol

from utils import logger
from vision.image_io import Image
from vision.model import BoardResult, ChainEndpointResult, ChainMovePositions, is_black_ghoul, is_linkable_ghoul

from .candidates import preferred_ghoul_touch_point
from .observation import BattleObservation
from .preview import BoardPreviewer
from .ranbu import (
    choose_ranbu_start,
    ghoul_color_at_point,
    ranbu_short_link_points,
    ranbu_start_color,
    ranbu_sweep_points,
)
from .ranbu_feedback import RanbuHoldFeedback
from .session import BattleState
from .strategy import BattlePhase, StageResolution, StrategyDecision


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

    ULTIMATE_TAP_POINTS = ((360, 450), (200, 730), (520, 730))
    ULTIMATE_SWIPE_PATHS = (((360, 350), (360, 930)), ((70, 640), (650, 640)))
    ULTIMATE_SWIPE_DURATION_MS = 500

    def __init__(
        self,
        controller: Any,
        previewer: BoardPreviewer,
        record_normal_link_success: Callable[[], None],
        can_start_ranbu_link: Callable[[], bool],
        is_stopping: Callable[[], bool],
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        inspect_ranbu_hold: Callable[[Image, BoardResult, str | None], RanbuHoldFeedback] | None = None,
    ) -> None:
        self._controller = controller
        self._previewer = previewer
        self._record_normal_link_success = record_normal_link_success
        self._can_start_ranbu_link = can_start_ranbu_link
        self._is_stopping = is_stopping
        self._clock = clock
        self._sleep = sleep
        self._inspect_ranbu_hold = inspect_ranbu_hold
        self._reverse_next_normal_link = False
        self._last_ranbu_color: str | None = None
        self._last_gesture_started = False
        self._last_ranbu_sweep_started = False
        self.last_gesture_released_at: float | None = None

    @property
    def last_ranbu_color(self) -> str | None:
        """返回上一次乱舞起手颜色，用于当前帧候选轮换。"""
        return self._last_ranbu_color

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
        elif decision.kind == "black_link":
            self._black_link(observation, decision)
        elif decision.kind == "ranbu_link":
            self._ranbu_link(observation, decision)
        elif decision.kind == "ranbu_short_link":
            self._ranbu_short_link(observation, decision)
        else:
            logger.debug("本轮不派发设备动作：阶段={}，建议={}。", resolution.stage.id, decision)

    def execute_ultimate_confirmation(self, state: BattleState) -> None:
        """按提示类型在720×1280画面执行三点点击或十字滑动。"""
        if state == BattleState.ULTIMATE_TAP_CONFIRMATION:
            for point in self.ULTIMATE_TAP_POINTS:
                if self._is_stopping():
                    return
                if not self._controller.post_click(*point).wait().succeeded:
                    raise RuntimeError("点击奥义位置失败")
            logger.debug("已在圆形范围内按三角分布点击三处奥义位置。")
            return
        if state == BattleState.ULTIMATE_SWIPE_CONFIRMATION:
            for start, end in self.ULTIMATE_SWIPE_PATHS:
                if self._is_stopping():
                    return
                if (
                    not self._controller.post_swipe(
                        *start,
                        *end,
                        self.ULTIMATE_SWIPE_DURATION_MS,
                    )
                    .wait()
                    .succeeded
                ):
                    raise RuntimeError("执行奥义十字滑动失败")
            logger.debug("已先竖滑、再横滑执行奥义十字手势。")
            return
        raise ValueError(f"不支持的奥义位置确认状态：{state}")

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
        """按当前帧路线整笔连线，仅在松手前检查一次菱形。"""
        if observation.board is None or decision.route is None or len(decision.route.points) < 2:
            logger.warning("当前帧普通连线缺少可信棋盘或路线，跳过触控。")
            return
        points = tuple(tuple(round(value) for value in point) for point in decision.route.points)
        if self._reverse_next_normal_link:
            points = tuple(reversed(points))
            logger.debug("上次整笔未确认成功，本次使用新帧路线的另一端起步。")
        logger.debug("普通连线目标：颜色={}，对象序列={}。", decision.route.color, decision.route.object_ids)
        found_marker = self._hold_and_move(points, observation)
        self._reverse_next_normal_link = False if found_marker else not self._reverse_next_normal_link
        if found_marker:
            self._record_normal_link_success()
            logger.debug("松手前已识别到连线菱形，本次普通连线手数递增一次。")
        else:
            logger.debug("松手前未确认连线菱形，本次不递增普通连线手数。")

    def _black_link(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """只连接当前帧黑色对象，乱舞期间不累计普通连线手数。"""
        route, board = decision.route, observation.board
        if board is None or route is None or route.color != "black" or len(route.points) < 2:
            return
        objects = {item.id: item for item in board.objects}
        if (
            len(route.object_ids) != len(route.points)
            or len(set(route.object_ids)) != len(route.object_ids)
            or any(
                identifier not in objects or not is_black_ghoul(objects[identifier]) for identifier in route.object_ids
            )
        ):
            logger.warning("黑色路线与当前帧对象不符，本轮不发送触控。")
            return
        for identifier, point in zip(route.object_ids, route.points, strict=True):
            item = objects[identifier]
            if (
                item.candidate_point is None
                or item.quality not in ("identified", "single_feature")
                or not all(math.isfinite(value) for value in point)
            ):
                return
            expected = preferred_ghoul_touch_point(item, board)
            if tuple(round(value) for value in point) != tuple(round(value) for value in expected):
                logger.warning("黑色路线落点已脱离本帧可信触点，本轮不发送触控。")
                return
        ranbu = observation.phase == BattlePhase.RANBU
        points = tuple(tuple(round(value) for value in point) for point in route.points)
        logger.debug("优先连接黑色饿鬼：本帧对象序列={}。", route.object_ids)
        confirmed = self._hold_and_move(
            points,
            observation,
            check_marker=not ranbu,
            check_ranbu_allowed=ranbu,
            hold_start_seconds=0.0 if ranbu else 0.5,
        )
        if confirmed and not ranbu:
            self._record_normal_link_success()

    def _ranbu_link(self, observation: BattleObservation, decision: StrategyDecision) -> None:
        """按乱舞规则选择起点并沿棋盘边界生成的固定往返轨迹扫动。"""
        if observation.board is None:
            logger.warning("当前帧没有乱舞棋盘识别结果，跳过扫动。")
            return
        # 保留独立起手图，避免后续截图复用缓冲区覆盖诊断现场。
        observation = replace(observation, image=observation.image.copy())
        assert observation.board is not None
        start = decision.ranbu_start
        if start is None:
            start = choose_ranbu_start(
                observation.board, decision.reserved_color, previous_color=self._last_ranbu_color
            )
        if start is None:
            logger.debug("当前帧没有符合保留颜色规则的乱舞起点，等待新截图。")
            return
        if decision.ranbu_frame_id is not None and (
            decision.ranbu_frame_id != observation.frame_id or observation.board.frame_id != observation.frame_id
        ):
            logger.debug("乱舞起点局部证据不属于当前帧，不发送触控。")
            return
        color = (
            decision.ranbu_color
            if decision.ranbu_frame_id is not None
            else (
                ghoul_color_at_point(observation.board, start)
                if decision.ranbu_start is not None
                else ranbu_start_color(observation.board, start)
            )
        )
        if decision.ranbu_start is not None and (color is None or color == decision.reserved_color):
            logger.debug("已确认起点在当前帧未命中非保留色，不更换落点或发送触控。")
            return
        points = (start, *ranbu_sweep_points(observation.board))
        previous_color = self._last_ranbu_color
        logger.debug(
            "乱舞整盘轨迹：起点颜色={}，上次颜色={}，保留颜色={}，本帧起点={}，完整折返点={}。",
            color,
            self._last_ranbu_color,
            decision.reserved_color,
            start,
            points[1:],
        )
        self._hold_and_move(
            points,
            observation,
            check_marker=False,
            check_ranbu_allowed=True,
            hold_start_seconds=0.0,
            gesture_duration=0.4,
            capture_ranbu=True,
            ranbu_color=color,
            ranbu_body_size=decision.ranbu_body_size,
        )
        # 预演图写盘移到松手后，避免颜色确认后再消耗一帧时间才按下。
        try:
            self._previewer.save_ranbu_report(
                observation.image, observation.board, start, color, previous_color, points
            )
        except Exception as exc:
            logger.warning("乱舞轨迹诊断保存失败：{}。", exc)
        if self._last_ranbu_sweep_started:
            self._last_ranbu_color = color
            logger.debug("乱舞扫动已结束，回到主循环重新截图；乱舞连线不计入普通手数。")
        else:
            logger.debug("乱舞扫动未开始，回到主循环重新截图。")

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
            check_ranbu_allowed=True,
            hold_start_seconds=0.0,
            gesture_duration=0.04,
        )
        logger.debug("乱舞保留色短连结束，回到主循环重新截图；不计入普通手数。")

    def _hold_and_move(
        self,
        points: tuple[tuple[int, int], ...],
        observation: BattleObservation,
        *,
        check_marker: bool = True,
        check_ranbu_allowed: bool = False,
        hold_start_seconds: float = 0.5,
        gesture_duration: float | None = None,
        capture_ranbu: bool = False,
        ranbu_color: str | None = None,
        ranbu_body_size: tuple[float, float] | None = None,
    ) -> bool:
        """整笔连续移动；普通连线只在松手前截图一次，出现菱形即确认一手。"""
        self._last_gesture_started = False
        self._last_ranbu_sweep_started = False
        self.last_gesture_released_at = None
        if len(points) < 2 or self._is_stopping():
            return False
        if check_ranbu_allowed and not self._can_start_ranbu_link():
            logger.debug("当前帧未确认允许乱舞，不再启动扫动。")
            return False
        gesture_started = self._clock()
        events: list[dict[str, object]] | None = [] if capture_ranbu else None
        if events is not None:
            events.append(
                {
                    "kind": "down",
                    "point": list(points[0]),
                    "sent_ms": 0.0,
                    "finished_ms": None,
                    "succeeded": None,
                }
            )
        completed, succeeded = False, False
        image: Image | None = None
        held: Image | None = None
        movement_finished_at: float | None = None
        capture_elapsed_ms = 0.0
        capture_status = "not_attempted"
        passes: list[dict[str, object]] = []
        pass_images: list[Image] = []
        stop_reason = "input_failed"
        result: ChainEndpointResult | None = None
        reason = "stopped"
        try:
            # 提交后等待异常时，设备仍可能已经按下，因此按下本身也必须处于释放保护内。
            try:
                down = self._controller.post_touch_down(*points[0]).wait()
            finally:
                if events is not None:
                    events[0]["finished_ms"] = round((self._clock() - gesture_started) * 1000, 3)
            if events is not None:
                events[0]["succeeded"] = bool(down.succeeded)
            if not down.succeeded:
                raise RuntimeError("按住连线起点失败")
            self._last_gesture_started = True
            stop_reason = "stopped"
            if hold_start_seconds > 0:
                self._sleep(hold_start_seconds)
            if gesture_duration is not None and capture_ranbu:
                self._last_ranbu_sweep_started = True
                pass_started = self._clock()
                stop_reason = "input_failed"
                try:
                    completed = self._move_timed_path(
                        points,
                        observation,
                        pass_started,
                        gesture_duration,
                        events=events,
                        event_origin=gesture_started,
                        body_size=ranbu_body_size,
                    )
                finally:
                    movement_finished_at = self._clock()
                stop_reason = "single_sweep_complete" if completed and not self._is_stopping() else "stopped"
                passes.append(
                    {
                        "index": 1,
                        "direction": "forward",
                        "movement_elapsed_ms": round((movement_finished_at - pass_started) * 1000, 3),
                        "capture_elapsed_ms": 0.0,
                        "state": None,
                        "state_checked": False,
                        "count": None,
                        "residual_status": "not_checked",
                        "residual_details": {},
                        "feedback_checked": False,
                        "feedback_skip_reason": "single_sweep",
                        "feedback_elapsed_ms": 0.0,
                    }
                )
            elif gesture_duration is not None:
                completed = self._move_timed_path(points, observation, gesture_started, gesture_duration, events=events)
                movement_finished_at = self._clock()
            else:
                for index, point in enumerate(points[1:], 1):
                    if not self._move_continuously(
                        points[index - 1],
                        point,
                        observation,
                        allowed=self._can_start_ranbu_link if check_ranbu_allowed else None,
                    ):
                        break
                else:
                    completed = True
            if check_marker and completed and observation.board is not None and not self._is_stopping():
                job = self._controller.post_screencap().wait()
                reason = "capture_failed"
                if not self._is_stopping() and job.succeeded:
                    current: Image = job.get()
                    reason = "capture_empty"
                    if not self._is_stopping() and current.size:
                        image = current
                        result = self._previewer.recognize_chain_endpoint(image, observation.board)
                        # 起点和终点均可能有菱形；只确认存在，不要求唯一或落在计划末点。
                        succeeded = (
                            not self._is_stopping()
                            and result is not None
                            and result.status in ("detected", "uncertain")
                            and bool(result.markers)
                        )
                        reason = "marker_detected" if succeeded else "marker_not_detected"
        finally:
            release_started = self._clock()
            up = self._controller.post_touch_up().wait()
            gesture_finished_at = self._clock()
            self.last_gesture_released_at = gesture_finished_at
            if events is not None:
                events.append(
                    {
                        "kind": "up",
                        "point": [],
                        "sent_ms": round((release_started - gesture_started) * 1000, 3),
                        "finished_ms": round((self._clock() - gesture_started) * 1000, 3),
                        "succeeded": bool(up.succeeded),
                    }
                )
                report: dict[str, object] = {
                    "frame_id": observation.frame_id,
                    "start": list(points[0]),
                    "start_color": ghoul_color_at_point(observation.board, points[0]) if observation.board else None,
                    "planned_points": points,
                    "target_duration_ms": round((gesture_duration or 0) * 1000),
                    "movement_elapsed_ms": round(
                        ((movement_finished_at or release_started) - gesture_started) * 1000, 3
                    ),
                    "capture_elapsed_ms": capture_elapsed_ms,
                    "capture_status": capture_status,
                    "total_elapsed_ms": round((gesture_finished_at - gesture_started) * 1000, 3),
                    "path_completed": completed,
                    "dispatch_completed": completed and bool(up.succeeded),
                    "game_connection_verified": False,
                    "quantity_recognition": "disabled_in_live_gesture",
                    "count_trend": "unknown",
                    "connection_progress_observed": any(
                        isinstance(a["count"], int) and isinstance(b["count"], int) and b["count"] > a["count"]
                        for a, b in zip(passes, passes[1:], strict=False)
                    ),
                    "passes": passes,
                    "start_selection_check": "disabled",
                    "touch_sampling": "body_relative_dense_points",
                    "start_body_size": ranbu_body_size,
                    "stop_reason": stop_reason,
                    "events": events,
                }
                if ranbu_color is not None:
                    report["start_color"] = ranbu_color
                try:
                    self._previewer.save_ranbu_execution_report(
                        observation.image, held, report, pass_images=pass_images
                    )
                except Exception as exc:
                    logger.warning("乱舞执行诊断保存失败：{}。", exc)
            if not up.succeeded:
                raise RuntimeError("释放连线触点失败")
        if check_marker:
            logger.debug(
                "普通连线末尾单次确认：计划目标={}，菱形数量={}，判定={}，耗时={:.2f}ms。",
                len(points),
                len(result.markers) if result else 0,
                reason,
                (self._clock() - gesture_started) * 1000,
            )
            if completed and not succeeded and not self._is_stopping() and observation.board is not None:
                self._previewer.save_link_failure(
                    image,
                    observation.board,
                    ChainMovePositions(points[0], points[-1]),
                    result,
                    {"planned_targets": len(points), "confirmation_mode": "after_full_gesture", "stop_reason": reason},
                )
        elif gesture_duration is not None:
            logger.debug(
                "乱舞手势耗时：单遍目标={}ms，整笔实际={:.2f}ms，完整派发={}，截图耗时={}ms，完成反馈={}，退出原因={}；松手后等待另计。",
                round(gesture_duration * 1000),
                (gesture_finished_at - gesture_started) * 1000,
                completed,
                capture_elapsed_ms,
                passes,
                stop_reason,
            )
        return succeeded and not self._is_stopping()

    def _move_timed_path(
        self,
        points: tuple[tuple[int, int], ...],
        observation: BattleObservation,
        started: float,
        duration: float,
        *,
        events: list[dict[str, object]] | None = None,
        event_origin: float | None = None,
        body_size: tuple[float, float] | None = None,
    ) -> bool:
        """按普通连线密度插入中间触点，整轮共享时间表，落后时也不省略触点。"""
        dimensions = (
            [
                size
                for item in observation.board.objects
                for size in item.body_box[2:]
                if is_linkable_ghoul(item) and math.isfinite(size) and size > 0
            ]
            if observation.board is not None
            else []
        )
        if body_size is not None:
            if not all(math.isfinite(size) and size > 0 for size in body_size):
                raise ValueError("乱舞触点采样尺寸必须为有限正数")
            dimensions.extend(body_size)
        # 确认帧对象表为空时仍使用当次确认尺寸，避免回退成跨度过大的坐标跳跃。
        step = max(2.0, min(32.0, min(dimensions) * 0.2)) if dimensions else 32.0
        segments = list(zip(points, points[1:], strict=False))
        total_distance = sum(math.dist(start, end) for start, end in segments)
        if total_distance <= 0:
            return True
        travelled = 0.0
        pending_moves: list[tuple[Any, tuple[int, int], float]] = []

        def finish_pending_moves() -> None:
            """等待当前小批触点完成，并按提交顺序检查执行结果。"""
            if not pending_moves:
                return
            pending_moves[-1][0].wait()
            for move, point, sent_at in pending_moves:
                move_succeeded = bool(move.succeeded)
                if events is not None:
                    events.append(
                        {
                            "kind": "move",
                            "point": list(point),
                            "sent_ms": round(
                                (sent_at - (event_origin if event_origin is not None else started)) * 1000, 3
                            ),
                            "finished_ms": round(
                                (self._clock() - (event_origin if event_origin is not None else started)) * 1000, 3
                            ),
                            "succeeded": move_succeeded,
                        }
                    )
                if not move_succeeded:
                    raise RuntimeError("移动连线触点失败")
            pending_moves.clear()

        def allowed() -> bool:
            """停止和本帧乱舞许可均按每个采样点检查。"""
            return not self._is_stopping() and self._can_start_ranbu_link()

        for start, end in segments:
            distance = math.dist(start, end)
            count = max(1, math.ceil(distance / step))
            for index in range(1, count + 1):
                if not allowed():
                    if not self._is_stopping():
                        finish_pending_moves()
                    return False
                due = started + duration * (travelled + distance * index / count) / total_distance
                now = self._clock()
                # 设备调用耗时计入时间表，超时只省去等待，不省去触点。
                if due > now:
                    self._sleep(due - now)
                    # 等待期间可能收到停止；没有等待时复用本触点刚核对的许可。
                    if not allowed():
                        if not self._is_stopping():
                            finish_pending_moves()
                        return False
                point = (
                    round(start[0] + (end[0] - start[0]) * index / count),
                    round(start[1] + (end[1] - start[1]) * index / count),
                )
                sent_at = self._clock()
                move = self._controller.post_touch_move(*point)
                pending_moves.append((move, point, sent_at))
                if len(pending_moves) >= 8:
                    finish_pending_moves()
            travelled += distance
        finish_pending_moves()
        return True

    def _move_continuously(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
        observation: BattleObservation,
        *,
        allowed: Callable[[], bool] | None = None,
    ) -> bool:
        """用短段连续拖动经过目标身体，段内响应停止和乱舞许可。"""
        dimensions = (
            [
                size
                for item in observation.board.objects
                for size in item.body_box[2:]
                if is_linkable_ghoul(item) and math.isfinite(size) and size > 0
            ]
            if observation.board is not None
            else []
        )
        step = max(2.0, min(32.0, min(dimensions) * 0.2)) if dimensions else 32.0
        count = max(1, math.ceil(math.dist(start, end) / step))
        # 常见相邻对象约用60至100毫秒经过身体，长段的采样总间隔最多160毫秒。
        interval = min(0.016, 0.16 / (count - 1)) if count > 1 else 0.0
        logger.debug("触控连续移动：起点={}，目标={}，采样点={}，最大步长={:.2f}像素。", start, end, count, step)

        def can_move() -> bool:
            """每个移动点都检查停止与当前乱舞许可。"""
            return not self._is_stopping() and (allowed is None or allowed())

        for index in range(1, count + 1):
            if not can_move():
                return False
            if index > 1:
                self._sleep(interval)
                if not can_move():
                    return False
            point = tuple(round(a + (b - a) * index / count) for a, b in zip(start, end, strict=True))
            if not self._controller.post_touch_move(*point).wait().succeeded:
                raise RuntimeError("移动连线触点失败")
        return True
