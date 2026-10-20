"""执行策略选出的单步战斗动作，并在动作后交还截图主循环。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Protocol

from utils import logger
from vision.model import ChainEndpointMarker

from .observation import BattleObservation
from .preview import BoardPreviewer
from .ranbu import choose_ranbu_start, ranbu_short_link_points, ranbu_sweep_points
from .strategy import StageResolution, StrategyDecision


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
    ) -> None:
        self._controller = controller
        self._previewer = previewer
        self._record_normal_link_success = record_normal_link_success
        self._can_start_ranbu_link = can_start_ranbu_link
        self._is_stopping = is_stopping

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
        baseline = observation.board.chain_endpoint.markers if observation.board.chain_endpoint else []
        found_marker = self._hold_and_move(points, observation, baseline)
        if found_marker:
            self._record_normal_link_success()
            logger.debug("连线过程中出现新菱形标志，确认普通连线成功并递增手数。")
        else:
            logger.debug("连线期间未观察到新菱形标志，本次不递增普通连线手数。")
        logger.debug("普通连线结束，回到主循环重新截图。")

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
            (),
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
            (),
            check_marker=False,
            check_ranbu_deadline=True,
            hold_start_seconds=0.0,
        )
        logger.debug("乱舞保留色短连结束，回到主循环重新截图；不计入普通手数。")

    def _hold_and_move(
        self,
        points: tuple[tuple[int, int], ...],
        observation: BattleObservation,
        baseline_markers: list[ChainEndpointMarker] | tuple[ChainEndpointMarker, ...],
        *,
        check_marker: bool = True,
        check_ranbu_deadline: bool = False,
        hold_start_seconds: float = 0.5,
    ) -> bool:
        """逐点完成候选路线；菱形只记录成功证据，不作为提前释放条件。"""
        if check_ranbu_deadline and not self._can_start_ranbu_link():
            logger.debug("乱舞已超过新一轮启动期限，不再开始扫动。")
            return False
        if self._is_stopping():
            return False
        down = self._controller.post_touch_down(*points[0]).wait()
        if not down.succeeded:
            raise RuntimeError("按住连线起点失败")
        succeeded = False
        try:
            if hold_start_seconds > 0:
                time.sleep(hold_start_seconds)
            succeeded = check_marker and self._new_marker_seen(observation, baseline_markers)
            for point in points[1:]:
                if self._is_stopping():
                    break
                if not self._controller.post_touch_move(*point).wait().succeeded:
                    raise RuntimeError("移动连线触点失败")
                # 每段仍重新截图检查，已取得的成功证据不能被后续漏检清除。
                marker_seen = check_marker and self._new_marker_seen(observation, baseline_markers)
                succeeded = succeeded or marker_seen
        finally:
            if not self._controller.post_touch_up().wait().succeeded:
                raise RuntimeError("释放连线触点失败")
        return succeeded

    def _new_marker_seen(
        self,
        observation: BattleObservation,
        baseline_markers: list[ChainEndpointMarker] | tuple[ChainEndpointMarker, ...],
    ) -> bool:
        """只将当前截图中新出现的大菱形作为连线成功证据。"""
        if observation.board is None:
            return False
        job = self._controller.post_screencap().wait()
        if not job.succeeded:
            return False
        image = job.get()
        if not image.size:
            return False
        result = self._previewer.recognize_chain_endpoint(image, observation.board)
        if result is None:
            return False
        return len(result.markers) > len(baseline_markers)
