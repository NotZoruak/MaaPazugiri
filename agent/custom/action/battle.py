"""提供整场战斗的唯一Maa自定义动作入口。"""

from __future__ import annotations

import time
from dataclasses import replace

from battle.candidates import build_route_preview
from battle.execution import MaaBattleExecutor
from battle.observation import BattleObserver
from battle.preview import BoardPreviewer
from battle.progress import BattleProgressReporter
from battle.ranbu import RanbuStartConfirmation
from battle.recognition import BattleFrameRecognizer, Image
from battle.session import BattleSession, BattleState
from battle.stability import NormalLinkStability, significantly_changed_route_objects
from battle.strategy import BattlePhase, StrategyRuntime, decide_action, load_strategy
from battle.ultimate import UltimateChargeRecognizer
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger
from utils.debug_options import load_debug_capture_options
from utils.params import parse_params
from utils.runtime_paths import get_runtime_paths
from vision.model import BoardResult, is_black_ghoul


def _normal_route_key(color: str, points: tuple[tuple[float, float], ...]) -> tuple[str, tuple[tuple[int, int], ...]]:
    """生成不区分起止方向的路线标识，供同一盘面过滤失败路线。"""
    quantized = tuple((round(x / 12), round(y / 12)) for x, y in points)
    return color, min(quantized, tuple(reversed(quantized)))


@AgentServer.custom_action("BattleAction")
class BattleAction(CustomAction):
    """在内部观察至胜负或连续未确认超时，外部只接收动作返回结果。"""

    def run(self, context: Context, argv: CustomAction.RunArg) -> CustomAction.RunResult:
        """胜利或游戏失败正常返回，超时与运行错误返回动作失败。"""
        try:
            controller = context.tasker.controller
            recognizer = BattleFrameRecognizer(context.run_recognition_direct)
            paths = get_runtime_paths()
            parameters = parse_params(argv.custom_action_param)
            strategy_id = parameters.get("strategy_id", "default")
            strategy = load_strategy(strategy_id, paths.config_dir)
            strategy_runtime = StrategyRuntime(strategy)
            progress = BattleProgressReporter(strategy)
            debug_capture_options = load_debug_capture_options(paths.config_dir)
            previewer = BoardPreviewer(
                paths.resource_dir / "base/image/board/development/profile.json",
                paths.debug_dir / "battle-preview",
                save_draw=debug_capture_options.save_draw,
            )
            ultimate_recognizer = UltimateChargeRecognizer(paths.resource_dir / "base/image/ultimate")
            observer = BattleObserver(
                context.run_recognition_direct,
                previewer,
                ultimate_recognizer,
                strategy.initial_stage_id,
            )
            last_state: BattleState | None = None

            normal_link_lightweight = False

            def confirm_normal_link_board_change(
                before_board: BoardResult, route_object_ids: tuple[int, ...]
            ) -> bool:
                """菱形未命中时，等待至少两个路线目标持续两帧明显变化。"""
                target_count = len(route_object_ids)
                required_count = 2
                if target_count < 2:
                    return False

                deadline = time.monotonic() + 6.0
                previous_changed_ids: frozenset[int] = frozenset()
                while not context.tasker.stopping and time.monotonic() < deadline:
                    time.sleep(0.15)
                    if context.tasker.stopping:
                        break
                    job = controller.post_screencap().wait()
                    if not job.succeeded:
                        previous_changed_ids = frozenset()
                        continue
                    image = job.get()
                    if not image.size or recognizer.analyze(image) not in (BattleState.OPERABLE, BattleState.RANBU):
                        previous_changed_ids = frozenset()
                        continue
                    current_board = previewer.recognize_board(
                        image,
                        f"link-board-confirm-{time.monotonic_ns()}",
                        lightweight=normal_link_lightweight,
                    )
                    changed_ids = significantly_changed_route_objects(
                        before_board, current_board, route_object_ids
                    )
                    if len(changed_ids) < required_count:
                        previous_changed_ids = frozenset()
                        continue
                    persistent_ids = previous_changed_ids & changed_ids
                    if len(persistent_ids) >= required_count:
                        return True
                    previous_changed_ids = changed_ids
                return False

            executor = MaaBattleExecutor(
                controller,
                previewer,
                observer.record_normal_link_success,
                lambda: last_state == BattleState.RANBU,
                lambda: context.tasker.stopping,
                confirm_normal_link_board_change=confirm_normal_link_board_change,
            )
            current_image: Image | None = None
            last_decision_at = float("-inf")
            force_next_decision = False
            observe_after: float | None = None
            last_ultimate_confirmation: BattleState | None = None
            normal_link_stability = NormalLinkStability()
            ranbu_start_confirmation = RanbuStartConfirmation()
            normal_route_failures: dict[tuple[str, tuple[tuple[int, int], ...]], int] = {}
            failed_normal_routes: set[tuple[str, tuple[tuple[int, int], ...]]] = set()
            stable_route_board_signature: tuple[object, ...] | None = None

            def observe() -> BattleState:
                """重新截图并核对状态，不在本阶段发送棋盘输入。"""
                nonlocal last_state, current_image, observe_after, force_next_decision
                nonlocal last_ultimate_confirmation
                current_image = None
                if observe_after is not None:
                    while not context.tasker.stopping:
                        remaining = observe_after - time.monotonic()
                        if remaining <= 0:
                            break
                        time.sleep(min(0.05, remaining))
                    observe_after = None
                    if context.tasker.stopping:
                        return BattleState.UNKNOWN
                job = controller.post_screencap().wait()
                if not job.succeeded:
                    normal_link_stability.reset()
                    ranbu_start_confirmation.reset()
                    last_state = BattleState.UNKNOWN
                    return BattleState.UNKNOWN
                image = job.get()
                state = BattleState.UNKNOWN if not image.size else recognizer.analyze(image)
                if state != BattleState.RANBU:
                    ranbu_start_confirmation.reset()
                if state != BattleState.OPERABLE:
                    normal_link_stability.reset()
                if state not in (
                    BattleState.ULTIMATE_TAP_CONFIRMATION,
                    BattleState.ULTIMATE_SWIPE_CONFIRMATION,
                    BattleState.UNKNOWN,
                ):
                    last_ultimate_confirmation = None
                if state in (
                    BattleState.OPERABLE,
                    BattleState.RANBU,
                    BattleState.ULTIMATE_TAP_CONFIRMATION,
                    BattleState.ULTIMATE_SWIPE_CONFIRMATION,
                ):
                    current_image = image
                if state != last_state:
                    force_next_decision = True
                    logger.debug("战斗内部状态：{}", state.value)
                    last_state = state
                return state

            def process_ultimate_confirmation(state: BattleState) -> None:
                """同一确认提示只执行一组固定手势，等待画面离开后再允许下一组。"""
                nonlocal current_image, last_ultimate_confirmation
                if current_image is None or state == last_ultimate_confirmation:
                    return
                last_ultimate_confirmation = state
                executor.execute_ultimate_confirmation(state)
                current_image = None

            def process_current_frame(phase: BattlePhase) -> None:
                """串联观测、阶段评估、行动选择和单步派发。"""
                nonlocal current_image, last_decision_at, force_next_decision, observe_after, normal_link_lightweight
                try:
                    now = time.monotonic()
                    if current_image is None or (
                        phase != BattlePhase.RANBU and not force_next_decision and now - last_decision_at < 1.0
                    ):
                        return
                    force_next_decision = False
                    image = current_image
                    configuration = context.get_node_data("Flag_LightweightRecognition")
                    lightweight = configuration.get("enabled", False) if configuration is not None else False
                    if not isinstance(lightweight, bool):
                        raise ValueError("轻量识别配置的enabled必须为布尔值")
                    normal_link_lightweight = lightweight
                    required_metrics = strategy_runtime.required_visual_metrics(phase)
                    observation = observer.observe(
                        image,
                        phase,
                        lightweight=lightweight,
                        required_metrics=required_metrics,
                    )
                    if phase == BattlePhase.RANBU:
                        logger.debug("乱舞本帧未检查的策略指标：{}", observation.unchecked_metrics)
                    board_stable = False
                    if phase == BattlePhase.OPERABLE:
                        board_stable = normal_link_stability.observe(observation.board)
                        previewer.save_stability_report({"phase": phase.value, **normal_link_stability.last_report})
                        logger.debug("普通连线棋盘稳定检查：{}", normal_link_stability.last_report["reason"])
                    resolution = strategy_runtime.evaluate_stage(observation.metrics)
                    observation = observer.apply_stage_resolution(observation, resolution)
                    route_preview = None
                    waiting_for_stability = False

                    def refresh_route_failures(route_board: BoardResult) -> None:
                        """只在当前规划盘面变化时清除旧路线失败记录。"""
                        nonlocal stable_route_board_signature
                        board_signature = (
                            tuple(
                                sorted(
                                    (
                                        item.kind,
                                        item.color,
                                        -1 if item.enhanced is None else int(item.enhanced),
                                        round(item.body_box[0] / 12),
                                        round(item.body_box[1] / 12),
                                        round(item.body_box[2] / 12),
                                        round(item.body_box[3] / 12),
                                    )
                                    for item in route_board.objects
                                )
                            ),
                            tuple(round(value / 12) for value in route_board.board_region),
                            tuple(
                                sorted(
                                    tuple(round(value / 12) for value in region)
                                    for region in route_board.route_exclusion_regions
                                )
                            ),
                        )
                        if board_signature != stable_route_board_signature:
                            normal_route_failures.clear()
                            failed_normal_routes.clear()
                            stable_route_board_signature = board_signature

                    def provide_normal_routes() -> tuple[object, ...]:
                        """仅在策略已选到普通连线时，规划当前帧路线。"""
                        nonlocal route_preview, waiting_for_stability, stable_route_board_signature
                        if not board_stable:
                            waiting_for_stability = True
                            return ()
                        route_board = normal_link_stability.route_board
                        if route_board is None:
                            return ()
                        refresh_route_failures(route_board)
                        route_preview = previewer.preview_routes(image, route_board)
                        available_routes = tuple(
                            route
                            for route in route_preview.routes
                            if _normal_route_key(route.color, route.points) not in failed_normal_routes
                        )
                        logger.debug(
                            "路线预演：候选边={}，不确定边={}，屏蔽失败路线={}，拟选编号={}",
                            len(route_preview.candidate_edges),
                            len(route_preview.uncertain_edges),
                            len(failed_normal_routes),
                            available_routes[0].object_ids if available_routes else (),
                        )
                        return available_routes

                    enhanced_routes = ()
                    enhanced_route_ready = phase == BattlePhase.RANBU or board_stable
                    enhanced_present = observation.board is not None and any(
                        item.kind == "ghoul" and item.enhanced is True for item in observation.board.objects
                    )
                    if enhanced_present and observation.board is not None:
                        if phase == BattlePhase.RANBU:
                            refresh_route_failures(observation.board)
                        enhanced_routes = tuple(
                            route
                            for route in build_route_preview(observation.board).routes
                            if route.enhanced_object_ids
                        )
                        if phase == BattlePhase.OPERABLE:
                            route_board = normal_link_stability.route_board
                            if board_stable and route_board is not None:
                                refresh_route_failures(route_board)
                                if route_board is not observation.board:
                                    enhanced_routes = tuple(
                                        route
                                        for route in build_route_preview(route_board).routes
                                        if route.enhanced_object_ids
                                    )
                                enhanced_routes = tuple(
                                    route
                                    for route in enhanced_routes
                                    if _normal_route_key(route.color, route.points) not in failed_normal_routes
                                )
                                enhanced_route_ready = True
                            else:
                                enhanced_route_ready = False
                        if enhanced_routes and not enhanced_route_ready:
                            waiting_for_stability = True

                    black_routes = ()
                    black_route_ready = phase == BattlePhase.RANBU or board_stable
                    if (
                        observation.board is not None
                        and sum(is_black_ghoul(item) for item in observation.board.objects) >= 2
                    ):
                        black_routes = tuple(build_route_preview(observation.board, black_only=True).routes)
                        route_board = normal_link_stability.route_board
                        if black_routes and phase == BattlePhase.OPERABLE:
                            if board_stable and route_board is not None and route_board is not observation.board:
                                allowed_black = tuple(build_route_preview(route_board, black_only=True).routes)
                                black_route_ready = bool(allowed_black)
                                if allowed_black:
                                    black_routes = allowed_black
                            waiting_for_stability = not black_route_ready and not enhanced_routes
                        logger.debug(
                            "黑色连线优先检查：候选路线={}，允许起手={}", len(black_routes), black_route_ready
                        )
                    decision = decide_action(
                        strategy_runtime,
                        phase,
                        metrics=observation.metrics,
                        ultimate_states=observation.metrics.ultimate_ready,
                        available_bomb_levels=observation.available_bomb_levels,
                        route_provider=provide_normal_routes,
                        enhanced_routes=enhanced_routes,
                        enhanced_route_ready=enhanced_route_ready,
                        black_routes=black_routes,
                        black_route_ready=black_route_ready,
                        stage_resolution=resolution,
                    )
                    if waiting_for_stability or (
                        phase == BattlePhase.OPERABLE and decision.kind == "normal_link" and not board_stable
                    ):
                        waiting_for_stability = True
                        decision = replace(decision, kind="wait", route=None, reason="等待棋盘稳定")
                    if decision.kind == "ranbu_link":
                        selected = ranbu_start_confirmation.observe(
                            observation.board,
                            decision.reserved_color,
                            image=image,
                        )
                        confirmed_body_size = ranbu_start_confirmation.confirmed_body_size
                        logger.debug("乱舞起点连续两帧确认：{}", ranbu_start_confirmation.last_report)
                        if selected is not None and confirmed_body_size is not None:
                            decision = replace(
                                decision,
                                ranbu_start=selected[0],
                                ranbu_color=selected[1],
                                ranbu_frame_id=observation.frame_id,
                                ranbu_body_size=confirmed_body_size,
                            )
                        else:
                            decision = replace(decision, kind="wait", reason="等待乱舞起点连续两帧稳定")
                    elif phase == BattlePhase.RANBU:
                        if decision.kind == "ranbu_short_link":
                            decision = replace(decision, kind="wait", reason="当前帧没有非保留色乱舞起点")
                    logger.debug(
                        "战斗决策：阶段={}，转变链={}，状态={}，整场普通手数={}，本阶段普通手数={}，敌方血量={}%，我方血量={}%，奥义={}，炸弹等级={}，行动={}",
                        resolution.stage.id,
                        resolution.transition_chain,
                        phase.value,
                        observation.metrics.battle_normal_links,
                        observation.metrics.stage_normal_links,
                        "未知"
                        if observation.metrics.enemy_hp_percent is None
                        else f"{observation.metrics.enemy_hp_percent:.2f}",
                        "未知"
                        if observation.metrics.ally_hp_percent is None
                        else f"{observation.metrics.ally_hp_percent:.2f}",
                        dict(observation.metrics.ultimate_ready),
                        observation.available_bomb_levels,
                        decision,
                    )
                    progress.report(phase, resolution, decision)
                    execution_observation = observation
                    if (
                        phase == BattlePhase.OPERABLE
                        and decision.kind in ("normal_link", "black_link")
                        and normal_link_stability.route_board is not None
                    ):
                        execution_observation = replace(observation, board=normal_link_stability.route_board)
                    executor.execute_once(execution_observation, resolution, decision)
                    if (
                        decision.kind == "normal_link"
                        and decision.route is not None
                        and getattr(executor, "last_normal_link_succeeded", None) is False
                    ):
                        route_key = _normal_route_key(decision.route.color, decision.route.points)
                        normal_route_failures[route_key] = normal_route_failures.get(route_key, 0) + 1
                        if normal_route_failures[route_key] >= 2:
                            failed_normal_routes.add(route_key)
                            logger.warning(
                                "当前盘面路线从两端尝试后仍未确认成功，暂时排除并重新规划：颜色={}，对象序列={}",
                                decision.route.color,
                                decision.route.object_ids,
                            )
                        else:
                            logger.warning(
                                "当前盘面路线未确认成功，将在新帧从另一端尝试：颜色={}，对象序列={}",
                                decision.route.color,
                                decision.route.object_ids,
                            )
                    if decision.kind != "wait":
                        normal_link_stability.reset()
                    force_next_decision = waiting_for_stability or decision.kind in (
                        "ultimate",
                        "bomb",
                        "normal_link",
                        "black_link",
                    )
                    if decision.kind in ("normal_link", "black_link", "ranbu_link", "ranbu_short_link"):
                        released_at = getattr(executor, "last_gesture_released_at", None)
                        after_release_seconds = 0.0 if phase == BattlePhase.RANBU else 1.0
                        if phase == BattlePhase.RANBU and isinstance(released_at, (int, float)):
                            observe_after = released_at + after_release_seconds
                        else:
                            observe_after = time.monotonic() + after_release_seconds
                    last_decision_at = time.monotonic()
                finally:
                    current_image = None

            logger.debug("进入战斗动作：已载入策略{}，每轮观测后最多执行一个动作", strategy.id)
            outcome = BattleSession().run(
                observe,
                stopped=lambda: context.tasker.stopping,
                on_operable=lambda: process_current_frame(BattlePhase.OPERABLE),
                on_ranbu=lambda: process_current_frame(BattlePhase.RANBU),
                on_ultimate_confirmation=process_ultimate_confirmation,
            )
        except Exception:
            logger.exception("战斗自定义动作发生运行错误")
            return CustomAction.RunResult(success=False)
        logger.debug("战斗动作结束：{}，正常完成={}", outcome.value, outcome.completed)
        return CustomAction.RunResult(success=outcome.completed)
