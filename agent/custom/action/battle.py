"""提供整场战斗的唯一Maa自定义动作入口。"""

from __future__ import annotations

import time
from dataclasses import asdict, replace

from battle.execution import MaaBattleExecutor
from battle.observation import BattleObserver
from battle.preview import BoardPreviewer
from battle.progress import BattleProgressReporter
from battle.ranbu import RanbuStartConfirmation
from battle.ranbu_feedback import RanbuHoldFeedback
from battle.recognition import BattleFrameRecognizer, Image
from battle.session import BattleSession, BattleState
from battle.stability import NormalLinkStability
from battle.strategy import BattlePhase, StrategyRuntime, decide_action, load_strategy
from battle.ultimate import UltimateChargeRecognizer
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger
from utils.params import parse_params
from utils.runtime_paths import get_runtime_paths
from vision.model import BoardResult
from vision.ranbu_coverage import inspect_ranbu_coverage


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
            previewer = BoardPreviewer(
                paths.resource_dir / "base/image/board/development/profile.json",
                paths.debug_dir / "battle-preview",
            )
            ultimate_recognizer = UltimateChargeRecognizer(paths.resource_dir / "base/image/ultimate")
            observer = BattleObserver(
                context.run_recognition_direct,
                previewer,
                ultimate_recognizer,
                strategy.initial_stage_id,
            )
            last_state: BattleState | None = None

            def inspect_ranbu_hold(image: Image, board: BoardResult, color: str | None) -> RanbuHoldFeedback:
                """先检查残留，仅在确实需要补扫时核对同帧状态。"""
                if context.tasker.stopping:
                    return RanbuHoldFeedback(BattleState.UNKNOWN, None, state_checked=False)
                coverage = inspect_ranbu_coverage(image, board.board_region, color)
                if coverage.status != "present" or context.tasker.stopping:
                    return RanbuHoldFeedback(
                        BattleState.UNKNOWN, None, coverage.status, asdict(coverage), state_checked=False
                    )
                state = recognizer.analyze(image)
                return RanbuHoldFeedback(state, None, coverage.status, asdict(coverage))

            executor = MaaBattleExecutor(
                controller,
                previewer,
                observer.record_normal_link_success,
                lambda: last_state == BattleState.RANBU,
                lambda: context.tasker.stopping,
                inspect_ranbu_hold=inspect_ranbu_hold,
            )
            current_image: Image | None = None
            last_decision_at = float("-inf")
            force_next_decision = False
            observe_after: float | None = None
            normal_link_stability = NormalLinkStability()
            ranbu_start_confirmation = RanbuStartConfirmation()

            def observe() -> BattleState:
                """重新截图并核对状态，不在本阶段发送棋盘输入。"""
                nonlocal last_state, current_image, observe_after, force_next_decision
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
                if state != BattleState.OPERABLE:
                    normal_link_stability.reset()
                if state != BattleState.RANBU:
                    ranbu_start_confirmation.reset()
                if state in (BattleState.OPERABLE, BattleState.RANBU):
                    current_image = image
                if state != last_state:
                    force_next_decision = True
                    logger.debug("战斗内部状态：{}", state.value)
                    last_state = state
                return state

            def process_current_frame(phase: BattlePhase) -> None:
                """串联观测、阶段评估、行动选择和单步派发。"""
                nonlocal current_image, last_decision_at, force_next_decision, observe_after
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
                    ranbu_layout = (
                        ranbu_start_confirmation.pending_layout_for(image) if phase == BattlePhase.RANBU else None
                    )
                    required_metrics = strategy_runtime.required_visual_metrics(phase)
                    observation = observer.observe(
                        image,
                        phase,
                        lightweight=lightweight,
                        ranbu_layout=ranbu_layout,
                        required_metrics=required_metrics,
                    )
                    if phase == BattlePhase.RANBU:
                        logger.debug("乱舞本帧未检查的策略指标：{}。", observation.unchecked_metrics)
                    board_stable = False
                    if phase == BattlePhase.OPERABLE:
                        board_stable = normal_link_stability.observe(observation.board)
                        previewer.save_stability_report({"phase": phase.value, **normal_link_stability.last_report})
                        logger.debug("普通连线棋盘稳定检查：{}。", normal_link_stability.last_report["reason"])
                    resolution = strategy_runtime.evaluate_stage(observation.metrics)
                    observation = observer.apply_stage_resolution(observation, resolution)
                    if resolution.applied_transitions:
                        ranbu_start_confirmation.reset()
                    route_preview = None
                    waiting_for_stability = False

                    def provide_normal_routes() -> tuple[object, ...]:
                        """仅在策略已选到普通连线时，规划当前帧路线。"""
                        nonlocal route_preview, waiting_for_stability
                        if not board_stable:
                            waiting_for_stability = True
                            return ()
                        if observation.board is None:
                            return ()
                        route_preview = previewer.preview_routes(image, observation.board)
                        logger.debug(
                            "路线预演：候选边={}，不确定边={}，拟选编号={}。",
                            len(route_preview.candidate_edges),
                            len(route_preview.uncertain_edges),
                            route_preview.selected_route.object_ids if route_preview.selected_route else (),
                        )
                        return tuple(route_preview.routes)

                    decision = decide_action(
                        strategy_runtime,
                        phase,
                        metrics=observation.metrics,
                        ultimate_states=observation.metrics.ultimate_ready,
                        available_bomb_levels=observation.available_bomb_levels,
                        route_provider=provide_normal_routes,
                        stage_resolution=resolution,
                    )
                    if waiting_for_stability or (decision.kind == "normal_link" and not board_stable):
                        waiting_for_stability = True
                        decision = replace(decision, kind="wait", route=None, reason="等待棋盘稳定")
                    if decision.kind == "ranbu_link":
                        confirmed = ranbu_start_confirmation.observe(
                            observation.board, decision.reserved_color, image=image
                        )
                        previewer.save_stability_report(
                            {"phase": phase.value, "check": "ranbu_start", **ranbu_start_confirmation.last_report}
                        )
                        logger.debug("乱舞预选点颜色检查：{}。", ranbu_start_confirmation.last_report)
                        if confirmed is not None:
                            decision = replace(
                                decision,
                                ranbu_start=confirmed[0],
                                ranbu_color=confirmed[1],
                                ranbu_frame_id=observation.frame_id,
                                ranbu_body_size=ranbu_start_confirmation.confirmed_body_size,
                            )
                        else:
                            decision = replace(decision, kind="wait", reason="等待起点颜色确认")
                    elif phase == BattlePhase.RANBU:
                        ranbu_start_confirmation.reset()
                        if decision.kind == "ranbu_short_link":
                            decision = replace(decision, kind="wait", reason="当前帧没有非保留色乱舞起点")
                    logger.debug(
                        "战斗决策：阶段={}，转变链={}，状态={}，整场普通手数={}，本阶段普通手数={}，敌方血量={}%，我方血量={}%，奥义={}，炸弹等级={}，行动={}。",
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
                    executor.execute_once(observation, resolution, decision)
                    if decision.kind != "wait":
                        normal_link_stability.reset()
                        ranbu_start_confirmation.reset()
                    force_next_decision = waiting_for_stability or decision.kind in (
                        "ultimate",
                        "bomb",
                        "normal_link",
                    )
                    if decision.kind in ("normal_link", "ranbu_link", "ranbu_short_link"):
                        released_at = getattr(executor, "last_gesture_released_at", None)
                        after_release_seconds = 0.5 if phase == BattlePhase.RANBU else 1.0
                        if phase == BattlePhase.RANBU and isinstance(released_at, (int, float)):
                            observe_after = released_at + after_release_seconds
                        else:
                            observe_after = time.monotonic() + after_release_seconds
                    last_decision_at = time.monotonic()
                finally:
                    current_image = None

            logger.debug("进入战斗动作：已载入策略{}，每轮观测后最多执行一个动作。", strategy.id)
            outcome = BattleSession().run(
                observe,
                stopped=lambda: context.tasker.stopping,
                on_operable=lambda: process_current_frame(BattlePhase.OPERABLE),
                on_ranbu=lambda: process_current_frame(BattlePhase.RANBU),
            )
        except Exception:
            logger.exception("战斗自定义动作发生运行错误")
            return CustomAction.RunResult(success=False)
        logger.debug("战斗动作结束：{}，正常完成={}", outcome.value, outcome.completed)
        return CustomAction.RunResult(success=outcome.completed)
