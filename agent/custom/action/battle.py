"""提供整场战斗的唯一Maa自定义动作入口。"""

from __future__ import annotations

import time
from dataclasses import replace

from battle.execution import MaaBattleExecutor
from battle.observation import BattleObserver
from battle.preview import BoardPreviewer
from battle.progress import BattleProgressReporter
from battle.ranbu import choose_ranbu_start, ranbu_short_link_points
from battle.recognition import BattleFrameRecognizer, Image
from battle.session import BattleSession, BattleState
from battle.strategy import BattlePhase, StrategyRuntime, decide_action, load_strategy
from battle.ultimate import UltimateChargeRecognizer
from maa.agent.agent_server import AgentServer
from maa.context import Context
from maa.custom_action import CustomAction
from utils import logger
from utils.params import parse_params
from utils.runtime_paths import get_runtime_paths


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
            ranbu_deadline: float | None = None
            executor = MaaBattleExecutor(
                controller,
                previewer,
                observer.record_normal_link_success,
                lambda: ranbu_deadline is None or time.monotonic() < ranbu_deadline,
                lambda: context.tasker.stopping,
            )
            last_state: BattleState | None = None
            current_image: Image | None = None
            last_decision_at = float("-inf")
            force_next_decision = False
            ranbu_observe_after: float | None = None

            def observe() -> BattleState:
                """重新截图并核对状态，不在本阶段发送棋盘输入。"""
                nonlocal last_state, current_image, ranbu_deadline, ranbu_observe_after
                current_image = None
                if ranbu_observe_after is not None:
                    while not context.tasker.stopping:
                        remaining = ranbu_observe_after - time.monotonic()
                        if remaining <= 0:
                            break
                        time.sleep(min(0.05, remaining))
                    ranbu_observe_after = None
                    if context.tasker.stopping:
                        return BattleState.UNKNOWN
                job = controller.post_screencap().wait()
                if not job.succeeded:
                    return BattleState.UNKNOWN
                image = job.get()
                state = BattleState.UNKNOWN if not image.size else recognizer.analyze(image)
                now = time.monotonic()
                if state == BattleState.RANBU and ranbu_deadline is None:
                    ranbu_deadline = now + 5.5
                elif state in (
                    BattleState.OPERABLE,
                    BattleState.INOPERABLE,
                    BattleState.VICTORY,
                    BattleState.DEFEAT,
                ):
                    ranbu_deadline = None
                if state in (BattleState.OPERABLE, BattleState.RANBU):
                    current_image = image
                if state != last_state:
                    logger.debug("战斗内部状态：{}", state.value)
                    last_state = state
                return state

            def process_current_frame(phase: BattlePhase) -> None:
                """串联观测、阶段评估、行动选择和单步派发。"""
                nonlocal current_image, last_decision_at, force_next_decision, ranbu_observe_after
                try:
                    now = time.monotonic()
                    if current_image is None or (not force_next_decision and now - last_decision_at < 1.0):
                        return
                    force_next_decision = False
                    image = current_image
                    configuration = context.get_node_data("Flag_LightweightRecognition")
                    lightweight = configuration.get("enabled", False) if configuration is not None else False
                    if not isinstance(lightweight, bool):
                        raise ValueError("轻量识别配置的enabled必须为布尔值")
                    observation = observer.observe(image, phase, lightweight=lightweight)
                    resolution = strategy_runtime.evaluate_stage(observation.metrics)
                    observation = observer.apply_stage_resolution(observation, resolution)
                    route_preview = None

                    def provide_normal_routes() -> tuple[object, ...]:
                        """仅在策略已选到普通连线时，规划当前帧路线。"""
                        nonlocal route_preview
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
                    if decision.kind == "ranbu_link" and (
                        observation.board is None
                        or choose_ranbu_start(observation.board, decision.reserved_color) is None
                    ):
                        if (
                            observation.board is not None
                            and ranbu_short_link_points(observation.board, decision.reserved_color) is not None
                        ):
                            decision = replace(decision, kind="ranbu_short_link", reason="当前帧普通饿鬼全部为保留色")
                        else:
                            decision = replace(decision, kind="wait", reason="当前帧没有符合保留颜色规则的乱舞起点")
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
                    if (
                        decision.kind in ("ranbu_link", "ranbu_short_link")
                        and ranbu_deadline is not None
                        and time.monotonic() >= ranbu_deadline
                    ):
                        logger.debug("乱舞已超过新一轮启动期限，不再开始扫动。")
                        progress.report(
                            phase,
                            resolution,
                            replace(decision, kind="wait", reason="本轮乱舞启动期限已过，等待新画面"),
                        )
                    else:
                        progress.report(phase, resolution, decision)
                        executor.execute_once(observation, resolution, decision)
                        force_next_decision = decision.kind in ("ultimate", "bomb", "normal_link")
                        if decision.kind in ("ranbu_link", "ranbu_short_link"):
                            ranbu_observe_after = time.monotonic() + 1.0
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
