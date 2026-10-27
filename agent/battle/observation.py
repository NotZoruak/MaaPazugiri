"""收集单张战斗截图的识别结果，不负责阶段切换或行动规划。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

from vision.model import BoardResult

from .preview import BoardPreviewer
from .recognition import Image, NativeRecognition, recognize_hp_percent
from .session import BattleState
from .strategy import BattlePhase, StageResolution, StrategyMetrics
from .ultimate import UltimateChargeRecognizer


@dataclass(frozen=True)
class BattleObservation:
    """保存同一帧的战斗阶段、棋盘与策略指标。"""

    frame_id: str
    phase: BattlePhase
    image: Image
    board: BoardResult | None
    metrics: StrategyMetrics
    available_bomb_levels: tuple[int, ...] = ()
    ultimate_click_points: dict[int, tuple[int, int]] = field(default_factory=dict)
    unchecked_metrics: tuple[str, ...] = ()


class BattleObserver:
    """只读取输入截图，不负责阶段判断、路线搜索或设备输入。"""

    def __init__(
        self,
        recognize: NativeRecognition,
        board_previewer: BoardPreviewer,
        ultimate_recognizer: UltimateChargeRecognizer,
        initial_stage_id: str,
    ) -> None:
        self._recognize = recognize
        self._board_previewer = board_previewer
        self._ultimate_recognizer = ultimate_recognizer
        self._current_stage_id = initial_stage_id
        self._battle_normal_links = 0
        self._stage_normal_links = 0

    def observe(
        self,
        image: Image,
        phase: BattlePhase,
        *,
        lightweight: bool = False,
        ranbu_layout: tuple[tuple[float, float, float, float], str] | None = None,
        required_metrics: frozenset[str] | None = None,
    ) -> BattleObservation:
        """在普通可操作或乱舞时识别棋盘，并读取血量、炸弹和奥义。"""
        frame_id = uuid4().hex
        if phase == BattlePhase.RANBU and ranbu_layout is not None:
            region, profile = ranbu_layout
            board = BoardResult(
                frame_id,
                (image.shape[1], image.shape[0]),
                region,
                profile,
                {"mode": "ranbu_start_point_only", "skipped_kinds": ["ghoul", "special_ghoul", "bomb", "obstacle"]},
            )
        else:
            board = self._board_previewer.recognize_board(
                image,
                frame_id,
                lightweight=lightweight or phase == BattlePhase.RANBU,
            )

        all_metrics = ("enemy_hp_percent", "ally_hp_percent", "ultimate_ready")
        required = frozenset(all_metrics) if required_metrics is None else required_metrics
        unchecked = tuple(name for name in all_metrics if name not in required)
        enemy_hp = recognize_hp_percent(self._recognize, image, "enemy") if "enemy_hp_percent" in required else None
        ally_hp = recognize_hp_percent(self._recognize, image, "ally") if "ally_hp_percent" in required else None
        ultimate_slots = (
            self._ultimate_recognizer.analyze(image, frame_id=frame_id, phase=BattleState(phase.value)).slots
            if "ultimate_ready" in required
            else []
        )
        ultimate_states = {
            slot.slot: True if slot.state.value == "ready" else False if slot.state.value == "charging" else None
            for slot in ultimate_slots
        }
        ultimate_points = {
            slot.slot: (
                round(slot.ready_box[0] + slot.ready_box[2] / 2),
                round(slot.ready_box[1] + slot.ready_box[3] / 2),
            )
            for slot in ultimate_slots
            if slot.state.value == "ready" and slot.ready_box is not None
        }
        bomb_levels = tuple(
            sorted(
                {
                    item.bomb_level
                    for item in board.objects
                    if item.kind == "bomb"
                    and item.bomb_level in (1, 2, 3)
                    and item.candidate_point is not None
                    and item.quality in ("identified", "single_feature")
                },
                reverse=True,
            )
        )
        metrics = StrategyMetrics(
            battle_normal_links=self._battle_normal_links,
            stage_normal_links=self._stage_normal_links,
            enemy_hp_percent=enemy_hp,
            ally_hp_percent=ally_hp,
            ultimate_ready=ultimate_states,
            frenzy_active=phase == BattlePhase.RANBU,
        )
        return BattleObservation(frame_id, phase, image, board, metrics, bomb_levels, ultimate_points, unchecked)

    def apply_stage_resolution(
        self,
        observation: BattleObservation,
        resolution: StageResolution,
    ) -> BattleObservation:
        """阶段切换后清零本阶段手数，并更新本帧供行动选择使用的指标。"""
        if resolution.stage.id != self._current_stage_id:
            self._current_stage_id = resolution.stage.id
            self._stage_normal_links = 0
        metrics = replace(
            observation.metrics,
            battle_normal_links=self._battle_normal_links,
            stage_normal_links=self._stage_normal_links,
        )
        return replace(observation, metrics=metrics)

    def record_normal_link_success(self) -> None:
        """仅在普通连线过程中识别到新菱形标志后递增手数。"""
        self._battle_normal_links += 1
        self._stage_normal_links += 1
