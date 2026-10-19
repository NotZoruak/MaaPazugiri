"""管理单次战斗的结束、连续未确认超时和用户停止。"""

from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class BattleState(StrEnum):
    """表示同一帧中独立确认的战斗状态。"""

    UNKNOWN = "unknown"
    INOPERABLE = "inoperable"
    OPERABLE = "operable"
    VICTORY = "victory"
    DEFEAT = "defeat"


class BattleOutcome(StrEnum):
    """区分游戏胜负与动作超时、停止。"""

    VICTORY = "victory"
    DEFEAT = "defeat"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"

    @property
    def completed(self) -> bool:
        """胜利或游戏失败均属于本次战斗正常结束。"""
        return self in (BattleOutcome.VICTORY, BattleOutcome.DEFEAT)


class BattleSession:
    """用单调时钟管理本次战斗，不把总战斗时长限制为120秒。"""

    def __init__(
        self,
        *,
        timeout_seconds: float = 120.0,
        poll_interval: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.poll_interval = poll_interval
        self.clock = clock
        self.sleep = sleep

    def run(
        self,
        observe: Callable[[], BattleState],
        *,
        stopped: Callable[[], bool],
        on_operable: Callable[[], None] | None = None,
    ) -> BattleOutcome:
        """观察至终局或退出，仅在期限内确认可操作后调用内部处理。"""
        last_operable = self.clock()
        while True:
            if stopped():
                return BattleOutcome.CANCELLED
            state = observe()
            now = self.clock()
            if stopped():
                return BattleOutcome.CANCELLED
            if state == BattleState.VICTORY:
                return BattleOutcome.VICTORY
            if state == BattleState.DEFEAT:
                return BattleOutcome.DEFEAT
            if now - last_operable >= self.timeout_seconds:
                return BattleOutcome.TIMEOUT
            if state == BattleState.OPERABLE:
                last_operable = now
                if on_operable is not None:
                    on_operable()
                    now = self.clock()
                    if stopped():
                        return BattleOutcome.CANCELLED
                    if now - last_operable >= self.timeout_seconds:
                        return BattleOutcome.TIMEOUT
            # 轮询间隔只控制观察频率，是否可操作始终由新截图判断。
            self.sleep(min(self.poll_interval, self.timeout_seconds - (now - last_operable)))
