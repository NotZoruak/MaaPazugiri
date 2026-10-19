"""验证战斗动作的正常结束、连续未确认计时和停止行为。"""

from __future__ import annotations

import unittest

from agent.battle.session import BattleOutcome, BattleSession, BattleState


class FakeClock:
    """提供无需实际等待的单调测试时钟。"""

    def __init__(self) -> None:
        self.now = 0.0

    def clock(self) -> float:
        """返回测试时间。"""
        return self.now

    def sleep(self, seconds: float) -> None:
        """推进测试时间。"""
        self.now += seconds


class BattleSessionTests(unittest.TestCase):
    """核对业务失败与动作失败，以及单次战斗的计时隔离。"""

    def setUp(self) -> None:
        self.clock = FakeClock()
        self.session = BattleSession(clock=self.clock.clock, sleep=self.clock.sleep, poll_interval=1.0)

    def test_victory_and_defeat_both_return_normal_completion(self) -> None:
        """游戏失败也应正常交还next。"""
        for state, expected in (
            (BattleState.VICTORY, BattleOutcome.VICTORY),
            (BattleState.DEFEAT, BattleOutcome.DEFEAT),
        ):
            with self.subTest(state=state):
                result = self.session.run(lambda state=state: state, stopped=lambda: False)
                self.assertEqual(result, expected)
                self.assertTrue(result.completed)

    def test_unknown_and_inoperable_do_not_reset_watchdog(self) -> None:
        """单纯识别到战斗或动画不能重置连续未确认计时。"""
        states = (BattleState.UNKNOWN, BattleState.INOPERABLE)
        result = self.session.run(lambda: states[int(self.clock.now) % 2], stopped=lambda: False)
        self.assertEqual(result, BattleOutcome.TIMEOUT)
        self.assertFalse(result.completed)
        self.assertEqual(self.clock.now, 120.0)

    def test_operable_confirmation_resets_continuous_watchdog(self) -> None:
        """120秒不是整场战斗总时长。"""
        result = self.session.run(
            lambda: BattleState.OPERABLE if self.clock.now == 100.0 else BattleState.UNKNOWN,
            stopped=lambda: False,
        )
        self.assertEqual(result, BattleOutcome.TIMEOUT)
        self.assertEqual(self.clock.now, 220.0)

    def test_late_operable_frame_cannot_rescue_expired_watchdog(self) -> None:
        """已经连续120秒未确认时不能被迟到的可操作帧重新放行。"""

        def observe() -> BattleState:
            self.clock.now += 120.0
            return BattleState.OPERABLE

        self.assertEqual(self.session.run(observe, stopped=lambda: False), BattleOutcome.TIMEOUT)

    def test_terminal_observation_takes_priority_over_watchdog(self) -> None:
        """已确认终局时直接正常返回，不误报为计时失败。"""

        def observe() -> BattleState:
            self.clock.now += 120.0
            return BattleState.DEFEAT

        self.assertEqual(self.session.run(observe, stopped=lambda: False), BattleOutcome.DEFEAT)

    def test_stop_does_not_capture_or_report_completion(self) -> None:
        """用户停止后不再观察画面，也不伪报战斗结束。"""

        def observe() -> BattleState:
            self.fail("停止后不应继续观察")

        result = self.session.run(observe, stopped=lambda: True)
        self.assertEqual(result, BattleOutcome.CANCELLED)
        self.assertFalse(result.completed)

    def test_stop_during_observation_is_honored(self) -> None:
        """观察期间收到停止时不继续循环。"""
        stopped = False

        def observe() -> BattleState:
            nonlocal stopped
            stopped = True
            return BattleState.OPERABLE

        self.assertEqual(self.session.run(observe, stopped=lambda: stopped), BattleOutcome.CANCELLED)

    def test_reused_session_starts_a_fresh_watchdog(self) -> None:
        """不同战斗不能继承上次战斗的最后可操作时间。"""
        self.session.run(lambda: BattleState.UNKNOWN, stopped=lambda: False)
        self.clock.now += 30.0
        self.assertEqual(self.session.run(lambda: BattleState.UNKNOWN, stopped=lambda: False), BattleOutcome.TIMEOUT)
        self.assertEqual(self.clock.now, 270.0)

    def test_preview_only_runs_after_operable_confirmation(self) -> None:
        """未知、不可操作和终局不调用棋盘预演。"""
        observations = iter((BattleState.UNKNOWN, BattleState.INOPERABLE, BattleState.OPERABLE, BattleState.VICTORY))
        calls: list[float] = []
        result = self.session.run(
            lambda: next(observations),
            stopped=lambda: False,
            on_operable=lambda: calls.append(self.clock.now),
        )
        self.assertEqual(result, BattleOutcome.VICTORY)
        self.assertEqual(calls, [2.0])

    def test_expired_confirmation_does_not_start_preview(self) -> None:
        """迟到的可操作帧不触发额外棋盘工作。"""
        calls: list[bool] = []

        def observe() -> BattleState:
            self.clock.now = 120.0
            return BattleState.OPERABLE

        result = self.session.run(observe, stopped=lambda: False, on_operable=lambda: calls.append(True))
        self.assertEqual(result, BattleOutcome.TIMEOUT)
        self.assertEqual(calls, [])

    def test_preview_processing_time_is_included_in_watchdog(self) -> None:
        """预演耗时不能被当成持续确认可操作。"""
        result = self.session.run(
            lambda: BattleState.OPERABLE,
            stopped=lambda: False,
            on_operable=lambda: self.clock.sleep(120.0),
        )
        self.assertEqual(result, BattleOutcome.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
