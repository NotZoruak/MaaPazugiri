"""验证已确认的黑黑连接规则及其高于全部策略动作的优先级。"""

from __future__ import annotations

import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import test_battle_execution as execution_fixtures
import test_battle_stability as stability_fixtures
from battle.candidates import CandidateRoute, build_route_preview
from battle.execution import MaaBattleExecutor
from battle.strategy import BattlePhase, StrategyDocument, StrategyRuntime, decide_action
from test_battle_candidates import board, ghoul
from vision.connectability import assess_ghoul_connectability
from vision.model import Hit
from vision.recognizer import BoardRecognizer


def black(identifier, x, y=400):
    """构造当前帧中具有可信触点的黑色饿鬼。"""
    return replace(ghoul(identifier, x, y, "black"), kind="special_ghoul", subtype="black", label="special_ghoul_black")


def black_route():
    """提供已通过当前帧几何检查的黑色路线。"""
    return CandidateRoute("black", (1, 2), ((100, 400), (190, 400)), 0.1, 90)


class BlackPriorityTests(unittest.TestCase):
    """黑色仅与黑色连接，且不能被奥义、炸弹或长彩色路线抢占。"""

    def test_black_uses_same_distance_boundary_as_ordinary_color(self) -> None:
        """黑黑几何判定与同尺寸普通同色对象一致，混色始终拒绝。"""
        for offset in (90, 144, 200):
            result = board(black(1, 100), black(2, 100 + offset))
            ordinary = board(ghoul(1, 100), ghoul(2, 100 + offset))
            self.assertEqual(
                assess_ghoul_connectability(result, 1, 2).status,
                assess_ghoul_connectability(ordinary, 1, 2).status,
            )
        mixed = board(black(1, 100), ghoul(2, 190))
        self.assertEqual(assess_ghoul_connectability(mixed, 1, 2).status.value, "rejected")

    def test_preview_prefers_two_black_over_a_longer_colored_route(self) -> None:
        """两只可连黑色优先于更多普通饿鬼，不混入任何彩色对象。"""
        result = build_route_preview(board(black(1, 100), black(2, 190), *(ghoul(i, 90 * i, 650) for i in range(3, 7))))
        self.assertIsNotNone(result.selected_route)
        self.assertEqual(result.selected_route.color, "black")
        self.assertEqual(set(result.selected_route.object_ids), {1, 2})

    def test_isolated_or_untrusted_black_does_not_block_ordinary_routes(self) -> None:
        """孤立黑色或没有可信触点时不能凭颜色制造可连接路线。"""
        for special in (black(1, 100), replace(black(1, 100), candidate_point=None)):
            result = build_route_preview(board(special, ghoul(3, 400), ghoul(4, 490)))
            self.assertEqual(result.selected_route.color, "red")

    def test_distant_black_pair_does_not_preempt_an_ordinary_connection(self) -> None:
        """两只黑色超出连接范围时，不能仅凭黑色数量抢占普通路线。"""
        result = build_route_preview(board(black(1, 100), black(2, 600), ghoul(3, 300, 650), ghoul(4, 390, 650)))
        self.assertEqual(result.selected_route.color, "red")

    def test_black_component_is_processed_before_colored_search_budget(self) -> None:
        """搜索预算很小时先保留可连黑色，不能被较大彩色分量耗尽预算。"""
        result = build_route_preview(
            board(black(99, 100), black(100, 190), *(ghoul(i, 90 * i, 650) for i in range(1, 7))),
            max_expansions=1,
        )
        self.assertEqual(result.selected_route.color, "black")
        self.assertEqual(set(result.selected_route.object_ids), {99, 100})

    def test_black_preempts_ready_ultimate_bomb_and_frenzy(self) -> None:
        """普通和乱舞阶段都先执行黑色连线，忽略普通颜色优先设置。"""
        for phase in (BattlePhase.OPERABLE, BattlePhase.RANBU):
            with self.subTest(phase=phase):
                decision = decide_action(
                    StrategyRuntime(StrategyDocument.builtin_default()),
                    phase,
                    ultimate_states={1: True, 2: True, 3: True},
                    available_bomb_levels=(3,),
                    routes=(black_route(),),
                )
                self.assertEqual(decision.kind, "black_link")
                self.assertEqual(decision.route.color, "black")

    def test_real_action_waits_for_stability_then_prioritizes_black(self) -> None:
        """真实普通入口先取得稳定证据，再执行当前帧黑色路线。"""
        replay = stability_fixtures.BattleStabilityActionTests()
        operations, _ = replay.run_replay([board(black(1, 100), black(2, 190), ghoul(3, 450)) for _ in range(2)])
        self.assertEqual([(index, decision.kind) for index, decision in operations], [(2, "black_link")])

    def test_black_in_uncertain_bottom_band_waits_until_full_stability(self) -> None:
        """底部容差不能放行穿越不确定带的黑色路线，也不能让彩色抢先连接。"""
        upper = [ghoul(3, 100), ghoul(4, 190)]
        blacks = [black(1, 100, 910), black(2, 190, 910)]
        replay = stability_fixtures.BattleStabilityActionTests()
        operations, _ = replay.run_replay(
            [board(*upper, *blacks, ghoul(5, 500, 910)), board(*upper, *blacks), board(*upper, *blacks)]
        )
        self.assertEqual([(index, decision.kind) for index, decision in operations], [(3, "black_link")])

    def test_unstable_black_route_waits_instead_of_using_ultimate(self) -> None:
        """检测到可连黑色但尚未稳定时，不能用奥义或炸弹抢先改变棋盘。"""
        provider = Mock(side_effect=AssertionError("不应规划普通颜色路线"))
        decision = decide_action(
            StrategyRuntime(StrategyDocument.builtin_default()),
            BattlePhase.OPERABLE,
            ultimate_states={1: True},
            available_bomb_levels=(3,),
            black_routes=(black_route(),),
            black_route_ready=False,
            route_provider=provider,
        )
        self.assertEqual(decision.kind, "wait")
        provider.assert_not_called()

    def test_without_black_connection_original_ultimate_priority_remains(self) -> None:
        """没有合法黑色路线时，原策略仍可直接释放就绪奥义。"""
        decision = decide_action(
            StrategyRuntime(StrategyDocument.builtin_default()),
            BattlePhase.OPERABLE,
            ultimate_states={1: True, 2: True, 3: True},
            available_bomb_levels=(3,),
        )
        self.assertEqual(decision.kind, "ultimate")

    def test_real_ranbu_action_uses_black_connection_before_full_sweep(self) -> None:
        """乱舞当前帧已有可连黑色时，不进行普通颜色全屏扫动。"""
        from battle.session import BattleState

        replay = stability_fixtures.BattleStabilityActionTests()
        operations, _ = replay.run_replay(
            [board(black(1, 100), black(2, 190), ghoul(3, 450))], states=[BattleState.RANBU, BattleState.VICTORY]
        )
        self.assertEqual([decision.kind for _, decision in operations], ["black_link"])

    def test_black_generates_visible_touch_point_and_avoids_bombs(self) -> None:
        """使用黑色当前帧模板锚点，锚点被炸弹覆盖时保留不可点击。"""
        recognizer = BoardRecognizer.__new__(BoardRecognizer)
        recognizer.profile = SimpleNamespace(board_roi=(0, 210, 720, 780), variable_ghoul_sizes=False)
        hit = Hit(
            "black_eye",
            "special_ghoul",
            "black",
            None,
            "eye",
            (100, 190),
            (100, 100),
            (100, 180),
            0.99,
            subtype="black",
        )
        result = recognizer._object([hit], 1, 1, 1, (720, 1280), [])
        self.assertEqual(result.candidate_point, (100, 390))
        bomb = Hit("bomb", "bomb", None, 1, "flame", (100, 180), (100, 100), (100, 180), 0.99)
        self.assertIsNone(recognizer._object([hit], 1, 1, 1, (720, 1280), [[bomb]]).candidate_point)

    def test_black_link_during_ranbu_does_not_count_as_ordinary_turn(self) -> None:
        """黑色手势仍在乱舞许可内执行，松手后不增加普通连线手数。"""
        fixture = execution_fixtures.BattleExecutionTests()
        fixture.setUp()
        observation = replace(fixture.upper_yellow_observation(), board=board(black(1, 100), black(2, 190)))
        controller, clock, count = execution_fixtures.ReplayController(), execution_fixtures.ReplayClock(), Mock()
        executor = MaaBattleExecutor(
            controller,
            execution_fixtures.MarkerReplay(1),
            count,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        decision = decide_action(
            StrategyRuntime(StrategyDocument.builtin_default()), BattlePhase.RANBU, routes=(black_route(),)
        )
        executor.execute_once(observation, fixture.resolution, decision)
        self.assertEqual(controller.actions[0], ("down", (100, 400)))
        self.assertEqual(controller.actions[-1], ("up", ()))
        count.assert_not_called()

    def test_normal_black_link_counts_once_and_samples_its_current_size(self) -> None:
        """普通状态确认连接后计数一次，运动步长随本帧黑色尺寸缩放。"""
        fixture = execution_fixtures.BattleExecutionTests()
        fixture.setUp()
        items = [
            replace(black(i, x), body_box=(x - 10, 390, 20, 20), size={"width": 20, "height": 20})
            for i, x in ((1, 100), (2, 118))
        ]
        route = replace(black_route(), points=((100, 400), (118, 400)))
        observation = replace(fixture.upper_yellow_observation(), phase=BattlePhase.OPERABLE, board=board(*items))
        controller, clock, count = execution_fixtures.ReplayController(), execution_fixtures.ReplayClock(), Mock()
        executor = MaaBattleExecutor(
            controller,
            execution_fixtures.MarkerReplay(0),
            count,
            lambda: True,
            lambda: False,
            clock=clock.clock,
            sleep=clock.sleep,
        )
        decision = decide_action(
            StrategyRuntime(StrategyDocument.builtin_default()), BattlePhase.OPERABLE, routes=(route,)
        )
        executor.execute_once(observation, fixture.resolution, decision)
        count.assert_called_once()
        moves = [point for kind, point in controller.actions if kind in ("down", "move")]
        self.assertGreater(len(moves), 4)

    def test_stale_black_points_are_not_dispatched(self) -> None:
        """即使编号相同，脱离本帧黑色身体的旧路线也不得发送输入。"""
        fixture = execution_fixtures.BattleExecutionTests()
        fixture.setUp()
        observation = replace(fixture.upper_yellow_observation(), board=board(black(1, 400), black(2, 490)))
        controller = execution_fixtures.ReplayController()
        executor = MaaBattleExecutor(
            controller, execution_fixtures.MarkerReplay(1), lambda: None, lambda: True, lambda: False
        )
        decision = decide_action(
            StrategyRuntime(StrategyDocument.builtin_default()), BattlePhase.RANBU, routes=(black_route(),)
        )
        executor.execute_once(observation, fixture.resolution, decision)
        self.assertEqual(controller.actions, [])
