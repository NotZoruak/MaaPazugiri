"""验证作战策略文件、条件树和血量读数的安全解析。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent.battle.strategy import (
    BattlePhase,
    StrategyFormatError,
    StrategyMetrics,
    StrategyRuntime,
    choose_normal_route,
    decide_action,
    evaluate_condition,
    load_strategy,
    parse_hp_percent,
)

STRATEGY_ID = "11111111111111111111111111111111"
STAGE_ID = "22222222222222222222222222222222"


def sample_strategy() -> dict[str, object]:
    """返回符合 MFAAvalonia v1 契约的最小完整策略。"""
    return {
        "schema_version": 1,
        "id": STRATEGY_ID,
        "name": "策略样例",
        "initial_stage_id": STAGE_ID,
        "stages": [
            {
                "id": STAGE_ID,
                "name": "全程作战",
                "actions": {
                    "priority": ["ultimate", "bomb"],
                    "bomb_levels": [3, 1],
                    "ultimates": [
                        {"slot": 2, "enabled": True},
                        {"slot": 1, "enabled": False},
                        {"slot": 3, "enabled": True},
                    ],
                    "normal_link": {
                        "prioritize_colors": True,
                        "long_link_threshold": 7,
                        "color_priority": ["red", "blue", "green", "yellow"],
                    },
                },
                "frenzy": {
                    "allow_ultimates": False,
                    "allow_bombs": False,
                    "reserved_color": "red",
                },
                "transitions": [],
            }
        ],
    }


class StrategyLoadingTests(unittest.TestCase):
    """验证版本、必填字段、标识和内置默认策略。"""

    def test_default_strategy_does_not_require_a_file(self) -> None:
        """内置 default 可用，且读取过程不要求 default.json。"""
        with tempfile.TemporaryDirectory() as directory:
            strategy = load_strategy("default", Path(directory))
            self.assertEqual(strategy.id, "default")
            self.assertGreaterEqual(len(strategy.stages), 1)
            self.assertFalse((Path(directory) / "strategies" / "default.json").exists())

    def test_custom_strategy_preserves_priority_and_stage_order(self) -> None:
        """用户策略从 config/strategies 按 GUID 文件名读取。"""
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(sample_strategy()), encoding="utf-8")

            strategy = load_strategy(STRATEGY_ID, Path(directory))

        self.assertEqual(strategy.stages[0].actions.priority, ("ultimate", "bomb"))
        self.assertEqual(strategy.stages[0].actions.bomb_levels, (3, 1))
        self.assertEqual([rule.slot for rule in strategy.stages[0].actions.ultimates], [2, 1, 3])

    def test_custom_strategy_file_id_must_match_requested_id(self) -> None:
        """不允许通过文件名读取另一个策略标识。"""
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            data = sample_strategy()
            data["id"] = "33333333333333333333333333333333"
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")

            with self.assertRaises(StrategyFormatError):
                load_strategy(STRATEGY_ID, Path(directory))

    def test_editor_color_priority_count_controls_ranked_and_unranked_routes(self) -> None:
        """编辑器只为前若干颜色编号，其余颜色同优先级并优先长连。"""
        routes = [
            {"color": "red", "object_ids": tuple(range(6))},
            {"color": "blue", "object_ids": tuple(range(7))},
            {"color": "green", "object_ids": tuple(range(8))},
            {"color": "yellow", "object_ids": tuple(range(10))},
        ]
        for count, expected_color in (
            (None, "blue"),
            (0, "yellow"),
            (1, "yellow"),
            (2, "blue"),
            (3, "blue"),
            (4, "blue"),
        ):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                strategy_dir = Path(directory) / "strategies"
                strategy_dir.mkdir()
                data = sample_strategy()
                if count is not None:
                    data["stages"][0]["actions"]["normal_link"]["color_priority_count"] = count  # type: ignore[index]
                (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")

                strategy = load_strategy(STRATEGY_ID, Path(directory))
                decision = decide_action(StrategyRuntime(strategy), BattlePhase.OPERABLE, routes=routes)

                self.assertEqual(decision.kind, "normal_link")
                self.assertEqual(decision.route["color"], expected_color)

    def test_invalid_editor_color_priority_count_is_rejected(self) -> None:
        """优先色数量只能是零至四的整数，不接受布尔、空值或越界值。"""
        for count in (-1, 5, True, 1.5, "2", None):
            with self.subTest(count=count), tempfile.TemporaryDirectory() as directory:
                strategy_dir = Path(directory) / "strategies"
                strategy_dir.mkdir()
                data = sample_strategy()
                data["stages"][0]["actions"]["normal_link"]["color_priority_count"] = count  # type: ignore[index]
                (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")

                with self.assertRaisesRegex(StrategyFormatError, "color_priority_count"):
                    load_strategy(STRATEGY_ID, Path(directory))

    def test_future_version_unknown_field_and_missing_required_field_fail_closed(self) -> None:
        """格式不兼容或字段不完整时拒绝加载，不静默补默认值。"""
        for mutation in (
            lambda data: data.update(schema_version=2),
            lambda data: data.update(unrecognized=True),
            lambda data: data["stages"][0]["actions"]["normal_link"].update(unrecognized=True),  # type: ignore[index]
            lambda data: data["stages"][0]["actions"].pop("priority"),  # type: ignore[index]
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                strategy_dir = Path(directory) / "strategies"
                strategy_dir.mkdir()
                data = sample_strategy()
                mutation(data)
                (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")

                with self.assertRaises(StrategyFormatError):
                    load_strategy(STRATEGY_ID, Path(directory))


class StrategyConditionTests(unittest.TestCase):
    """验证条件树的 all、any 与未知值传播。"""

    def test_unknown_numeric_value_does_not_satisfy_or_invert_condition(self) -> None:
        """缺少血量观测时，数值判断为未知，不会变成满足。"""
        condition = {
            "type": "predicate",
            "metric": "enemy_hp_percent",
            "comparison": "less_or_equal",
            "value": 30,
        }
        self.assertIsNone(evaluate_condition(condition, StrategyMetrics(enemy_hp_percent=None)))

    def test_nested_any_and_all_match_documented_semantics(self) -> None:
        """组内未知值仅在无确定答案时向上传播。"""
        condition = {
            "type": "group",
            "match": "all",
            "children": [
                {
                    "type": "predicate",
                    "metric": "stage_normal_links",
                    "comparison": "greater_or_equal",
                    "value": 2,
                },
                {
                    "type": "group",
                    "match": "any",
                    "children": [
                        {
                            "type": "predicate",
                            "metric": "ultimate_ready",
                            "slots": [1],
                            "slot_match": "all",
                            "expected": False,
                        },
                        {
                            "type": "predicate",
                            "metric": "ultimate_ready",
                            "slots": [2],
                            "slot_match": "all",
                            "expected": False,
                        },
                    ],
                },
            ],
        }
        metrics = StrategyMetrics(stage_normal_links=2, ultimate_ready={1: None, 2: False})
        self.assertTrue(evaluate_condition(condition, metrics))
        metrics.ultimate_ready = {1: None, 2: None}
        self.assertIsNone(evaluate_condition(condition, metrics))

    def test_runtime_applies_first_transition_and_resets_stage_count(self) -> None:
        """阶段转变按数组顺序检查，进入新阶段后阶段手数归零。"""
        data = sample_strategy()
        data["stages"][0]["transitions"] = [  # type: ignore[index]
            {
                "id": "33333333333333333333333333333333",
                "target_stage_id": "44444444444444444444444444444444",
                "condition": {
                    "type": "group",
                    "match": "all",
                    "children": [
                        {
                            "type": "predicate",
                            "metric": "battle_normal_links",
                            "comparison": "greater_or_equal",
                            "value": 1,
                        }
                    ],
                },
            },
            {
                "id": "55555555555555555555555555555555",
                "target_stage_id": STAGE_ID,
                "condition": {
                    "type": "group",
                    "match": "all",
                    "children": [
                        {
                            "type": "predicate",
                            "metric": "battle_normal_links",
                            "comparison": "greater_or_equal",
                            "value": 1000,
                        }
                    ],
                },
            },
        ]
        next_stage = sample_strategy()["stages"][0].copy()  # type: ignore[index]
        next_stage.update(id="44444444444444444444444444444444", name="第二阶段", transitions=[])
        data["stages"].append(next_stage)  # type: ignore[union-attr]
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")
            runtime = StrategyRuntime(load_strategy(STRATEGY_ID, Path(directory)))

        runtime.record_normal_link_success()
        selected, chain, looped = runtime.resolve_stage()
        self.assertFalse(looped)
        self.assertEqual(chain, ("44444444444444444444444444444444",))
        self.assertEqual(selected.id, "44444444444444444444444444444444")
        self.assertEqual(runtime.metrics.stage_normal_links, 0)

    def test_same_observation_transition_cycle_stops_before_action(self) -> None:
        """无新观察可打破的阶段循环不继续执行行动。"""
        data = sample_strategy()
        second_id = "44444444444444444444444444444444"
        first = data["stages"][0]  # type: ignore[index]
        first["transitions"] = [  # type: ignore[index]
            {
                "id": "33333333333333333333333333333333",
                "target_stage_id": second_id,
                "condition": {
                    "type": "group",
                    "match": "all",
                    "children": [
                        {
                            "type": "predicate",
                            "metric": "battle_normal_links",
                            "comparison": "greater_or_equal",
                            "value": 0,
                        }
                    ],
                },
            }
        ]
        second = first.copy()  # type: ignore[union-attr]
        second.update(id=second_id, name="第二阶段")
        second["transitions"] = [  # type: ignore[index]
            {
                "id": "55555555555555555555555555555555",
                "target_stage_id": STAGE_ID,
                "condition": {
                    "type": "group",
                    "match": "all",
                    "children": [
                        {
                            "type": "predicate",
                            "metric": "battle_normal_links",
                            "comparison": "greater_or_equal",
                            "value": 0,
                        }
                    ],
                },
            }
        ]
        data["stages"].append(second)  # type: ignore[union-attr]
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")
            runtime = StrategyRuntime(load_strategy(STRATEGY_ID, Path(directory)))

        decision = decide_action(
            runtime,
            BattlePhase.OPERABLE,
            ultimate_states={1: True, 2: True, 3: True},
            available_bomb_levels=[3],
            routes=[],
        )
        self.assertEqual(decision.kind, "wait")
        self.assertTrue(decision.transition_loop)

    def test_priority_controls_ultimate_and_bomb_order(self) -> None:
        """奥义与炸弹严格按策略 priority 检查。"""
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            data = sample_strategy()
            data["stages"][0]["actions"]["priority"] = ["bomb", "ultimate"]  # type: ignore[index]
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")
            runtime = StrategyRuntime(load_strategy(STRATEGY_ID, Path(directory)))

        decision = decide_action(
            runtime,
            BattlePhase.OPERABLE,
            ultimate_states={1: True, 2: True, 3: True},
            available_bomb_levels=[1, 3],
            routes=[],
        )
        self.assertEqual((decision.kind, decision.level), ("bomb", 3))

    def test_ultimate_rule_order_and_unknown_precondition_are_respected(self) -> None:
        """奥义位置按策略顺序检查，未知前置不会允许释放。"""
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            data = sample_strategy()
            data["stages"][0]["actions"]["ultimates"][0]["condition"] = {  # type: ignore[index]
                "type": "group",
                "match": "all",
                "children": [{"type": "predicate", "metric": "enemy_hp_percent", "comparison": "less", "value": 50}],
            }
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")
            runtime = StrategyRuntime(load_strategy(STRATEGY_ID, Path(directory)))

        decision = decide_action(
            runtime,
            BattlePhase.OPERABLE,
            metrics=StrategyMetrics(enemy_hp_percent=None),
            ultimate_states={1: True, 2: True, 3: True},
            available_bomb_levels=[],
            routes=[],
        )
        self.assertEqual((decision.kind, decision.slot), ("ultimate", 3))

    def test_frenzy_permissions_gate_actions_and_use_color_reservation(self) -> None:
        """乱舞额外许可关闭时跳过炸弹和奥义，并保留配置颜色。"""
        with tempfile.TemporaryDirectory() as directory:
            strategy_dir = Path(directory) / "strategies"
            strategy_dir.mkdir()
            data = sample_strategy()
            data["stages"][0]["actions"]["priority"] = ["ultimate", "bomb"]  # type: ignore[index]
            (strategy_dir / f"{STRATEGY_ID}.json").write_text(json.dumps(data), encoding="utf-8")
            runtime = StrategyRuntime(load_strategy(STRATEGY_ID, Path(directory)))

        decision = decide_action(
            runtime,
            BattlePhase.RANBU,
            ultimate_states={1: True, 2: True, 3: True},
            available_bomb_levels=[3],
            routes=[],
        )
        self.assertEqual(decision.kind, "ranbu_link")
        self.assertEqual(decision.reserved_color, "red")

    def test_successful_normal_link_is_the_only_counter_increment(self) -> None:
        """仅显式确认成功的普通连线同时增加本场和阶段手数。"""
        strategy = load_strategy("default", Path("missing"))
        runtime = StrategyRuntime(strategy)
        runtime.record_normal_link_success()
        self.assertEqual(runtime.metrics.battle_normal_links, 1)
        self.assertEqual(runtime.metrics.stage_normal_links, 1)


class HealthRatioTests(unittest.TestCase):
    """验证 OCR 的 A/B 血量结果及未知读数处理。"""

    def test_parses_ratio_to_exact_percentage(self) -> None:
        """百分比保留精确小数供策略阈值比较。"""
        self.assertEqual(parse_hp_percent(" 125 / 2000 "), 6.25)
        self.assertEqual(parse_hp_percent("0/2000"), 0)
        self.assertEqual(parse_hp_percent("2000/2000"), 100)

    def test_invalid_ratio_is_unknown(self) -> None:
        """OCR 缺字、除数为零或当前值超界时不制造血量结论。"""
        for text in ("", "1/0", "2001/2000", "A/2000", "12/20/3"):
            with self.subTest(text=text):
                self.assertIsNone(parse_hp_percent(text))


class RoutePriorityTests(unittest.TestCase):
    """验证七连门槛只在启用颜色优先时改变颜色筛选。"""

    def test_long_link_threshold_filters_colors_only_when_a_qualifying_route_exists(self) -> None:
        """达标路线存在时只在策略优先色中选，否则回退全部候选。"""
        routes = [
            {"color": "red", "object_ids": (1, 2, 3, 4, 5, 6)},
            {"color": "blue", "object_ids": (1, 2, 3, 4, 5, 6, 7)},
        ]
        selected = choose_normal_route(
            routes,
            prioritize_colors=True,
            long_link_threshold=7,
            color_priority=("red", "blue", "green", "yellow"),
        )
        self.assertEqual(selected["color"], "blue")
        fallback = choose_normal_route(
            routes[:1],
            prioritize_colors=True,
            long_link_threshold=7,
            color_priority=("red", "blue", "green", "yellow"),
        )
        self.assertEqual(fallback["color"], "red")

    def test_partial_color_priority_keeps_short_route_fallback_and_disabled_behavior(self) -> None:
        """无七连时仍可选已编号短连，关闭颜色优先后按长度选择。"""
        routes = [
            {"color": "red", "object_ids": tuple(range(4))},
            {"color": "blue", "object_ids": tuple(range(6))},
        ]
        for prioritize_colors, expected_color in ((True, "red"), (False, "blue")):
            with self.subTest(prioritize_colors=prioritize_colors):
                selected = choose_normal_route(
                    routes,
                    prioritize_colors=prioritize_colors,
                    long_link_threshold=7,
                    color_priority=("red", "blue", "green", "yellow"),
                    color_priority_count=1,
                )
                self.assertEqual(selected["color"], expected_color)


if __name__ == "__main__":
    unittest.main()
