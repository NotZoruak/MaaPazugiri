"""统一 MaaPazugiri Pipeline 节点结构并按固定顺序排列字段。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from compress_json import (
    ACTION_RANK,
    FIELD_RANK,
    RECOGNITION_RANK,
    clean_empty_param,
    format_json,
    reorder,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "resource" / "base" / "pipeline"
RECOGNITION_PARAM_KEYS = set(RECOGNITION_RANK)
ACTION_CONFIG_KEYS = {"custom_action", "custom_action_param"}
ACTION_PARAM_KEYS = set(ACTION_RANK) - ACTION_CONFIG_KEYS
ACTION_SECTION_RANK = {"type": 0, "custom_action": 1, "custom_action_param": 2, "param": 3}


def reorder_recognition_params(params: dict[str, Any]) -> dict[str, Any]:
    """按 MaaFramework 识别参数顺序排列字段。"""
    return reorder(params, RECOGNITION_RANK)


def normalize_recognition(node: dict[str, Any], recognition: Any) -> Any:
    """将字符串式或扁平式识别配置归一化为 type 与 param。"""
    if isinstance(recognition, str):
        # Config.json 中的开关节点使用 DirectHit，不按普通识别节点转换。
        if recognition == "DirectHit" and "enabled" in node and "action" not in node:
            return recognition
        params = {key: node.pop(key) for key in list(node) if key in RECOGNITION_PARAM_KEYS}
        result: dict[str, Any] = {"type": recognition}
        if params:
            result["param"] = reorder_recognition_params(params)
        return reorder(result, {"type": 0, "param": 1})

    if isinstance(recognition, dict):
        result = dict(recognition)
        recognition_type = result.get("type")
        if "param" in result:
            params = result["param"]
            if isinstance(params, dict):
                result["param"] = reorder_recognition_params(params)
            return reorder(result, {"type": 0, "param": 1})

        params = {key: value for key, value in result.items() if key != "type"}
        if not params:
            return result
        normalized = {"type": recognition_type}
        normalized["param"] = reorder_recognition_params(params)
        return reorder(normalized, {"type": 0, "param": 1})

    return recognition


def normalize_action(node: dict[str, Any], action: Any) -> Any:
    """将字符串式或扁平式动作配置归一化为 type 与 param。"""
    if isinstance(action, str):
        result: dict[str, Any] = {"type": action}
        for key in ACTION_CONFIG_KEYS:
            if key in node:
                result[key] = node.pop(key)
        params = {key: node.pop(key) for key in list(node) if key in ACTION_PARAM_KEYS or key == "target"}
        if params:
            result["param"] = reorder(params, ACTION_RANK)
        return reorder(result, ACTION_SECTION_RANK)

    if not isinstance(action, dict):
        return action

    result = dict(action)
    params = result.get("param")
    if isinstance(params, dict):
        result["param"] = reorder(params, ACTION_RANK)
        return reorder(result, ACTION_SECTION_RANK)

    extras = {
        key: value for key, value in result.items() if key not in {"type", "custom_action", "custom_action_param"}
    }
    if extras:
        result = {key: value for key, value in result.items() if key not in extras}
        result["param"] = reorder(extras, ACTION_RANK)
    return reorder(result, ACTION_SECTION_RANK)


def normalize_node(node: Any) -> Any:
    """归一化单个 Pipeline 节点，并重排节点字段。"""
    if not isinstance(node, dict):
        return node
    result = dict(node)
    if "recognition" in result:
        result["recognition"] = normalize_recognition(result, result["recognition"])
    if "action" in result:
        result["action"] = normalize_action(result, result["action"])
    return reorder(result, FIELD_RANK)


def normalize_pipeline(value: Any) -> Any:
    """归一化 Pipeline 根对象中的所有节点。"""
    if not isinstance(value, dict):
        return value
    return {name: normalize_node(node) for name, node in value.items()}


def collect_files(arguments: list[str]) -> list[Path]:
    """收集指定文件或目录；无参数时递归发现项目 Pipeline JSON。"""
    paths: list[Path] = []
    if arguments:
        for argument in arguments:
            path = Path(argument)
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.json")))
            else:
                paths.append(path)
    else:
        paths.extend(sorted(PIPELINE_ROOT.rglob("*.json")))

    unique_paths: dict[Path, None] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved not in unique_paths:
            unique_paths[resolved] = None
    return list(unique_paths)


def main() -> int:
    parser = argparse.ArgumentParser(description="归一化 MaaPazugiri Pipeline 节点。")
    parser.add_argument("--dry-run", action="store_true", help="只预览会变化的节点，不修改文件")
    parser.add_argument("paths", nargs="*", help="要处理的 Pipeline JSON 文件或目录")
    args = parser.parse_args()

    paths = collect_files(args.paths)
    if not paths:
        print("错误：没有找到 Pipeline JSON 文件。", file=sys.stderr)
        return 1

    total_changed = 0
    for path in paths:
        try:
            original_text = path.read_text(encoding="utf-8")
            original = json.loads(original_text)
            if not isinstance(original, dict):
                raise ValueError("Pipeline 根节点必须是对象")
            normalized = clean_empty_param(normalize_pipeline(original))
            output = format_json(normalized) + "\n"
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"错误：{path}：{error}", file=sys.stderr)
            return 1

        changed_nodes = [
            name
            for name, node in original.items()
            if isinstance(node, dict)
            and json.dumps(node, ensure_ascii=False) != json.dumps(normalized.get(name), ensure_ascii=False)
        ]
        total_changed += len(changed_nodes)

        if args.dry_run:
            print(f"[预览] {path.name}：{len(changed_nodes)} 个节点需要调整")
            for name in changed_nodes[:5]:
                print(f"  {name}")
        else:
            path.write_text(output, encoding="utf-8", newline="\n")
            print(f"已归一化：{path}（{len(changed_nodes)} 个节点调整）")

    if args.dry_run:
        print(f"共 {total_changed} 个节点需要调整")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
