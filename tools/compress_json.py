"""排版 MaaPazugiri 的 Pipeline JSON，支持按 MaaFramework 字段顺序排序。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_ROOT = PROJECT_ROOT / "resource" / "base" / "pipeline"

FIELD_ORDER = [
    "desc",
    "doc",
    "enabled",
    "max_hit",
    "sub_name",
    "recognition",
    "inverse",
    "pre_wait_freezes",
    "pre_delay",
    "action",
    "anchor",
    "repeat",
    "repeat_wait_freezes",
    "repeat_delay",
    "post_wait_freezes",
    "post_delay",
    "timeout",
    "rate_limit",
    "next",
    "on_error",
    "focus",
    "attach",
]
FIELD_RANK = {name: index for index, name in enumerate(FIELD_ORDER)}

RECOGNITION_ORDER = [
    "custom_recognition",
    "custom_recognition_param",
    "roi",
    "roi_offset",
    "template",
    "green_mask",
    "method",
    "detector",
    "ratio",
    "lower",
    "upper",
    "connected",
    "expected",
    "replace",
    "only_rec",
    "model",
    "color_filter",
    "labels",
    "threshold",
    "count",
    "all_of",
    "any_of",
    "box_index",
    "order_by",
    "index",
]
RECOGNITION_RANK = {name: index for index, name in enumerate(RECOGNITION_ORDER)}

ACTION_ORDER = [
    "custom_action",
    "custom_action_param",
    "target",
    "target_offset",
    "begin",
    "begin_offset",
    "end",
    "end_offset",
    "end_hold",
    "only_hover",
    "duration",
    "contact",
    "pressure",
    "swipes",
    "dx",
    "dy",
    "key",
    "input_text",
    "package",
    "exec",
    "args",
    "detach",
    "cmd",
    "shell_timeout",
    "filename",
    "format",
    "quality",
]
ACTION_RANK = {name: index for index, name in enumerate(ACTION_ORDER)}
SECTION_RANK = {"type": 0, "param": 1}
SWIPE_ORDER = [
    "starting",
    "begin",
    "begin_offset",
    "end",
    "end_offset",
    "duration",
    "end_hold",
    "only_hover",
    "contact",
    "pressure",
]
SWIPE_RANK = {name: index for index, name in enumerate(SWIPE_ORDER)}


def clean_empty_param(value: Any) -> Any:
    """递归删除值为空对象的 param 字段。"""
    if isinstance(value, dict):
        return {key: clean_empty_param(item) for key, item in value.items() if not (key == "param" and item == {})}
    if isinstance(value, list):
        return [clean_empty_param(item) for item in value]
    return value


def reorder(value: dict[str, Any], rank: dict[str, int]) -> dict[str, Any]:
    """按给定顺序重排已知键，未知键保留原有顺序并排在末尾。"""
    ordered = sorted((key for key in value if key in rank), key=rank.__getitem__)
    unknown = [key for key in value if key not in rank]
    return {key: value[key] for key in ordered + unknown}


def sort_node(node: Any) -> Any:
    """重排单个节点及识别、动作参数中的字段。"""
    if not isinstance(node, dict):
        return node

    result = dict(node)
    for section_name, rank in (
        ("recognition", RECOGNITION_RANK),
        ("action", ACTION_RANK),
    ):
        section = result.get(section_name)
        if not isinstance(section, dict):
            continue
        section = dict(section)
        params = section.get("param")
        if isinstance(params, dict):
            params = reorder(params, rank)
            swipes = params.get("swipes")
            if isinstance(swipes, list):
                params["swipes"] = [reorder(item, SWIPE_RANK) if isinstance(item, dict) else item for item in swipes]
            section["param"] = params
        result[section_name] = reorder(section, SECTION_RANK)
    return reorder(result, FIELD_RANK)


def sort_pipeline_root(value: Any) -> Any:
    """重排 Pipeline 根对象中的所有节点。"""
    if not isinstance(value, dict):
        return value
    return {name: sort_node(node) for name, node in value.items()}


def sort_interface_overrides(value: Any) -> Any:
    """递归重排 interface 或任务文件中的 pipeline_override。"""
    if isinstance(value, dict):
        return {
            key: sort_pipeline_root(item)
            if key == "pipeline_override" and isinstance(item, dict)
            else sort_interface_overrides(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sort_interface_overrides(item) for item in value]
    return value


def format_json(value: Any, indent: int = 0) -> str:
    """以紧凑方式排版 JSON，同时保留有效的 JSON 字面量。"""
    space = "  " * indent
    child_space = "  " * (indent + 1)

    if isinstance(value, dict):
        if not value:
            return "{}"
        if len(value) == 1:
            key, item = next(iter(value.items()))
            if not isinstance(item, (dict, list)):
                encoded_key = json.dumps(key, ensure_ascii=False)
                return f"{{{encoded_key}: {format_json(item)}}}"
        entries = [
            f"{child_space}{json.dumps(key, ensure_ascii=False)}: {format_json(item, indent + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(entries) + f"\n{space}}}"

    if isinstance(value, list):
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            if any(isinstance(item, float) and not math.isfinite(item) for item in value):
                raise ValueError("JSON 数组不能包含非有限浮点数")
            return "[" + ", ".join(json.dumps(item) for item in value) + "]"
        if not value:
            return "[]"
        if all(isinstance(item, str) for item in value):
            if len(value) <= 5:
                return "[" + ", ".join(json.dumps(item, ensure_ascii=False) for item in value) + "]"
            entries = [f"{child_space}{json.dumps(item, ensure_ascii=False)}" for item in value]
            return "[\n" + ",\n".join(entries) + f"\n{space}]"
        entries = [f"{child_space}{format_json(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(entries) + f"\n{space}]"

    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON 不能包含非有限浮点数")
    return json.dumps(value, ensure_ascii=False)


def collect_files(arguments: list[str], include_interface: bool) -> list[Path]:
    """收集命令行路径，或递归发现项目 Pipeline 与接口文件。"""
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

    if include_interface:
        interface_path = PROJECT_ROOT / "interface.json"
        if interface_path.exists():
            paths.append(interface_path)
        tasks_root = PROJECT_ROOT / "tasks"
        if tasks_root.exists():
            paths.extend(sorted(tasks_root.rglob("*.json")))

    unique_paths: dict[Path, None] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved not in unique_paths:
            unique_paths[resolved] = None
    return list(unique_paths)


def is_interface_file(path: Path) -> bool:
    """判断文件是否包含接口或任务配置。"""
    return path.name.casefold() in {"interface.json", "interface.jsonc"} or "tasks" in path.parts


def main() -> int:
    parser = argparse.ArgumentParser(description="排版 MaaPazugiri 的 Pipeline JSON。")
    parser.add_argument("--sort-keys", action="store_true", help="按 MaaFramework 字段顺序排序")
    parser.add_argument("--interface", action="store_true", help="同时处理 interface.json 和 tasks 下的 JSON")
    parser.add_argument("--dry-run", action="store_true", help="只报告是否需要改写，不修改文件")
    parser.add_argument("paths", nargs="*", help="要处理的 JSON 文件或目录")
    args = parser.parse_args()

    paths = collect_files(args.paths, args.interface)
    if not paths:
        print("错误：没有找到要处理的 JSON 文件。", file=sys.stderr)
        return 1

    for path in paths:
        try:
            original = path.read_text(encoding="utf-8")
            value = json.loads(original)
            if args.sort_keys:
                value = sort_interface_overrides(value) if is_interface_file(path) else sort_pipeline_root(value)
            value = clean_empty_param(value)
            output = format_json(value) + "\n"
        except (OSError, json.JSONDecodeError, ValueError) as error:
            print(f"错误：{path}：{error}", file=sys.stderr)
            return 1

        if args.dry_run:
            state = "需要更新" if output != original else "无需更新"
            print(f"[预览] {path}：{state}")
        else:
            path.write_text(output, encoding="utf-8", newline="\n")
            print(f"已排版：{path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
