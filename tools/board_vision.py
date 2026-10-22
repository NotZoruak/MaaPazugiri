"""制作局部模板并运行离线棋盘识别，不连接模拟器。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.vision.image_io import protect_sources, read_image, write_text_atomic  # noqa: E402
from agent.vision.recognizer import BoardRecognizer  # noqa: E402
from agent.vision.report import write_report  # noqa: E402
from agent.vision.templates import prepare_templates  # noqa: E402


def main() -> int:
    """解析离线命令，错误明确返回非零状态。"""
    parser = argparse.ArgumentParser(description="饿鬼、炸弹、障碍物与青铜钟离线识别原型")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="根据来源清单生成模板和配置")
    prepare.add_argument("--catalog", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    analyze = commands.add_parser("analyze", help="输出单张或多张原图的识别诊断")
    analyze.add_argument("--profile", required=True, type=Path)
    analyze.add_argument("--workers", type=int, help="模板并行数1至16；默认不超过8且受可用逻辑处理器数限制，1表示串行")
    analyze.add_argument("--full-search", action="store_true", help="使用全区域精细扫描，供漏检核对与性能对比")
    analyze.add_argument("--lightweight", action="store_true", help="仅识别所有饿鬼与炸弹，跳过障碍物（含青铜钟）")
    analyze.add_argument("--no-color-prefilter", action="store_true", help="关闭颜色前置筛选，保留模板与帧内预处理缓存")
    analyze.add_argument("--input", nargs="+", required=True, type=Path)
    analyze.add_argument("--output", type=Path, default=ROOT / "debug" / "board-vision")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            profile = prepare_templates(args.catalog, args.output)
            print(f"已生成模板配置：{profile}")
            return 0
        sources: list[Path] = args.input
        protect_sources(args.output, sources)
        if len({source.stem.casefold() for source in sources}) != len(sources):
            raise ValueError("输入文件名重复，无法生成互不覆盖的报告")
        recognizer = BoardRecognizer(
            args.profile,
            full_search=args.full_search,
            color_prefilter=not args.no_color_prefilter,
            workers=args.workers,
        )
        summaries: list[dict[str, Any]] = []
        for source in sources:
            image = read_image(source)
            result = recognizer.analyze(image, source.name, lightweight=args.lightweight)
            counts = write_report(image, result, args.output, source.stem)
            summaries.append(
                {
                    "file": source.name,
                    "counts": counts,
                    "elapsed_ms": result.elapsed_ms,
                    "issues": len(result.issues),
                    "missing_templates": result.coverage["missing_templates"],
                    "recognition_mode": result.coverage["recognition_mode"],
                    "skipped_kinds": result.coverage["skipped_kinds"],
                    "skipped_variants": result.coverage["skipped_variants"],
                }
            )
            print(json.dumps(summaries[-1], ensure_ascii=False))
        lines = [
            "# 离线识别开发报告",
            "",
            "这些结果用于逐个核对，不代表独立评估准确率或可自动操作。",
            "识别范围以各图结果的coverage为准；主动跳过的类别与外观未检查，未检查的区域不能视为空地。",
            "",
            "诊断图数字对应 JSON 中的对象编号；问号表示证据单一、等级冲突或尚未解释的区域。",
            "延伸至区域边界的残留颜色仅在 JSON 保留，避免将大片背景画成对象候选框。",
            "",
        ]
        for summary in summaries:
            lines.extend(
                [
                    f"## {summary['file']}",
                    "",
                    f"候选数量：{summary['counts']}",
                    "",
                    f"处理耗时：{summary['elapsed_ms']} 毫秒；诊断问题：{summary['issues']} 项。",
                    "",
                    f"缺少模板：{summary['missing_templates']}。",
                    "",
                    f"未检查类别：{summary['skipped_kinds']}；未检查外观：{summary['skipped_variants']}。",
                    "",
                ]
            )
        write_text_atomic(args.output / "README.md", "\n".join(lines))
        return 0
    except (ValueError, OSError, KeyError, TypeError, cv2.error) as error:
        print(f"离线识别失败：{error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
