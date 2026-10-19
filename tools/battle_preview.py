"""离线运行战斗状态门禁与候选路线预演，不连接设备或执行输入。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.preview import BoardPreviewer  # noqa: E402
from battle.recognition import BattleFrameRecognizer, Image  # noqa: E402
from battle.session import BattleState  # noqa: E402
from maa.controller import CustomController  # noqa: E402
from maa.define import RecognitionDetail  # noqa: E402
from maa.pipeline import JRecognitionParam, JRecognitionType  # noqa: E402
from maa.resource import Resource  # noqa: E402
from maa.tasker import Tasker  # noqa: E402
from maa.toolkit import Toolkit  # noqa: E402
from vision.image_io import protect_sources, read_image, write_text_atomic  # noqa: E402


class RecognitionOnlyController(CustomController):
    """只提供原生识别所需的诊断标识，不实现任何输入操作。"""

    def request_uuid(self) -> str:
        """返回离线识别标识。"""
        return "battle-preview-offline"


def main() -> int:
    """只为通过状态门禁的输入截图生成候选预演。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="待检查的官方日文界面截图")
    parser.add_argument("--output", type=Path, default=ROOT / "debug/battle-preview/offline", help="诊断输出目录")
    parser.add_argument("--lightweight", action="store_true", help="跳过障碍物识别，报告明确保留跳过范围")
    args = parser.parse_args()
    sources = [path.resolve(strict=True) for path in args.images]
    output = args.output.resolve()
    protect_sources(output, sources)
    Toolkit.init_option(ROOT / "debug/battle-preview/maa")
    resource = Resource()
    if not resource.post_bundle(ROOT / "resource/base").wait().succeeded:
        raise RuntimeError("无法加载本项目Maa资源")
    controller = RecognitionOnlyController()
    tasker = Tasker()
    if not tasker.bind(resource, controller):
        raise RuntimeError("无法初始化离线识别")

    def recognize(method: JRecognitionType, parameters: JRecognitionParam, image: Image) -> RecognitionDetail | None:
        """将显式传入图像交给原生识别，不请求控制器截图。"""
        detail = tasker.post_recognition(method, parameters, image).wait().get()
        if detail is None or len(detail.nodes) != 1:
            return None
        return detail.nodes[0].recognition

    gate = BattleFrameRecognizer(recognize)
    previewer = BoardPreviewer(ROOT / "resource/base/image/board/development/profile.json", output)
    records: list[dict[str, object]] = []
    for source in sources:
        image = read_image(source)
        state = gate.analyze(image)
        record: dict[str, object] = {"source": str(source), "state": state.value, "preview_generated": False}
        if state == BattleState.OPERABLE:
            preview = previewer.analyze(image, uuid4().hex, lightweight=args.lightweight, stem=source.stem)
            route = preview.selected_route
            record.update(
                preview_generated=True,
                candidate_edges=len(preview.candidate_edges),
                uncertain_edges=len(preview.uncertain_edges),
                selected_ids=list(route.object_ids) if route is not None else [],
                report=str(output / f"{source.stem}.preview.json"),
                image=str(output / f"{source.stem}.preview.png"),
            )
        records.append(record)
    summary = {"mode": "preview_only", "device_actions_executed": False, "results": records}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    write_text_atomic(output / "summary.json", text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
