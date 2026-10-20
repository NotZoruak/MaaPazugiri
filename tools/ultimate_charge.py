"""先识别战斗状态，再离线观测奥义充能；不执行角色操作或读取策略。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "agent"))

from battle.recognition import BattleFrameRecognizer, Image  # noqa: E402
from battle.ultimate import UltimateChargeRecognizer  # noqa: E402
from battle.ultimate_report import write_ultimate_report  # noqa: E402
from maa.controller import CustomController  # noqa: E402
from maa.define import RecognitionDetail  # noqa: E402
from maa.pipeline import JRecognitionParam, JRecognitionType  # noqa: E402
from maa.resource import Resource  # noqa: E402
from maa.tasker import Tasker  # noqa: E402
from maa.toolkit import Toolkit  # noqa: E402
from vision.image_io import protect_sources, read_image, write_text_atomic  # noqa: E402


class ObservationController(CustomController):
    """只提供原生识别所需标识，不连接设备或实现输入。"""

    def request_uuid(self) -> str:
        """返回离线充能验证标识。"""
        return "ultimate-charge-offline"


def main() -> int:
    """显式传入图片进行状态门禁和同帧奥义识别，保存逐图证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+", help="本项目实机截图")
    parser.add_argument("--output", type=Path, default=ROOT / "debug/battle-ultimate/offline", help="诊断输出目录")
    args = parser.parse_args()
    sources = [source.resolve(strict=True) for source in args.images]
    output = args.output.resolve()
    protect_sources(output, sources)
    if len({source.stem for source in sources}) != len(sources):
        raise ValueError("输入图片文件名重复，请分批输出以免报告互相覆盖")
    Toolkit.init_option(ROOT / "debug/battle-ultimate/maa")
    resource = Resource()
    if not resource.post_bundle(ROOT / "resource/base").wait().succeeded:
        raise RuntimeError("无法加载本项目Maa资源")
    tasker = Tasker()
    controller = ObservationController()
    if not tasker.bind(resource, controller):
        raise RuntimeError("无法初始化离线状态识别")

    def recognize(method: JRecognitionType, parameters: JRecognitionParam, image: Image) -> RecognitionDetail | None:
        """将当前显式图片送入原生识别，不请求设备截图。"""
        detail = tasker.post_recognition(method, parameters, image).wait().get()
        return detail.nodes[0].recognition if detail is not None and len(detail.nodes) == 1 else None

    gate = BattleFrameRecognizer(recognize)
    charge = UltimateChargeRecognizer(ROOT / "resource/base/image/ultimate")
    records: list[dict[str, object]] = []
    for source in sources:
        image = read_image(source)
        start = time.perf_counter()
        phase = gate.analyze(image)
        gate_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        observation = charge.analyze(image, frame_id=uuid4().hex, phase=phase)
        charge_ms = (time.perf_counter() - start) * 1000
        write_ultimate_report(image, observation, output, source.stem)
        records.append(
            {
                "source": str(source),
                "phase": phase.value,
                "checked": observation.checked,
                "states": [slot.state.value for slot in observation.slots],
                "gate_ms": round(gate_ms, 3),
                "charge_ms": round(charge_ms, 3),
                "report": str(output / f"{source.stem}.ultimate.json"),
            }
        )
    summary = {"mode": "gated_charge_observation_only", "device_actions_executed": False, "results": records}
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    write_text_atomic(output / "summary.json", text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
