"""验证压缩后仍包含正确补丁，缺失或被覆盖时必须阻止发布。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from build_maafw_cpu import DEPENDENCIES, PATCH, REVISION, VERSION, digest, verify_archive  # noqa: E402


class ReleaseCpuArchiveTests(unittest.TestCase):
    """对真正的 zip 内容作检查，避免只校验压缩前的目录。"""

    def make_archive(self, root: Path, mode: str) -> Path:
        """生成正常、缺清单、被覆盖和过期补丁四类压缩包。"""
        payload = b"patched-runtime"
        manifest = {
            "version": VERSION,
            "sourceRevision": REVISION,
            "patchSha256": digest(PATCH),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "dependencies": {name: hashlib.sha256(b"dependency").hexdigest() for name in DEPENDENCIES},
        }
        if mode == "stale":
            manifest["patchSha256"] = "old-patch"
        target = root / (mode + ".zip")
        prefix = "runtimes/win-x64/native/"
        with zipfile.ZipFile(target, "w") as package:
            if mode != "missing":
                package.writestr(prefix + "cpu-ocr-patch.json", json.dumps(manifest))
            package.writestr(prefix + "MaaFramework.dll", b"official" if mode == "overwritten" else payload)
            for name in DEPENDENCIES:
                package.writestr(prefix + name, b"dependency")
        return target

    def test_verified_archive_passes(self) -> None:
        """清单、补丁和依赖匹配时通过。"""
        with tempfile.TemporaryDirectory() as folder:
            verify_archive(self.make_archive(Path(folder), "valid"))

    def test_missing_overwritten_and_stale_archives_fail(self) -> None:
        """原版 DLL 混入或补丁过期时不能发布。"""
        with tempfile.TemporaryDirectory() as folder:
            for mode in ("missing", "overwritten", "stale"):
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    verify_archive(self.make_archive(Path(folder), mode))

    def test_cli_uses_utf8_with_western_windows_encoding(self) -> None:
        """英文 Windows runner 的默认编码不能使成功校验变成失败。"""
        with tempfile.TemporaryDirectory() as folder:
            archive = self.make_archive(Path(folder), "valid")
            result = subprocess.run(
                [sys.executable, str(PATCH.parents[3] / "tools/build_maafw_cpu.py"), "--verify-archive", str(archive)],
                env={**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"},
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr.decode("utf-8", errors="replace"))
            self.assertIn("压缩包中的 CPU OCR 补丁", result.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
