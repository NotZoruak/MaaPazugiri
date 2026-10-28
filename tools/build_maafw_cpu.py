"""在独立目录构建 Windows x64 OCR 补丁，并校验最终发布压缩包。"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[1]
REVISION = "8963191215ca49d6b037802acd07d20a4213160a"
VERSION = "v5.14.3+cpu-nospin"
PATCH = ROOT / "patches/maafw/v5.14.3/disable-ocr-cpu-spinning.patch"
OUTPUT = ROOT / ".cache/maa-tools/maafw-cpu/win-x64"
DEPENDENCIES = ("fastdeploy_ppocr_maa.dll", "onnxruntime_maa.dll", "opencv_world4_maa.dll")
DEPS_SHA256 = "3638891ca4b9424808598ab917ef73d8811b235ef2dd2246e3af25edd416dfd6"


def digest(path: Path) -> str:
    """计算构建输入或产物的实际摘要。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(*arguments: str, cwd: Path) -> str:
    """执行固定构建命令，任一步失败即停止发布。"""
    result = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    print(result.stdout, end="", flush=True)
    result.check_returncode()
    return result.stdout.strip()


def verify_archive(archive: Path) -> None:
    """直接读取压缩包中的 DLL 和清单，防止后续打包步骤覆盖补丁。"""
    with zipfile.ZipFile(archive) as package:
        names = [name for name in package.namelist() if name.endswith("/cpu-ocr-patch.json")]
        if len(names) != 1:
            raise ValueError("Windows 发布包必须恰好包含一份 CPU OCR 补丁清单")
        prefix = names[0].removesuffix("cpu-ocr-patch.json")
        manifest = json.loads(package.read(names[0]))
        if (
            manifest["version"] != VERSION
            or manifest["sourceRevision"] != REVISION
            or manifest["patchSha256"] != digest(PATCH)
        ):
            raise ValueError("压缩包的 CPU OCR 补丁版本或源码不符")
        if hashlib.sha256(package.read(prefix + "MaaFramework.dll")).hexdigest() != manifest["sha256"]:
            raise ValueError("压缩包中的补丁 DLL 摘要不符")
        for name in DEPENDENCIES:
            if hashlib.sha256(package.read(prefix + name)).hexdigest() != manifest["dependencies"][name]:
                raise ValueError("压缩包中的补丁依赖不符：" + name)
    print("[OK] 压缩包中的 CPU OCR 补丁和依赖摘要校验通过：", archive.name)


def build(source: Path, build_dir: Path) -> None:
    """固定源码与依赖版本；只产出主框架 DLL，不更换客户端。"""
    project = json.loads((ROOT / "maa-project.json").read_text(encoding="utf-8"))
    if project["maafw"]["version"] != "v5.14.3" or sys.platform != "win32":
        raise ValueError("当前补丁构建仅支持固定 v5.14.3 的 Windows x64")
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run(
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "v5.14.3",
            "https://github.com/MaaXYZ/MaaFramework.git",
            str(source),
            cwd=ROOT,
        )
    if run("git", "rev-parse", "HEAD", cwd=source) != REVISION:
        raise ValueError("框架源码提交不符，拒绝覆盖现有目录")
    run(
        "git", "submodule", "update", "--init", "--depth", "1", "source/MaaUtils", "3rdparty/EmulatorExtras", cwd=source
    )
    if (
        subprocess.run(["git", "apply", "--reverse", "--check", str(PATCH)], cwd=source, capture_output=True).returncode
        != 0
    ):
        run("git", "apply", "--check", str(PATCH), cwd=source)
        run("git", "apply", str(PATCH), cwd=source)
    changed = run("git", "diff", "--name-only", cwd=source)
    if changed != "source/MaaFramework/Resource/OCRResMgr.cpp":
        raise ValueError("构建源码含补丁以外的改动")
    deps = source / "source/MaaUtils/MaaDeps"
    if not (deps / "maadeps.cmake").exists():
        archive = source.parent / "MaaDeps-x64-windows.tar.xz"
        if not archive.exists() or digest(archive) != DEPS_SHA256:
            url = "https://github.com/MaaXYZ/MaaDeps/releases/download/v3.0.0/MaaDeps-x64-windows.tar.xz"
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if digest(archive) != DEPS_SHA256:
            raise ValueError("官方 MaaDeps 依赖包摘要不符")
        deps.mkdir(exist_ok=True)
        with tarfile.open(archive) as package:
            package.extractall(deps, filter="data")
    run(
        "cmake",
        "-S",
        source.as_posix(),
        "-B",
        build_dir.as_posix(),
        "-G",
        "Ninja",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
        "-DCMAKE_TRY_COMPILE_CONFIGURATION=RelWithDebInfo",
        "-DMAADEPS_TRIPLET=maa-x64-windows",
        "-DMAA_HASH_VERSION=" + VERSION,
        "-DWITH_NODEJS_BINDING=OFF",
        "-DWITH_QUICKJS_BINDING=OFF",
        "-DWITH_MAA_AGENT=OFF",
        "-DBUILD_PICLI=OFF",
        cwd=ROOT,
    )
    run("cmake", "--build", str(build_dir), "--target", "MaaFramework", "-j", "4", cwd=ROOT)
    dll = build_dir / "bin/MaaFramework.dll"
    with os.add_dll_directory(str(dll.parent)):
        library = ctypes.CDLL(str(dll))
        library.MaaVersion.restype = ctypes.c_char_p
        version = cast(bytes, library.MaaVersion()).decode()
        if version != VERSION:
            raise ValueError("构建产物未实际加载补丁版本：" + version)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shutil.copy2(dll, OUTPUT / dll.name)
    manifest: dict[str, Any] = {
        "version": version,
        "sourceRevision": REVISION,
        "patchSha256": digest(PATCH),
        "sha256": digest(dll),
        "dependencies": {name: digest(deps / "runtime/maa-x64-windows" / name) for name in DEPENDENCIES},
    }
    (OUTPUT / "cpu-ocr-patch.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("[OK] 补丁已构建并加载验证：", version)


def main() -> None:
    """构建和压缩包校验共用固定版本与摘要约束。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=OUTPUT / "source")
    parser.add_argument("--build-dir", type=Path, default=OUTPUT / "build")
    parser.add_argument("--verify-archive", type=Path)
    args = parser.parse_args()
    if args.verify_archive:
        verify_archive(args.verify_archive)
    else:
        build(args.source_dir.resolve(), args.build_dir.resolve())


if __name__ == "__main__":
    main()
