import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync} from "node:fs";
import {tmpdir} from "node:os";
import {join} from "node:path";
import {test} from "node:test";
import {installCpuOcrPatch, assertCpuOcrPatch, PATCH_VERSION, SOURCE_REVISION} from "../tools/maafw-cpu-patch.mjs";

const hash = (value) => createHash("sha256").update(value).digest("hex");

function fixture() {
    const root = mkdtempSync(join(tmpdir(), "map-release-patch-"));
    const native = join(root, "package", "runtimes", "win-x64", "native");
    const artifact = join(root, ".cache", "maa-tools", "maafw-cpu", "win-x64");
    const patchDir = join(root, "patches", "maafw", "v5.14.3");
    for (const path of [
        native,
        artifact,
        patchDir,
    ])
        mkdirSync(path, {recursive: true});
    writeFileSync(join(root, "maa-project.json"), JSON.stringify({maafw: {version: "v5.14.3"}}));
    writeFileSync(join(patchDir, "disable-ocr-cpu-spinning.patch"), "测试补丁");
    writeFileSync(join(native, "MaaFramework.dll"), "原版");
    writeFileSync(join(native, "onnxruntime_maa.dll"), "依赖");
    writeFileSync(join(native, "fastdeploy_ppocr_maa.dll"), "依赖");
    writeFileSync(join(native, "opencv_world4_maa.dll"), "依赖");
    writeFileSync(join(artifact, "MaaFramework.dll"), "补丁版");
    const manifest = {
        version: PATCH_VERSION,
        sourceRevision: SOURCE_REVISION,
        patchSha256: hash("测试补丁"),
        sha256: hash("补丁版"),
        dependencies: {
            "onnxruntime_maa.dll": hash("依赖"),
            "fastdeploy_ppocr_maa.dll": hash("依赖"),
            "opencv_world4_maa.dll": hash("依赖"),
        },
    };
    writeFileSync(join(artifact, "cpu-ocr-patch.json"), JSON.stringify(manifest));
    return {root, native, artifact, cleanup: () => rmSync(root, {recursive: true, force: true})};
}

test("Windows 包必须注入补丁，后续覆盖 DLL 会被拦截", () => {
    const f = fixture();
    try {
        installCpuOcrPatch(f.native, "win-x64", f.root);
        assert.equal(readFileSync(join(f.native, "MaaFramework.dll"), "utf8"), "补丁版");
        assertCpuOcrPatch(f.native, "win-x64", f.root);
        writeFileSync(join(f.native, "MaaFramework.dll"), "原版");
        assert.throws(() => assertCpuOcrPatch(f.native, "win-x64", f.root), /摘要/);
    } finally {
        f.cleanup();
    }
});

test("缺少构建产物、依赖不匹配、补丁变更均阻止打包", () => {
    for (const change of [
        (f) => rmSync(join(f.artifact, "MaaFramework.dll")),
        (f) => writeFileSync(join(f.native, "onnxruntime_maa.dll"), "其他依赖"),
        (f) => writeFileSync(join(f.root, "patches/maafw/v5.14.3/disable-ocr-cpu-spinning.patch"), "新补丁"),
        (f) => writeFileSync(join(f.root, "maa-project.json"), JSON.stringify({maafw: {version: "v5.15.0"}})),
    ]) {
        const f = fixture();
        try {
            change(f);
            assert.throws(() => installCpuOcrPatch(f.native, "win-x64", f.root));
        } finally {
            f.cleanup();
        }
    }
});

test("未验证的平台保留官方运行库，不冒充已打补丁", () => {
    const f = fixture();
    try {
        installCpuOcrPatch(f.native, "osx-arm64", f.root);
        assertCpuOcrPatch(f.native, "osx-arm64", f.root);
        assert.equal(readFileSync(join(f.native, "MaaFramework.dll"), "utf8"), "原版");
    } finally {
        f.cleanup();
    }
});
