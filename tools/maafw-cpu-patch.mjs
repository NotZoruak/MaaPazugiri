import {createHash} from "node:crypto";
import {copyFileSync, readFileSync, writeFileSync} from "node:fs";
import {join} from "node:path";

export const PATCH_VERSION = "v5.14.3+cpu-nospin";
export const SOURCE_REVISION = "8963191215ca49d6b037802acd07d20a4213160a";
const manifestName = "cpu-ocr-patch.json";
const digest = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");

function verifyManifest(manifest, dll, projectRoot) {
    const project = JSON.parse(readFileSync(join(projectRoot, "maa-project.json"), "utf8"));
    if (project.maafw?.version !== "v5.14.3") throw new Error("CPU OCR 补丁需要固定 MaaFW v5.14.3，升级须重新适配");
    const patch = join(projectRoot, "patches/maafw/v5.14.3/disable-ocr-cpu-spinning.patch");
    if (
        manifest.version !== PATCH_VERSION ||
        manifest.sourceRevision !== SOURCE_REVISION ||
        manifest.patchSha256 !== digest(patch)
    ) {
        throw new Error("CPU OCR 构建清单与当前源码补丁不符");
    }
    if (!manifest.dependencies || Object.keys(manifest.dependencies).length === 0)
        throw new Error("补丁清单缺少依赖摘要");
    if (manifest.sha256 !== digest(dll)) throw new Error("CPU OCR 补丁 DLL 摘要不符");
}

function verifyDependencies(manifest, nativeDir) {
    for (const name of [
        "fastdeploy_ppocr_maa.dll",
        "onnxruntime_maa.dll",
        "opencv_world4_maa.dll",
    ]) {
        if (typeof manifest.dependencies[name] !== "string") throw new Error(`补丁清单缺少依赖摘要：${name}`);
    }
    for (const [
        name,
        expected,
    ] of Object.entries(manifest.dependencies)) {
        if (!/^[A-Za-z0-9_]+\.dll$/.test(name) || digest(join(nativeDir, name)) !== expected) {
            throw new Error(`CPU OCR 补丁依赖不匹配：${name}`);
        }
    }
}

export function installCpuOcrPatch(nativeDir, platform, projectRoot = process.cwd()) {
    if (platform !== "win-x64") return;
    const artifact = join(projectRoot, ".cache/maa-tools/maafw-cpu/win-x64");
    const manifest = JSON.parse(readFileSync(join(artifact, manifestName), "utf8"));
    const dll = join(artifact, "MaaFramework.dll");
    verifyManifest(manifest, dll, projectRoot);
    verifyDependencies(manifest, nativeDir);
    copyFileSync(dll, join(nativeDir, "MaaFramework.dll"));
    writeFileSync(join(nativeDir, manifestName), JSON.stringify(manifest, null, 2) + "\n");
    console.log(`[OK] 已注入 CPU OCR 补丁：${manifest.version}`);
}

export function assertCpuOcrPatch(nativeDir, platform, projectRoot = process.cwd()) {
    if (platform !== "win-x64") return;
    const manifest = JSON.parse(readFileSync(join(nativeDir, manifestName), "utf8"));
    verifyManifest(manifest, join(nativeDir, "MaaFramework.dll"), projectRoot);
    verifyDependencies(manifest, nativeDir);
}
