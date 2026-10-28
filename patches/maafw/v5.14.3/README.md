# CPU OCR 关闭空闲自旋补丁

本目录保存 MaaPazugiri 在 MaaFramework v5.14.3 上验证过的原生补丁。它修改框架源码，需要重新编译 `MaaFramework.dll`；仅更新 Python Agent 不会启用这项优化。

## 自动发布接入

Windows x64 的 tag 发布与 Package Smoke 流程已加入 `.github/actions/build-cpu-ocr-patch`。该步骤使用 `tools/build_maafw_cpu.py` 检出精确源码、初始化固定子模块、下载并校验 MaaDeps v3.0.0，再应用补丁并编译；构建后通过原生 API 核对版本标识。

`maa-project.json` 的 MaaFW 版本固定为 `v5.14.3`，避免同步运行库时把新版本依赖与旧补丁混用。构建产物写入忽略目录 `.cache/maa-tools/maafw-cpu/win-x64`，不需要把 DLL 提交到 Git。

打包脚本在 GUI、资源、运行库复制完成后注入补丁，并检查补丁源码、DLL 以及 FastDeploy、ONNX Runtime、OpenCV 的摘要。Windows 包包含 `cpu-ocr-patch.json`，记录版本、源码提交、补丁和文件摘要。缺少产物、依赖不匹配或补丁变化时构建失败，不回退到官方 DLL。

Release 流程还会直接检查最终 ZIP 内的清单和原生文件，Package Smoke 则使用包内 Python 实际加载运行库并核对版本。Linux/macOS 暂未构建验证补丁，继续使用固定版本的官方运行库，打包时明确输出提示。

这些改动只影响包含本次修改的新构建，不会修改已发布的 beta.12 资产。旧 tag 仍指向旧工作流和代码。

## 问题与修复

Maa 的 CPU OCR 分别为文字检测、文字识别创建四线程 ONNX Runtime 会话。推理线程在等待下一次请求时仍会自旋；战斗持续发起 OCR，等待开销反复出现。实机 ETW 采样中，ONNX Runtime 占 MAP 主进程 CPU 采样约74.1%；独立实验也确认 OCR 返回后、没有新请求时仍消耗 CPU。

补丁仅为 CPU OCR 配置安装 FastDeploy 会话回调，保留四个推理线程，并将以下两个会话配置值设为字符串 `0`：

- `session.intra_op.allow_spinning`
- `session.inter_op.allow_spinning`

回调会完整接管会话配置，因此显式恢复四线程。原 CPU 路径的其他选项为默认值，沿用 ONNX Runtime 默认设置；修改只作用于创建会话时的局部 CPU 配置副本。现有推理后端选择、模型、OCR 前后处理、ROI、阈值、奥义文字识别、轮询及棋盘规则均保留。

项目 Agent 同时包含两项独立优化：在传给原生识别前裁剪本帧静态界面 ROI，减少进程间整图传输；根据策略所有可达阶段的依赖按需读取血量和奥义指标。动态棋盘对象仍逐帧独立定位。

## 对应版本

- MaaFramework：`v5.14.3`，提交 `8963191215ca49d6b037802acd07d20a4213160a`。
- MaaUtils：提交 `f8994a38ddc6858800dfea8a04e958a0adbfb184`。
- EmulatorExtras：提交 `54d3a3ad448f0541df3759ae91b571a6762daaf1`。
- MaaDeps：`v3.0.0`，其中 FastDeploy 固定提交 `e11935e4595360e7038573db1a7ab08dd0e096ec`。
- 已验证平台：Windows x64；本次构建使用 MSVC19.44、Windows SDK26100、CMake3.31，构建类型 `RelWithDebInfo`。

原生补丁文件为本目录的 `disable-ocr-cpu-spinning.patch`。构建产物版本标识使用 `v5.14.3+cpu-nospin`，便于确认实际加载的运行库。

## 构建和部署

1. 在独立目录检出上述 MaaFramework 精确提交，初始化对应 MaaUtils、EmulatorExtras 子模块，按框架官方流程准备 MaaDeps v3.0.0 的 x64 Windows 依赖。
2. 在框架源码根目录使用 `git apply --check` 检查补丁，再执行 `git apply`。补丁路径使用本项目文件的绝对路径。
3. 在已配置好 x64 编译器和 SDK 的环境中，按下面命令构建。默认控制器能力保持启用，只关闭无关入口和绑定的构建。

```powershell
cmake -S . -B build-cpu-nospin -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_TRY_COMPILE_CONFIGURATION=RelWithDebInfo -DMAADEPS_TRIPLET=maa-x64-windows -DMAA_HASH_VERSION=v5.14.3+cpu-nospin -DWITH_NODEJS_BINDING=OFF -DWITH_QUICKJS_BINDING=OFF -DWITH_MAA_AGENT=OFF -DBUILD_PICLI=OFF
cmake --build build-cpu-nospin --target MaaFramework -j 6
```

4. 先复制目标发布版原生库到独立验证目录，仅用新 DLL 替换其中的 `MaaFramework.dll`，检查架构、导出、依赖、资源加载及完整 OCR 结果。
5. 关闭 MAP，备份运行目录原 DLL 并记录新旧文件摘要，再替换 `runtimes/win-x64/native/MaaFramework.dll`。原 EXE、其余运行库、模型和用户配置保留。
6. 使用原 EXE 验证战斗功能、CPU 和耗时。需要回退时，关闭 MAP，再恢复原 DLL。

构建工具链、原生 DLL、ETL、训练截图和本机备份不纳入此目录。使用官方未打补丁的运行库时，Python 优化仍有效，但不会自动获得关闭原生自旋的收益。Windows CI 会按上一节构建并注入补丁；手动打包也需要先准备同样的构建产物。升级 MaaFramework 后应重新审查、应用补丁并编译，同时更新版本锁定及保护检查，不能把旧版 DLL 混入新版本；官方提供相同修复或配置入口后，优先移除此补丁并使用官方实现。

## 验证结果

2026年10月28日，使用31张已有实机截图的3个静态区域，共93组完整 OCR 对照，新旧文字、框、命中及分数严格一致。新旧 DLL 均为x64，172个导出名称完全一致，没有新增导入 DLL；独立验证只替换 `MaaFramework.dll`，其余发布文件逐字节一致。

30次 OCR、每次返回后间隔200毫秒的对照结果：

| 项目                     |        原版 |     补丁版 |
| ------------------------ | ----------: | ---------: |
| 进程累计 CPU 时间        | 38640.625ms | 1484.375ms |
| 整批耗时                 |   6553.91ms |  6636.61ms |
| 单次平均识别耗时         |     18.11ms |    20.88ms |
| 随后空闲1.5秒的 CPU 时间 |    781.25ms |        0ms |

本次离线累计 CPU 时间减少约96.2%，整批耗时增加约1.3%；这些数值不能直接换算成实战 CPU 百分比降幅。

部署后，原发布 EXE 的60.515秒实测共30个两秒样本，按16逻辑核归一化：MAP平均6.093%，Agent平均2.892%，合计8.985%，合计两秒采样峰值15.417%。此前15分钟合计平均27.62%；前后场景、时长未严格配对。用户同时目视任务管理器，确认当前占用稳定在10%以下。

尚未覆盖长期运行、GPU回归或独立标注的奥义点击/滑动提示正样本。

## 原始来源

- [MaaFramework CPU OCR 会话配置](https://github.com/MaaXYZ/MaaFramework/blob/8963191215ca49d6b037802acd07d20a4213160a/source/MaaFramework/Resource/OCRResMgr.cpp)
- [FastDeploy 会话回调及完整接管语义](https://github.com/MaaXYZ/FastDeploy/blob/e11935e4595360e7038573db1a7ab08dd0e096ec/fastdeploy/runtime/backends/ort/option.h)
- [ONNX Runtime 会话配置键](https://github.com/microsoft/onnxruntime/blob/v1.29.0/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h)
