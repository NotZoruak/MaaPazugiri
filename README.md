<div align="center">

<img alt="MaaPazugiri 标志" src="./assets/logo.png" width="240" />

# MaaPazugiri

<br>

<img alt="平台" src="https://img.shields.io/badge/Platform-Android%20%7C%20Emulator-3DDC84?style=flat-square&logo=android&logoColor=white" />
<img alt="语言" src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img alt="框架" src="https://img.shields.io/badge/Framework-MaaFramework-6750A4?style=flat-square" />
<img alt="许可证" src="https://img.shields.io/badge/License-AGPL--3.0--or--later-blue?style=flat-square" />

基于 MaaFramework 的《刀剑乱舞ぱずぎり》自动化项目，面向官方日文界面，使用 Python Agent 与 Project Interface V2，通过 ADB 连接 Android 设备或模拟器。

</div>

> [!TIP]
>
> 项目仍在开发中。当前已实现游戏启动、常驻作战流程、战斗状态检查和候选路线预演；棋盘自动触控尚未实现，视觉识别也尚未完成独立评估。功能范围会随实机验证逐步扩展。

## ⚠️ 使用说明与风险提示

> [!NOTE]
>
> 本项目是由社区开发者制作的第三方自动化工具，与游戏运营方无关。项目基于 MaaFramework，通过识别游戏画面和 ADB 执行界面操作。
>
> 使用前请了解并遵守游戏服务条款及所在地适用规定。自动化操作可能导致意外的游戏内结果；请先检查任务配置，并在首次运行及新增功能验证时保持观察。
>
> 项目目前仍在开发中，页面识别和流程仅覆盖已核对的场景。未验证的布局、动画和游戏规则可能导致任务停止或识别失败。

## 下载与安装

项目目前没有正式发布包。开发阶段请从源码运行，参阅下方[参与开发](#参与开发)中的环境准备与启动命令。正式发布包准备完成后，会在此处补充下载方式。

## 首次使用

1. 准备 Android 设备或模拟器，启用 ADB 连接；当前优先使用 MuMu 模拟器。
2. 在设备中安装并启动《刀剑乱舞ぱずぎり》，使用官方日文界面。
3. 使用支持 MaaFramework Project Interface V2 的客户端加载项目，并连接 Android / Emulator 控制器。
4. 首次运行时观察客户端日志与设备画面，确认当前分辨率、页面状态和任务配置符合预期。

目前已核对的战斗截图为 720×1280。其他分辨率和界面布局尚未完成验证。不要在无人观察的情况下长时间运行尚未验证的任务。

## 功能介绍

当前项目聚焦游戏启动、常驻作战流程和战斗棋盘视觉研究，主要功能如下：

- **启动游戏**：通过 ADB 重启游戏并识别标题界面。
- **常驻作战**：导航至当前配置的章节与关卡，选择目标部队并进入战斗。
- **战斗状态检查**：在已核对的界面布局中识别可操作状态、胜利结算和失败续关提示。
- **候选路线预演**：根据当前截图识别饿鬼并生成候选连线路线，保存诊断图和 JSON 报告；不会执行点击或拖动。
- **离线棋盘视觉原型**：分析四色普通及强化饿鬼、特殊黑色饿鬼、一至三级炸弹、贝壳、石头、箱子、树枝和固定位置的青铜钟。
- **轻量识别**：跳过障碍物识别以减少计算；结果会明确标记未检查的障碍物类别。

棋盘视觉结果仍是待核对的候选对象，不能作为自动触控或路径可通行的依据。路线触控、完整实机战斗以及独立识别评估尚未完成。

## 使用说明

### 任务执行

在 MaaFramework 客户端连接设备后，选择需要执行的任务并启动。常驻作战会尝试进入当前配置的章节和关卡；目标关卡、部队及 Pipeline 流程参数以项目当前配置为准。

战斗动作目前负责检查画面状态并生成路线预演，不会触控棋盘。任务无法确认画面状态时可能停止或交还流程错误分支。运行时请查看客户端日志，并在画面与预期不符时及时停止任务。

### 棋盘识别与预演

离线棋盘识别可使用项目内截图运行，无需连接设备或启动 Agent：

```powershell
uv sync
uv run python tools/board_vision.py prepare --catalog training/board-vision/template_sources.json --output resource/base/image/board/development
uv run python tools/board_vision.py analyze --profile resource/base/image/board/development/profile.json --input "training/board-samples/raw/001_bombs_lv1x1_lv2x4.png" --output debug/board-vision/development
```

候选结果和诊断资料用于开发核对，不等于识别准确率。未识别区域不能视为空地或可通行路径；轻量模式会跳过全部障碍物，包括青铜钟。

## 参与开发

需要 Python 3.13、Node.js 22.13 或更高版本、pnpm 11 和 `uv`。在项目目录安装依赖并启动 Agent：

```powershell
uv sync
uv run python agent/main.py
```

安装前端开发工具依赖并运行资源、格式和 Python 检查：

```powershell
pnpm install
pnpm check
pnpm check:py
```

新增 Python 依赖时，使用 `uv add <包名>` 后运行：

```powershell
create-maa-project --update python-deps
```

这样会同步发布依赖清单。项目结构和 Project Interface 由 `create-maa-project` 维护；Pipeline、Agent 业务逻辑和视觉资源按项目开发约定修改。

## 致谢

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework) — 提供自动化框架与 Project Interface。
- [create-maa-project](https://github.com/Windsland52/create-maa-project) — 提供 Maa 标准 Agent 项目创建与维护工具。
- [OpenCV](https://opencv.org/) — 用于离线棋盘视觉原型。

## 文档语言

English documentation: [README.en.md](README.en.md)
