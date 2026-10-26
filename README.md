<div align="center">

<img alt="MaaPazugiri 标志" src="./assets/logo.png" width="240" />

# MaaPazugiri

<br>

<img alt="平台" src="https://img.shields.io/badge/Platform-Emulator-3DDC84?style=flat-square" />
<img alt="语言" src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img alt="框架" src="https://img.shields.io/badge/Framework-MaaFramework-6750A4?style=flat-square" />
<img alt="许可证" src="https://img.shields.io/badge/License-AGPL--3.0--or--later-blue?style=flat-square" />

基于 MaaFramework 的《刀剑乱舞ぱずぎり》自动化项目，面向官方日文界面，使用 Python Agent 与 Project Interface V2，当前以 MuMu 模拟器和 ADB 连接为开发与核对环境。

</div>

桌面 GUI 使用 [MFAA](https://github.com/NotZoruak/MFAA)，这是基于上游 [MaaXYZ/MFAAvalonia](https://github.com/MaaXYZ/MFAAvalonia) 为 MaaPazugiri 定制的版本，界面功能和行为以 MFAA 仓库为准。

> [!TIP]
>
> 项目状态：持续维护与迭代
> 本项目已完成每日主要流程（启动游戏、常驻作战、领取奖励等核心循环）的完整实现， 现进入持续维护与迭代阶段，后续以**Bug 修复、游戏版本适配、依赖更新**为主，同时按需评估新功能、新任务、新流程。
> 目前存在的问题：乱舞期间无法精准保留指定颜色，同时对于强化饿鬼的识别还不是很准确。

## ⚠️ 使用说明与风险提示

> [!NOTE]
>
> 本项目是由社区开发者制作的第三方自动化工具，与游戏运营方无关。项目基于 MaaFramework，通过识别游戏画面和 ADB 执行界面操作。
>
> 使用前请了解并遵守游戏服务条款及所在地适用规定。自动化操作可能导致意外的游戏内结果；请先检查任务配置，并在首次运行及新增功能验证时保持观察。
>
> 项目目前仍在完善中，可能存在诸多 Bugs。

## 下载与安装

从 [GitHub Releases](https://github.com/NotZoruak/MaaPazugiri/releases) 下载 MaaPazugiri 发布压缩包。请下载 Release 附带的发布包，不要下载 GitHub 自动生成的 `Source code` 源码包。

1. 将发布压缩包解压到单独的英文名目录下。
2. 双击MaaPazugiri.exe启动。第一次启动可能耗时较长，请耐心等待。
3. 启动 MuMu 模拟器并连接 ADB 。

> 启动时如提示缺少“.NET Desktop Runtime 10.0”或 `VCRUNTIME140.dll`，请右键 `DependencySetup_依赖库安装_win.bat`，选择“以管理员身份运行”；安装完成后重新启动 MaaPazugiri。

## 功能介绍

当前项目聚焦游戏启动、常驻作战和动态棋盘识别，主要功能如下：

- **启动游戏**：通过 ADB 重启游戏并识别标题界面。
- **常驻作战**：选择已配置的章节、关卡和部队，进入战斗并按任务次数重复；可设置出战前返回主页面、使用体力回复药和碎玉换体。策略选择当前仅提供默认策略。
- **战斗识别与决策**：识别普通可操作、乱舞、胜负及奥义确认状态；按当前策略选择奥义、炸弹、强化或黑色饿鬼路线、普通四色连线或乱舞扫动。
- **战斗触控**：已实现奥义和炸弹点击、普通及特殊路线滑动。普通连线需先通过连续画面稳定检查；每次操作后重新读取画面并确认结果。未确认的普通路线会在当前盘面尝试另一方向，仍失败时暂时排除并重算。
- **棋盘识别**：使用 OpenCV 识别自由布局中的四色普通及强化饿鬼、特殊黑色饿鬼、一至三级炸弹、贝壳、石头、箱子、树枝和固定区域的青铜钟，并生成路线候选及诊断图、JSON。
- **轻量识别**：全局开关默认关闭；开启后跳过全部障碍物识别。结果会明确标记未检查的类别，未检查对象不会被当作不存在。

### 棋盘识别与预演

离线棋盘识别可使用项目内截图运行，无需连接模拟器或启动 Agent：

```powershell
uv sync
uv run python tools/board_vision.py prepare --catalog training/board-vision/template_sources.json --output resource/base/image/board/development
uv run python tools/board_vision.py analyze --profile resource/base/image/board/development/profile.json --input "training/board-samples/raw/001_bombs_lv1x1_lv2x4.png" --output debug/board-vision/development
```

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
