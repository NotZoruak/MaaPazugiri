<div align="center">

<img alt="MaaPazugiri logo" src="./assets/logo.png" width="240" />

# MaaPazugiri

<br>

<img alt="Platform" src="https://img.shields.io/badge/Platform-Emulator-3DDC84?style=flat-square" />
<img alt="Language" src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img alt="Framework" src="https://img.shields.io/badge/Framework-MaaFramework-6750A4?style=flat-square" />
<img alt="License" src="https://img.shields.io/badge/License-AGPL--3.0--or--later-blue?style=flat-square" />

A MaaFramework automation project for the official Japanese version of _Touken Ranbu Pazugiri_. It uses a Python Agent and Project Interface V2, with development and verification currently targeting the MuMu emulator over ADB.

</div>

The desktop GUI uses [MFAA](https://github.com/NotZoruak/MFAA), a customized version of upstream [MaaXYZ/MFAAvalonia](https://github.com/MaaXYZ/MFAAvalonia) for MaaPazugiri. MFAA will evolve independently around this project's needs and may gradually diverge from upstream; refer to the MFAA repository for GUI features and behavior.

> [!TIP]
>
> Project status: Ongoing maintenance and iteration
> The main daily workflows are implemented, including game startup, resident battles, and reward claims. The project is now in a maintenance and iteration phase, focusing on bug fixes, game-version compatibility, and dependency updates, while evaluating new features, tasks, and workflows as needed.
> Current known issues: the selected color cannot yet be preserved accurately during Ranbu, and enhanced ghoul recognition is not yet accurate enough.

## ⚠️ Usage and risk notice

> [!NOTE]
>
> This is a community-developed third-party automation tool and is not affiliated with the game operator. It is built on MaaFramework and uses screen recognition and ADB to interact with the game interface.
>
> Before using it, review the game's terms of service and the rules that apply in your region. Automated actions may produce unintended in-game results. Review task settings and observe the emulator during first runs and while validating new features.
>
> The project is still being improved and may contain bugs.

## Download and installation

Download the MaaPazugiri release archive from [GitHub Releases](https://github.com/NotZoruak/MaaPazugiri/releases). Choose the release package attached to a release; do not download GitHub's automatically generated `Source code` archives.

1. Extract the release archive into a directory with an English name.
2. Double-click `MaaPazugiri.exe` to launch. The first launch may take longer; please wait.
3. Start the MuMu emulator and connect it through ADB.

> If you see a message that `.NET Desktop Runtime 10.0` or `VCRUNTIME140.dll` is missing, right-click `DependencySetup_依赖库安装_win.bat` and select **Run as administrator**. Restart MaaPazugiri after installation.

## Features

The project currently focuses on game startup, resident battles, and dynamic board recognition:

- **Start the game**: Restart the game through ADB and recognize the title screen.
- **Resident battle**: Choose a configured chapter, stage, and team, enter battle, and repeat according to the task count. Options include returning to the main page before sortie, using stamina recovery items, and spending jade to restore stamina. Only the default strategy is currently available in the task selector.
- **Battle recognition and decisions**: Recognize normal actionable state, ranbu, results, and ultimate confirmation states; choose an ultimate, bomb, enhanced or black ghoul route, normal-color link, or ranbu sweep according to the active strategy.
- **Battle input**: Ultimate and bomb taps, normal links, and special route gestures are implemented. Normal links require stable-frame checks; after each action, the screen is read again and the result is confirmed. An unconfirmed normal route is retried from the other end on the current board, then temporarily excluded if it still fails.
- **Board recognition**: OpenCV recognizes freely arranged normal and enhanced ghouls in four colors, special black ghouls, level-one through level-three bombs, shells, rocks, boxes, branches, and the bronze bell in its fixed region. It generates route candidates and diagnostic images and JSON reports.
- **Lightweight recognition**: The global switch is off by default. When enabled, it skips all obstacle recognition. Results identify unchecked categories, which are not treated as absent.

### Board recognition and route preview

Offline board recognition can use project screenshots without connecting the emulator or starting the Agent:

```powershell
uv sync
uv run python tools/board_vision.py prepare --catalog training/board-vision/template_sources.json --output resource/base/image/board/development
uv run python tools/board_vision.py analyze --profile resource/base/image/board/development/profile.json --input "training/board-samples/raw/001_bombs_lv1x1_lv2x4.png" --output debug/board-vision/development
```

## Development

Requirements: Python 3.13, Node.js 22.13 or later, pnpm 11, and `uv`. Install dependencies and start the Agent from the project directory:

```powershell
uv sync
uv run python agent/main.py
```

Install the development tooling and run resource, formatting, and Python checks:

```powershell
pnpm install
pnpm check
pnpm check:py
```

When adding a Python dependency, run `uv add <package>`, then synchronize the release dependency manifests:

```powershell
create-maa-project --update python-deps
```

`create-maa-project` maintains the project structure and Project Interface. Pipeline flows, Agent logic, and vision resources are maintained according to the project's development conventions.

## Acknowledgements

- [MaaFramework](https://github.com/MaaXYZ/MaaFramework) — automation framework and Project Interface.
- [create-maa-project](https://github.com/Windsland52/create-maa-project) — Maa standard Agent project scaffolding and maintenance.
- [OpenCV](https://opencv.org/) — offline board-vision prototype.

中文文档：[README.md](README.md)
