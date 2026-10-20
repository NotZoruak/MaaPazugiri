<div align="center">

<img alt="MaaPazugiri logo" src="./assets/logo.png" width="240" />

# MaaPazugiri

<br>

<img alt="Platform" src="https://img.shields.io/badge/Platform-Android%20%7C%20Emulator-3DDC84?style=flat-square&logo=android&logoColor=white" />
<img alt="Language" src="https://img.shields.io/badge/Language-Python-3776AB?style=flat-square&logo=python&logoColor=white" />
<img alt="Framework" src="https://img.shields.io/badge/Framework-MaaFramework-6750A4?style=flat-square" />
<img alt="License" src="https://img.shields.io/badge/License-AGPL--3.0--or--later-blue?style=flat-square" />

A MaaFramework automation project for the official Japanese version of _Touken Ranbu Pazugiri_. It uses a Python Agent and Project Interface V2, and connects to Android devices or emulators through ADB.

</div>

> [!TIP]
>
> This project is under active development. Game startup, the resident battle flow, battle-state checks, and candidate route previews are implemented. Automatic board touch input is not implemented, and board vision has not had an independent evaluation. Features will expand as they are verified on the game.

## ⚠️ Usage and risk notice

> [!NOTE]
>
> This is a community-developed third-party automation tool and is not affiliated with the game operator. It is built on MaaFramework and uses screen recognition and ADB to interact with the game interface.
>
> Before using it, review the game's terms of service and the rules that apply in your region. Automated actions may produce unintended in-game results. Review task settings and observe the device during first runs and while validating new features.
>
> The project is still under development, and page recognition and flows cover only verified situations. Unverified layouts, animations, and game rules may cause a task to stop or recognition to fail.

## Download and installation

There is currently no formal release package. Run from source during development; see [Development](#development) for environment setup and commands. Download instructions will be added when a release package is available.

## First use

1. Prepare an Android device or emulator with ADB enabled. MuMu is the current preferred emulator.
2. Install and launch _Touken Ranbu Pazugiri_ on the device, using the official Japanese interface.
3. Load the project in a client that supports MaaFramework Project Interface V2 and connect the Android / Emulator controller.
4. On the first run, observe the client logs and device screen to confirm that the resolution, page state, and task settings are suitable.

The currently verified battle screenshots use a 720×1280 layout. Other resolutions and interface layouts have not been fully verified. Do not leave unverified tasks running unattended for long periods.

## Features

The project currently focuses on game startup, the resident battle flow, and board-vision research:

- **Start the game**: Restart the game through ADB and recognize the title screen.
- **Resident battle**: Navigate to the currently configured chapter and stage, select the target team, and enter battle.
- **Battle-state checks**: Recognize actionable state, victory results, and failure continue prompts in verified layouts.
- **Candidate route preview**: Detect ghouls in the current screenshot and generate candidate connection routes with diagnostic images and JSON reports. It does not click or drag.
- **Offline board-vision prototype**: Analyze four colors of normal and enhanced ghouls, special black ghouls, level-one through level-three bombs, shells, rocks, boxes, branches, and the bronze bell in its fixed screen region.
- **Lightweight recognition**: Skip obstacle recognition to reduce computation. Results explicitly indicate which obstacle categories were not checked.

Board-vision results are candidates for review, not authorization for touch input or proof that a route is traversable. Touching routes, complete in-game battle validation, and independent recognition evaluation are still pending.

## Usage

### Running tasks

Connect the device in a MaaFramework client, choose a task, and start it. Resident battle attempts to navigate to the currently configured chapter and stage; the target stage, team, and Pipeline parameters are determined by the project configuration.

The battle action currently checks screen state and generates route previews, but does not interact with the board. If the screen state cannot be confirmed, the task may stop or follow its error branch. Watch the client logs and stop the task if the screen differs from expectations.

### Board recognition and route preview

Offline board recognition can use project screenshots without connecting a device or starting the Agent:

```powershell
uv sync
uv run python tools/board_vision.py prepare --catalog training/board-vision/template_sources.json --output resource/base/image/board/development
uv run python tools/board_vision.py analyze --profile resource/base/image/board/development/profile.json --input "training/board-samples/raw/001_bombs_lv1x1_lv2x4.png" --output debug/board-vision/development
```

Candidate results and diagnostics are for development review and do not establish recognition accuracy. Unrecognized areas cannot be assumed empty or traversable. Lightweight mode skips every obstacle, including the bronze bell.

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
