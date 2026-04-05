# NNStudio — VS Code Developer Setup Guide

**Target audience:** developer opening this repository for the first time.  
**Goal:** build, run, and debug the engine tests and the Qt app in under 15 minutes.

This guide covers VS Code primarily, then Visual Studio and Qt Creator/Design Studio.

---

## Table of contents

1. [Prerequisites](#1-prerequisites)
2. [Recommended VS Code extensions](#2-recommended-vs-code-extensions)
3. [Opening the project](#3-opening-the-project)
4. [CMake kits and presets](#4-cmake-kits-and-presets)
5. [IntelliSense mode selection](#5-intellisense-mode-selection)
6. [Run / Debug profiles (`launch.json`)](#6-run--debug-profiles-launchjson)
   - 6.1 [GDB (MSYS2 MinGW) — primary on Windows](#61-gdb-msys2-mingw--primary-on-windows)
   - 6.2 [LLDB (MSYS2 LLVM / Clang)](#62-lldb-msys2-llvm--clang)
   - 6.3 [cppvsdbg (MSVC, Visual Studio debugger)](#63-cppvsdbg-msvc-visual-studio-debugger)
   - 6.4 [GTest filter input](#64-gtest-filter-input)
7. [Testing sidebar (GTest adapter)](#7-testing-sidebar-gtest-adapter)
8. [Qt `PATH` and DLL resolution](#8-qt-path-and-dll-resolution)
9. [Switching between GCC, Clang, and MSVC](#9-switching-between-gcc-clang-and-msvc)
10. [Visual Studio (Windows)](#10-visual-studio-windows)
11. [Qt Creator / Qt Design Studio](#11-qt-creator--qt-design-studio)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. Prerequisites

All paths below match the configuration recorded in `ai_priv/ai_priv.md`.
Adjust for your own installation locations.

| Tool | Version | Default location (Windows) |
|---|---|---|
| CMake | 3.30.5 | `C:\Users\<you>\Documents\Dev\Qt\Tools\CMake_64\bin` |
| Ninja | bundled with Qt | `C:\Users\<you>\Documents\Dev\Qt\Tools\Ninja` |
| GCC (MSYS2 MinGW-w64) | 15.2.0 | `C:\msys64\mingw64\bin\g++.exe` |
| Clang (MSYS2 LLVM) | install via `pacman -S mingw-w64-x86_64-clang` | `C:\msys64\mingw64\bin\clang++.exe` |
| GDB | bundled with MSYS2 MinGW | `C:\msys64\mingw64\bin\gdb.exe` |
| LLDB | bundled with MSYS2 LLVM | `C:\msys64\mingw64\bin\lldb.exe` |
| Qt | 6.10.1 | `C:\Users\<you>\Documents\Dev\Qt\6.10.1\mingw_64` |
| OpenSSL | 3.6.0 | `C:\Program Files\OpenSSL-Win64` |
| MSVC (optional) | VS 2022 | installed with Visual Studio |

> **Quick path check** — open a PowerShell terminal and run:
> ```powershell
> & 'C:\Users\plachy\Documents\Dev\Qt\Tools\CMake_64\bin\cmake.exe' --version
> & 'C:\msys64\mingw64\bin\g++.exe' --version
> ```
> If either fails, add the missing directory to your user `PATH`.

---

## 2. Recommended VS Code extensions

When you open the workspace VS Code will prompt you to install recommended extensions
(defined in `.vscode/extensions.json`). Accept. Manual install:

```
Ctrl+Shift+X  →  search and install each ID:
```

| Extension ID | Purpose |
|---|---|
| `ms-vscode.cmake-tools` | CMake configure / build / test / kit selection |
| `llvm-vs-code-extensions.vscode-clangd` | Fast IntelliSense via clangd (preferred over `ms-vscode.cpptools` for large C++ projects) |
| `ms-vscode.cpptools` | Fallback IntelliSense + **cppvsdbg** debugger (needed for MSVC debugging) |
| `vadimcn.vscode-lldb` | CodeLLDB: LLDB-based debugger for Clang/MinGW builds |
| `twxs.cmake` | CMake syntax highlighting |
| `josetr.cmake-language-support` | CMake Language Server (completion, hover, go-to-def) |
| `qt-official.qt-cpp-extension-pack` | Qt official pack: QML support, Qt paths, `.qml` syntax |
| `hbenl.vscode-test-explorer` + `matepek.vscode-catch2-test-adapter` | GTest in Testing sidebar |

> **Conflict warning:** `clangd` and `ms-vscode.cpptools` IntelliSense conflict.
> Disable `ms-vscode.cpptools` IntelliSense (keep the extension for `cppvsdbg` debugger only):
> add `"C_Cpp.intelliSenseEngine": "disabled"` to `.vscode/settings.json`.

---

## 3. Opening the project

```
File → Open Folder → …/Studio
```

VS Code opens the workspace root. The engine lives in `nnstudio/`. CMake Tools
detects `nnstudio/CMakePresets.json` automatically (no manual configuration file
path needed).

On first open you will see a notification: **"Would you like to configure CMake?"**  
Click **Yes**. CMake Tools opens the kit selector.

---

## 4. CMake kits and presets

### Kits (toolchains)

CMake Tools discovers kits from compilers on your `PATH`. The three you will use:

| Kit name (as shown in VS Code) | Generator | Compiler | Use case |
|---|---|---|---|
| `GCC 15.2.0 x86_64-w64-mingw32` | Ninja | `C:\msys64\mingw64\bin\g++.exe` | **Primary development** |
| `Clang x.y.0 x86_64` | Ninja | `C:\msys64\mingw64\bin\clang++.exe` | Cross-check warnings, ASan |
| `Visual Studio Community 2022 — amd64` | VS 2022 | MSVC `cl.exe` | Windows ABI compatibility check |

Switch kit: **status bar bottom → kit name → pick from list**.

### Presets

`nnstudio/CMakePresets.json` defines the configure presets. CMake Tools reads them
and populates the **Configure Preset** picker in the status bar:

| Preset | Description |
|---|---|
| `engine-ninja` | Ninja + GCC; engine library + tests; no Qt app |
| `engine-clang-ninja` | Ninja + Clang; same targets |
| `engine-vs` | Visual Studio 17 generator + MSVC; engine + tests |
| `app-ninja` | Ninja + GCC; all targets including Qt app (`nnstudio/app/`) |
| `app-vs` | Visual Studio 17 + MSVC; all targets including Qt app |

Select preset: **status bar → preset name → pick from list**.  
VS Code re-configures automatically (or press **Ctrl+Shift+P → CMake: Configure**).

### Build

```
Ctrl+Shift+P → CMake: Build       (builds selected target in selected preset)
Ctrl+Shift+B                       (runs the default build task from tasks.json)
```

---

## 5. IntelliSense mode selection

### Option A — clangd (recommended)

clangd reads the `compile_commands.json` that CMake generates:

```jsonc
// .vscode/settings.json
{
  "clangd.arguments": [
    "--compile-commands-dir=${workspaceFolder}/nnstudio/build/engine-ninja",
    "--header-insertion=never",
    "--clang-tidy"
  ],
  "C_Cpp.intelliSenseEngine": "disabled"   // keeps cpptools for debugger only
}
```

After configuring the preset, clangd auto-indexes. No other setup needed.

### Option B — ms-vscode.cpptools

If you prefer the Microsoft C/C++ extension IntelliSense, edit
`.vscode/c_cpp_properties.json` to set `compilerPath` to your active compiler:

```jsonc
{
  "configurations": [
    {
      "name": "GCC-MinGW-MSYS2",
      "compilerPath": "C:/msys64/mingw64/bin/g++.exe",
      "intelliSenseMode": "windows-gcc-x64",
      "cppStandard": "c++17",
      "includePath": [
        "${workspaceFolder}/nnstudio/include/**",
        "${workspaceFolder}/nnstudio/build/engine-ninja/_deps/eigen-src",
        "${workspaceFolder}/nnstudio/build/engine-ninja/_deps/googletest-src/googletest/include",
        "C:/Users/${env:USERNAME}/Documents/Dev/Qt/6.10.1/mingw_64/include/**"
      ],
      "defines": ["_DEBUG", "UNICODE"]
    },
    {
      "name": "Clang-LLVM-MSYS2",
      "compilerPath": "C:/msys64/mingw64/bin/clang++.exe",
      "intelliSenseMode": "windows-clang-x64",
      "cppStandard": "c++17",
      "includePath": [
        "${workspaceFolder}/nnstudio/include/**",
        "${workspaceFolder}/nnstudio/build/engine-ninja/_deps/eigen-src"
      ]
    },
    {
      "name": "MSVC-x64",
      "compilerPath": "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.xx.xxxxx/bin/Hostx64/x64/cl.exe",
      "intelliSenseMode": "windows-msvc-x64",
      "cppStandard": "c++17",
      "includePath": [
        "${workspaceFolder}/nnstudio/include/**",
        "${workspaceFolder}/nnstudio/build/engine-ninja/_deps/eigen-src"
      ]
    }
  ],
  "version": 4
}
```

---

## 6. Run / Debug profiles (`launch.json`)

`.vscode/launch.json` ships with the following profiles. Press **F5** to launch
the selected profile, or open the **Run and Debug** sidebar (`Ctrl+Shift+D`) and
pick from the dropdown.

### 6.1 GDB (MSYS2 MinGW) — primary on Windows

Uses the `cppdbg` debug adapter from `ms-vscode.cpptools` driving `gdb.exe`.

```jsonc
{
  "name": "Engine tests (GDB/MinGW)",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/engine-ninja/tests/test-core.exe",
  "args": [],
  "stopAtEntry": false,
  "cwd": "${workspaceFolder}/nnstudio",
  "environment": [
    { "name": "PATH", "value": "C:\\msys64\\mingw64\\bin;${env:PATH}" }
  ],
  "MIMode": "gdb",
  "miDebuggerPath": "C:\\msys64\\mingw64\\bin\\gdb.exe",
  "preLaunchTask": "Build engine-ninja",
  "setupCommands": [
    { "description": "Enable pretty-printing", "text": "-enable-pretty-printing" }
  ]
},
{
  "name": "Engine tests — filter (GDB/MinGW)",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/engine-ninja/tests/test-core.exe",
  "args": ["--gtest_filter=${input:gtestFilter}"],
  "stopAtEntry": false,
  "cwd": "${workspaceFolder}/nnstudio",
  "environment": [
    { "name": "PATH", "value": "C:\\msys64\\mingw64\\bin;${env:PATH}" }
  ],
  "MIMode": "gdb",
  "miDebuggerPath": "C:\\msys64\\mingw64\\bin\\gdb.exe",
  "preLaunchTask": "Build engine-ninja"
},
{
  "name": "NNStudio App (GDB/MinGW)",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/app-ninja/app/NNStudio.exe",
  "args": [],
  "cwd": "${workspaceFolder}",
  "environment": [
    { "name": "PATH",       "value": "C:\\msys64\\mingw64\\bin;C:\\Users\\${env:USERNAME}\\Documents\\Dev\\Qt\\6.10.1\\mingw_64\\bin;${env:PATH}" },
    { "name": "NN_BACKEND", "value": "cpu" }
  ],
  "MIMode": "gdb",
  "miDebuggerPath": "C:\\msys64\\mingw64\\bin\\gdb.exe",
  "preLaunchTask": "Build app-ninja"
},
{
  "name": "NNStudio App — CUDA backend (GDB/MinGW)",
  "type": "cppdbg",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/app-ninja/app/NNStudio.exe",
  "args": [],
  "cwd": "${workspaceFolder}",
  "environment": [
    { "name": "PATH",       "value": "C:\\msys64\\mingw64\\bin;C:\\Users\\${env:USERNAME}\\Documents\\Dev\\Qt\\6.10.1\\mingw_64\\bin;${env:PATH}" },
    { "name": "NN_BACKEND", "value": "cuda" }
  ],
  "MIMode": "gdb",
  "miDebuggerPath": "C:\\msys64\\mingw64\\bin\\gdb.exe",
  "preLaunchTask": "Build app-ninja"
}
```

> **Breakpoints in GCC builds:** because we compile with `-O2 -g`, some lines may
> be optimised away. Change to `-O0 -g3` in a local `CMakeUserPresets.json` override
> for a pure debug build (do not commit `-O0` to shared presets).

### 6.2 LLDB (MSYS2 LLVM / Clang)

Uses the `CodeLLDB` extension (`vadimcn.vscode-lldb`). Prerequisite: build with
the `engine-clang-ninja` preset.

```jsonc
{
  "name": "Engine tests (LLDB/Clang)",
  "type": "lldb",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/engine-clang-ninja/tests/test-core.exe",
  "args": [],
  "cwd": "${workspaceFolder}/nnstudio",
  "env": {
    "PATH": "C:\\msys64\\mingw64\\bin;${env:PATH}"
  },
  "preLaunchTask": "Build engine-clang-ninja"
}
```

CodeLLDB automatically pretty-prints STL containers. For custom types, add LLDB
formatters in `docs/lldb-formatters.py` (Phase 3.5 task).

### 6.3 cppvsdbg (MSVC, Visual Studio debugger)

Uses the `cppvsdbg` debug adapter bundled with `ms-vscode.cpptools`. Build with
the `engine-vs` preset first (generates a Visual Studio solution in
`build/engine-vs/`). `cppvsdbg` is Windows-only and does not work with
GCC/MinGW binaries — only MSVC-compiled executables.

```jsonc
{
  "name": "Engine tests (MSVC / cppvsdbg)",
  "type": "cppvsdbg",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/engine-vs/tests/Debug/test-core.exe",
  "args": [],
  "cwd": "${workspaceFolder}/nnstudio",
  "environment": [],
  "preLaunchTask": "Build engine-vs"
},
{
  "name": "NNStudio App (MSVC / cppvsdbg)",
  "type": "cppvsdbg",
  "request": "launch",
  "program": "${workspaceFolder}/nnstudio/build/app-vs/app/Debug/NNStudio.exe",
  "args": [],
  "cwd": "${workspaceFolder}",
  "environment": [
    { "name": "PATH", "value": "C:\\Users\\${env:USERNAME}\\Documents\\Dev\\Qt\\6.10.1\\msvc2022_64\\bin;${env:PATH}" }
  ],
  "preLaunchTask": "Build app-vs"
}
```

> **NatVis:** `nnstudio.natvis` (in the repo root) is automatically loaded by
> `cppvsdbg` and by Visual Studio. It makes `Tensor`, `Shape`, `Result<T>`, and
> `Parameter` show meaningful summaries in Locals/Watch. No configuration needed.

### 6.4 GTest filter input

The "Engine tests — filter" profile uses a VS Code `input` variable. Define it at
the bottom of `launch.json` (in the top-level `"inputs"` array):

```jsonc
"inputs": [
  {
    "id": "gtestFilter",
    "type": "promptString",
    "description": "GTest filter expression  (e.g.  LayerTest.*  or  *Backward*)",
    "default": "*"
  }
]
```

When you launch the profile you get an inline text box — type a filter like
`LayerTest.Dropout*` and press Enter. The test binary receives `--gtest_filter=LayerTest.Dropout*`.

---

## 7. Testing sidebar (GTest adapter)

Install `hbenl.vscode-test-explorer` and a GTest adapter. The adapter discovers
test binaries from CMake build output and lists all GTest cases in the
**Testing** sidebar (`Ctrl+Shift+P → Test: Focus on Test Explorer View`).

Add to `.vscode/settings.json`:

```jsonc
{
  "testExplorer.useNativeTesting": false,
  "cmake.ctest.testExplorerIntegration": true
}
```

CMake Tools 1.17+ has its own native CTest integration that populates the Testing
sidebar directly — prefer this over the third-party adapter if available.
Right-click any test → **Debug Test** to attach the debugger to a single case.

---

## 8. Qt `PATH` and DLL resolution

Qt DLLs must be on `PATH` when running the app from VS Code; otherwise the
executable starts and immediately crashes with a missing DLL error.

**Option A — per-profile `environment` in `launch.json`** (shown above): explicit
and portable; adjust the Qt version path per machine.

**Option B — `.env` file** (cleaner for large teams): create `.vscode/.env`
(gitignored — add to `.gitignore`):
```
PATH=C:\Users\plachy\Documents\Dev\Qt\6.10.1\mingw_64\bin;C:\msys64\mingw64\bin;${env:PATH}
NN_BACKEND=cpu
```
Reference it in `launch.json`:
```jsonc
"envFile": "${workspaceFolder}/.vscode/.env"
```

**Option C — `windeployqt`**: run `windeployqt NNStudio.exe` once after build to
copy all required Qt DLLs next to the executable. The app then runs without
modifying `PATH`. Suitable for CI and distribution.

---

## 9. Switching between GCC, Clang, and MSVC

| Goal | Steps in VS Code |
|---|---|
| Change compiler (kit) | Status bar → kit name → pick new kit → CMake reconfigures |
| Change preset | Status bar → preset name → pick preset → CMake reconfigures |
| Force reconfigure | `Ctrl+Shift+P → CMake: Delete Cache and Reconfigure` |
| Use `-O0 -g3` locally | Create `nnstudio/CMakeUserPresets.json` (gitignored) inheriting from a base preset; override `CMAKE_BUILD_TYPE=Debug` or add `CMAKE_CXX_FLAGS=-O0 -g3` |
| Run ASan (Clang) | `engine-clang-ninja` preset + add `-fsanitize=address,undefined` to `CMAKE_CXX_FLAGS` in a local user preset |
| Cross-compile | Create a new preset with `CMAKE_TOOLCHAIN_FILE` pointing to your cross-toolchain |

### `CMakeUserPresets.json` example (local debug overrides, never commit)

```jsonc
{
  "version": 6,
  "include": ["CMakePresets.json"],
  "configurePresets": [
    {
      "name": "engine-debug",
      "inherits": "engine-ninja",
      "displayName": "Engine (GCC, Debug, -O0)",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE":  "Debug",
        "CMAKE_CXX_FLAGS":   "-O0 -g3 -fno-omit-frame-pointer"
      }
    },
    {
      "name": "engine-asan",
      "inherits": "engine-clang-ninja",
      "displayName": "Engine (Clang, ASan+UBSan)",
      "cacheVariables": {
        "CMAKE_BUILD_TYPE": "Debug",
        "CMAKE_CXX_FLAGS":  "-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer"
      }
    }
  ],
  "buildPresets": [
    { "name": "engine-debug", "configurePreset": "engine-debug" },
    { "name": "engine-asan",  "configurePreset": "engine-asan"  }
  ],
  "testPresets": [
    { "name": "engine-debug", "configurePreset": "engine-debug", "output": { "outputOnFailure": true } },
    { "name": "engine-asan",  "configurePreset": "engine-asan",  "output": { "outputOnFailure": true } }
  ]
}
```

Add to `.gitignore`: `CMakeUserPresets.json`.

---

## 10. Visual Studio (Windows)

Visual Studio 2022 supports "Open Folder" CMake projects natively — **no `.sln`
file required**.

1. `File → Open → Folder` → select `nnstudio/`
2. VS detects `CMakePresets.json` and shows a **Configuration** dropdown in the
   toolbar. Select `engine-vs` or `app-vs`.
3. **Build:** `Build → Build All` (or `Ctrl+Shift+B`)
4. **Test:** `Test → Test Explorer` — GTest cases appear after a successful build.
5. **Breakpoints:** set in the editor; press `F5` to start debugging with the VS
   debugger (MSVC `cl.exe` builds only; MinGW builds under VS won't use `cppvsdbg`).

### NatVis pretty-printer

`nnstudio.natvis` (repo root `docs/`) is loaded automatically by Visual Studio
when the project folder contains it. Verify: open a debug session, inspect a
`Tensor` variable in **Locals** — you should see `shape=[4] data=[0.1, 0.2, ...]`
instead of a raw struct dump.

### MSYS2 DLL friction

If you build with the MinGW kit inside VS (via CMake `engine-ninja` preset),
the produced `.exe` links against MSYS2 DLLs. Visual Studio's built-in debugger
(`cppvsdbg`) can attach but may not resolve symbols correctly. **Use GDB via VS
Code instead** for MinGW builds; reserve Visual Studio for MSVC (`engine-vs` /
`app-vs`) builds.

Workaround for running the exe from VS without PATH trouble:
add `C:\msys64\mingw64\bin` to the system or user `PATH` permanently.

---

## 11. Qt Creator / Qt Design Studio

### Qt Creator — full project

1. `File → Open File or Project` → select `nnstudio/CMakeLists.txt`
2. Qt Creator asks to configure a kit. Select:
   - **Qt version:** Qt 6.10.1 MinGW (detected from your Qt installation)
   - **Compiler (C++):** MinGW GCC 15.2.0 (`C:\msys64\mingw64\bin\g++.exe`)
   - **Debugger:** GDB (`C:\msys64\mingw64\bin\gdb.exe`)
3. Click **Configure** → CMake runs → project tree appears.
4. **Build:** `Ctrl+B` (uses the `engine-ninja` or `app-ninja` preset mapped to the kit).
5. **Run configuration:** click the green play button; select `test-core` or `NNStudio`
   from the run configuration dropdown.
6. **Debug:** `F5`; breakpoints work as expected.

### Qt Design Studio — QML-only workflow

Qt Design Studio is for pure UI/QML design work without building the C++ engine.

1. Open the `.qmlproject` file in `nnstudio/app/ui/` (created in Phase 3.5).
2. Design Studio creates a live preview of `.qml` files using a mock
   `ModelController` QML singleton that provides stub data.
3. Any `.qml` changes are immediately reflected in the live preview — no build needed.
4. Designed components are committed normally; the C++ backend is not involved.

> **Version note:** Qt Design Studio 4.x supports Qt 6.x projects natively.
> Older versions (3.x) had limited Qt 6 support — use 4.x or later.

---

## 12. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| CMake Tools says "No kit found" | GCC/Clang not on PATH | Add `C:\msys64\mingw64\bin` to user PATH; restart VS Code |
| `clangd` reports missing headers | `compile_commands.json` path wrong | Set `--compile-commands-dir` to active preset's build dir |
| App crashes on launch (DLL not found) | Qt or MSYS2 bin not in PATH | Add Qt bin + MSYS2 bin to `.vscode/.env` or `PATH` in launch.json |
| GDB says "no debug info" | Built with a Release preset | Use `engine-debug` (local user preset) or `CMAKE_BUILD_TYPE=Debug` |
| GTest tests not in Testing sidebar | CTest integration not enabled | Add `"cmake.ctest.testExplorerIntegration": true` to settings.json |
| MSVC build fails on `__builtin_trap` | `QuantumBackend.cpp` uses GCC intrinsic | Replace with `__assume(false); __debugbreak();` in an MSVC `#ifdef` |
| IntelliSense slow / incorrect | Competing with clangd | Set `"C_Cpp.intelliSenseEngine": "disabled"` when using clangd |
| NatVis not loading in VS Code | `cppvsdbg` only loads `.natvis` from project | Add `"visualizerFile": "${workspaceFolder}/docs/nnstudio.natvis"` to the launch profile |
