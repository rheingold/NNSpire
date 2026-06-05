# ADR-010 — CMake 3.21+ as the Build System

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNSpire must build on Windows (MSVC), Linux (GCC/Clang), and macOS (Apple Clang).
The build system must integrate with:
- Qt 6 (which requires CMake 3.21+ for its `qt_add_executable` / `qt_add_qml_module` commands).
- pybind11 (ships its own CMake helpers).
- Eigen (header-only, CMake `find_package` or bundled include path).
- GoogleTest, ONNX protobuf (via FetchContent or bundled in `third-party/`).

Alternatives:

| System | Notes |
|---|---|
| **CMake** | Industry standard; Qt 6 requires it; first-class IDE support (VS, Qt Creator, CLion) |
| Meson | Good cross-platform; Qt 6 Meson support is unofficial and lagging |
| Bazel | Excellent for monorepos; steep learning curve; poor Qt integration |
| qmake | Qt-specific; being deprecated by Qt; no future |
| xmake | Niche; limited Qt 6 integration |

---

## Decision

**CMake 3.21+** is the build system for NNSpire.

- Top-level `CMakeLists.txt` lives in `NNSpire/` (the source root).
- Build directory: `build/` (gitignored).
- `CMakePresets.json` provides named configurations:
  - `engine-ninja` (current) — Ninja generator, Debug, for engine development.
  - Additional presets for Release, MSVC, Clang, etc. to be added per phase.
- CMake target names:
  - `NNSpire-core` — static library
  - `NNSpire-plugin-api` — interface (header-only) library
  - `NNSpire` — executable (Qt 6 app)
  - `NNSpire-runner` — optional sidecar executable
  - `NNSpire-onnxrt-custom-ops` — ONNX Runtime custom-op shared library
  - `NNSpire_py` — pybind11 Python extension module
- Qt binary path is stored in the developer's `ai_priv/ai_priv.md` (gitignored) and passed
  via `CMAKE_PREFIX_PATH`.

---

## Consequences

**Positive**
- Qt Creator, VS 2022, CLion, and VS Code (CMake Tools) all work natively.
- `CMakePresets.json` makes CI configuration and developer onboarding consistent.
- `FetchContent` / `find_package` for dependencies is well-supported.

**Negative / constraints**
- CMake DSL is notoriously verbose and has several footguns (link vs interface includes, etc.).
- Qt 6.x patches occasionally require minimum CMake bumps; must track Qt changelog.

**Follow-on**
- `CMakePresets.json` should be extended with Release, ASAN, and CI presets in Phase 1.
- Consider using CPM.cmake or vcpkg for dependency management in later phases.
- Add a `cmake/` directory for reusable CMake helper modules.
