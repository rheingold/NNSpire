# ADR-001 — Qt 6 as the GUI Framework

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio must run natively on Windows, macOS, and Linux with a single source tree.
The UI is expected to host complex, data-rich panels: real-time topology graphs,
weight heatmaps, training loss charts, and neuron-level visualisations.
The application must also integrate tightly with a C++ engine core.

Candidate frameworks considered:

| Framework | Notes |
|---|---|
| **Qt 6** | Mature, cross-platform, excellent C++ and QML support, OpenGL/Vulkan via Qt Quick |
| Electron | Node.js-based; large memory overhead; poor C++ integration; not suitable for compute-adjacent work |
| wxWidgets | Native widgets; lacks modern GPU-accelerated scene rendering |
| Dear ImGui | Immediate mode; great for tools but limited layout/theming capabilities |
| Flutter | Requires Dart; C++ FFI is possible but not first-class |

---

## Decision

**Qt 6.5 LTS** is the minimum primary target for NNStudio.

- Primary UI language: QML + Qt Quick (see ADR-018).
- All Qt-version-specific API calls are isolated behind
  `nnstudio/app/qt_version_helpers.h`.
- The codebase is written with disciplined version isolation so that **compiling against
  an older Qt version (including Qt 5.15 LTS) remains feasible** without a major rewrite,
  should that ever be needed for a specific platform or deployment target.
  This is a **coding discipline**, not a separately maintained build configuration.
  No second CI lane is kept green for Qt 5.x by default.

### Primary target OS matrix (Qt 6.5+)

| Platform | Minimum version |
|---|---|
| Windows | 10 (1809+) |
| macOS | 11 Big Sur |
| Linux | Ubuntu 20.04 |

### Backwards-compat reach (if `qt_version_helpers.h` isolation holds)

| Platform | Potentially reachable |
|---|---|
| Windows | 7 SP1+ (x64) via Qt 5.15 |
| macOS | 10.15 Catalina via Qt 5.15 |
| Linux | Ubuntu 18.04 via Qt 5.15 |

These are **aspirational fallback targets**, not committed support tiers.
Pre-Win7 and pre-Catalina are out of scope — Qt 5.15 LTS itself dropped them.

---

## Consequences

**Positive**
- Single C++ codebase for UI, confirmed cross-platform.
- GPU-accelerated rendering available via Qt Quick Scene Graph / RHI.
- Strong signal/slot integration with C++ engine objects.
- Qt's CMake integration (Qt::Core, Qt::Quick, etc.) is stable at CMake 3.21+.

**Negative / constraints**
- GPL v3 app binary due to Qt 6 LGPL and app-layer GPL (see ADR-006). Acceptable.
- Qt 5 compatibility layer adds minor ongoing maintenance cost.
- Requires Qt installation on developer machines (recorded in `ai_priv/ai_priv.md`).

**Follow-on**
- `nnstudio/app/qt_version_helpers.h` must be created and kept up to date.
- Qt binary path must be recorded in `ai_priv/ai_priv.md` (gitignored).
- When Qt 7 is released, revisit Qt 5 secondary config.
- Windows 7 support via Qt 5.15 secondary must not introduce any workarounds in
  shared engine code — isolation in `qt_version_helpers.h` only.
