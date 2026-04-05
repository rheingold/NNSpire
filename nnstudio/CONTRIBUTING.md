# Contributing to NNStudio — Repository Architecture

> *The most dangerous moment is when everything looks fine.*  
> — cave diving safety rule; adopted as this project's guiding principle

> This file explains **why the repository is structured the way it is**.
> For what NNStudio does as software, read [README.md](README.md) (once created).
> For the full architectural walkthrough, read [../blueprints.md](../blueprints.md).

---

## Reading order

If you are new to the codebase, read the folders in this order.
This is the dependency order — lower items build on higher items.

1. `include/core/` — the engine contract: Tensor, ILayer, IBackend, ILoss, IOptimizer, Trainer
2. `src/core/` — the engine implementation (mirrors `include/core/` exactly)
3. `include/builtin/` + `src/builtin/` — NNStudio's own implementations: Dense, CpuBackend, Adam, MSE
4. `plugins/` — third-party extension slot (placeholder until Phase 2)
5. `pipeline/` — data flow: input adapters, tokenization, RAG, execution stages (Phase 4)
6. `python-bridge/` — pybind11 bridge wrapping `nnstudio-core` for Python (Phase 2)
7. `app/` — Qt 6 QML Studio UI (Phase 3+)
8. `deployment/` — ONNX export, packaging, remote runner (Phase 5)
9. `tests/` — GTest suites; organised to mirror the component they test
10. `third-party-deps/` — FetchContent-managed deps (Eigen, GTest); do not edit manually

CMake targets are defined directly in `CMakeLists.txt` (root level — no subdirectory cmake files for core or builtin). Targets that require optional features (`pipeline`, `deployment`, `app`, `python-bridge`) are behind `add_subdirectory` guards.

---

## Namespace tiers

See `blueprints.md § Appendix — Namespace Map` for the full design.
Short version:

| Tier | Namespace | What lives here |
|---|---|---|
| 1 — public contract | `nnstudio::core::` | Interfaces (`ILayer`, `IBackend`, …), data types (`Tensor`, `Result<T>`) |
| 2 — engine guts | `nnstudio::internal::` | Graph, training loop, file I/O — not for plugin authors |
| 3 — bundled implementations | `nnstudio::builtin::` | Dense, CpuBackend, Adam, MSE — treated same as any plugin |

---

## Folder structure conventions

### The one-directional mirror rule

`include/` is the **Plugin SDK surface** — the exact set of headers installed by `cmake --install`
and shipped to plugin authors. `src/` is build-only.

Every subfolder in `include/` **must** have a matching subfolder in `src/`.
The reverse does not hold: `src/` may gain private subfolders (e.g. `src/internal/graph/`)
that have no `include/` counterpart because they are never part of the public surface.

```
include/core/         ← exists  →  src/core/         ← must exist
include/builtin/      ← exists  →  src/builtin/      ← must exist
src/internal/graph/   ← future   →  include/.../graph/ ← NOT required
```

### Plural folder names

Domain collection folders use plural names by convention (`layers/`, `backends/`, `losses/`).
This is a convention, not a rule — use natural language where singular fits better (`formats/`, `graph/`).
The `I` prefix on filenames (`ILayer.h`, `IBackend.h`) is the complete signal that a file is an abstract interface.
No separate `interfaces/` subfolder is needed.

### Per-folder documentation

Every folder that contains primarily subfolders (rather than source files directly) should carry:
- `README.md` — what this folder contains in terms of **software concept** (runtime behaviour)
- `CONTRIBUTING.md` — **this kind of file** — why this folder exists here, why it is structured this way

---

## Namespace migration status

Complete — all migrations applied:
- Phase 1: `nnstudio::layers::Dense` → `nnstudio::builtin::layers::Dense` (and all builtin types)
- Phase 2: `nnstudio::Tensor` → `nnstudio::core::Tensor` (and all Tier 1 types)
- `include/` + `src/` unified to a single shared root under `nnstudio/` (Option B layout)

All 16 `ctest` tests pass.

---

## Toolchain

| Tool | Version | Location |
|---|---|---|
| CMake | 3.30.5 | `C:\Users\plachy\Documents\Dev\Qt\Tools\CMake_64\bin` |
| Ninja | 1.12.1 | `C:\msys64\mingw64\bin` |
| GCC | 15.2.0 | `C:\msys64\mingw64\bin` |
| Qt | 6.10.1 | `C:\Users\plachy\Documents\Dev\Qt\6.10.1\mingw_64` |
| Eigen | 3.4.0 | FetchContent (auto-downloaded) |
| GTest | 1.14.0 | FetchContent (auto-downloaded) |

Build command: `cmake --preset engine-ninja && cmake --build --preset engine-ninja`  
Test command: `ctest --preset engine-ninja`
