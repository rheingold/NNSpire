# ai.md — NNSpire AI Assistant Knowledge Base

This file is the persistent memory for AI coding assistants working on NNSpire.
Update it whenever a significant architectural decision is taken, a convention is established,
or a non-obvious implementation detail is discovered. Do NOT commit secrets here — use `ai_priv/ai_priv.md`.

---

## Project identity

- **Name**: NNSpire (full), Studio (short internal reference)
- **Purpose**: Multiplatform Qt 6 neural-network design/training/deployment/learning workbench
- **Project root**: `Studio/` (the folder containing this file)
- **Source root**: `Studio/NNSpire/`
- **Standards KB**: `Studio/docs/ai-standards-kb/` — read-only reference; do not modify its structure
- **Inception date**: 2026-03-31

---

## Terminology conventions (dialogue + documentation only)

These terms are used in `ai.md`, `README.md`, `ARCHITECTURE.md`, ADRs, chapter titles,
and verbal descriptions. They have **no effect on namespaces, folder names, CMake targets,
or any code artefact** — those are governed by the rules in "Standing architectural decisions" below.

| Term | Meaning | Code materialisation |
|---|---|---|
| **NNSpire** | The overall product — application + engine + plugins | Repo root, product name |
| **Engine** (capital E) | Collective shorthand for everything under `NNSpire::{core,builtin}` and their backends, plugins, and related signatures. Used when referring to the compute/runtime layer as a whole in documentation and dialogue. | `NNSpire::core`, `NNSpire::builtin`, `NNSpire-core`, `NNSpire-builtin` CMake targets |
| **CpuBackend** | Reference / didactical backend (Eigen) | `NNSpire::builtin::backends::CpuBackend` |
| **CudaBackend** | Our own cuBLAS/cuDNN backend (transparent, didactical) | `NNSpire::builtin::backends::CudaBackend` |
| **LibTorchBackend** | Production opt-in backend delegating to LibTorch | `NNSpire::builtin::backends::LibTorchBackend` |
| **Studio** (capital S, no "NN") | Short internal reference to the application / UI layer | `NNSpire/app/`, `NNSpire` Qt executable target |
| **Plugin** | Any loadable extension (C++ `.dll`/`.so` or Python `.py`) conforming to the NNSpire plugin ABI | `NNSpire::plugins::*`, `NNSpire_plugin.h` |
| **Engine API** | The public C++ interface surface of the Engine (`NNSpire::core` interfaces + `NNSpire::builtin`) | `include/core/`, `include/builtin/` |
| **torch-compat surface** | The subset of the Engine API that maps 1:1 to PyTorch/Keras names; usable as a drop-in | `include/NNSpire/torch_compat.h`, `NNSpire.nn.*` Python |

> **No codename** for the Engine beyond "Engine" / "NNSpire Engine". Inventing a separate brand
> (e.g. "NNCore", "Synapse") would create a two-name problem everywhere and add posh connotations
> implying competition with mature frameworks. Decision recorded 2026-04-06.

---

## Standing architectural decisions

### Language stack
- **Core engine** (`NNSpire/core/`, `NNSpire/backends/`, `NNSpire/plugin-api/`): C++17
- **UI**: Qt 6 + QML (primary); Qt 5.15 LTS secondary build config for older Linux/macOS
- **Python plugins & bridge**: pybind11 exposing `NNSpire-core` to Python
- **Dual-language principle**: every plugin, export, script, sample, and runner client MUST ship both a Python and a C++ implementation. Neither is optional. C++ API is defined first; pybind11 mirrors it exactly.

### Build system
- CMake 3.21+ (required for Qt 6 CMake integration)
- Top-level `CMakeLists.txt` in `NNSpire/`
- Build directory: `build/` (gitignored)
- Target names: `NNSpire-core` (static lib), `NNSpire-plugin-api` (interface lib/headers), `NNSpire` (executable), `NNSpire-runner` (optional sidecar)

### Qt version policy
- Primary target: Qt 6.5 LTS or Qt 6.7+
- All Qt 6-only APIs isolated behind `NNSpire/app/qt_version_helpers.h`
- Minimum OS targets (Qt 6 build): Windows 10, macOS 11, Ubuntu 20.04
- Qt 5.15 LTS build config maintained for older Linux/macOS — `CMake -DNN_QT_VERSION=5`
- Windows 9x/XP/Vista/7: explicitly out of scope, hard no, never revisit

### C++ standard and style
- C++17 minimum; C++20 features only behind `#if __cplusplus >= 202002L` guards
- Naming: `PascalCase` classes, `camelCase` methods, `snake_case` local variables, `SCREAMING_SNAKE` constants/macros
- Headers: `.h` extension, `#pragma once`
- All public API headers are in `include/NNSpire/` subdirs — no implementation in public headers except templates
- No exceptions in the engine core (use `Result<T, Error>` return type) — exceptions allowed in the Qt app layer
- RTTI disabled in core (`-fno-rtti`)

### Python conventions
- Python 3.10+ minimum (matches embeddable Python distribution)
- Type hints required on all public functions
- `pyproject.toml` (PEP 517/518) for all Python packages
- Runtime Python lives in `<install>/runtime/python/` — never pollutes system Python

### File-only content at project root
Only `.md` files, `.gitignore`, and eventual `CMakeLists.txt` live at the project root.
No scripts, no binaries, no data files at root. Everything else in a named subdirectory.

---

## Phase execution order

| Phase | Scope | Status |
|---|---|---|
| 0 | Project scaffolding — docs, directory layout, .gitignore | **IN PROGRESS** |
| 1 | NN engine core C++ library — Tensor, Layer, Graph, Trainer, formats | not started |
| 2 | Plugin SDK — C ABI, TrustVerifier, TrustStore, PluginLoader, pybind11 bridge | not started |
| 3 | Qt 6 QML Studio UI — layer editor, training dashboard, weight viewer, KB help | not started |
| 4 | Full pipeline — input adapters, tokenization, context DB, chained execution, output | not started |
| 5 | Deployment & runners — export wizards, runner connectors, `.nnsr` bundle, Registry API | not started |
| 6 | Quantum backend — hybrid classical-quantum graph, Qiskit bridge | not started |

---

## Component map (one-line each)

| Path | What it is |
|---|---|
| `NNSpire/core/tensor/` | `Tensor<T>` — shape, strides, dtype, device, ops |
| `NNSpire/core/layers/` | Abstract `Layer` + built-ins (Dense, Conv2D, MHA, etc.) |
| `NNSpire/core/activations/` | Activation functions with forward value AND derivative |
| `NNSpire/core/losses/` | Loss functions (MSE, CrossEntropy, Huber, BCE) |
| `NNSpire/core/optimizers/` | SGD, Adam, AdamW, RMSProp with LR schedulers |
| `NNSpire/core/graph/` | `ComputeGraph` — DAG of ops, autograd, drives visual editor |
| `NNSpire/core/training/` | `Trainer` — training loop, callbacks, checkpoint save/load |
| `NNSpire/core/formats/` | `OnnxIO`, `NNSFormat` (`.nns` project file) |
| `NNSpire/core/features/` | `FeatureFlags.h` — all flags start `FREE` |
| `NNSpire/backends/` | `CpuBackend`, `CudaBackend`, `QuantumBackend` (stub), `BackendRegistry` |
| `NNSpire/plugin-api/` | `NNSpire_plugin.h` (C ABI), `TrustVerifier`, `TrustUpdateHandler`, `PluginLoader` |
| `NNSpire/plugin-api/trust/` | `root_ca.pem` (embedded seed), TrustStore management, TUP schema |
| `docs/adr/` | Architecture Decision Records — ADR-001 to ADR-019; index at `docs/adr/README.md` |
| `NNSpire/plugins/` | Reference plugins: BPE tokenizer, FAISS index, example custom activation |
| `NNSpire/pipeline/` | Input adapters, tokenization chain, context stage, execution, output adapters |
| `NNSpire/deployment/` | Export wizards, runner clients (Triton/TFS/KServe/ORT), bundle builder |
| `NNSpire/python-bridge/` | pybind11 module; `runners/` sub-package mirrors deployment connectors |
| `NNSpire/app/` | Qt 6 QML application — `main.cpp`, controllers, QML components |
| `NNSpire/tests/` | GoogleTest unit + integration tests |
| `docs/` | All documentation: Doxygen config, ADRs, design docs, blueprints, history, standards KB |
| `docs/ai-standards-kb/` | Reference KB — 18 MD docs covering NN math, ONNX, serving, RAG, tokenization |

---

## Feature flag policy

All Studio features are registered in `NNSpire/core/features/FeatureFlags.h`.
Each flag carries a tier: `FREE | PRO | ENTERPRISE`.

**Standing commitment**: the following categories are permanently declared `FREE` in the codebase:
- All learning/wizard/walkthrough features
- All visualization features (weight heatmaps, neuron viewer, loss charts)
- Full pipeline (all input/output modalities)
- All standard runner connectors
- KB help system

The commercial plugin economy is the primary intended revenue stream.
A `PRO` tier is a last resort and should not be introduced lightly.
Any flag change from `FREE` to `PRO` must be a deliberate, documented architectural decision recorded here.

---

## Plugin trust levels at runtime

| Level | Condition | UI behaviour |
|---|---|---|
| Trusted | Valid cert chain to Root CA, not revoked | Silent load |
| Community | Valid chain, registered free/open | One-time notice |
| Unverified | No cert / self-signed / unknown | Modal warning every session + persistent yellow banner |
| Revoked | On CRL/OCSP | Hard block, no force-load |

---

## Dual-language implementation checklist

Before marking any feature complete, verify both exist and are behaviourally equivalent:
- [ ] C++ class/function in `NNSpire/core/` or `NNSpire/plugin-api/` or `NNSpire/backends/`
- [ ] Python binding in `NNSpire/python-bridge/`
- [ ] C++ plugin scaffold template in `NNSpire/plugin-api/templates/cpp/`
- [ ] Python plugin scaffold template in `NNSpire/plugin-api/templates/python/`
- [ ] Code export (sandbox "generate code" action) produces both Python and C++ tabs
- [ ] All whitepaper .md docs show dual code blocks for non-trivial examples

---

## AI Standards KB — usage rules

- Path: `docs/ai-standards-kb/` (moved into docs/ folder)
- Treat as **read-only reference**. Do not restructure or rename files.  
- All in-app help, wizard content, and context-help references point here by relative path.
- When implementing a mathematical function (activation, loss, optimizer step), always cross-reference the relevant KB section in the source code `// @kb:` comment, e.g.:
  ```cpp
  // @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#activation-functions
  ```
- Gaps identified in the KB (to be filled as new .md docs added to the KB):
  - Graph editor / node-graph UI patterns
  - Custom operator authoring guide
  - Training loop state machine
  - Hyperparameter search (Optuna, Ray Tune)
  - Quantization/pruning workflows (TensorRT/OpenVINO depth)
  - Real-time tensor visualization patterns
  - Quantum computing for ML (Qiskit, Pennylane)

---

## Dependency policy

### Bundled (legally cleared)
| Dep | License | How |
|---|---|---|
| Qt 6 runtime | LGPL v3 | `windeployqt` / `macdeployqt` / `linuxdeployqt` |
| Embeddable Python | PSF | Downloaded to `runtime/python/` on first run |
| MinGW-w64 (Windows) | GPL + runtime exception | Optional, `runtime/mingw64/` |
| pybind11 | MIT | Headers only, compile-time |
| ONNX protobuf | Apache 2.0 | Statically linked |
| Eigen | MPL 2.0 | Headers only |
| GoogleTest | BSD 3-Clause | Test builds only |
| OpenSSL | Apache 2.0 (3.x) | For TrustVerifier; dynamically linked |

### NOT bundled — facilitated via Dependency Manager UI
- CUDA toolkit + drivers (nvidia-proprietary)
- Full Python SDK (user development)
- Triton / TF Serving / KServe (user-deployed servers)
- System BLAS/LAPACK (Linux)

### Install size targets
| Component | Goal |
|---|---|
| Core app binary + Qt | < 130 MB |
| Bundled embeddable Python | ~15 MB |
| Total base install (no CUDA) | < 200 MB |
| CUDA backend plugin | ~10 MB + user-installed CUDA |

---

## Process architecture

- **One main process** — Qt app. Training on worker threads (`QThreadPool`).
- **Dynamic plugin loading** — `QPluginLoader` / `dlopen`. Unused backends/bridges do not load.
- **`NNSpire-runner` sidecar** — optional, crash-isolated inference process for untrusted plugins. Off by default.
- **No background services** — nothing survives app exit; no auto-start; no system tray by default.
- **External runner connection** — Studio connects to already-running Triton/TFS/KServe over gRPC/REST; Studio does not own those processes.

---

## Portability targets

| Platform | Distribution | Notes |
|---|---|---|
| Windows 10+ | Xcopy-deployable folder + optional installer | Portable mode: settings in `<app_folder>/settings/` |
| macOS 11+ | `.app` bundle | `@rpath` for all dylibs |
| Linux (glibc ≥ 2.31) | AppImage | Distro-agnostic |
| Older Linux / macOS | Qt 5.15 LTS build config | Separate CMake config |

---

## Key external references

- ONNX spec: `ai-standards-kb/standards/02-ONNX.md`
- NN fundamentals (math): `ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md`
- Tokenization deep dive: `ai-standards-kb/annexes/A06-Tokenization-Deep-Dive.md`
- RAG patterns: `ai-standards-kb/annexes/A03-RAG-Patterns.md`
- gRPC serving: `ai-standards-kb/standards/08-gRPC-ML-Serving.md`
- Triton: `ai-standards-kb/standards/10-Triton-Inference-Server.md`
- KServe: `ai-standards-kb/standards/09-KServe.md`
- TF Serving: `ai-standards-kb/standards/05-TensorFlow-Serving.md`
- LLM APIs: `ai-standards-kb/annexes/A05-LLM-API-Standards.md`
- Vector DBs: `ai-standards-kb/annexes/A02-Vector-Databases.md`
