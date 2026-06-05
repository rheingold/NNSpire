# NNSpire — Master Project Checklist

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

> **Product pillars** — three components, one repo, clearly separated:
>
> | Pillar | Folder | Status (June 2026) |
> |---|---|---|
> | **NNSpire Engine** | `nnspire/` (excl. `app/`) | ✅ Phases 1–2 complete — 170/170 tests green |
> | **NNSpire Studio** | `nnspire/app/` | ⏳ Phase 3 starting — ADR-030–033 decisions gate all UI work |
> | **NNSpire Agent** | `nnagent/` | ⏳ Phase 0 starting — parallel with Studio |
>
> Studio and Agent develop **in parallel**. See the _Parallel Development Tracks_ section below.

---

## Phase 0 — Project Scaffolding

### Root layout and documentation
- [x] Establish `Studio/` as project root (singular workspace folder)
- [x] Create `.gitignore` (build artefacts, `ai_priv/`, Python caches, Qt artefacts)
- [x] Create `README.md` — master signpost to all documents
- [x] Create `ai.md` — AI assistant conventions, decisions, key paths
- [x] Create `ai_priv/ai_priv.md` — local private config template (gitignored)
- [x] Create `TODO.md` — this file
- [x] Create `ARCHITECTURE.md` — system component whitepaper
- [x] Create `PIPELINE.md` — full chain design
- [x] Create `PLUGIN-SDK.md` — plugin architecture specification
- [x] Create `DEPLOYMENT.md` — runner/deployment strategy
- [x] Create `TRUST-ARCHITECTURE.md` — PKI trust chain design
- [x] Create `LICENSING.md` — license split and plugin charter
- [x] Verify no non-.md files exist at project root (except .gitignore) — `.gitattributes` is standard git infrastructure

### Source tree skeleton (empty dirs + placeholder CMakeLists)
- [x] Create `NNSpire/` top-level `CMakeLists.txt`
- [x] Create `NNSpire/core/` directory structure
- [x] Create `NNSpire/backends/` directory structure
- [x] Create `NNSpire/plugin-api/` directory structure
- [x] Create `NNSpire/plugins/` directory structure
- [x] Create `NNSpire/pipeline/` directory structure
- [x] Create `NNSpire/deployment/` directory structure
- [x] Create `NNSpire/python-bridge/` directory structure
- [x] Create `NNSpire/app/` directory structure
- [x] Create `NNSpire/tests/` directory structure
- [x] Create `docs/` directory with Doxygen `Doxyfile` stub

### Third-party / dependency setup
- [x] Copy/fetch Eigen headers — resolved via CMake FetchContent (3.4.0); no manual copy needed
- [x] Copy/fetch GoogleTest — resolved via CMake FetchContent (1.14.0); no manual copy needed
- [x] Copy/fetch pybind11 into `NNSpire/third-party/pybind11/` — resolved via pip site-packages (3.0.3) → FetchContent fallback; no manual copy needed
- [ ] Copy/fetch ONNX protobuf into `NNSpire/third-party/onnx/` — deferred to Format I/O phase
- [x] Confirm OpenSSL 3.x available — OpenSSL 3.6.0 at `C:\Program Files\OpenSSL-Win64`; MSYS2 also has 3.1.4
- [x] Verify Qt 6.5+ installation and record path in `ai_priv/ai_priv.md` — Qt 6.10.1 recorded

---

## Phase 1 — NN Engine Core (C++ library `NNSpire-core`)

### Tensor subsystem (`NNSpire/core/tensor/`)
- [x] `Tensor<T>` class: shape, strides, dtype enum (float32/float16/int8/int32)
- [x] Device tag: `CPU | CUDA | QUANTUM`
- [x] Basic ops: element-wise add/mul, reshape, slice, transpose, broadcast
- [x] `matmul()` dispatching to backend
- [x] Serialization: save/load to raw binary + metadata header (`NNS1` magic, binary little-endian)
- [x] Python binding via pybind11
- [x] Unit tests: shape arithmetic, op correctness vs NumPy reference values

### Layer subsystem (`NNSpire/core/layers/`)
- [x] Abstract `Layer` base: `forward()`, `backward()`, `parameters()`, `serialize()`, `docRef()`
- [x] `Dense` (fully connected)
- [x] `Conv2D`
- [x] `BatchNorm` → `BatchNorm1d` in `NormLayers.h`
- [x] `Dropout` → `NormLayers.h`
- [x] `Embedding`
- [x] `MultiHeadAttention`
- [x] `LayerNorm` → `NormLayers.h`
- [x] Python bindings for all layers
- [ ] C++ export template for each layer type
- [x] Unit tests: forward pass vs reference, backward pass gradient check

### Activations (`NNSpire/core/activations/`)

> ⚠️ **HIGH PRIORITY — complete before Phase 2 Plugin SDK ABI freeze**  
> ADR-020 mandates that `IActivation` (not `ActivationBase`) is the published plugin extension point.  
> The current `ActivationBase : ILayer` design is correct for internal use but **must not** be shipped as the SDK contract — it is stateful between `forward()`/`backward()` and therefore not reentrant (see Invariant I-1, I-2 in `blueprints.md`).  
> **Work required:**
> - [x] Add `core/include/NNSpire/core/IActivation.h` with `ActivationForward` struct and `IActivation` interface (Option C: functional pair — see ADR-020)
> - [x] Add `ActivationsFnLayer.h` adapter wrapping `IActivation` into a full `ILayer` (`ActivationsFnLayer<Fn>` owning + `ActivationsFnLayerPtr` non-owning)
> - [x] Migrate `ReLU`, `LeakyReLU`, `Sigmoid`, `TanhAct`, `Softmax`, `GELU` to implement `IActivation` (remove `lastInput_`/`lastOutput_`; return as `ctx` in `ActivationForward`)
  - [x] Update `blueprints.md §3.8` to reflect the new interface (IActivation / ActivationForward / ActivationsFnLayer)
> - [x] Add unit tests: `ActivationsFnLayer_ReLU_Forward`, `_Backward`, `_NoParams`

- [x] `ReLU` + derivative
- [x] `LeakyReLU` + derivative
- [x] `Sigmoid` + derivative
- [x] `Tanh` + derivative
- [x] `Softmax` + derivative (Jacobian)
- [x] `GELU` + derivative
- [x] `@kb:` comments linking to `ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md`
- [x] Python bindings
- [x] Unit tests

### Loss functions (`NNSpire/core/losses/`)
- [x] `MSE` + gradient
- [x] `CrossEntropy` + gradient
- [x] `BinaryCrossEntropy` + gradient
- [x] `HuberLoss` + gradient
- [x] Python bindings
- [x] Unit tests (11 tests: MSE×3, BCE×3, CrossEntropy×2, Huber×4)

### Optimizers (`NNSpire/core/optimizers/`)
- [x] `SGD` (with momentum)
- [x] `Adam`
- [x] `AdamW`
- [x] `RMSProp`
- [x] LR scheduler base + `StepLR`, `CosineAnnealingLR`
- [x] Python bindings
- [x] Unit tests: parameter update step correctness (14 tests: SGD×5, Adam×3, AdamW×2, RMSProp×3, StepDecay×1)

### Compute graph (`NNSpire/core/graph/`)
- [x] `ComputeGraph` DAG: node registration during forward pass
- [x] Autograd: `backward()` traversal, gradient accumulation
- [x] Graph serialization (JSON)
- [x] Graph visualization data export (for UI consumption)
- [x] `EvalTrace` struct: opt-in per-layer capture of inputs, outputs, and gradients
- [x] `ILayer::forward()` optional `EvalTrace*` parameter (`nullptr` = zero-cost no-op in normal training)
- [x] `ILayer::backward()` optional `EvalTrace*` parameter — captures `grad_output` and computed `grad_input`
- [x] `Trainer::setTraceMode(bool)` — delegates to `ComputeGraph::setTraceMode()`; traces accessible via `ComputeGraph::traces()`
- [x] Python bindings

### Training loop (`NNSpire/core/training/`)
- [x] `Trainer` class: graph + optimizer + loss + dataset
- [x] Callback interface: `onEpochStart/End`, `onBatchStart/End`, `onMetric`
- [x] Checkpoint save/load to `.nns` format
  - [x] Save model weights (W, b) into embedded ONNX blob
  - [x] Save **optimizer state** into `.nns` extension: Adam `m`, `v`, step counter `t` per parameter
        (without this, Adam loses its accumulated momentum on resume and must rebuild from scratch)
  - [x] Save epoch + batch-within-epoch counters for exact resume position
  - [x] Raw gradients are intentionally NOT saved — they are zeroed at the start of every step
        and recomputed by one forward+backward pass; losing them costs at most one step
- [x] Early stopping callback → `EarlyStoppingCallback` in `EarlyStopping.h`
- [x] Python bindings (`NNSpire.keras.Model.fit/predict/evaluate`, pure-Python training loop)

### Format I/O (`NNSpire/core/formats/`)
- [ ] `OnnxIO::import()` — ONNX protobuf → ComputeGraph
- [ ] `OnnxIO::export()` — ComputeGraph → ONNX protobuf
- [ ] `NNSFormat` — `.nns` project file (JSON/MessagePack): weights + metadata + plugin refs + UI hints
- [ ] Python bindings

### Feature flags (`NNSpire/core/features/`)
- [x] `FeatureFlags.h` — `inline constexpr FeatureFlag` pattern; all flags declared `FREE`; `isEnabled()` returns true for all FREE flags

### Backend abstraction (`NNSpire/backends/`)
- [x] `IBackend` interface: `matmul()`, `elementWise()`, `memAlloc()`, `memFree()`, `sync()`
- [x] `CpuBackend` — Eigen-based matmul reference implementation (reference + didactical; retained permanently)
- [ ] `CudaBackend` — our own cuBLAS/cuDNN implementation; didactical, makes every CUDA op visible; conditional compile (`NN_ENABLE_CUDA=ON`); disabled if CUDA toolkit not found
- [ ] `LibTorchBackend` — **additive production backend** alongside `CudaBackend`; delegates all ops to LibTorch optimised CUDA/cuDNN/TensorRT kernels; opt-in via `NN_ENABLE_LIBTORCH=ON`; gives full cuDNN perf without reimplementing kernels
- [x] `QuantumBackend` — stub; interface compiles, all methods call `__builtin_trap()`; registered in BackendRegistry
- [x] `BackendRegistry` — runtime registration and selection by device tag
- [x] Dynamic loading: each backend is a separate shared library loaded on demand

### Deferred: Training orchestration delegation (ADR pending)

> ⚠️ **When touching `Trainer` for any reason: take a split-second check that the change does not close the door on delegating to Lightning/Accelerate/DeepSpeed when `LibTorchBackend` is active. Specifically: keep `trainStep()`, `trainEpoch()`, `train()` as thin virtual-dispatch-friendly calls; avoid embedding loop logic that would be hard to replace with a delegate.**

- [ ] Evaluate whether `Trainer` should become a facade over **PyTorch Lightning** / **HuggingFace Accelerate** when `LibTorchBackend` is in use:
  - Our `TrainCallbacks` translate to Lightning `Callback` hooks
  - Our `Dataset` / `DataBatch` translate to Lightning `DataModule`
  - Benefit: multi-GPU, mixed-precision, gradient accumulation for free
  - Cost: Lightning as a dependency in the LibTorch-enabled build
- [ ] Evaluate **DeepSpeed** for ZeRO optimizer / model parallelism (LLM-scale training only)
- [ ] Evaluate **Ray Train** for multi-node orchestration (post-v1.0)
- [ ] Decision: does Studio's own `Trainer` always run, or can it be replaced by a backend-specific orchestrator?
  - Recommendation: `Trainer` stays as the *interface* (callbacks, metrics, stop signal) even when an
    external orchestrator runs underneath — the Studio remains the controller, not a passenger.

### Phase 1 milestone verification
- [x] `cmake --build && ctest` passes all unit tests — **63/63 green** (15 new LayerTest + 48 prior)
- [x] Can construct 3-layer MLP in C++, run forward pass on XOR dataset, verify output shape ← `test_trainer_xor.cpp`
- [x] Same via Python bindings: `import NNSpire; ...` works in embedded Python

---

## ⚡ Cross-Framework Compatibility Layer (PyTorch / Keras drop-in)

> **Priority: UTMOST — must be designed before Phase 2 ABI freeze and before any pybind11 bindings are written.**
>
> ### Goal
> A user who writes code using only the *standard* PyTorch or Keras API signatures and names
> must be able to swap the include/import between NNSpire and the real framework with no code changes:
>
> ```cpp
> // C++ — swap this include to switch underlying framework
> #include <NNSpire/torch_compat.h>   // uses NNSpire engine
> // #include <torch/torch.h>          // uses LibTorch
>
> auto model = torch::nn::Sequential(torch::nn::Linear(4, 8), torch::nn::ReLU());
> ```
>
> ```python
> # Python — swap this import to switch underlying framework
> import NNSpire.torch_compat as torch   # uses NNSpire engine
> # import torch                           # uses real PyTorch
>
> layer = torch.nn.Linear(4, 8)
> x = layer(torch.zeros(2, 4))
> ```
>
> Any use of an **NNSpire extension** (beyond the torch/keras standard surface) triggers
> a UI/CLI warning: *"This project uses NNSpire-only features: [list]. Export to PyTorch
> will require adapters."*
>
> ### Framework evaluation
> | Framework | C++ drop-in | Python drop-in | Decision |
> |---|---|---|---|
> | PyTorch / LibTorch | ✅ via `torch_compat.h` alias shim | ✅ pybind11 naming by design | **Implement** |
> | Keras | N/A (Python only) | ✅ `model.compile/fit/predict` aliases | **Implement** |
> | JAX | N/A | ❌ pure-functional, incompatible paradigm | **Skip** |
> | TensorFlow C++ | ❌ Google deprecated it | ❌ not worth the cost | **Skip** |
> | MLX (Apple) | N/A | 🟡 growing — revisit post-v1.0 | **Defer** |

### Prerequisite: Tensor dtype-generic buffer ~~`[! BLOCKS ALL BELOW]`~~ ✓ **DONE**

- [x] Change `Tensor` internal buffer from `shared_ptr<float[]>` to `shared_ptr<void>` + `itemsize_` derived from `DType`
  - `DType` enum already exists; `dtypeBytes()` already exists
  - `rawData()` type-erased accessor added; `itemsize()` public getter added
  - All existing tests continue to pass (Float32 is still the default; 98/98 green)
  - `CpuBackend` unchanged — uses `data()` which still returns `float*` via `static_cast`
  - This was a self-contained change: only `Tensor.cpp` + `Tensor.h` changed; public API surface unchanged
- [x] Add `Tensor::item<T>()` accessor (type-safe scalar extraction) replacing raw `data()[i]`
- [x] Unit tests: `Itemsize_Float32`, `Itemsize_Int8`, `Itemsize_Int32`, `RawData_NonNull`, `Dtype_RoundTrip` (5 new tests)
- [x] Unit tests: dtype mismatch returns `Result::error()` in CpuBackend dispatch

### C++ PyTorch-compatible shim (`include/NNSpire/torch_compat.h`)

- [x] `namespace torch` → alias block mapping to `NNSpire::*`:
  - `torch::Tensor` → `NNSpire::core::Tensor`
  - `torch::zeros / torch::ones / torch::rand` → our factory functions
  - `torch::nn::Module` → `NNSpire::core::Layer`
  - `torch::nn::Linear` → `NNSpire::builtin::layers::Dense`
  - `torch::nn::Conv2d` → `NNSpire::builtin::layers::Conv2D`
  - `torch::nn::Embedding` → `NNSpire::builtin::layers::Embedding`
  - `torch::nn::MultiheadAttention` → `NNSpire::builtin::layers::MultiHeadAttention`
  - `torch::nn::BatchNorm1d / LayerNorm / Dropout` → our NormLayers
  - `torch::nn::ReLU / Sigmoid / Tanh / GELU / Softmax` → our activations (wrapped in `ActivationsFnLayer`)
  - `torch::nn::MSELoss / CrossEntropyLoss / BCELoss` → our losses
  - `torch::optim::SGD / Adam / AdamW / RMSProp` → our optimizers
  - `torch::nn::functional::relu / sigmoid / softmax / ...` → our standalone activation callables
- [x] `torch::nn::Sequential` — thin wrapper building `ComputeGraph` from initializer list
- [x] The shim is **header-only** — zero link-time cost; only aliases, no new compiled symbols
- [x] Unit test: compile a simple 3-layer MLP using only `torch::` names against NNSpire

### Python pybind11 bindings — torch-compatible naming (design constraint, not retrofit)

> ⚠️ These bindings do not exist yet — but they **must be designed with torch naming** from day one.
> Do not expose `NNSpire.core.Dense`; expose `NNSpire.nn.Linear` with `Dense` as an alias.

- [x] Top-level module: `import NNSpire` → available sub-namespaces: `NNSpire.nn`, `NNSpire.optim`, `NNSpire.nn.functional`
- [x] `NNSpire.Tensor` matches `torch.Tensor` public surface: `.shape`, `.dtype`, `.device`, `.item()`, `.numpy()` (CPU copy), `__add__/__mul__/...`
- [x] `NNSpire.nn.Linear(in, out)` — default torch constructor signature
- [x] `NNSpire.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)` — torch signature
- [x] `NNSpire.nn.Embedding(num_embeddings, embedding_dim)` — torch signature
- [x] `NNSpire.nn.MultiheadAttention(embed_dim, num_heads)` — torch signature
- [x] `NNSpire.nn.Sequential(*layers)` — torch constructor
- [x] `NNSpire.nn.ReLU / Sigmoid / Tanh / GELU / Softmax / Dropout / BatchNorm1d / LayerNorm` — torch naming
- [x] `NNSpire.nn.MSELoss / CrossEntropyLoss / BCEWithLogitsLoss` — torch naming
- [x] `NNSpire.optim.SGD / Adam / AdamW / RMSProp` — torch constructor signatures (params, lr, weight_decay, ...)
- [x] `NNSpire.nn.functional.relu / sigmoid / softmax / gelu / dropout` — functional API
- [x] `NNSpire.torch_compat` re-export: `import NNSpire.torch_compat as torch` works as drop-in

### Python Keras-compatible aliases (additive, thin wrapper)

- [x] `NNSpire.keras.layers.Dense / Conv2D / Embedding / LSTM / MultiHeadAttention / BatchNormalization / Dropout / LayerNormalization`
- [x] `NNSpire.keras.losses.MeanSquaredError / CategoricalCrossentropy / BinaryCrossentropy`
- [x] `NNSpire.keras.optimizers.SGD / Adam / AdamW / RMSprop`
- [x] `NNSpire.keras.Model.compile(optimizer, loss, metrics)` — wraps training loop
- [x] `NNSpire.keras.Model.fit(x, y, epochs, batch_size, validation_data, callbacks)` — pure-Python training loop
- [x] `NNSpire.keras.Model.predict(x)` — single forward pass, no grad
- [x] `NNSpire.keras.callbacks.EarlyStopping / ModelCheckpoint` — pure-Python callback system

### Compatibility warning system

- [x] `CompatibilityChecker` — static analysis pass over `ComputeGraph` nodes:
  - Classifies each node as: `standard_torch | standard_keras | NNSpire_extension`
  - Reports list of non-standard ops used in the project
- [ ] CLI: `NNSpire-sign verify --compat=torch` — exits non-zero if any extension ops present
- [ ] UI: yellow warning banner in Layer Inspector when an NNSpire-extension layer is used
- [ ] `.nns` project file: `"compat_level": "torch_standard" | "keras_standard" | "NNSpire_extended"` field
- [ ] Export dialog: blocks ONNX-standard export if extension ops present; offers "export with custom ops sidecar" alternative

### ONNX export alignment

- [x] Map every torch-compatible layer to a standard ONNX op (no custom domain required):
  - `Linear` → `Gemm`, `Conv2d` → `Conv`, `Embedding` → `Gather`, `LayerNorm` → `LayerNormalization`,
    `BatchNorm1d` → `BatchNormalization`, `ReLU/Sigmoid/Tanh/GELU/Softmax` → standard ONNX ops
- [x] NNSpire extension layers: register under `com.NNSpire.*` custom op domain; export with sidecar `.onnx_ops.dll`
- [ ] Plugin ONNX adapter contract: each plugin optionally provides `onnx_export()` → custom op node + sidecar impl

---

## Architectural Debt — Namespace Migration (before Phase 2 ABI freeze)

> ⚠️ **ABI freeze is soft until v1.0 ships.**  
> All phases (1 through 5; Phase 6 — Quantum — is post-v1.0) are developed in-house before the first public release. This means any rename or ABI change can always be healed by finding all usages and updating them atomically in one commit. There is no external plugin ecosystem to break yet.  
> Treat "ABI freeze" as: *fix it now so it doesn't accumulate as debt*, not as: *this is irreversible*. Once v1.0 ships publicly, **that** is the real freeze point for the C plugin ABI. Until then, apply the "broken-window" rule: fix the namespace on sight, but do not block Phase 2 work if doing so perfectly would take more than a day.
>
> Do this as **one dedicated commit** before Plugin SDK work begins — or incrementally as each subsystem is touched, whichever keeps the build green.  
> After v1.0 ships, renaming **will** be a breaking change for external plugin authors.

### Source renames

- [x] `NNSpire::layers::Dense` → `NNSpire::builtin::layers::Dense`
- [x] `NNSpire::activations::*` (ReLU, Sigmoid, Tanh, Softmax, GELU, …) → `NNSpire::builtin::layers::*` *(activations are layers)*
- [x] `NNSpire::losses::*` (MSE, CrossEntropy, …) → `NNSpire::builtin::losses::*`
- [x] `NNSpire::optimizers::*` (SGD, Adam, AdamW) → `NNSpire::builtin::optimizers::*`
- [x] `CpuBackend` in `NNSpire::` → `NNSpire::builtin::backends::CpuBackend`
- [x] `Trainer` internals → `NNSpire::internal::training::*`

### Folder renames (to match namespaces)

- [ ] `NNSpire/core/include/NNSpire/core/` → split into `api/` (Tier 1 public) and `internal/` (Tier 2)
  <!-- DEFERRED: 60+ #include <core/...> references across src, tests, python-bridge. Scope > 1 day.
       Do as Phase 2 SDK prep step 0 (before NNSpire_plugin.h is written), so plugin authors
       see the correct include paths from day one. Namespace migration is already done. -->
- [x] `NNSpire/core/` layers/activations/losses/optimizers/backends → `NNSpire/builtin/`
- [x] Update all `#include` paths in source files and tests

### Verification

- [x] `cmake --build && ctest` — all 16 tests still pass after migration
- [x] No `NNSpire::layers::`, `NNSpire::activations::`, `NNSpire::losses::`, `NNSpire::optimizers::` appear outside of a `using` alias (grep check)
- [x] Only `NNSpire::` (Tier 1 interfaces + data types) and `NNSpire::builtin::` remain after rename

---

## Phase 2 — Plugin SDK

### C ABI contract (`NNSpire/plugin-api/`)
- [x] `NNSpire_plugin.h` — C-linkage structs/function pointers; no C++ in public ABI
- [x] `PluginDescriptor` — name, version, type, factory functions, capabilities
- [x] Plugin types: `LAYER | TOKENIZER | OPTIMIZER | BACKEND | UI_PANEL | TRUST_UPDATE`
- [x] `LayerPlugin` interface — custom `forward()`/`backward()`, operator registration
- [x] `TokenizerPlugin` interface — `encode()`, `decode()`, vocab management
- [x] `UIPlugin` interface — QML component path, property declarations

### Plugin manifests
- [x] `plugin.manifest.json` schema — name, version, author, license, type, capabilities, binary hash, detached signature path
- [x] `TUP` (Trust Update Package) manifest schema — type: `TRUST_UPDATE`, timestamp, cert add/revoke lists
- [x] Schema validation library (C++ + Python)

### Trust system (`NNSpire/plugin-api/trust/`)
- [x] `TrustStore` class — manages `<app_data>/truststore/` (roots/, intermediates/, crls/, history/)
- [x] First-run seed from embedded `seed_roots/root_ca.pem`
- [x] Append-only `history/` audit log
- [x] Atomic TUP application (write temp → verify → rename)
- [x] `TrustVerifier` class — X.509 chain verification, CRL/OCSP check, offline CRL cache fallback
- [x] `TrustUpdateHandler` — validates and applies TUP; compiled into core, NOT a loadable plugin
- [x] `PluginLoader` — wraps `QPluginLoader` + Python importer; runs `TrustVerifier` before any code executes
- [x] Root CA public key embedded as `seed_roots/root_ca.pem` (published openly)
- [x] TUP validation rules: signature valid, timestamp not replayed, no self-referential removal, user confirmation modal required

### `NNSpire-sign` CLI tool
- [x] `keygen` subcommand — generate plugin key pair + CSR
- [x] `sign` subcommand — sign plugin binary + manifest using issued cert
- [x] `verify` subcommand — verify plugin signature offline
- [ ] `submit` subcommand — submit CSR to registry for community/commercial cert
- [x] `create-tup` subcommand — build and sign a Trust Update Package
- [x] `issue-enterprise-ca` subcommand — project owner issues Enterprise Intermediate CA cert (admin only)

### pybind11 bridge (`NNSpire/python-bridge/`)
- [x] Python module `NNSpire` exposing: `Tensor`, all `Layer` subclasses, `ComputeGraph`, `Trainer`, `BackendRegistry`
- [x] `runners/` sub-package: Python-side runner clients mirroring `NNSpire/deployment/`
- [x] `pyproject.toml` for installable Python package
- [ ] Wheel build in CI

### Plugin scaffolds (`NNSpire/plugin-api/templates/`)
- [x] `cpp/` — CMakeLists.txt + `.h`/`.cpp` template (layer type; other types follow same pattern)
- [x] `python/` — `pyproject.toml` + `.py` module template (layer type; other types follow same pattern)
- [x] Both templates include `plugin.manifest.json` generator script (`generate_manifest.py`)
- [x] `README.md` per template explaining the plugin type

### Built-in reference plugins (`NNSpire/plugins/`)
- [x] BPE tokenizer plugin (C++ + Python, from KB A06) — C++ ✓ (319-token vocab, full encode/decode C ABI); Python binding deferred to Phase 3
- [ ] FAISS vector index plugin (C++ + Python) — deferred to Phase 3.5
- [x] Example custom activation plugin (demonstrates both C++ and Python plugin path) — C++ ✓ (Swish/SiLU forward+backward); Python binding deferred to Phase 3

### Phase 2 milestone
- [x] `PluginLoader` successfully loads BPE tokenizer plugin at runtime — validated via 17 whitebox C ABI tests (170/170 green)
- [ ] Unverified plugin triggers modal warning every session — requires Phase 3 UI
- [ ] TUP can be applied and is logged as immutable audit entry — trust disabled (`NN_ENABLE_TRUST=OFF`); integration test deferred

---

## 🏗️ Parallel Development Tracks — Studio + Agent

> **Where we are (June 2026):** Engine (Phases 1–2) is complete and fully tested.
> Studio and Agent now develop **in parallel** — they share integration seams that gate
> each other at the points listed below. Neither blocks the other for its own internal work.

### Cross-pillar integration seams (delivery order)

| Seam | NNSpire Studio deliverable | NNSpire Agent deliverable | Unlocks |
|---|---|---|---|
| **S1 — NMID + Runner** | Runner sidecar serving `.nnsp` inference (Studio Phase 5) | NMID reader + Runner API client (Agent Phase 9) | Agent headlessly tests any Studio model |
| **S2 — Agent Panel in Studio** | Embedded Agent Panel QML host (Studio Phase 3.5) | `nnagent-core` packaged as embeddable library | User chats with LLM while editing a model |
| **S3 — MCP boundary node** | Studio exposes MCP server (ADR-044, Studio Phase 5.7) | Agent connects as MCP client (Agent Phase 9–10) | Agent triggers training / export / inspection |
| **S4 — Orchestration driver** | Studio pipeline canvas hosts Agent orchestration node | Agent L6 orchestration engine (Agent Phase 10) | Full bidirectional Studio ↔ Agent control |

### What to build right now (parallel)

**Studio track — start with Phase 3:**
- Resolve ADR-030 (canonical representation), ADR-031 (metadata storage), ADR-032 (unrecognized code), ADR-033 (two-way WYSIWYG)
- First working window: project open/close, layer list panel, basic canvas render
- Record ADR decisions in `ARCHITECTURE.md` and unblock the editor implementation

**Agent track — start with Phase 0–1:**
- `CMakeLists.txt` scaffold for `nnagent-core` (mirrors `nnspire/` preset conventions)
- NMID parser (`nmid_reader.h/cpp`) — needed for Seam S1
- LLM API client stub (OpenAI-compatible streaming via chunked HTTP)
- `nnagent-core` GTest suite (mirrors `nnspire/tests/` patterns)
- CLI shell stub (Phase 2): stdin/stdout JSON line-protocol, static binary target

**Shared / cross-cutting:**
- Finalise `.nnsp` project format (gates Studio file-open _and_ Agent NMID load)
- Define Runner sidecar protocol — document as `docs/RUNNER-PROTOCOL.md` (gates Seam S1)
- Draft ADR-044: MCP typed boundary node spec (gates Seams S2–S4)

---

## Phase 3 — NNSpire Studio (Qt 6 QML)

> ⚠️ **DESIGN PRINCIPLE — resolve this before writing a single line of UI code**
>
> ### UI Reentrant Idempotency
>
> The NNSpire UI state machine must be designed around a dominant **`ReadyToEdit`**
> state. This principle governs every panel, dialog, wizard, and control added in
> Phase 3 and beyond.
>
> **What this means:**
>
> - **Maximise time in `ReadyToEdit`** — the user should nearly always be directly
>   editing the neural network (adding/removing/reordering layers, tuning
>   hyperparameters, inspecting weights) without having had to complete prior wizard
>   steps to get there.
>
> - **Minimise step-dependent contingency.** The following patterns are UX debt and
>   require explicit justification before use:
>   - `Step1 → Step2 → … → Stepx → ReadyToEdit` (linear wizard-gate before editing)
>   - `ReadyToEdit → SubStep1 → SubStep2 → SubStep3 → … → ReadyToEdit`
>     (multi-step sub-wizard inside the main flow)
>
> - **Target state diagram for every feature:**
>   `ReadyToEdit → SingleCommand / SingleEdit → ReadyToEdit`
>   After any operation — combobox selection, dialog confirm, export, compile, training
>   run — the UI must return to `ReadyToEdit` as quickly as possible (dialog closes,
>   progress bar finishes, selection confirms).
>
> - **"Idempotency" is used in the UI sense**, not the mathematical sense.  The system
>   as a whole is *not* idempotent — the NN data is permanently altered by training,
>   export, weight import, etc.  But the *UI* must recover to its primary editing state
>   immediately after each action, with no follow-up navigation required of the user.
>
> - **Practical gate before implementing any UI flow:** *Could a motivated expert user
>   accomplish this in a single gesture (click / dropdown / drag / keyboard shortcut)
>   once they know the tool?*  If not, redesign it.

> ---
>
> ### Design Decisions — Editor Paradigm (must be resolved before Phase 3 implementation)
>
> The model editor should be primarily a **Visual Code Editor & Runner/Debugger** —
> not a pure drag-and-drop canvas and not a pure text editor, but a **two-view
> environment** where the code and the visual graph are always in sync — analogous
> to how FrontPage Express let you switch between HTML source and WYSIWYG layout
> with real-time bidirectional reflection of every change.
>
> The following four architectural questions must be decided (and recorded as ADRs)
> before implementation begins.  Recommendations are given; use `[! decision needed]`
> items to track the votes.
>
> ---
>
> #### ADR-030 — Canonical Model Representation  `[! decision needed]`
>
> **Options:**
>
> **(a) Abstract IR only** — the `.nnsx` project contains an abstract
> XML/JSON/MessagePack model description.  All code (engine, torch, keras, ONNX)
> is generated from it at compile/export time.  The user never edits code directly
> in the primary workflow.
>
> **(b) Native NNSpire / torch-compat source code** — the source file
> (`model.py` or `model.cpp`) *is* the canonical representation.  The visual canvas
> is a live rendering of the parsed AST.  Export to other frameworks is a well-defined
> code-to-code translation.
>
> **(c) Multi-framework native editing** — the user can choose which framework language
> is "live" (NNSpire engine, PyTorch, Keras …) and the editor hides/shows features
> that are not expressible in the currently selected framework.  Switching framework
> re-parses the same file under a different grammar.
>
> **Recommendation: (b) — native code as source of truth.**
>
> Rationale: NNSpire already has a `torch_compat.h` C++ shim and a
> `NNSpire.torch_compat` Python module whose public API is *syntactically identical*
> to PyTorch.  This means "NNSpire engine code" and "torch-compatible code" are the
> same text.  There is no IR translation for the primary path — code IS the model.
> Export to real PyTorch is then a mechanical projection, not a lossy round-trip.
> Option (c) is additive on top of (b): grammar switching can be layered in later
> without changing the core data model.  Option (a) creates a private IR language
> that users must learn and that must track all framework evolution forever.
>
> **Decision gate items:**
> - [ ] Confirm ADR-030 choice (a / b / c) and record in `ARCHITECTURE.md §ADR-030`
> - [ ] If (b) or (c): choose Python or C++ as the *default* code surface for new
>   projects (recommendation: Python — lower barrier, identical in both cases thanks
>   to `torch_compat`)
> - [ ] If (a): design the IR schema and its versioning strategy before any other work
>
> ---
>
> #### ADR-031 — UI Metadata Storage  `[! decision needed]`
>
> Applies when the canonical representation is code (ADR-030 b or c).  The UI needs
> to store information that does not belong in the model definition: canvas node
> positions, colour groups, user annotations, folding state, wizard bookmarks.
>
> **Option I — Sidecar file** — a companion `.nnsx.view.json` (or a `view/` entry
> inside the `.nnsx` zip bundle) keyed on stable node identifiers derived from the
> AST (layer variable names / positional indices, not line numbers).
>
> **Option II — Structured comments** — `# @NNSpire {"x":120,"y":80}` comments
> embedded directly above each layer construction call in the source file.
>
> **Recommendation: Option I — sidecar inside the `.nnsx` bundle.**
>
> Rationale: source code stays clean and runnable outside NNSpire without any
> stripping step.  The sidecar is completely invisible to Python/C++ tooling
> (formatters, linters, type checkers).  The key stability problem ("positions go
> stale when the user rearranges code") is solved by keying entries on the AST
> node's fully-qualified name (e.g. `layer_0_Linear`, `layer_1_ReLU`) rather than
> on line numbers — the same convention tree-sitter uses for named captures.
> Structured comments (Option II) are fragile: formatters remove them, diffs are
> noisy, and a parse error in the comment annotation can corrupt the whole file.
>
> **Decision gate items:**
> - [ ] Confirm ADR-031 choice (I / II) and record in `ARCHITECTURE.md §ADR-031`
> - [ ] If (I): define the `.nnsx` bundle format — zip with `model.py`, `view.json`,
>   `manifest.json`, `weights/` — and the view-JSON schema version field
> - [ ] Define the node-identity key algorithm (variable name > positional index >
>   hash of construction-call arguments, in that priority order)
>
> ---
>
> #### ADR-032 — Unparsed / Unrecognized Code Visualization  `[! decision needed]`
>
> When the model editor's grammar recognizes layer/optimizer/loss construction calls,
> any source code that does NOT match a known pattern must be handled gracefully —
> no AI, algorithm-only.
>
> **Proposed approach (concrete grammar + AST walk):**
>
> 1. Use **tree-sitter** with the existing Python (or C++) grammar plus a set of
>    named query patterns covering all recognized NNSpire/torch constructs.
> 2. Walk the concrete syntax tree; tag every matched node as `recognized`.
> 3. Any top-level statement or expression that is NOT tagged `recognized` is
>    classified as `opaque` and rendered in the code editor with an **amber
>    left-border gutter marker** + a **subtle amber background tint**.
> 4. `opaque` nodes are preserved byte-for-byte; the code generator never touches
>    them.  They participate in execution normally — only the visual canvas cannot
>    represent them as layer nodes.
> 5. A "Show opaque" toggle in the canvas toolbar reveals `opaque` blocks as
>    grey "custom code" placeholder nodes, showing the first line of source text.
> 6. Hovering an amber node in the code editor shows a tooltip:
>    *"Not recognized as a known layer/optimizer — will run as-is but cannot be
>    visualised. See PLUGIN-SDK.md to wrap it as a plugin."*
>
> **Decision gate items:**
> - [ ] Confirm tree-sitter as the parser foundation (alternative: ANTLR4 grammar)
>   and record in `ARCHITECTURE.md §ADR-032`
> - [ ] Define the query-pattern file format and where it lives in the repo
>   (`NNSpire/app/grammar/NNSpire_torch.scm` is the suggested path)
> - [ ] Decide whether `opaque` blocks appear as placeholder nodes on the canvas
>   (recommendation: yes, opt-out toggle) or are hidden entirely (opt-in toggle)
>
> ---
>
> #### ADR-033 — Two-Way WYSIWYG Editor  `[! decision needed]`
>
> The central UX bet: the code pane and the visual canvas are **two views of the
> same in-memory model graph**, both writable, both always in sync.
>
> **Architecture:**
>
> ```
> Code editor keystroke ──► tree-sitter incremental re-parse
>                                    │
>                           [model graph diff]
>                                    │
>                     ┌─────────────▼──────────────┐
>                     │   In-memory Model Graph     │
>                     │ (single source of truth)    │
>                     └───────┬───────────┬─────────┘
>                             │           │
>              canvas redraw ◄┘           └► code re-emit (minimal diff)
>         (Qt Quick SceneGraph)            (preserves unrecognized nodes
>                                          verbatim via CST edit API)
> ```
>
> - **Code → graph**: tree-sitter's edit API applies incremental re-parses (sub-
>   millisecond for typical model files).  The grammar extracts layer nodes and
>   edges; the model graph is updated structurally, not by full re-parse.
> - **Graph → code**: mutations from canvas interactions (add layer, change property,
>   reorder) emit minimal code insertions/deletions using tree-sitter's node-edit
>   API, preserving all surrounding code and comments unchanged.  This is the same
>   mechanism language servers use for code actions.
> - **Round-trip fidelity**: `code → parse → emit` must produce output ≤ one
>   whitespace-normalization pass away from the input.  This is a hard requirement;
>   validated by a round-trip fuzz test suite (Phase 3.5).
> - **Conflict resolution**: if both panes are modified within a single event tick
>   (impossible in single-threaded Qt, but guarded anyway), the code editor wins.
>
> **Recommendation: implement.** The FrontPage Express analogy is exact and the
> technical path is clear.  tree-sitter already ships C++ and Python grammars;
> the Qt text editor component (QPlainTextEdit or a Monaco WebEngine embed) can
> apply AST-driven decorations via `ExtraSelection`; the canvas is a Qt Quick
> `Canvas` or `ShaderEffect` scene.  The main risk is round-trip fidelity for
> complex files — mitigated by the fuzz test suite and by making the canvas
> generate idiomatic, predictable code patterns.
>
> **Decision gate items:**
> - [ ] Confirm ADR-033 (implement two-way WYSIWYG vs. one-way canvas-generates-code)
>   and record in `ARCHITECTURE.md §ADR-033`
> - [ ] Choose text editor component: `QPlainTextEdit` + custom highlighter (lighter),
>   or Monaco via `QWebEngineView` (full LSP support, heavier); recommendation:
>   `QPlainTextEdit` for Phase 3, Monaco as an opt-in for Phase 3.5+
> - [ ] Confirm tree-sitter bindings strategy: compile tree-sitter as a C library
>   and call from C++ (recommended — no runtime dependency, statically linkable),
>   or use the WASM build inside Monaco (only applicable if Monaco is chosen)
> - [ ] Define the fuzz-test harness for round-trip fidelity (runs at every CI build)
>
> ---
>
> #### ADR-034 — `ParametricLayer<Fn>`: Generic Parametric Layer + KAN-Style Non-Linear Synapses  `[! decision needed — Phase 4]`
>
> **Background**
>
> Standard MLP layers (e.g. `Dense`) compute an affine map:
>
> ```
> y = x @ W^T + b      (linear synapse)
> ```
>
> The linear synapse is a *computational convenience* — BLAS GEMM is fast — not a
> mathematical requirement. The Universal Approximation Theorem (Cybenko 1989,
> Hornik et al. 1991) places no constraint on the synapse transform; only the
> activation function is required to be non-constant and bounded. This motivates
> two interlinked questions:
>
> 1. **Architecture**: Should `Dense` be refactored into `ParametricLayer<Fn>`, a
>    generic base where `Fn` describes both the forward computation and which tensors
>    it requires as learnable parameters?
> 2. **Research**: If we lift the linear synapse assumption, what alternatives are
>    known and useful?
>
> **Proposed architecture**
>
> ```cpp
> template<typename Fn>
> class ParametricLayer : public ILayer {
>     std::vector<Parameter> params_;   // W, b, spline control points, …
>     Fn fn_;
> public:
>     Result<Shape>  build   (const Shape& s) override { return fn_.build(s, params_); }
>     Result<Tensor> forward (const Tensor& x, ...) override { return fn_.forward(x, params_); }
>     Result<Tensor> backward(const Tensor& g, ...) override {
>         auto [dX, dP] = fn_.backward(g, params_, lastInput_);
>         for (int i = 0; i < (int)params_.size(); ++i)
>             params_[i].tensor.accumulateGrad(dP[i]);
>         return dX;
>     }
>     std::vector<Parameter*> parameters() override { /* expose params_ */ }
> };
>
> // Dense becomes a type alias — ZERO caller breakage, identical public interface
> using Dense = ParametricLayer<LinearFn>;
>
> // KAN layer (Liu et al. 2024) — per-synapse learnable B-spline
> using KANLayer = ParametricLayer<SplineFn>;
>
> // SIREN layer (Sitzmann et al. 2020) — sinusoidal synapses
> using SIRENLayer = ParametricLayer<SinusoidalFn>;
> ```
>
> `torch_compat.h` needs zero changes. All existing tests continue to pass.
>
> **Known non-linear synapse functions from the literature**
>
> | Name | Synapse fn | Params / synapse | Reference | Status |
> |---|---|---|---|---|
> | Linear (current) | w*x + b | 2 | — | ✅ `Dense` |
> | B-spline (KAN) | spline(x; control_points) | k+1 (order k) | Liu et al., *KAN: Kolmogorov-Arnold Networks*, arXiv:2404.19756, MIT/CalTech (Apr 2024) | 📖 Research |
> | Sinusoidal (SIREN) | sin(w * omega * x + phi) | 3 | Sitzmann et al., *Implicit Neural Representations with Periodic Activation Functions*, NeurIPS 2020 | 📖 Research |
> | Log-bilinear | exp(w * log(x)) — multiplicative in linear space | 1 | Mikolov et al., word2vec (2013); neural matrix factorisation | 📖 Established |
> | Quadratic | w1*x^2 + w2*x + b | 3 | Various (1990s–2010s) | 📖 Established |
>
> The theoretical foundation for KANs is the **Kolmogorov-Arnold Representation
> Theorem** (Kolmogorov 1957, Arnold 1957): any multivariate continuous function
> f:[0,1]^n -> R decomposes as a finite composition of univariate continuous
> functions. The outer sum corresponds to node activations (as in a standard MLP);
> the inner functions phi_{q,p}(x_p) are per-synapse — the KAN novelty.
>
> ---
>
> **Advocatus Diaboli I — Software Engineer: "The costs are very high. Design elegance is not runtime feasibility."**
>
> 1. **GEMM is sacred.** `cuBLAS`, `Eigen`, `MKL`, `Metal Performance Shaders` all
>    reduce `y = xW^T` to a single hardware-optimal kernel call. A B-spline per
>    synapse on a 4096×4096 layer = 16,777,216 non-linear evaluations, none fuseable
>    into a cuBLAS call without custom CUDA kernels. Expected throughput loss:
>    **100×–1000×** vs. cuBLAS SGEMM.
>
> 2. **`IBackend` vtable does not support per-element functors.** Adding
>    `applyFunctor1D` to `IBackend` is a breaking API change (plugin minor-version
>    bump) before any `KANLayer` can accelerate on CUDA or non-CPU backends.
>
> 3. **Checkpoint format must be extended.** `NNSC` currently encodes `WS`/`OS`/`TC`/`EN`
>    sections. `ParametricLayer<SplineFn>` needs a new `FN` section: functor type tag +
>    hyperparameters (spline order, knot vector, frequency scale). Checkpoints without
>    this section cannot be resumed.
>
> 4. **Autograd complexity.** The gradient of a B-spline w.r.t. its control points is
>    itself a spline evaluation. The backward pass returns dX plus k+1 gradient tensors
>    per synapse; memory use grows with spline order.
>
> 5. **Debugging surface explosion.** "My loss isn't converging" now has many new
>    causes: knot placement, spline order, per-synapse vs. global learning rates,
>    frequency initialisation. Standard MLPs are already hard to debug; per-synapse
>    parametric functions multiply the degrees of freedom.
>
> **Mitigation**: `ParametricLayer<LinearFn>` (pure internal refactor, identical
> behaviour to current `Dense`) has **none** of these problems. All issues above apply
> only to non-linear `Fn` variants. Build in order: (0) refactor, (1) CPU-only KAN,
> (2) accelerated backend extension.
>
> ---
>
> **Advocatus Diaboli II — Mathematical Theory of NNs: "The theory is deliberately silent here — and that silence is the interesting part."**
>
> 1. **The UAT does not constrain synapse form.** Cybenko (1989) and Hornik et al.
>    (1991) require only a non-constant, bounded, monotone *activation* on neurons.
>    The synapse transform appears nowhere in the theorem's statement or proof.
>    Linearity was chosen because it maps to available hardware, not because it is
>    theoretically optimal.
>
> 2. **The Kolmogorov-Arnold Representation Theorem** (Kolmogorov 1957, Arnold 1957)
>    guarantees the same expressiveness class as the UAT, but via a path where the
>    non-linearity lives on the *edges* (synapses) rather than the *nodes* (neurons).
>    Both families span the same function class; they trade parameters for compute
>    differently depending on the problem geometry.
>
> 3. **SIREN** (Sitzmann et al., NeurIPS 2020) demonstrates that sinusoidal synapses
>    sin(omega * W*x + phi) are provably superior for targets with periodic structure
>    (audio, images, physics PDE solutions): every derivative of a sinusoid is again
>    a sinusoid, so SIREN networks can match arbitrarily many derivatives of the
>    target function — a capability ReLU networks cannot replicate without drastically
>    increasing depth.
>
> 4. **Log-bilinear / multiplicative synapses** are well-established, not exotic:
>    word2vec skip-gram, neural matrix factorisation, and physics-informed NNs for
>    multiplicative PDEs all use them. They emerged naturally from task structure.
>
> 5. **What is mathematically forbidden?** Nothing categorically. Constraints are:
>    - Gradient or sub-gradient must exist (non-differentiability on a measure-zero
>      set is fine — ReLU being the canonical example)
>    - The Jacobian chain must not collapse pathologically beyond known issues
>      (residual connections and normalisation mitigate this for any reasonable Fn)
>    - Training dynamics must be stable (engineering constraint, not a theorem)
>    None of these exclude B-splines, sinusoids, or log-linear transforms.
>
> 6. **The deep open question (2024–2026 research frontier):** Is there a synapse
>    function that is simultaneously (a) GPU-fuseable, (b) stably trainable at scale,
>    and (c) provably more expressive per-FLOP than the linear synapse?
>    KANs (Liu et al. 2024) claim a parameter-efficiency advantage for *scientific*
>    tasks (symbolic regression, physics PDE fitting). Follow-up 2024–2025 benchmarks
>    on ImageNet and GPT-class language models have not reproduced a consistent
>    advantage for standard ML tasks. **The question is genuinely open.**
>
> ---
>
> **Recommendation: three-step opt-in rollout**
>
> | Step | What | Phase | Risk |
> |---|---|---|---|
> | 0 | Refactor `Dense` → `ParametricLayer<LinearFn>` · `using Dense = ParametricLayer<LinearFn>;` | Phase 4 | Zero — identical behaviour, no API change |
> | 1 | Add CPU-only `KANLayer` + `SIRENLayer`; mark `NNSpire_extension` in `CompatibilityChecker`; expose `torch::nn::KANLinear` alias in `torch_compat.h` | Phase 4 | Low — isolated, off by default |
> | 2 | Add optional `IBackend::applyFunctor1D` (backend minor-version bump); GPU backends may fuse custom kernels | Phase 5 | Medium — vtable change, all backend plugins must update |
>
> This preserves the **didactic aim**: `ParametricLayer<LinearFn>` reads as a textbook
> definition of a parametric affine layer. KAN/SIREN variants then demonstrate
> exactly what changes when you swap the synapse function — one template parameter,
> everything else identical. NNSpire becomes both technically state-of-the-art *and*
> maximally transparent about what the difference actually is.
>
> **Decision gate items:**
> - [ ] Confirm Phase 4 step 0: refactor `Dense` → `ParametricLayer<LinearFn>` (pure,
>   no behaviour change) — record decision in `ARCHITECTURE.md §ADR-034`
> - [ ] Confirm Phase 4 step 1: add experimental `KANLayer` (CPU-only, `NNSpire_extension`
>   scope; `torch::nn::KANLinear` alias in `torch_compat.h`)
> - [ ] Confirm backend vtable extension strategy for GPU-accelerated functor dispatch
>   (step 2; requires IBackend minor-version bump)
> - [ ] Design `NNSC` `FN` section format for non-linear synapse hyperparameter
>   persistence (spline order, knot vector, frequency initialisation)
> - [ ] Track research: re-evaluate KAN FLOP-efficiency claim on standard ML tasks
>   at Phase 4 start; adjust step 1 priority accordingly

### App shell (`NNSpire/app/`)
- [ ] `main.cpp` — Qt app init, backend detection, plugin loader, dependency check, QML engine setup
- [ ] `controllers/` — `ModelController`, `TrainingController`, `BackendController`, `PluginController`, `HelpController`, **`CodeGraphSyncController`** (owns the parser ↔ model-graph ↔ canvas sync loop; see ADR-033)
- [ ] Dockable panel system — every panel (model editor, training dashboard, weight viewer,
  neuron viewer, tokenizer, KB help, output/debugger, ecosystem architect) is independently:
  - **Movable** — drag by title bar to any position
  - **Anchorable** — snap to left / right / top / bottom edge or centre tabbed area
  - **Detachable** — float as an independent OS window (multi-monitor support)
  - **Stackable** — drop onto another panel to create a tab group
  - **Collapsible** — minimise to icon strip; panel remembers its last size on expand
- [ ] **Workspace layout presets** — switchable from View menu and toolbar dropdown:
  - `Code-first` — code pane wide, canvas narrow (default for experienced users)
  - `Canvas-first` — visual canvas dominant, code pane collapsible side panel
  - `Training` — training dashboard + weight viewer prominent, editor minimised
  - `Minimal` — code pane + output/debugger only (terminal-programmer style)
  - `Ecosystem` — AI Ecosystem Architect canvas full-width (Phase 5.5)
- [ ] **Named layout save/load** — user saves current arrangement as a named profile;
  all layouts listed in View menu; last-used layout restored on next launch
- [ ] **Per-project layout override** — optional; stored in `view.json` sidecar;
  takes precedence over global default only for that project
- [ ] **Toolbox panel** — floating or dockable palette of layer/component types;
  drag from toolbox to canvas or code cursor to insert; grouped by category;
  user can pin/hide individual categories; toolbox position persisted per layout
- [ ] Persistent settings: `<app_folder>/settings/` (portable mode) or OS config dir fallback
- [ ] Application menu (File/Edit/View/Tools/Help)
- [ ] `.nnsx` project bundle format — zip containing: `model.py` (canonical source), `view.json` (ADR-031 sidecar), `manifest.json`, `weights/` directory

### First-run Dependency Manager
- [ ] Detect: Python runtime, C++ compiler, CUDA, system BLAS on startup
- [ ] Non-blocking notice for optional missing deps + one-click download button
- [ ] Blocking error for required missing deps with download link
- [ ] Accessible post-startup via `Tools → Dependency Manager`
- [ ] Embeddable Python auto-install to `<install>/runtime/python/`
- [ ] MinGW-w64 auto-install to `<install>/runtime/mingw64/` (Windows)

### Model editor panel (`ui/ModelEditor.qml`)

> The model editor is a **split-pane Visual Code Editor & Runner/Debugger** — not a
> pure canvas, not a pure text editor.  Left pane: code editor.  Right pane: visual
> canvas graph.  Both are always live.  Editing either pane updates the other in real
> time (see ADR-033).  The user can collapse either pane to work full-screen in either
> mode.  The split is a draggable `SplitView`; ratio is persisted in settings.

#### Parser subsystem (`NNSpire/app/parser/`)
- [ ] Integrate **tree-sitter** as a static C library (`third-party-deps/tree-sitter/`)
- [ ] Bundle `tree-sitter-python` grammar (primary) and `tree-sitter-cpp` grammar (secondary)
- [ ] Implement `NNSpire_torch.scm` query file — named captures for all recognized
  layer/optimizer/loss construction patterns (see ADR-032)
- [ ] `NNParser` C++ class — wraps tree-sitter; exposes `parse(source)`,
  `applyEdit(edit)`, `recognizedNodes()`, `opaqueNodes()`
- [ ] `ModelGraphBuilder` — walks recognized nodes → produces `ModelGraph` (in-memory
  DAG of `ModelNode` structs: type, params, connections, AST range)
- [ ] Round-trip fuzz test: `source → parse → emit → parse` must produce identical
  `ModelGraph` (not identical text — opaque nodes and comments are preserved verbatim)

#### In-memory Model Graph
- [ ] `ModelGraph` — DAG of `ModelNode`, `ModelEdge`; owns the single source of truth
  while the editor is open
- [ ] `ModelNode` — `{id, type, params, astRange, canvasPos, canvasGroup, userNote}`;
  `id` is the AST variable name if present, otherwise `layer_<index>`
- [ ] Change notifications via Qt signals: `nodeAdded`, `nodeRemoved`, `nodeChanged`,
  `edgeAdded`, `edgeRemoved` — both panes subscribe and update independently
- [ ] `ModelGraphDiff` — structural diff between two `ModelGraph` snapshots; used by
  the code emitter to produce minimal textual edits

#### Code editor pane (`ui/CodeEditorPane.qml` / `CodeEditorWidget`)
- [ ] `QPlainTextEdit`-based editor with custom `QSyntaxHighlighter` for Python/C++
  (Phase 3); option to swap in Monaco via `QWebEngineView` (Phase 3.5, see ADR-033)
- [ ] Layer-aware syntax highlighting on top of the base grammar — recognized nodes
  get a distinct foreground colour per node type (layer=blue, optimizer=green,
  loss=orange); **opaque nodes get an amber left-gutter bar + amber background tint**
- [ ] Hover tooltip on amber (opaque) regions: *"Not recognized as a known construct —
  will run as-is but cannot be visualised. See PLUGIN-SDK.md."*
- [ ] Gutter icons: ▶ run-to-here, 🔍 inspect activations at this layer, ⚠ warning
- [ ] Code edits feed `NNParser::applyEdit()` → `ModelGraphBuilder` → `ModelGraph`
  signals → canvas redraws (target latency: < 50 ms for files < 5 000 lines)
- [ ] **Opaque-node toggle**: toolbar button shows/hides opaque blocks on the canvas
  (they are always present and unmodified in code regardless of toggle state)

#### Visual canvas pane (`ui/CanvasPane.qml`)
- [ ] Qt Quick `Canvas`/`ShaderEffect` scene; nodes are QML `Item` delegates
- [ ] Auto-layout via Sugiyama/layered-graph algorithm (C++, no external lib needed)
- [ ] Node cards: show type name + key params (e.g. `Linear(128→64)`), coloured by type
- [ ] Opaque nodes rendered as grey placeholder cards labelled **"Custom code"** with
  first line of source text; eye icon toggles their visibility
- [ ] Drag to reorder nodes → `ModelGraph.nodeReordered` → code emitter moves
  corresponding source lines (all surrounding code preserved verbatim)
- [ ] Click node → Properties Panel updates; double-click → code editor scrolls to
  and highlights that node's AST range
- [ ] Add node from palette → code emitter inserts construction call after last
  recognized node (or at code-editor cursor if the code pane has focus)
- [ ] Delete node → code emitter removes corresponding lines; confirms if any opaque
  code depends on the deleted variable
- [ ] Connection lines auto-drawn from `ModelEdge` list; sequential models: straight
  vertical flow; branching models: curved bezier edges
- [ ] Mini-map for large graphs (> 20 nodes)

#### Properties panel (shared, context-sensitive)
- [ ] Updates when a node is selected in either pane
- [ ] All `ModelNode.params` shown as type-appropriate form fields (spinbox / checkbox
  / combobox / text); edits flow back through `ModelGraph` → code emitter → code editor
- [ ] "?" button per parameter opens KB deep-link for that parameter
- [ ] **Warning banner**: "every two consecutive Dense layers must have a non-linear
  activation between them" — educational, not a build block
- [ ] **Backend acceleration status badge** — shown per layer _and_ per activation in the
  Properties panel:
  - Green badge **"Fully accelerated"** when all forward/backward operations are routed
    exclusively through `IBackend` vtable calls (e.g. `Dense`, `Sigmoid`).
  - Amber badge **"Partial — CPU loops"** when any path falls back to raw `flat(i)` scalar
    loops (e.g. `TanhAct`, `GELU`, `Softmax`, `LeakyReLU`); shows a tooltip listing the
    offending paths; links to `§D.8` of `blueprints.md` for remediation guidance.
  - Grey badge **"Unknown / plugin"** for plugin-supplied layers or activations that do not
    implement `isFullyVtableDispatched()`.
  - Acceleration status re-evaluated when the backend selector in the Training Dashboard
    changes (e.g. switching from `CPU` to `CUDA`), since a method's effective speedup only
    exists if the active backend implements it.
  - _Prerequisite_: `IActivation` and `ILayer` expose `backendAccelerationProfile()` →
    returns a small struct: `{fullyVtable: bool, rawLoopOps: vector<string>}`.
    See `blueprints.md §D.9` for the plugin API design.
- [ ] **User note** field — free text stored in `view.json` sidecar (ADR-031); never
  emitted into source code

#### Architecture preset gallery
- [ ] 🥪 sandwich toolbar icon opens the gallery
- [ ] Presets: MLP (classifier, regressor), Autoencoder, CNN (image classifier),
  Transformer encoder block, GPT decoder block, ResNet skip-connection block
- [ ] Loading a preset inserts the template source at the cursor; canvas reflects immediately
- [ ] See `blueprints.md § Annex — Architecture Templates` for specs and typical sizes

#### Import / export
- [ ] Import ONNX model → populate `ModelGraph` → emit `model.py` source
- [ ] Export to: `.nnsx` bundle, ONNX, plain `model.py`, plain `model.cpp`
- [ ] **Run** button: executes `model.py` in the embedded Python runtime; stdout/stderr
  shown in the integrated Runner/Debugger panel
- [ ] **Debugger** panel (Phase 3 stub, Phase 3.5 full): set breakpoint at any recognized
  layer call; `EvalTrace` captures that layer's output; activations shown in Weight Viewer

### Training dashboard panel (`ui/TrainingDashboard.qml`)
- [ ] Live loss/metric chart (Qt Charts or custom Canvas)
- [ ] Epoch + batch progress indicators
- [ ] Start / Pause / Resume / Stop training controls
- [ ] Early stopping configuration
- [ ] Checkpoint save/load UI
- [ ] Backend selector — dropdown covering all registered backends:
  - `CPU` (always available)
  - `CUDA` (shown only when CudaBackend loads successfully)
  - `Quantum` (stub — grayed-out until Phase 6)
  - `Remote (gRPC)` — forwards tensor ops to a remote CUDA worker via `RemoteBackend`;
    shows a sub-form for endpoint URL + auth token (see **External Compute Services** annex)
- [ ] Remote training panel — shown when `RemoteTrainer` mode is active (not `RemoteBackend`):
  - [ ] Service selector: Vast.ai / RunPod / SageMaker / Vertex AI / custom endpoint
  - [ ] Pod/instance size selector (GPU type, VRAM, vcpu count) — populated from service API
  - [ ] Estimated cost calculator (price/hr × estimated epochs × batch time)
  - [ ] Pod lifecycle controls: Launch → Monitor → Stop / Terminate
  - [ ] Job status: queued / provisioning / running / done / failed, with elapsed time
  - [ ] Auto-import weights on job completion (poll → download → load into project)
  - [ ] API key entry (written to OS keychain; never stored in `.nnsp`)
  > **Cross-reference:** the backend-level and trainer-level integration points, service
  > table, pricing, and pilot sequence are documented in the **External Compute Services**
  > annex below. The UI here surfaces those integration points — it does not re-specify them.

### Weight/activation viewer (`ui/WeightViewer.qml`)
- [ ] Heatmap of weight matrices per layer (powered by `EvalTrace` weight snapshots)
- [ ] Activation distribution histograms (powered by `EvalTrace` output capture)
- [ ] Gradient magnitude per layer during training (powered by `EvalTrace` gradient capture)
- [ ] Language toggle: Python | C++ generated code for selected layer
- [ ] Flash-on-change animation: weight cells that changed most in last step pulse red/blue
- [ ] Vanishing/exploding gradient warning: gradient magnitude bar turns amber/red automatically

### Neuron viewer — learning/wizard only (`ui/NeuronViewer.qml`)
- [ ] Auto-layout visualization of neurons and weighted connections
- [ ] Maximum network size guard (do not render > N neurons)
- [ ] Animate forward pass propagation (driven by `EvalTrace` per-layer output tensors)
- [ ] Shown only in Wizard Mode or when user explicitly enables it for small networks
- [ ] Step-by-step mode: pause after each of the 6 `trainStep` phases; user clicks "Next" to advance
- [ ] Colour-code neuron nodes by activation value (blue=near-zero, red=saturated)
- [ ] Arrow thickness = absolute weight magnitude between connected neurons
- [ ] Arrow colour = gradient direction (positive=green, negative=red, near-zero=grey)

### Tokenizer panel (`ui/TokenizerPanel.qml`)
- [ ] Select tokenizer plugin
- [ ] Enter sample text → see token IDs + decoded tokens live
- [ ] Vocabulary browser
- [ ] Special token editor

### KB help system (`ui/HelpPanel.qml`)
- [ ] Render Markdown from `ai-standards-kb/` (QTextBrowser or WebEngineView)
- [ ] Deep-link anchor navigation (`kb://standards/01#backpropagation`)
- [ ] Triggered by `?` buttons throughout the UI
- [ ] Full-text search across KB

### Wizard / learning mode (`ui/WizardMode.qml`)
- [ ] Step-by-step guided walkthroughs
- [ ] Each step has `kbRef` property → auto-opens matching KB section
- [ ] Code snippets show **Python | C++ toggle**
- [ ] Neuron viewer enabled for small demo networks
- [ ] Walkthroughs: XOR MLP, MNIST CNN, Transformer block, BPE tokenizer

### Phase 3 milestone
- [ ] NNSpire launches on Windows/macOS/Linux
- [ ] Create 3-layer MLP → view topology → run XOR training → watch live loss chart
- [ ] KB help opens from layer "?" button showing correct KB section

---

## Phase 3.5 — IDE Integration & Developer Experience

> **Trigger:** complete once Phase 3 produces a launchable app shell — even if panels
> are stubs — so that new contributors can open the repository in their IDE of choice
> and immediately build, run, and debug.
>
> The canonical reference for hands-on configuration is
> **[`docs/VSCODE-DEV-SETUP.md`](docs/VSCODE-DEV-SETUP.md)** — see that document for
> step-by-step compiler/linker/debugger setup and the complete set of recommended
> VS Code run/debug profiles. IDE-specific files referenced below live in the repo
> and are kept up to date as the build system evolves.

### VS Code — primary IDE (emphasis here first)

**`.vscode/` workspace files**
- [ ] `extensions.json` — recommended extensions: `ms-vscode.cmake-tools`,
      `llvm-vs-code-extensions.vscode-clangd` (or `ms-vscode.cpptools`), `ms-vscode.cpptools-themes`,
      `webfreak.debug` (GDB/LLDB native), `myriad-dreamin.tinymist` (Doxygen), `twxs.cmake`,
      `josetr.cmake-language-support`; Qt: `qt-official.qt-cpp-extension-pack`
- [ ] `settings.json` — CMake: `cmake.configureOnOpen=true`, `cmake.buildDirectory=…/build/\${preset}`,
      `cmake.useCMakePresets="always"`, clangd path (MSYS2 GCC / LLVM), IntelliSense mode
- [ ] `c_cpp_properties.json` — one configuration per toolchain: `GCC-MinGW-MSYS2`,
      `Clang-LLVM-MSYS2`, `MSVC-x64`; correct `includePath` (Qt, Eigen, GTest, project includes),
      `compilerPath`, `cStandard`/`cppStandard`, `intelliSenseMode`
- [ ] `tasks.json` — already present for build/configure/test; add: `Configure (all presets)`,
      `Rebuild (clean + build)`, `Run app`, `Generate Doxygen docs`
- [ ] `launch.json` — run/debug profiles (see detailed list below)

**`launch.json` profiles (one `.vscode/launch.json` entry each)**
- [ ] `Engine tests (GDB/MinGW)` — launch `build/engine-ninja/tests/test-core.exe` under
      `gdb` (from MSYS2 MinGW); `cwd` = `NNSpire/`; env: `PATH` includes Qt bin + MSYS2 bin;
      `stopAtEntry=false`
- [ ] `Engine tests — filter (GDB/MinGW)` — same as above with `"args": ["--gtest_filter=${input:gtestFilter}"]`
      and a `${input}` prompt so the user types a filter expression without editing the file
- [ ] `NNSpire App (GDB/MinGW)` — launch the Qt app executable; pre-launch task = build; working dir = project root
- [ ] `NNSpire App (no GPU)` — same with `"env": {"NN_BACKEND": "cpu"}` override
- [ ] `NNSpire App (CUDA backend)` — same with `"env": {"NN_BACKEND": "cuda"}`
- [ ] `Engine tests (LLDB/Clang)` — alternate clang/LLDB profile using `CodeLLDB` extension
- [ ] `Engine tests (MSVC / cppvsdbg)` — Windows-only; requires `ms-vscode.cpptools` debugger;
      preset `engine-vs` (MSVC x64 generator)
- [ ] `NNSpire App (MSVC / cppvsdbg)` — app launch via MSVC debug adapter

**CMake presets**
- [ ] `engine-vs` preset — Visual Studio 17 generator, MSVC x64 (mirrors `engine-ninja` logic, different generator)
- [ ] `engine-clang-ninja` preset — Ninja + clang++ (MSYS2 LLVM package), same flags minus `-fno-exceptions` for
      libcxx compatibility check
- [ ] `app-ninja` preset — includes `NNSpire/app/` target; `CMAKE_PREFIX_PATH` pointing to Qt installation
- [ ] `app-vs` preset — same with Visual Studio generator

**Extensions and toolchain validation guide**
- [ ] `docs/VSCODE-DEV-SETUP.md` written and kept current — full guided tour:
  compiler selection, kit switching in CMake Tools, IntelliSense modes,
  debugging GDB vs LLDB vs cppvsdbg, `.env` file for Qt `PATH`, GTest adapter
  integration, Doxygen task, remote SSH target for GPU build/test (Phase 5+)

### Visual Studio (Windows, secondary)

- [ ] `engine-vs` CMake preset (above) enables "Open Folder" workflow in VS 2022 — no `.sln` needed
- [ ] `NNSpire.natvis` — NatVis visualiser file so VS debugger pretty-prints `Tensor`, `Shape`,
      `Result<T>`, `Parameter` in the Autos/Watch/Locals pane instead of raw byte dumps
- [ ] `NNSpire.props` — shared property sheet: Qt install path, MSYS2 path, include dirs;
      developers override via `NNSpire.user.props` (gitignored) for local paths
- [ ] Verify: right-click `CMakeLists.txt` → "Add to View" → CMake Targets appear; build + test
      via VS Test Explorer (GTest adapter)
- [ ] Document the one known friction point: MSYS2 DLLs not in VS `PATH` → workaround in
      `docs/VSCODE-DEV-SETUP.md` §Visual Studio

### Qt Creator / Qt Design Studio

- [ ] Confirm `CMakeLists.txt` opens cleanly in Qt Creator 13+ via "Open as CMake project"
- [ ] Kit configuration: MinGW 15.2.0 (MSYS2) + Qt 6.10.1; document kit JSON in `docs/VSCODE-DEV-SETUP.md`
- [ ] `.qmlproject` stub for `NNSpire/app/ui/` — allows pure QML/UI work in **Qt Design Studio**
      without building the C++ engine; mock `ModelController` QML singleton provides stub data
- [ ] Verify QML live-preview works in Qt Design Studio 4+ for all `.qml` files in `ui/`
- [ ] Document: Qt Creator run configurations for `test-core` and `NNSpire app`

### Phase 3.5 milestone
- [ ] A contributor on a fresh Windows machine can clone → open in VS Code → press F5 → debugger
      hits a breakpoint in `Dense::forward()` in under 15 minutes, following `docs/VSCODE-DEV-SETUP.md`
- [ ] Same contributor can run and filter GTest cases directly from the VS Code Testing sidebar
- [ ] `NNSpire.natvis` makes `Tensor` readable in the VS 2022 debugger Locals pane
- [ ] Qt Design Studio opens the `ui/` QML project and live-previews the model editor panel stub

---

## Phase 4 — Full Pipeline

### Input adapters (`NNSpire/pipeline/input/`)
- [ ] Text (UTF-8 string)
- [ ] Image (via Qt image loading → tensor)
- [ ] Audio (WAV/FLAC → waveform tensor via libsndfile or similar)
- [ ] Structured data (CSV → tensor, Apache Arrow pass-through)
- [ ] MCP message format adapter
- [ ] Python + C++ implementation for each

### Tokenization chain (`NNSpire/pipeline/tokenize/`)
- [ ] `TokenizationChain` — ordered list of `TokenizerPlugin`s
- [ ] Produces: token-ID tensor, attention mask, position IDs
- [ ] Cache layer for repeated vocabulary lookups
- [ ] Python + C++ implementation

### Context / RAG stage (`NNSpire/pipeline/context/`)
- [ ] `ContextStage` — optional; takes query tensor, returns retrieved chunk tensors
- [ ] Embedded FAISS connector (no server needed)
- [ ] Qdrant HTTP connector
- [ ] Chroma HTTP connector
- [ ] Reranker interface (cross-encoder plugin slot)
- [ ] Python + C++ implementation

### Execution stage (`NNSpire/pipeline/execution/`)
- [ ] `ExecutionStage` — runs one `ComputeGraph` or a chain of chained graphs
- [ ] Output of graph A → input of graph B wiring
- [ ] Streaming output support (token-by-token)
- [ ] Python + C++ implementation

### Output adapters (`NNSpire/pipeline/output/`)
- [ ] Text decode (de-tokenize)
- [ ] Image synthesis stub
- [ ] Audio synthesis stub
- [ ] Structured JSON serialization
- [ ] Python + C++ implementation

### Pipeline Studio panel (`ui/PipelinePanel.qml`)
- [ ] Coarse node graph: drag pipeline stages, wire them
- [ ] Each stage node opens a configuration panel
- [ ] Execute pipeline end-to-end from within the UI
- [ ] Display intermediate tensor shapes and sample values at each stage

### Phase 4 milestone
- [ ] Text input → BPE tokenizer → 2-layer transformer → decoded text output, visible in UI end-to-end

---

## Phase 5 — Deployment & Runners

### Export wizards (`NNSpire/deployment/export/`)
- [ ] ONNX export wizard (uses `OnnxIO::export()`)
- [ ] TorchScript export (via Python bridge + `torch.jit.script`)
- [ ] TFLite export stub
- [ ] `.nnsr` runner bundle builder: zip of weights + `manifest.json` + `runner.py` + `runner.cpp` + precompiled `.so`/`.dll`
- [ ] **C++ library export** — generates self-contained, NNSpire-free compilable package:
  `model.h` + `model.cpp` + `CMakeLists.txt` + `README.md`; output uses only stdlib +
  Eigen (header-only); no `NNSpire-core` link dependency in the exported build
- [ ] **Python package export** — generates pip-installable package: `pyproject.toml` +
  `model.py` (uses real PyTorch or raw NumPy, no `NNSpire` import); `model.py` is the
  round-trip of the canonical source with the `NNSpire.torch_compat` import replaced by
  `import torch`
- [ ] **"Compile to binary" one-click** — runs CMake + Ninja on the exported C++ package;
  output is a `.dll`/`.so` or standalone executable; build stdout/stderr shown in the
  integrated terminal panel; artifact path opened in OS file manager on success
- [ ] **Multi-language code preview** — all generated code (export wizard, architecture
  presets, sandbox snippets) shown with a `Python | C++` tab switcher; chosen language
  persisted per-user in settings; both tabs always kept in sync with the model graph

### Runner connectors (`NNSpire/deployment/runners/`) — C++ + Python each
- [ ] Triton gRPC client (HTTP/2 gRPC, model load/infer/health — KB doc 10 + 08)
- [ ] TF Serving REST client (KB doc 05)
- [ ] KServe V2 client (KB doc 09)
- [ ] ONNX Runtime embedded client
- [ ] Generic OpenAI-compatible REST client (KB annex A05)

### Deployment panel (`ui/DeploymentPanel.qml`)
- [ ] Export format selector + wizard launcher
- [ ] Runner endpoint configuration (URL, auth, TLS cert)
- [ ] Model push to runner
- [ ] Health / latency / throughput monitor (Prometheus scrape)
- [ ] Optional: launch local Triton Docker container helper
- [ ] **Remote pod manager** (for Vast.ai / RunPod / LambdaLabs Level-0 services):
  - [ ] List active pods (status, GPU type, price/hr, elapsed time, SSH endpoint)
  - [ ] Launch new pod from template (image, GPU type, disk size, open ports)
  - [ ] Terminate / interrupt pod
  - [ ] Port-forward helper: opens a local gRPC tunnel to the pod for `RemoteBackend`
  - [ ] SSH terminal shortcut: opens VS Code Remote-SSH or system terminal to the pod
- [ ] **Managed job submitter** (for SageMaker / Vertex AI / Azure ML Level-2 services):
  - [ ] Choose project variant + dataset → submit training job
  - [ ] Job list: status, cost so far, ETA, logs tail
  - [ ] Download + import weights from completed job

### `NNSpire-cli` command-line tool (`NNSpire/app/cli/`)

A headless CLI companion to the Qt app. Same C++ engine, no Qt dependency in
the CLI binary itself — links only `NNSpire-core` + `NNSpire-builtin`.
Useful for CI pipelines, scripting, and SSH sessions on remote pods.

- [ ] `NNSpire-cli train <project.nnsp> [--preset <name>] [--epochs N] [--remote <service>]`
  — run training locally or submit to a remote service; streams loss to stdout
- [ ] `NNSpire-cli export <project.nnsp> --format <onnx|nns|nnsr> -o <out>`
  — export model; no UI required
- [ ] `NNSpire-cli run <model.nnsx> --input <file|stdin> --output <file|stdout>`
  — single inference pass; reads NMID for I/O format
- [ ] `NNSpire-cli remote list` — list active pods on configured services
- [ ] `NNSpire-cli remote launch --service vast.ai --gpu RTX4090`
- [ ] `NNSpire-cli remote stop <pod-id>`
- [ ] Common flags: `--backend cpu|cuda|remote`, `--config <settings.json>`,
      `--api-key <env-var-name>` (never accept key as positional arg — shell history)
- [ ] Machine-readable output: `--json` flag on all commands for scripting
- [ ] Built as a separate CMake target `NNSpire-cli`; included in `app-ninja` and
      `app-vs` presets; single static binary goal on Linux/macOS (musl / libc++)

### Plugin Registry server spec (`NNSpire/deployment/registry/`)
- [ ] REST API spec (OpenAPI 3.1): `POST /plugin/register`, `GET /plugin/{id}`, `POST /enterprise/ca/request`, `GET /crl`
- [ ] Reference server implementation (Python FastAPI)
- [ ] `NNSpire-sign submit` integration

### `.nnsp` project file spec (finalise)
- [ ] Adopt ZIP container format (ECMA OPC principle — like .docx/.xlsx)
- [ ] `project.json`: project metadata, NNSpire version, author, signature/encryption header
- [ ] `networks/<variant>/blueprint.json`: layer stack, sizes, activations, loss fn, optimizer config, hyperparameters
- [ ] `networks/<variant>/manifest.json`: parts references (embedded or sidecar paths)
- [ ] `networks/<variant>/graph.onnx`: OPTIONAL embedded weights
- [ ] `networks/<variant>/training/`: OPTIONAL optimizer state + progress
- [ ] Support multiple network variants in one project (A/B comparison, scale testing)
- [ ] `pipeline/interconnect.json`: multi-network wiring graph (LLM pipeline, plugin slots, RAG stage)
- [ ] `plugins/refs.json`: required plugins with name, version, hash, signature
- [ ] `analysis/loss_curves.bin`: metric histories (OPTIONAL, omit when shipping without weights)
- [ ] `analysis/cluster_viz.json`: cluster analysis config and display settings (OPTIONAL)
- [ ] `analysis/weight_clusters.bin`: cluster computation results (large, OPTIONAL)
- [ ] `ui/layout.json`: panel arrangement, tabs, zoom levels
- [ ] `model_card.json`: KB doc 04 fields
- [ ] **"Homebrew shipping mode"**: project with blueprints+signatures but NO weights
      Studio opens and says "untrained — run training or import weights"
      Enables designer-to-customer distribution of NN architectures without large data
- [ ] Sidecar reference scheme: any part can be `embedded:file`, `file:rel/path`, `file:/abs/path`, or `container:../file`
      - `container:` = up-reference to root of the enclosing ZIP (used when .nnsp is embedded inside .nnsx)
      - When .nnsp is embedded in .nnsx: .nnsp carries only blueprints/UI/metadata; weights stay in outer .nnsx, never duplicated
- [ ] Schema validation for all JSON parts

### `.nnsx` model exchange file spec (finalise)
- [ ] ZIP container: manifest.json + graph.onnx + training/ + tokenizer/ + plugins/ + model_card.json
- [ ] `graph.onnx` is a valid standalone ONNX blob (extractable: `unzip model.nnsx graph.onnx`)
- [ ] All large parts (graph.onnx, adam.bin) support sidecar references via manifest
- [ ] `.nnsx` can be imported into a `.nnsp` project or exported from one
- [ ] `.nns` extension alias: Studio opens as `.nnsp`; runner treats as `.nnsx`

### Training checkpoint fields (inside `training/`)
- [ ] `training/adam.bin`: m[], v[], t per named parameter — **required for true resume**
- [ ] `training/progress.json`: epoch, batch_within_epoch, last_loss, optional rng_state
- [ ] Raw gradients intentionally NOT stored (recomputed in one step on resume)

### `.nnsr` runner bundle spec (finalise)
- [ ] Zip structure documented
- [ ] `manifest.json` schema: model metadata, tokenizer config, required plugins, hardware hints, input/output modality declarations, min Studio version
- [ ] `runner.py` + `runner.cpp` template generated by export wizard

### Phase 5 milestone
- [ ] Export XOR MLP trained in Phase 3 to `.onnx` → load in ONNX Runtime embedded → verify matching output
- [ ] Export same model as `.nnsr` bundle → verify bundle structure and manifest validity

---

## Phase 5.5 — AI Ecosystem Architecture Studio

> **Goal: enterprise-architecture didactic** — the user can visually design, configure,
> and learn a complete multi-component AI ecosystem (LLM runner, agents, vector stores,
> crawlers, pipelines, NN models from the model editor) at the architectural level;
> understand how the components interconnect, what each does, and how to configure it;
> and export runnable configuration artefacts from the design.
>
> This is NOT a deployment automation platform — it is a **learning and design tool**
> with export capability.  The visual language is ArchiMate-inspired (components,
> interfaces, connectors, typed ports) but domain-specific and simplified.
>
> NNSpire models built in the model editor appear as first-class components in the
> ecosystem canvas — the NN is one node among others in the full AI system picture.

### Component library (`NNSpire/ecosystem/components/`)

Each component type has: a C++ descriptor (name, category, ports, config schema),
a QML node card, a KB reference, and a configuration panel.

#### LLM Runners
- [ ] **Ollama** — local; config: model tag, host:port, context length, GPU layers, NUMA pinning
- [ ] **llama.cpp server** — local; config: GGUF path, threads, context, quantisation level
- [ ] **vLLM** — local / pod; config: model HF ID, tensor-parallel degree, max batch
- [ ] **LiteLLM proxy** — config: upstream provider list, routing policy, rate limits
- [ ] **HuggingFace TGI** — local/container; config: HF model ID, dtype, sharding
- [ ] **OpenAI-compatible endpoint** — generic; config: base URL, API key env-var, model name
- [ ] **NNSpire model (local runner)** — inline reference to any model built in Phase 3;
  runs via the embedded Python runtime; no separate server required

#### Vector Stores / Retrieval
- [ ] **Elasticsearch** — config: URL, index name, embedding field, k-NN metric
- [ ] **Qdrant** — config: URL / gRPC, collection name, vector size, distance metric
- [ ] **Chroma** — local (embedded) or server; config: persist dir or server URL, collection
- [ ] **Weaviate** — config: URL, class name, vectorizer module
- [ ] **Milvus / Zilliz** — config: URI, collection, index type
- [ ] **FAISS (embedded)** — no server; config: index file path, metric, nprobe

#### Agents / Orchestrators
> These are architectural components — Studio draws how they are wired; it does NOT
> execute Python orchestration code itself.  Export produces the orchestration script.
- [ ] **RAG chain** — query → retriever → reranker → LLM; configurable at each stage
- [ ] **ReAct agent** — tool-calling loop with configurable tool list
- [ ] **Multi-agent supervisor** — one orchestrator, N worker agents; round-robin or priority routing
- [ ] **LangChain chain blueprint** — maps to LangChain LCEL; exported as `chain.py`
- [ ] **CrewAI crew** — agents + tasks + process type (sequential / hierarchical)
- [ ] **Custom agent** (opaque node) — placeholder for bespoke orchestration code

#### Data Ingestion / Crawlers
- [ ] **Web crawler** — config: seed URLs, depth, rate limit, domain whitelist, output format
- [ ] **Document loader** — source type (PDF, DOCX, PPTX, HTML, Markdown, plain text);
  config: glob pattern or directory, chunking strategy, chunk size / overlap
- [ ] **Database extract** — SQL query → rows → embedding; config: DSN, query, text column
- [ ] **API poller** — periodic REST/GraphQL fetch; config: URL, interval, JSONPath selector, auth

#### Connectors / Interfaces
- [ ] **HTTP REST connector** — typed port; carries: request/response schema, base URL, auth method
- [ ] **gRPC connector** — typed port; carries: proto service name, endpoint, TLS config
- [ ] **Unix socket connector** — local IPC; carries: socket path, protocol (JSON-lines / length-prefix)
- [ ] **Embedding dimension guard** — validates that the vector size produced by a model
  matches the vector size expected by a vector store; raises a compatibility error on mismatch
- [ ] **Token format guard** — validates tokenizer output format across connected components

#### Embedding models (standalone, or embedded inside a runner)
- [ ] **Sentence-Transformers** — local; config: model name, batch size, device
- [ ] **Ollama embed** — via Ollama `/api/embeddings`; config: model tag
- [ ] **OpenAI embeddings** — config: API key env-var, model (`text-embedding-3-small` etc.)
- [ ] **Custom embedding** — NNSpire model from the model editor functioning as an encoder

### Ecosystem canvas (`ui/EcosystemCanvas.qml`)
- [ ] Separate canvas from the model editor — full panel occupying the `Ecosystem` layout preset
- [ ] Component palette (toolbox) docked to left; grouped by category (Runners / Stores /
  Agents / Ingestion / Connectors / NNSpire Models)
- [ ] Drag component from palette → node card placed on canvas; double-click → config panel
- [ ] Wire two ports by drag-connect; connector type inferred from port types; incompatible
  types highlighted red with tooltip explaining the mismatch
- [ ] **Interface compatibility enforcement** (algorithm-only, no AI):
  - Port type system: `{LLM_TEXT_IN, LLM_TEXT_OUT, EMBED_VEC[dim], TOKEN_IDS, HTTP_REST, GRPC, UNIXSOCK, DOCS}`
  - Each component declares its input/output port types; wiring validates at connection time
  - Dimension mismatches (embedding size) shown as amber warning, not blocked (user may know what they're doing)
  - Protocol mismatches (REST target → gRPC source) shown as red error, connector refused
- [ ] **Architecture pattern gallery** — pre-wired templates the user can load as starting points:
  - Naive RAG (crawler → chunk → embed → store → retriever → LLM)
  - Agentic RAG (retriever + tools + ReAct loop)
  - Multi-agent supervisor (planner + specialist agents + shared vector store)
  - Local-only stack (Ollama + FAISS + llama.cpp; zero cloud)
  - Hybrid local+cloud (Ollama for fast drafts, OpenAI for final answers)
  - NNSpire model as a component (Studio-trained NN acts as a classifier inside the pipeline)
- [ ] **Learn panel** — clicking any component opens a KB panel explaining: what it is, when
  to use it, typical configuration parameters, links to official docs, cost/resource estimates
- [ ] **Resource estimator** — per node: estimated VRAM (GPU), RAM (CPU), disk; totalled for
  the whole design; warning shown if total exceeds user-configured budget

### Configuration panel (per component)
- [ ] All config fields shown as type-appropriate form controls (same widget system as model-editor Properties panel)
- [ ] Sensitive fields (API keys, passwords) stored in OS keychain; form shows `****` populated
  from keychain; never written to any project file
- [ ] "Test connection" button per network component — sends a health-check request and shows
  latency / error in the panel footer
- [ ] VRAM/RAM usage estimated from config values (model size × quantisation factor × context length)

### Export artefacts (`ui/EcosystemExportDialog.qml`)
- [ ] **Docker Compose** — `docker-compose.yml` configuring all server components (Ollama,
  Qdrant, Elasticsearch, etc.) with the configured versions, ports, volumes, environment vars;
  NNSpire-model components mapped to a `NNSpire-runner` container
- [ ] **Kubernetes Helm values** — `values.yaml` for a generic AI-stack chart; one entry per
  component; secret refs replace inline key values
- [ ] **Ollama Modelfile + run script** — `Modelfile` + `run_ollama.sh`/`.ps1` for the
  configured model tag and parameters; one file per Ollama node in the design
- [ ] **`.env` template** — all required environment variables listed with placeholder values
  and comments; sensitive vars marked `# REQUIRED — set in CI/keychain, do not commit`
- [ ] **Python orchestration script** — `run_pipeline.py` matching the designed architecture;
  uses real library imports (LangChain, requests, qdrant-client, etc.);  no NNSpire import;
  comments explain each section; NNSpire model nodes emitted as ONNX Runtime inference calls
- [ ] **Architecture diagram (Mermaid)** — auto-generated `architecture.md` with a
  Mermaid flowchart of all components and connections; suitable for documentation

### Storage inside `.nnsp`
- [ ] Ecosystem design stored under `ecosystem/design.json` in the project bundle
- [ ] Port wiring, component configs (except secrets), and canvas positions all persisted
- [ ] Multiple ecosystem designs per project supported (e.g. "local dev", "production K8s");
  active design selected from a dropdown in the canvas toolbar

### Phase 5.5 milestone
- [ ] Design a local RAG pipeline (Ollama + FAISS + document loader) on the ecosystem canvas
- [ ] Export Docker Compose + `.env` template; verify containers start from the export
- [ ] Export Python orchestration script; verify it runs end-to-end with a PDF input

---

## Phase 5.7 — Semantic Composition & Macro-Architecture Studio

> **Goal: a ComfyUI-for-every-architecture.** Phase 3 gives the user a *micro* editor —
> the graph of primitive layers (`Dense`, `ReLU`, `GELU`, `Conv2D`, `MultiHeadAttention`).
> Phase 5.5 gives the user an *ecosystem* editor — the graph of whole running services
> (Ollama, Qdrant, agents). **Phase 5.7 fills the missing middle**: the *semantic*
> composition layer, where the user designs and learns a model by its **meaning** —
> "this is a text-embedding model", "this is an inference-assistant LLM", "this is a
> latent diffuser" — and drills down through named **layer groups** ("token recognition",
> "positional awareness", "syntax learning", "denoiser block") into the primitives.
>
> ### The unifying insight (see `docs/modern_ai_systems_ontology.md` §0, ch. 1–2)
>
> An LLM is **not** a fundamentally different kind of object from a diffuser. Both are a
> **typed semantic message-passing graph over tensors** (ontology Level 5). A
> text-to-image diffuser is `text → encoder → conditioning → [denoiser ⟲ scheduler] →
> latent → VAE → image`. An LLM is `text → tokenizer → embedding → [transformer block ×N]
> → LM head → sampler → text`. The autoregressive decode loop and the diffusion denoise
> loop are *the same pattern* — a learned transform inside a controller loop — with
> different typed messages and a different preset of blocks. **So the same node-graph
> editor can express both.** An "LLM" is one **macro-architecture preset**; a "diffuser"
> is another; a "VAE", a "CLIP encoder", a "reranker", a "query-rewrite head" are smaller
> presets that plug into them. ComfyUI proved this works for diffusion; NNSpire
> generalises it across **all** architecture families *and* across wrapping formats
> (`.onnx`, `.safetensors`, GGUF, `.nnsx`).
>
> ### Everything is a canvas template — and canvases nest (ADR-041)
>
> The diffuser/LLM duo is only the headline example. **Every semantic concept in
> `NN_bricks_overview.md` and the ontology is a *canvas template*** — a typed, named box
> (or graph of boxes) with ComfyUI-style input/output ports. The spectrum runs from the
> trivial to the elaborate:
>
> - A concept that genuinely *is* a single network (e.g. a bare `TextEmbeddingModel`)
>   becomes a template whose canvas has **exactly one fillable slot** — one box, one
>   position, fillable only with content valid for that role (native subgraph, imported
>   weights, or a remote endpoint).
> - A composite concept (LLM, diffuser, RAG) is a template with many wired slots.
>
> **Canvases nest recursively.** Any canvas can be wrapped by an *outer* canvas that
> serves as a **"running landscape"** for the model inside it — an orchestration/
> deployment scene (data sources, clients, service bindings, monitoring) hosting the
> *inner* model canvas. The inner canvas can itself host a deeper canvas, to arbitrary
> depth. A nested canvas exposes typed ports to its parent, exactly like a node. **No
> external format (ONNX/GGUF/safetensors) can express this nesting — only NNSpire
> project files (`.nnsp`/`.nnsx`) store it.** Export to a standard format flattens or
> extracts the runnable leaf where the format allows; the recursive landscape stays
> Studio-native.
>
> ### Two-level zoom (the core UX)
>
> ```text
>  MACRO graph (this phase)            MICRO graph (Phase 3 model editor)
>  ┌───────────────────────────┐      ┌──────────────────────────────────┐
>  │ [Text In] → (Tokenizer)    │      │  drill into "Transformer block":  │
>  │   → (Embedding) →          │  ⇄   │  Attention → LayerNorm →          │
>  │   ((Transformer ×N)) →     │ open │  Dense → GELU → Dense → +skip     │
>  │   (LM Head) → (Sampler) →  │      │  (the Phase-3 primitive graph)    │
>  │   [Text Out]               │      └──────────────────────────────────┘
>  └───────────────────────────┘
> ```
>
> Double-clicking any macro/group node opens its definition: either a built-in/native
> NNSpire subgraph (drill down to primitives), an imported weights blob, or a remote
> API endpoint. **Macro and micro are two views of one model; the canonical source is
> still code (ADR-030).**
>
> ### One navigable tree, three orthogonal axes (the corrected mental model)
>
> The whole composition is **one tree the user navigates top-to-bottom** — from the
> outermost running landscape down to the lowest primitive op / functoid — and the
> namespaces and `.nnsp` layout mirror that tree (ADR-047). The key correction: three
> things that *look* like "layers" are **not** levels of the tree; they are **orthogonal
> attributes** hanging off any node or subtree:
>
> - **Execution axis** (ontology L0/L1) — *where* a node runs: `local-cpu` / `local-cuda`
>   / `local-qpu` (Phase 6) / `remote-grpc` / `api-service`. A wrapper, not a child.
> - **Packaging axis** (ontology L2) — *how* a subtree is wrapped for export: `.onnx` /
>   GGUF / `.safetensors` / `.nnsx`. A decorator on a node, not a child. **This answers
>   the "is packaging above or below orchestration?" question: neither — packaging is not
>   a tree level at all** (ADR-045).
> - **Process axis** (ADR-046) — *which procedure* drives a fixed model scaffold: train /
>   infer / evaluate / distil / RLHF — the request's "2/3-and-a-half".
>
> **Axes can also be *surfaced* as canvases** (still attributes underneath). A
> **packaging-typed canvas** (ADR-045) is a virtual *lens* instance of a generic canvas
> that shows a subtree *through* one export format and live-greys what that format cannot
> package — constraining editors up-front instead of dumping the conflict on the human at
> export time. A **procedural canvas** (ADR-046) is a canvas *kind* that *instantiates* a
> structural (layer / flow / diffusion) canvas to run it under one procedure.
>
> **Tree orientation (UI):** the **root** is the outermost generic canvas (the user's
> "layer 1"); the **leaves** are the primitive ops / functoids. Hardware / runtime is an
> attribute, never a leaf. This is the *inverse* of the ontology's L0 (hardware) → L6
> numbering, which is kept only in the conceptual / blueprint docs.
>
> The structural containment tree is only: **landscape canvas → orchestration / pipeline
> (Tier D) → model role (Tier C) → block (Tier B) → primitive op (Tier A) → functoid.**
> Two **cross-cutting views** re-slice the same data: a **Repository view** (definitions
> vs. instances — class/instance disambiguation, ADR-047) and a **Message-type inventory**
> (the Level-5 typed messages in play, ADR-048). The user's "layer 0" (Client / Foundry /
> Agents) is **ontology L6**, lives in the separate `nnagent` project *outside* this tree,
> and appears only as one typed orchestration-boundary node (ADR-044).

### Phase 5.7 — implementation chunks (incremental delivery, one per agent session)

> **For cost-bounded incremental agents (e.g. Claude Sonnet).** Each row is a
> self-contained work unit: implement it, run its slice of the Phase 5.7 milestone, then
> stop. Do **not** load the whole phase at once. Read the listed refs for that chunk only.
> Decision-gate ADRs (035–050) must be accepted (promoted to `docs/adr/`) before the
> implementation chunks that depend on them.

| # | Chunk (sub-section) | Deliverable | Read first (refs) |
|---|---|---|---|
| 0 | 5.7.0 | Accept ADR-035–049 (conceptual gates) | `docs/ARCHITECTURE.md` §12, `docs/blueprints.md` §3.10, `docs/modern_ai_systems_ontology.md` "Application to NNSpire" |
| 1 | 5.7.1 + 5.7.1a | Tier A/B/C/D descriptors + `NN_bricks` realisation map | ADR-036, ADR-042, `docs/NN_bricks_overview.md` |
| 2 | 5.7.2 | `SemanticPortType` + compat check | ADR-037, ontology L5 |
| 3 | 5.7.4 | `BackingModel` (native / imported / remote) | ADR-035, `docs/DEPLOYMENT.md` (`.nnsp` spec) |
| 4 | 5.7.5 | `ExecutionBinding` + `RunnerRepository` + `ExecutionPlan` | ADR-038, ADR-039, `docs/DEPLOYMENT.md`, External Compute Services annex |
| 5 | 5.7.6 | `UniversalClient` (type→editor) | ADR-040, `docs/PIPELINE.md` |
| 6 | 5.7.3 | `MacroCanvas.qml` (+ nesting, lens, single-slot) | ADR-035/041/045/046, `docs/ARCHITECTURE.md` §12.4 |
| 7 | 5.7.9 | `CompositionTree` + `RepositoryView` | ADR-044, ADR-047 |
| 8 | 5.7.10 | `ProcedureTemplate`s + run-mode selector | ADR-046 |
| 9 | 5.7.11 | `MessageInventory` view | ADR-048, 5.7.2 |
| 10 | 5.7.7 + 5.7.12 | `.nnsp`/`.nnsx` format + namespace/layout mirroring | ADR-045, ADR-047, ADR-021, `docs/DEPLOYMENT.md` |
| 11 | Theming | `ThemeDescriptor` + built-in themes | ADR-049, `nnagent/TODO.md` skin descriptor |

### 5.7.0 — Conceptual model & decision gates (ADRs)

> These follow the inline-ADR convention used by ADR-030…034 above: recorded here as
> `[! decision needed]` until accepted, then promoted to a file in `docs/adr/`.

#### ADR-035 — Macro/micro two-level graph; "LLM is a preset, not a type" `[! decision needed]`
- [ ] Confirm that NNSpire has **one** graph model rendered at two zoom levels (macro =
  semantic blocks, micro = primitive layers), not two separate editors — record in
  `ARCHITECTURE.md §12`
- [ ] Confirm macro-architectures (LLM, diffuser, VAE, encoder-only, RAG head) are
  **presets** (typed node templates + default wiring), not hard-coded model classes
- [ ] Define the macro↔micro "open definition" interaction (double-click drills down;
  breadcrumb to zoom back up)

#### ADR-036 — Semantic composition taxonomy (four tiers) `[! decision needed]`
- [ ] Confirm the four-tier taxonomy and record in `blueprints.md` (new chapter) +
  `ARCHITECTURE.md §12`:
  - **Tier A — Primitives** (DONE, Phase 1): `Dense`, `ReLU`, `GELU`, `Conv2D`,
    `LayerNorm`, `MultiHeadAttention`, … — the `ILayer` / `IActivation` set
  - **Tier B — Layer groups / semantic blocks** (NEW): named, parameterised *subgraphs*
    of Tier A with a documented *function* — e.g. `TokenEmbeddingBlock`,
    `PositionalEncodingBlock`, `SelfAttentionBlock`, `FeedForwardBlock`,
    `TransformerEncoderBlock`, `DenoiserBlock`, `VAEEncoderBlock`, `LMHead`,
    `SyntaxLearningStack`. Each maps to a section of
    `NN_vnitrni_struktury_…architektury.pdf` and `NN_bricks_overview.md`
  - **Tier C — Model roles / semantic model types** (NEW): full networks defined by
    *purpose* — `TextEmbeddingModel`, `InferenceAssistantLLM`, `QueryRewriteModel`,
    `MultimodalPerceptionModel`, `Reranker`, `LatentDiffuser`, `VAE`, `CLIPEncoder`,
    `Classifier` — composed of Tier B blocks
  - **Tier D — Macro-architectures / pipelines** (NEW): typed graphs wiring Tier C
    models together — `LLMPipeline`, `DiffusionPipeline`, `RAGPipeline`,
    `AgenticPipeline` (this is where Phase 4 pipeline + Phase 5.5 ecosystem meet)
- [ ] Decide where each tier is *authored*: B/C/D as native subgraph templates vs.
  plugin-contributed vs. imported

#### ADR-037 — Typed semantic port system (ontology Level 5 messages) `[! decision needed]`
- [ ] Define the port type set from ontology Level 5: `TEXT`, `TOKENS`, `TOKEN_IDS`,
  `EMBEDDING[dim]`, `HIDDEN_STATE`, `LOGITS`, `LATENT`, `NOISE`, `CONDITIONING`,
  `KV_CACHE`, `ATTENTION_MASK`, `IMAGE`, `AUDIO`, `VIDEO`, `TAGS`, `DOCS`, `TOOL_CALL`
- [ ] Extend the Phase 5.5 ecosystem `PortType` system (currently `{LLM_TEXT_IN/OUT,
  EMBED_VEC[dim], TOKEN_IDS, HTTP_REST, GRPC, UNIXSOCK, DOCS}`) into this shared set so
  the macro canvas and the ecosystem canvas use **one** type system
- [ ] Connection-time validation: dimension mismatch = amber warning; type mismatch =
  red error (reuse the `CompatibilityChecker` machinery from the cross-framework layer)

#### ADR-038 — Execution bindings & federated offload `[! decision needed]`
- [ ] Confirm every macro/micro node carries an **`ExecutionBinding`**: `local-cpu` |
  `local-cuda` | `remote-grpc` (`RemoteBackend`) | `remote-job` (`RemoteTrainer`) |
  `api-service` (a `ServiceConnector` plugin endpoint) — record in `DEPLOYMENT.md`
- [ ] Confirm a single project may **split execution across bindings**: e.g. train the
  embedding head locally, offload LLM fine-tune to a RunPod job, run the diffusion step
  on a hosted API — all recorded in the project file
- [ ] Define the `ExecutionPlan` scheduler that walks the macro graph and dispatches each
  node to its bound runner, moving typed messages between them
- [ ] Add a **project Runner Repository**: a project-scoped registry of runner definitions
  (local devices, remote gRPC hosts, API services) seedable from the app's global runner
  repository; an `ExecutionBinding` **references a runner entry by id**, not inline config.
  Runner configs (URLs, driver paths, memory addresses, key-refs) are **project parameters**;
  secrets stay in the OS keychain only

#### ADR-039 — `ServiceConnector` plugin type (standardised external APIs) `[! decision needed]`
- [ ] Add `SERVICE_CONNECTOR` to the Plugin SDK plugin-type enum (`PLUGIN-SDK.md`,
  `NNSpire_plugin.h`) alongside `LAYER | TOKENIZER | OPTIMIZER | BACKEND | UI_PANEL |
  TRUST_UPDATE`
- [ ] `ServiceConnector` contract: `capabilities()` (train / infer / embed / generate),
  `submit(typed message + model ref)`, `poll(job)`, `fetchWeights(job)`, `estimateCost()`,
  `healthCheck()` — installed, activated, configured, and **paid** per the existing trust +
  keychain rules (no secrets in project files)
- [ ] Reference connectors (design only this phase): OpenAI-compatible, HuggingFace
  Inference Endpoints, Replicate, a generic "diffusion API", a generic "LLM API"

#### ADR-040 — Universal client as a type-parameterised I/O component `[! decision needed]`
- [ ] Confirm the "agentic clients" that feed diffusers/LLMs (the encoders/decoders /
  dataset providers) are instances of **one** `UniversalClient` component whose editor
  UI is chosen by its declared port type — record in `PIPELINE.md`
- [ ] Confirm it wraps the Phase 4 input/output adapters (text, image, audio, CSV,
  Apache Arrow, MCP) and the dataset loaders, surfaced as a single draggable node
- [ ] Define the type→editor mapping table (see 5.7.6)

#### ADR-041 — Recursive canvas types: every semantic concept is a (nestable) canvas template `[! decision needed]`
- [ ] Confirm the **canvas itself is a first-class, typed, versioned template** — not a
  blank surface — and that *all* semantic concepts (Tier B/C/D **and** a single NN) are
  expressed as canvas templates; record in `ARCHITECTURE.md §12`
- [ ] Define `CanvasTemplate.kind` as **four families** (user-confirmed taxonomy):
  - **`generic`** — recursive container / folder; roles: `landscape` (outer running
    scene), `single-slot` (one fillable box), `custom`
  - **`structural`** — defines NN structure (Tier B blocks + Tier C model role); sub-kind
    **`orchestration`** wires *multiple* models/structures (Tier D: diffusion, LLM, RAG)
  - **`procedural`** — *instantiates* a `structural`/`orchestration` canvas to run it under
    one procedure (train / infer / …, ADR-046)
  - **`packaging`** — virtual *lens* over a subtree for one export format (ADR-045)
- [ ] Confirm **recursive nesting**: a canvas may contain a nested canvas node; the nested
  canvas exposes ComfyUI-style typed ports to its parent; arbitrary depth
- [ ] Confirm an **outer `landscape` canvas** can wrap an inner model canvas as its
  "running landscape" (data sources, clients, `ExecutionBinding`s, monitors) — usable as
  a standalone NNSpire project
- [ ] Confirm nesting is **Studio-only persistence** (`.nnsp`/`.nnsx`); no external format
  stores it; export flattens/extracts the runnable leaf where the target format allows
- [ ] Single-NN templates expose exactly one slot, fillable only with role-valid content
  (native subgraph | imported weights | remote endpoint — see 5.7.4)

#### ADR-042 — `NN_bricks_overview.md` realisation: Prim.X channels, Beh.X as templates `[! decision needed]`
- [ ] Confirm **Prim.X (the 20 primitives) → Tier A**, each realised via one of three
  channels (see 5.7.1a):
  1. **Builtin** — compiled into `NNSpire::builtin::*` (the prominent ones; mostly done
     in Phase 1: Dense, ReLU, GELU, Sigmoid, Softmax, MHA, LayerNorm, Conv2D, …)
  2. **Native plugin** — compiled, distributed, signed reference plugins (heavier / less
     common: DiffusionStep, PPO/RL, Routing/Gating, CrossAttention, …)
  3. **Script** — Python/scripting-engine plugins; includes deliberate **duplicates** of
     builtins purely to exercise and validate the scripting path
- [ ] Confirm **Beh.X (the 21 semantic behaviours) are *emergent functions*, not single
  ops → realised as Tier B/C templates**, never as primitives (e.g. "Token Recognition"
  = `TokenEmbeddingBlock`; "Positional Awareness" = `PositionalEncodingBlock`; "Syntax
  Learning" = early `TransformerEncoderBlock`). This answers the open question in the
  request: *yes, Beh.X = templates.*
- [ ] Define the channel-assignment policy (which Prim goes builtin vs native-plugin vs
  script) and the Beh.X → template-id map; record in `blueprints.md` (new chapter)

#### ADR-044 — One navigable composition tree; `nnagent`/L6 boundary `[! decision needed]`
- [ ] Confirm a single **Composition Tree / Project Explorer** presents every level in one
  outline — landscape → pipeline (Tier D) → model role (Tier C) → block (Tier B) →
  primitive (Tier A) → functoid — with nested canvases expandable inline; record in
  `ARCHITECTURE.md §12`
- [ ] Confirm the tree is the navigation spine: selecting a node focuses it on the active
  canvas and vice-versa; drill-down + breadcrumb (5.7.3) are tree traversal
- [ ] Confirm the user's **"level 0" (Client / Foundry / Agents) is ontology L6**, living
  in the separate `nnagent` project *outside* NNSpire's structural tree, embeddable only
  as a single typed **orchestration-boundary node** with an I/O interface into a canvas;
  note the numbering inversion (user-top "0" = ontology-top "L6"; hardware L0 is the floor)

#### ADR-045 — Three orthogonal axes vs. the structural tree `[! decision needed]`
- [ ] Confirm **semantic orchestration (L3 / Tier D) is the parent of model packaging
  (L2)** — an orchestration wires *many* packaged models — so they are **not** the same
  level and the order is **not** reversible
- [ ] Confirm **packaging / wrapping format is NOT a tree level**: it is an *export
  attribute* of a node or subtree (any subtree — a Tier C model or a whole Tier D macro —
  can be wrapped to `.onnx` / GGUF / `.safetensors` / `.nnsx`)
- [ ] Confirm the **execution binding** (L0/L1) and the **process** (train/infer/…) are
  likewise orthogonal attributes, not children; the structural tree contains only canvas
  / pipeline / model / block / op / functoid (ADR-044); record in `ARCHITECTURE.md §12` +
  `DEPLOYMENT.md`
- [ ] Add an optional **packaging-typed canvas (lens)**: a *virtual* instance of a generic
  canvas that projects a subtree through a chosen export format and **live-disables /
  flags** nodes the format cannot represent (constrain-early, not resolve-at-export); the
  underlying packaging attribute stays canonical, so one subtree can be viewed through
  several formats. Resolves the request's "let-the-user-do-anything vs. solve-at-export"
  dilemma in favour of early, per-format feedback

#### ADR-046 — Process/Procedure templates: one scaffold, many procedures `[! decision needed]`
- [ ] Confirm a model's **structure is fixed scaffolding** but its **procedure varies** —
  the *same* Tier C/D model is consumed by distinct **`ProcedureTemplate`s**: `train`,
  `fine-tune`, `infer`, `evaluate`, `distil`, `rlhf` — each binding its own datasets
  (`UniversalClient`), losses/optimisers, metrics, and `ExecutionBinding`s
- [ ] Confirm a `ProcedureTemplate` **references** a model definition (never copies it),
  so switching train↔infer reuses one scaffold (the request's "2/3-and-a-half")
- [ ] Model procedures as a **`procedural` canvas kind** that *instantiates* a structural
  (layer / flow / diffusion) canvas — structure reused, procedure wraps it; add `procedure`
  to `CanvasTemplate.kind` (ADR-041). This is how the "from-aside" axis also reads as a
  *layer/folder* in the tree
- [ ] Define the descriptor `{id, kind, modelRef, io[], dataset, loss, optimizer,
  metrics[], bindings}`; persist under `composition/procedures.json`
- [ ] Surface procedures as a switchable **run-mode** selector on the canvas and as
  `Procedures` entries in the Repository view (ADR-047)

#### ADR-047 — Definition vs. instance: the Repository view + layout/namespace mirroring `[! decision needed]`
- [ ] Confirm every template / network / resource is a **definition (class)** stored once,
  and canvases hold **instances (references, possibly parameterised)** of it — multi-use
  across canvases shares a single definition
- [ ] Add a **Repository view** listing definitions (NN structures, semantic ontology
  definitions, datasets, service endpoints, **runners**, procedures) and, per definition,
  every instance and its reference sites — the user-facing class/instance disambiguation
- [ ] Confirm the view mirrors the **physical project layout**: `.nnsp` `definitions/`
  (canonical, hashed, signed) + `composition/*_ref.json` (instances by id + version +
  hash); editing a definition once updates all instances
- [ ] Confirm **namespace = path (ADR-021)** so a definition's id *is* its tree location;
  `.nnsp` internal folders mirror the composition tree

#### ADR-048 — Message/data-type inventory view (Level-5 typed messages) `[! decision needed]`
- [ ] Add a **Message Inventory** view: the catalog of `SemanticPortType` instances in
  use — what typed data flows on each connector between nodes (diffusion: `LATENT`,
  `CONDITIONING`, `NOISE`; LLM: `TOKENS`, `HIDDEN_STATE`, `LOGITS`, `KV_CACHE`)
- [ ] Two scopes: **per-canvas** (the canvas being edited) and **suite-wide** (all types
  across the project); filter + jump-to-edge
- [ ] Pure query over `SemanticPortType` edges (5.7.2) — a view, not new persisted data

### 5.7.1 — Semantic composition taxonomy (`NNSpire/builtin/blocks/` + catalog)
- [ ] `BlockTemplate` descriptor (Tier B): `{id, displayName, tier, function, inputPorts,
  outputPorts, params, body: native-subgraph | code-snippet, kbRef, provenance}`
- [ ] `ModelRoleTemplate` descriptor (Tier C): `{id, role, inputPorts, outputPorts,
  blocks[], defaultWiring, exportTargets[]}`
- [ ] `MacroTemplate` descriptor (Tier D): `{id, family, nodes[], edges[], defaultBindings}`
- [ ] Built-in Tier B blocks (native subgraphs of existing primitives): `TokenEmbeddingBlock`,
  `PositionalEncodingBlock`, `SelfAttentionBlock`, `FeedForwardBlock`,
  `TransformerEncoderBlock`, `TransformerDecoderBlock`, `LMHead`, `VAEEncoderBlock`,
  `VAEDecoderBlock`, `DenoiserBlock` (UNet/DiT-style), `ClassifierHead`
- [ ] Built-in Tier C roles: `TextEmbeddingModel`, `InferenceAssistantLLM`,
  `QueryRewriteModel`, `Reranker`, `LatentDiffuser`, `MultimodalPerceptionModel`
- [ ] Built-in Tier D macros: `LLMPipeline`, `DiffusionPipeline`, `RAGPipeline`
- [ ] **Template catalog**: repo-shipped catalog + user catalog (`<app_data>/catalog/`) +
  plugin-contributed; each template versioned, signed (reuse Phase 2 trust), and
  classified by `CompatibilityChecker` (`torch_standard` / `keras_standard` /
  `NNSpire_extended`)
- [ ] Each Tier B/C template carries `@kb:` deep-links and a paragraph mapping it to
  `NN_vnitrni_struktury_…architektury.pdf` / `NN_bricks_overview.md`
- [ ] Dual-language: every template emits both Python (`torch_compat`) and C++ source

### 5.7.1a — `NN_bricks_overview.md` realisation map (ADR-042)

> The 20 `Prim.X` are Tier A ops; the 21 `Beh.X` are Tier B/C templates. The three
> realisation channels deliberately exercise the whole extensibility surface — builtin,
> native plugin, and script — so the scripting/plugin engine is validated against real
> content (including duplicate-of-builtin scripts).

- [ ] **Prim → channel** assignments (initial proposal; finalise in ADR-042):
  - *Builtin* (`NNSpire::builtin::*`): Prim.1 Dense, Prim.2 Embedding, Prim.3 ReLU,
    Prim.4 GELU, Prim.5 Sigmoid, Prim.6 Softmax, Prim.7 MHA, Prim.9 ResidualAdd,
    Prim.10 LayerNorm, Prim.11 BatchNorm, Prim.12 Convolution, Prim.13 MaxPooling,
    Prim.14 PositionalEncoding
  - *Native plugin* (compiled, signed): Prim.8 CrossAttention, Prim.15 Top-k/Top-p,
    Prim.16 ContrastiveLoss, Prim.17 PPO/RL, Prim.18 Routing/Gating,
    Prim.19 DiffusionStep, Prim.20 Recurrence/Loop
  - *Script* (Python plugin; some as deliberate builtin duplicates for engine testing):
    e.g. a scripted ReLU/GELU/Softmax duplicate + scripted Top-k sampler
- [ ] **Beh → template** map (Beh.X are Tier B/C templates, not ops): Beh.1→`TokenEmbeddingBlock`,
  Beh.2→`PositionalEncodingBlock`, Beh.3→early `TransformerEncoderBlock`,
  Beh.4→`SemanticTransformerBlock`, Beh.5→`DeepAttentionStack`, Beh.6→`FeedForwardBlock`,
  Beh.7→`AutoregressiveDecoder`, Beh.8/9→`LMHead`+sampler, Beh.10→`ResidualPathways`,
  Beh.11→`NormStack`, Beh.12→`QueryRewriteModel`, Beh.13→`Reranker`,
  Beh.14/15→`CNNFeatureStack`/`ViTEncoder`, Beh.16→`CrossmodalEncoder`,
  Beh.17→`DenoiserBlock`, Beh.18→`PolicyHead`, Beh.19→`PlannerLoop`,
  Beh.20→`ToolInvocationDecoder`, Beh.21→`EntityExtractionHead`
- [ ] Each `Prim`/`Beh` entry records its channel, namespace/plugin id, and `@kb:` link

### 5.7.2 — Typed semantic port system (`NNSpire/builtin/ports/`)
- [ ] `SemanticPortType` enum + `dim`/shape carrier (ADR-037 set)
- [ ] `PortCompat::check(out, in)` → `{ok | dimWarning | typeError}` with human message
- [ ] Unify with the Phase 5.5 ecosystem `PortType` enum (single source of truth)
- [ ] Adapter inference: when a `TOKENS → EMBEDDING` edge needs an embedding block, the
  editor offers to insert the matching Tier B block automatically

### 5.7.3 — Macro-architecture canvas (`ui/MacroCanvas.qml`)
- [ ] New panel + `Macro` layout preset (sibling of the Phase 5.5 `Ecosystem` preset);
  shares the dockable panel system and the typed-port wiring engine
- [ ] Palette grouped by tier: **Roles** (Tier C) · **Blocks** (Tier B) · **Macros** (Tier D)
  · **Clients** (universal I/O, 5.7.6) · **External** (service-connector nodes, 5.7.5)
- [ ] Drag template → typed node card; wire by typed ports; incompatible wires flagged
- [ ] **Drill-down**: double-click a Tier B/C node → opens its definition in the Phase 3
  micro model editor (native subgraph) **or** an "imported weights" inspector **or** a
  "remote endpoint" config panel, depending on the node's backing (5.7.4); breadcrumb
  navigates back to the macro view (`CodeGraphSyncController` extended to multi-level)
- [ ] Macro pattern gallery (load-as-starting-point), one entry per Tier D family:
  - `LLM (decoder-only)`, `Encoder-only embedding model`, `Latent diffusion (txt→img)`,
    `Naive RAG`, `Query-rewrite + retrieve + LLM`, `Multimodal perception → LLM`
- [ ] **Wrapping-format templates**: load an existing `.onnx` / `.safetensors` / GGUF /
  `.nnsx` as a macro node (introspect graph/metadata → show as a typed black-box with
  drill-down where the graph is available)
- [ ] **Recursive / nested canvases** (ADR-041): any canvas may be dropped onto another as
  a node; nested canvas exposes typed ports to its parent; breadcrumb + zoom navigate
  across nesting levels; arbitrary depth
- [ ] **`landscape` canvas kind**: an outer "running landscape" canvas (data sources,
  `UniversalClient`s, `ExecutionBinding`s, monitors) that hosts an inner model canvas and
  is itself a standalone NNSpire project
- [ ] **Single-slot templates**: a concept that is one network renders as a one-box canvas;
  the box accepts only role-valid content (native subgraph | imported weights | endpoint)
- [ ] Nesting persisted **only** in `.nnsp`/`.nnsx` (no external-format equivalent)

### 5.7.4 — Backing models: native / imported weights / remote (`NNSpire/builtin/backing/`)
- [ ] `BackingModel` abstraction — a Tier B/C node is backed by exactly one of:
  - **(a) Native subgraph** — drills down to Phase 3 primitives; trainable in-engine
  - **(b) Imported weights** — `.safetensors` / `.onnx` / GGUF / `.nnsx`; run/fine-tune
    where the format allows; "untrained / homebrew shipping mode" supported (blueprints
    only, no weights — see Phase 5 `.nnsp` spec)
  - **(c) Remote endpoint** — a `ServiceConnector` (5.7.5); the node is a typed proxy
- [ ] Drill-down policy per backing type (native = full edit; imported = inspect + adapt;
  remote = configure only)
- [ ] "Pre-defined network" library: browse/import known role models (embedding, LLM,
  VAE, CLIP) by id; attach weights; mark as run-only or fine-tunable

### 5.7.5 — Federated execution & external-service offload
- [ ] `ExecutionBinding` struct on every macro/micro node (ADR-038 set), default = inherit
  from parent / project default
- [ ] `RunnerRepository` (project-scoped): runner entries `{id, kind, endpoint/driverPath/
  memAddr, capabilities, keyRef}` seeded from the global app runner repo; an
  `ExecutionBinding` references an entry by id; **secrets in OS keychain only** (the file
  keeps endpoint + key-ref, never the secret)
- [ ] `ServiceConnector` plugin type (ADR-039) — new entry in the Plugin SDK; reference
  connectors implemented as design stubs (OpenAI-compatible, HF Inference, Replicate,
  generic LLM API, generic diffusion API)
- [ ] `ExecutionPlan` — topologically walks the macro graph, groups nodes by binding,
  dispatches each group to its runner (`CpuBackend` / `CudaBackend` / `RemoteBackend` /
  `RemoteTrainer` / `ServiceConnector`), and streams typed messages across boundaries
- [ ] Mixed-binding scenario validated: **train embedding head locally → offload LLM
  fine-tune to a remote job → run diffusion step on an API service**, all in one project
- [ ] Per-node execution badge in the UI (extends the Phase 3 "backend acceleration status
  badge"): shows binding + live status (local / queued / running on `<service>` / done)
- [ ] Credentials: API keys in OS keychain only; project stores **endpoint + key-ref**,
  never the secret (consistent with Phase 5.5 and the External Compute Services annex)
- [ ] Cost surfacing: `estimateCost()` per remote/api node; project-level total before run

### 5.7.6 — Universal client as a parameterised I/O component (`ui/UniversalClient.qml`)
- [ ] One `UniversalClient` node whose **input editor is selected by its declared port
  type** — the type→editor mapping:

  | Declared port type | Rendered editor |
  |---|---|
  | `TEXT` | multi-line text editor |
  | `TAGS` | single/multi-word tag editor (chips) |
  | `TOKENS` / `TOKEN_IDS` | token-id field + live tokenizer preview |
  | `IMAGE` | image picker + thumbnail + drop target |
  | `AUDIO` | waveform picker / recorder |
  | `EMBEDDING[dim]` | vector paste / file picker |
  | `DOCS` | document/folder loader + chunking controls |
  | dataset (training) | dataset loader: CSV / Arrow / image-folder / text-corpus |

- [ ] Wraps the Phase 4 input adapters (text, image, audio, CSV/Arrow, MCP) and output
  adapters (text decode, image/audio synth, JSON) — surfaced as **one** draggable node
  with an input side and an output side
- [ ] Acts as the "agentic encoder/decoder" that feeds Tier C/D nodes (the role ComfyUI
  assigns to its CLIP-encode / VAE-decode / text-input nodes)
- [ ] Training-set mode: the same component supplies *batched* typed datasets to a
  training run (drives `Trainer` / `RemoteTrainer`), not just single-shot inference input
- [ ] Dual-language: emits matching Python and C++ adapter code

### 5.7.7 — Project & package format impact
- [ ] `.nnsp` extension: `composition/macro_graph.json` (Tier D graph), `composition/
  templates_ref.json` (Tier B/C template ids + versions + hashes), `composition/
  bindings.json` (per-node `ExecutionBinding`), `composition/services.json` (endpoints +
  key-refs, **no secrets**)
- [ ] `.nnsx` export: a macro node backed by a native subgraph exports to standard ONNX;
  remote/api nodes export as a documented proxy descriptor (not runnable offline)
- [ ] Output packages unchanged in spirit (ONNX / GGUF wrap / `.nnsr` / `.nnsx`), now
  selectable **per Tier C model** inside a Tier D macro (e.g. export the embedding model
  as ONNX, keep the LLM as an API reference)
- [ ] Schema validation for all new JSON parts; `compat_level` recorded per macro

### 5.7.8 — Cross-references (no new work; wiring of existing items)
- [ ] Phase 3 model editor = the **micro** view drilled into from the macro canvas
- [ ] Phase 4 pipeline input/output adapters = the **Universal Client** internals
- [ ] Phase 5 `RemoteBackend` / `RemoteTrainer` + External Compute Services annex =
  the **offload** runners bound via `ExecutionBinding`
- [ ] Phase 5.5 ecosystem canvas = the **Tier D / deployment** neighbour; shares the
  typed-port engine and the `ServiceConnector` plugin type
- [ ] Cross-framework `CompatibilityChecker` = the template classifier (standard vs.
  extension)

### 5.7.9 — Unified composition tree & Repository view (`ui/CompositionTree.qml`)
- [ ] `CompositionTree` dock: one outline across all tiers (landscape → D → C → B → A →
  functoid); nested canvases expandable inline; node selection ⇄ canvas focus (ADR-044)
- [ ] `RepositoryView` dock: definitions (classes) vs. instances (references); per
  definition show usage sites; edit-once-update-all (ADR-047)
- [ ] Backed by `composition/definitions/` + `*_ref.json`; namespace = path (ADR-021, ADR-047)

### 5.7.10 — Procedure / process templates (`NNSpire/builtin/procedures/`)
- [ ] `ProcedureTemplate` kinds: `train`, `fine-tune`, `infer`, `evaluate`, `distil`,
  `rlhf` — each *references* one model definition (ADR-046)
- [ ] Run-mode selector on the canvas swaps the active procedure without altering the model
- [ ] Persist `composition/procedures.json`; each procedure carries its own datasets,
  losses, metrics, and `ExecutionBinding`s

### 5.7.11 — Message/data-type inventory view (`ui/MessageInventory.qml`)
- [ ] `MessageInventory` dock: catalog of Level-5 typed messages on the edges (ADR-048)
- [ ] Per-canvas filter and suite-wide filter; jump from a type to the edges carrying it
- [ ] Pure query over `SemanticPortType` edges (5.7.2); no new persisted data

### 5.7.12 — Namespace & `.nnsp` layout mirroring (ADR-047)
- [ ] Confirm `.nnsp` internal layout mirrors the composition tree (one folder per nested
  canvas; `definitions/` for classes; `composition/*` for instances + axis attributes)
- [ ] Confirm built-in template namespaces follow path = namespace (ADR-021):
  `NNSpire::builtin::blocks::*`, `::roles::*`, `::macros::*`, `::procedures::*`
- [ ] Schema-validate the mirrored layout; round-trip test (tree ⇄ files)

### Phase 5.7 milestone
- [ ] On the macro canvas, build an `InferenceAssistantLLM` from Tier B blocks, drill into
  `TransformerEncoderBlock`, see the Phase-3 primitive graph, and run XOR-scale training
- [ ] Load a `LatentDiffusion (txt→img)` macro template; confirm it is the *same* editor
  expressing a different preset (validates ADR-035 "LLM is a preset")
- [ ] Bind one node to `local-cpu` and one to a stub `ServiceConnector`; run the
  `ExecutionPlan` and confirm typed messages cross the boundary
- [ ] Feed the macro with a `UniversalClient` set to `TAGS` and to `IMAGE`; confirm the
  editor UI changes with the declared port type
- [ ] Place a non-LLM concept (e.g. `TextEmbeddingModel`) as a **single-slot** canvas and
  fill it with imported weights (validates "everything is a canvas template", ADR-041)
- [ ] Nest an inner model canvas inside an outer `landscape` canvas; save + reopen the
  `.nnsp`; confirm the recursive structure round-trips (Studio-only persistence)
- [ ] Realise one `Prim` per channel — builtin, native plugin, **and** a script duplicate
  of a builtin — and confirm all three appear identically as Tier A nodes (ADR-042)
- [ ] Open the **Composition Tree** and navigate from a `landscape` canvas down to a single
  `Dense` op and a functoid in one outline (validates ADR-044 one-tree)
- [ ] Reference one `TextEmbeddingModel` **definition** in two canvases, edit it once in the
  **Repository view**, and confirm both instances update (validates ADR-047 class/instance)
- [ ] Switch a model between `train` and `infer` **procedures** without changing its
  structure; confirm datasets/losses/bindings swap (validates ADR-046 process axis)
- [ ] Open the **Message Inventory** filtered to a diffusion canvas; confirm `LATENT`,
  `CONDITIONING`, `NOISE` edges are listed (validates ADR-048)

---

## Phase 6 — Quantum Backend

### Quantum backend (`NNSpire/backends/quantum/`)
- [ ] `QuantumBackend` implementation: `execute()` dispatches to Qiskit Python via bridge
- [ ] Quantum layer type: `QuantumCircuitLayer` — parameterised quantum gate sequence
- [ ] Hybrid graph support: classical `ComputeGraph` with embedded `QuantumCircuitLayer` nodes
- [ ] Qiskit Python plugin bridge: circuit construction from layer parameters
- [ ] IonQ REST API connector (alternative to local Qiskit simulator)
- [ ] IBM Quantum REST API connector

### Quantum studio panel (`ui/QuantumPanel.qml`)
- [ ] Quantum circuit visualizer for `QuantumCircuitLayer`
- [ ] Simulator vs. real hardware toggle
- [ ] Job queue monitor for hardware execution

### Phase 6 milestone
- [ ] Simple 2-qubit parameterised circuit runs via Qiskit simulator, driven from Studio UI

---

## Cross-cutting concerns (all phases)

### UI frontend portability — desktop-first, but web/mobile-ready (do not lock out)

> **Architecture guard.** NNSpire ships as a **desktop application**, and that is the
> right call. But the *UI* (only the UI — never the runners) should remain *convertible*
> in the future into an **Android / iOS** app or a **Web** front-end (Qt for WebAssembly /
> `QtWebEngine`), without a rewrite. **Runners are always a HW/VM/host concern and stay
> native** — a phone or browser would talk to a remote runner via the existing
> `RemoteBackend` / `ServiceConnector` paths, not run CUDA locally. The goal here is to
> avoid *accidentally* welding the QML UI to desktop-only assumptions.

#### ADR-043 — Keep the UI portable to web/mobile (runners stay native) `[! decision needed]`
- [ ] Confirm the rule and record in `ARCHITECTURE.md §7`; QML (ADR-018) already targets
  Qt for WebAssembly, Android, and iOS — the choice is preserved, not changed
- [ ] **Strict UI/engine separation**: QML and controllers must call the engine only
  through a transport-abstractable façade (in-process today; gRPC/WebSocket-capable
  tomorrow) so a WASM/mobile UI can drive a remote engine unchanged
- [ ] **No desktop-only assumptions in QML**: no raw local-filesystem paths in UI logic
  (go through a `FileService` abstraction), no blocking calls on the UI thread, no
  hard dependency on multi-window/dock features that WASM/mobile cannot provide
- [ ] **Feature-capability flags** for surfaces that cannot exist on web/mobile (local
  plugin loading, OS keychain, embedded Python) so the UI degrades gracefully instead of
  failing to build/run on those targets
- [ ] Keep heavy/native-only work (plugin loading, trust store, CUDA, embedded Python)
  behind the engine façade, never in QML — these are exactly the parts a WASM build drops
- [ ] Add a CMake `app-wasm` preset **stub** (not built in CI yet) as a standing reminder
  the target must keep compiling-in-principle; revisit post-v1.0
- [ ] Document the portability contract in `docs/VSCODE-DEV-SETUP.md` / `ARCHITECTURE.md`
  so every new panel is reviewed against it

### UI theming / skinning — full visual customization (parity with `nnagent` skins)

> **Goal.** Every visible surface in the Studio is themeable from a declarative theme
> descriptor — connectors, node / module outline shapes, fonts, parameter inputs / setters,
> the composition / layer tree graph, and all foreground / background / accent colours —
> using only what Qt / QML exposes directly (palettes, `Material` / `Universal` styles,
> `FontLoader`, `Qt Quick Controls` styling, `Shape` / `Canvas` for connectors). The token
> vocabulary is shared in spirit with the `nnagent` skin descriptor so the two products
> feel like one family.

#### ADR-049 — Declarative UI theming descriptor `[! decision needed]`
- [ ] Define a `ThemeDescriptor` (JSON) with token tables: **colour roles** (fg / bg /
  accent / warning / error per element class), **typography** (family / size / weight /
  line-height per class: node-title, port-label, param-input, code, tree-row, breadcrumb),
  **node / module outline shape** (rect / rounded / hex / pill + border width), **connector
  style** (bezier / orthogonal / straight, width, arrow, per-`SemanticPortType` colour),
  and **canvas / grid background**
- [ ] Theme scope covers the node-graph canvas, the `CompositionTree` / `RepositoryView`
  docks, the `MessageInventory`, parameter inspectors, and all standard panels
- [ ] Built-in themes (reference): `light`, `dark`, `high-contrast`, `blueprint`; user
  themes under `<app_data>/themes/`; plugin-contributed themes allowed (signed, reuse
  Phase 2 trust)
- [ ] Hot-reload: edit a theme file → UI updates without restart (mirrors `nnagent`)
- [ ] Qt-native means only (QML `palette`, `Qt Quick Controls` style, `FontLoader`,
  `Shape` / `Canvas`) — **no custom rendering engine**, stays WASM/mobile-portable (ADR-043)
- [ ] Per-`SemanticPortType` colour map shared with the Message Inventory legend (5.7.11)
- [ ] Align the token vocabulary with the `nnagent` skin descriptor so themes are
  conceptually portable between the two apps

### Dual-language compliance
- [ ] Policy in `ai.md` enforced: code review checklist item before merge
- [ ] All exported code (sandbox) shows Python | C++ toggle
- [ ] All whitepaper .md docs use dual code blocks

### KB integration
- [ ] `@kb:` source comments on all mathematical implementations
- [ ] In-app help deep-links verified for all wizard steps
- [ ] KB gap documents written as new `.md` files added to `ai-standards-kb/`

### CI / packaging
- [ ] GitHub Actions (or CI of choice) — build + test on Windows (MSVC + MinGW), macOS, Ubuntu
- [ ] Qt 5.15 LTS secondary build matrix
- [ ] AppImage build for Linux
- [ ] Installer build (optional) for Windows
- [ ] Doxygen API doc generation
- [ ] Plugin signing step in release pipeline using `NNSpire-sign`

### Performance & optimization
- [ ] CUDA backend benchmarks (matmul, conv) vs CPU baseline
- [ ] Memory layout profiling (cache-friendly tensor strides)
- [ ] Training throughput: tokens/sec, samples/sec metrics exposed via `Trainer` callbacks
- [ ] ONNX model profiling panel in UI

### Security
- [ ] TrustStore permissions hardened on all three platforms
- [ ] `NNSpire-runner` sidecar sandboxing (process isolation, limited syscalls)
- [ ] No eval / dynamic code execution in the Qt app process without sandbox
- [ ] OpenSSL kept up to date; SBOM generated for each release

### Legal, licensing & branding
> Refs: `LICENSING.md` §4–6, `LICENSES/third-party/README.md`

- [ ] **LICENSES/third-party/** — populate with required license texts as each dep is vendored (see checklist in `LICENSES/third-party/README.md`); wire into `Help → About → Licenses` dialog
- [ ] **CUDA attribution** *(Phase 4)* — when CudaBackend ships: add NVIDIA CUDA Toolkit attribution to `Help → About`; confirm EULA redistribution terms; do NOT vendor NVIDIA runtime files without legal review; add `LICENSES/third-party/NVIDIA-CUDA-EULA.txt`
- [ ] **Qiskit attribution** *(Phase 6)* — when QuantumBackend ships: add `Qiskit-LICENSE` + `Qiskit-NOTICE` to `LICENSES/third-party/`; add attribution to `Help → About`
- [ ] **CLA tooling** — set up GitHub CLA bot or DCO sign-off (`Signed-off-by:`) before first external contribution is accepted; CLA text must include the patent warranty clause (see `LICENSING.md §5`)
- [ ] **Trademark search** *(before public beta)* — perform formal USPTO / EUIPO / WIPO wordmark search for "NNSpire" in Nice classes 9 + 42; see `LICENSING.md §6` for findings and action items
- [ ] **Domain registration** *(before public marketing)* — `NNSpire.com` is taken (graphic design freelancer, different field); register `NNSpire.io` / `.ai` / `.dev` or chosen alternative; align with trademark filing
- [ ] **Trademark filing** *(before commercial plugin sales)* — file in US (class 9 + 42) and EU; engage IP attorney
- [ ] **GELU patent watch** — US application 20210110229A1 (Hendrycks 2019); confirm current prosecution status with attorney before commercial release; consensus is non-patentable as a mathematical function but confirm

---

## External Compute Services

NNSpire must support offloading computation to remote GPU services at multiple architectural levels.
This is critical during development (no local CUDA GPU available) and for production scale-out.

### Where external compute surfaces in the NNSpire UI and CLI

| Integration point | Where it appears | Phase |
|---|---|---|
| **Backend selector** (CPU / CUDA / Remote-gRPC) | Training Dashboard panel — backend dropdown | Phase 3 |
| **Remote training panel** (pod launch, job monitor, cost estimate, weight import) | Training Dashboard panel — shown when Remote mode active | Phase 3 |
| **Remote pod manager** (Vast.ai / RunPod / LambdaLabs pod lifecycle) | Deployment panel | Phase 5 |
| **Managed job submitter** (SageMaker / Vertex AI / Azure ML job dispatch) | Deployment panel | Phase 5 |
| **`NNSpire-cli remote *`** (pod list / launch / stop from terminal) | `NNSpire-cli` command-line tool | Phase 5 |
| **`NNSpire-cli train --remote`** (headless training submission) | `NNSpire-cli` command-line tool | Phase 5 |
| **Runner health monitor** (Triton / KServe / TF Serving inference endpoints) | Deployment panel — health/latency/throughput | Phase 5 |

### Integration points in NNSpire architecture

| NNSpire level | What integrates | What it enables |
|---|---|---|
| **Backend level** | New `IBackend` implementation (`RemoteBackend`) | Individual tensor ops (`matmul`, etc.) forwarded via gRPC to a remote CUDA worker; training loop runs locally |
| **Trainer level** | `RemoteTrainer` / job dispatch | Entire training job serialised and sent to a remote service; local machine only submits + receives trained weights |
| **Inference level** | Existing Triton / KServe / TF Serving connectors | Already in TODO Phase 5; model deployed remotely, NNSpire queries it |

### Services sorted by abstraction level (lowest = most control / cheapest)

#### Level 0 — Raw GPU Marketplace (P2P / bare metal, you manage everything)

Best for: development, experimentation, budget training. SSH + Docker container access.

| Service | API type | Sample prices (per GPU/hr) | Notes |
|---|---|---|---|
| **Vast.ai** | REST + Python SDK ([docs.vast.ai](https://docs.vast.ai)) | RTX 4090 from $0.28, A100 from $0.53, H100 from $1.54 | P2P marketplace; prices fluctuate by supply. **Cheapest option. Recommended first pilot.** |
| **RunPod** | REST + GraphQL ([docs.runpod.io](https://docs.runpod.io)) | RTX 3090 $0.46, RTX 4090 $0.59, A100 $1.39, H100 $2.39 | Community Cloud + Secure Cloud tiers. Serverless (per-second) also available from $0.00019/s |
| **LambdaLabs** | REST ([docs.lambdalabs.com](https://docs.lambdalabs.com)) | A100 ~$1.10–1.30, H100 ~$2.00–2.50 | Fixed prices, no marketplace. Simple SSH access. Very reliable. |

#### Level 1 — IaaS Cloud GPU (enterprise VMs, SLA, Kubernetes-native)

Best for: larger training runs needing reliability guarantees or Kubernetes orchestration.

| Service | API type | Sample prices (per GPU/hr) | Notes |
|---|---|---|---|
| **CoreWeave** | Kubernetes API / REST ([docs.coreweave.com](https://docs.coreweave.com)) | L40S $2.25, H100 $6.16, A100 $2.70 (all single-GPU from 8-GPU nodes) | Kubernetes-native; Slurm-on-Kubernetes (SUNK) available. NVIDIA-focused. No free tier. |
| **Hyperstack** | REST + Kubernetes ([hyperstack.cloud](https://hyperstack.cloud)) | Contact sales for large; competitive with CoreWeave | UK/EU-based NVIDIA DGX Cloud partner. Good for EU data compliance. |
| **AWS EC2** (p/g instances) | AWS SDK / boto3 ([aws.amazon.com](https://aws.amazon.com)) | V100 ~$3.06, A100 ~$4.00 (p4d), H100 (p5) ~$12+ | Full IaaS control; integrate with S3, IAM, VPC. Most complex but most flexible. |
| **Google Cloud** (GPU VMs) | Google Cloud SDK / REST ([cloud.google.com](https://cloud.google.com)) | A100 ~$2.93–3.67, H100 ~$9.98; TPU v4 also available | Strong for large runs; Vertex AI train jobs sit on top |
| **Azure** (NC/ND series) | Azure SDK / REST ([azure.microsoft.com](https://azure.microsoft.com)) | A100 ~$3.06–6.12 by region | Integrated with Azure ML and DevOps. Good if already in Microsoft ecosystem. |

#### Level 2 — Managed ML Platform (job submission, no VM management)

Best for: submitting training jobs without managing infrastructure. Higher cost per compute hour.

| Service | API type | Sample prices | Notes |
|---|---|---|---|
| **Paperspace / DigitalOcean Gradient** | REST ([docs.paperspace.com](https://docs.paperspace.com)) | Free tier (M4000 $0/hr quota); A100 ~$3.09/hr | Notebooks + GPU jobs. **Free tier is a good no-cost pilot.** Gradient Workflows = training job API. |
| **AWS SageMaker** | boto3 SageMaker client | Training instance overhead + EC2 cost (~15–20% premium) | Submit script + data; SageMaker manages spin-up/down. Integrates with S3 datasets, MLflow, etc. |
| **Google Vertex AI** | Google Cloud Vertex SDK | Training job overhead + VM cost (~15% premium) | Custom training jobs; good for TPU access. |
| **Azure ML** | Azure ML SDK v2 | AML compute overhead + VM cost | Jobs API; MLflow integration; best in Azure ecosystem. |

#### Level 3 — Enterprise Reserved / DGX Cloud (highest abstraction, capacity commitment)

Best for: production-scale training with guaranteed capacity. Enterprise contracts only.

| Service | API type | Notes |
|---|---|---|
| **NVIDIA DGX Cloud** | Kubernetes / Slurm via cloud partner (Azure, GCP, Oracle) | DGX SuperPOD capacity; ~$30–40k+/month enterprise contract. No public per-hour price. |
| **CoreWeave Reserved** | Same as CoreWeave on-demand | Up to 60% discount with 1/3/6-month commitment. |

### Recommended pilot sequence (budget-conscious)

1. **Paperspace Free Tier** — zero cost, immediate; verify the Trainer-level job dispatch design.
2. **Vast.ai RTX 4090** (~$0.30/hr) — cheapest CUDA GPU with 24 GB VRAM; test Backend-level RemoteBackend over SSH tunnel.
3. **RunPod Serverless** (~$0.00031/s for 4090) — test per-request inference dispatch; useful for the inference-level connector.
4. **LambdaLabs A100** (~$1.15/hr) — first "real" training run at scale; stable fixed pricing, no surprises.
5. **CoreWeave / AWS** — only once NNSpire's remote training pipeline is proven at smaller scale.

### TODOs — architecture items to add
- [ ] Design `RemoteBackend : IBackend` — forwards individual tensor ops to a remote CUDA worker via gRPC (Phase 1 extension)
- [ ] Design `RemoteTrainer` / job-dispatch adapter — serialises model + dataset, submits to SageMaker/Vertex/RunPod job API, polls for completion, imports weights (Phase 5 extension)
- [ ] `DEPLOYMENT.md` section: external compute service integration guide (which service maps to which NNSpire integration point)
- [ ] Add backend selector dropdown in Phase 3 UI to include "Remote (gRPC)" option alongside CPU/CUDA
- [ ] Credential/secret management for API keys (none stored in `.nnsp`; use OS keychain or env var)
- [ ] Vast.ai / RunPod CLI wrapper in `deployment/runners/` for spinning up and tearing down training pods from within Studio

---

## Icebox — Deferred Ideas

> Good ideas that are too valuable to discard and too far out to schedule. Revisit before each major phase.
> *(Original working definition: "follow-up spin-off TODO leftovers" — things that spun off from active discussions, do not belong in any current phase backlog, yet must not be forgotten.)*

- [ ] **Explorer ordering plugin** — a VS Code extension (and ideally a Windows Explorer shell extension) that reads the `## Reading order` numbered list from each folder's `CONTRIBUTING.md` and reorders the file tree accordingly for first-time navigators. The reserved heading name is `## Reading order`; the format is a standard Markdown ordered list. Authors set the intended order; developers can switch back to alphabetical at will.
- [ ] **Plugin SDK distribution** — once the Plugin SDK ABI is frozen (post-Phase 2), evaluate packaging `core/include/` as a standalone installable SDK (vcpkg port, Conan recipe, or plain zip). Linked to the namespace migration ticket.
- [ ] **ADR folder (`decisions/`)** — Architecture Decision Records: one short document per major architectural decision (namespace tier design, `backends/`-as-sibling-of-`core/`, Plugin SDK ABI plan, colocated-headers convention, etc.). These decisions are currently scattered in `blueprints.md`; ADRs make each "why" individually discoverable without reading the full document. Also the primary human-continuity mechanism if the AI session that made the decision is no longer available. See `docs/ai-standards-kb/CODE-ONBOARDING-ESSAY.md` for rationale.
- [ ] **C4 diagram (Level 1 + Level 2)** — one Mermaid diagram in `README.md` showing system context (Level 1) and major components/targets (Level 2). Serves all non-developer stakeholders without requiring code knowledge. Low maintenance: only update on major structural change.

- [ ] **`ParametricLayer<Fn>` / KAN-style non-linear synapses** — generic parametric
  layer template where `Dense = ParametricLayer<LinearFn>` (pure refactor, zero
  breakage) unlocks KAN (B-spline synapses, Liu et al. arXiv:2404.19756 2024),
  SIREN (sinusoidal synapses, Sitzmann et al. NeurIPS 2020), and log-bilinear
  variants (Mikolov et al. word2vec 2013) as first-class opt-in layer types.
  See **ADR-034** for full design rationale, both advocatus diaboli arguments,
  reference papers, and the recommended three-step Phase 4 rollout.
  Low priority until Phase 3 UI is stable; high strategic value —
  NNSpire becomes a transparent didactic bridge from standard linear layers
  to cutting-edge research architectures with a single template parameter change.

- [ ] **Primitivistic educational `nnprimitive` engine** — a small, fully isolated
  sub-project: its own CMake target, include tree (`include/nnprimitive/`), source
  tree (`src/nnprimitive/`), and C++ namespace `nnprimitive`. No dependency on
  `NNSpire-core`, no `Tensor`, no `IBackend`, no BLAS, no template metaprogramming.
  Implements the identical NN concepts (neuron, layer, forward pass, backward pass,
  gradient descent) as explicit `for`-loops over `std::vector<float>` — exactly
  how a motivated beginner encountering NNs for the first time would write them.
  **Purpose: purely educational.** Acts as the missing stepping-stone between
  "NN mathematical theory" and the matrix-optimised production implementation in
  `NNSpire`: a learner who finds the leap from textbook equations to `Dense::forward()`
  too large can read `nnprimitive` first, then see the same computation generalised
  and optimised in `NNSpire`. A dedicated blueprints.md chapter (to be written)
  will explain every generalisation step and why it was made.
  **Low priority** — implement only once the main engine and Phase 3 UI are stable.
