# NNStudio — Master Project Checklist

Legend: `[ ]` not started · `[~]` in progress · `[x]` done · `[!]` blocked/decision needed

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
- [x] Create `nnstudio/` top-level `CMakeLists.txt`
- [x] Create `nnstudio/core/` directory structure
- [x] Create `nnstudio/backends/` directory structure
- [x] Create `nnstudio/plugin-api/` directory structure
- [x] Create `nnstudio/plugins/` directory structure
- [x] Create `nnstudio/pipeline/` directory structure
- [x] Create `nnstudio/deployment/` directory structure
- [x] Create `nnstudio/python-bridge/` directory structure
- [x] Create `nnstudio/app/` directory structure
- [x] Create `nnstudio/tests/` directory structure
- [x] Create `docs/` directory with Doxygen `Doxyfile` stub

### Third-party / dependency setup
- [x] Copy/fetch Eigen headers — resolved via CMake FetchContent (3.4.0); no manual copy needed
- [x] Copy/fetch GoogleTest — resolved via CMake FetchContent (1.14.0); no manual copy needed
- [x] Copy/fetch pybind11 into `nnstudio/third-party/pybind11/` — resolved via pip site-packages (3.0.3) → FetchContent fallback; no manual copy needed
- [ ] Copy/fetch ONNX protobuf into `nnstudio/third-party/onnx/` — deferred to Format I/O phase
- [x] Confirm OpenSSL 3.x available — OpenSSL 3.6.0 at `C:\Program Files\OpenSSL-Win64`; MSYS2 also has 3.1.4
- [x] Verify Qt 6.5+ installation and record path in `ai_priv/ai_priv.md` — Qt 6.10.1 recorded

---

## Phase 1 — NN Engine Core (C++ library `nnstudio-core`)

### Tensor subsystem (`nnstudio/core/tensor/`)
- [x] `Tensor<T>` class: shape, strides, dtype enum (float32/float16/int8/int32)
- [x] Device tag: `CPU | CUDA | QUANTUM`
- [x] Basic ops: element-wise add/mul, reshape, slice, transpose, broadcast
- [x] `matmul()` dispatching to backend
- [x] Serialization: save/load to raw binary + metadata header (`NNS1` magic, binary little-endian)
- [x] Python binding via pybind11
- [x] Unit tests: shape arithmetic, op correctness vs NumPy reference values

### Layer subsystem (`nnstudio/core/layers/`)
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

### Activations (`nnstudio/core/activations/`)

> ⚠️ **HIGH PRIORITY — complete before Phase 2 Plugin SDK ABI freeze**  
> ADR-020 mandates that `IActivation` (not `ActivationBase`) is the published plugin extension point.  
> The current `ActivationBase : ILayer` design is correct for internal use but **must not** be shipped as the SDK contract — it is stateful between `forward()`/`backward()` and therefore not reentrant (see Invariant I-1, I-2 in `blueprints.md`).  
> **Work required:**
> - [x] Add `core/include/nnstudio/core/IActivation.h` with `ActivationForward` struct and `IActivation` interface (Option C: functional pair — see ADR-020)
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

### Loss functions (`nnstudio/core/losses/`)
- [x] `MSE` + gradient
- [x] `CrossEntropy` + gradient
- [x] `BinaryCrossEntropy` + gradient
- [x] `HuberLoss` + gradient
- [x] Python bindings
- [x] Unit tests (11 tests: MSE×3, BCE×3, CrossEntropy×2, Huber×4)

### Optimizers (`nnstudio/core/optimizers/`)
- [x] `SGD` (with momentum)
- [x] `Adam`
- [x] `AdamW`
- [x] `RMSProp`
- [x] LR scheduler base + `StepLR`, `CosineAnnealingLR`
- [x] Python bindings
- [x] Unit tests: parameter update step correctness (14 tests: SGD×5, Adam×3, AdamW×2, RMSProp×3, StepDecay×1)

### Compute graph (`nnstudio/core/graph/`)
- [x] `ComputeGraph` DAG: node registration during forward pass
- [x] Autograd: `backward()` traversal, gradient accumulation
- [x] Graph serialization (JSON)
- [x] Graph visualization data export (for UI consumption)
- [x] `EvalTrace` struct: opt-in per-layer capture of inputs, outputs, and gradients
- [x] `ILayer::forward()` optional `EvalTrace*` parameter (`nullptr` = zero-cost no-op in normal training)
- [x] `ILayer::backward()` optional `EvalTrace*` parameter — captures `grad_output` and computed `grad_input`
- [x] `Trainer::setTraceMode(bool)` — delegates to `ComputeGraph::setTraceMode()`; traces accessible via `ComputeGraph::traces()`
- [x] Python bindings

### Training loop (`nnstudio/core/training/`)
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
- [x] Python bindings (`nnstudio.keras.Model.fit/predict/evaluate`, pure-Python training loop)

### Format I/O (`nnstudio/core/formats/`)
- [ ] `OnnxIO::import()` — ONNX protobuf → ComputeGraph
- [ ] `OnnxIO::export()` — ComputeGraph → ONNX protobuf
- [ ] `NNSFormat` — `.nns` project file (JSON/MessagePack): weights + metadata + plugin refs + UI hints
- [ ] Python bindings

### Feature flags (`nnstudio/core/features/`)
- [x] `FeatureFlags.h` — `inline constexpr FeatureFlag` pattern; all flags declared `FREE`; `isEnabled()` returns true for all FREE flags

### Backend abstraction (`nnstudio/backends/`)
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
- [x] Same via Python bindings: `import nnstudio; ...` works in embedded Python

---

## ⚡ Cross-Framework Compatibility Layer (PyTorch / Keras drop-in)

> **Priority: UTMOST — must be designed before Phase 2 ABI freeze and before any pybind11 bindings are written.**
>
> ### Goal
> A user who writes code using only the *standard* PyTorch or Keras API signatures and names
> must be able to swap the include/import between NNStudio and the real framework with no code changes:
>
> ```cpp
> // C++ — swap this include to switch underlying framework
> #include <nnstudio/torch_compat.h>   // uses NNStudio engine
> // #include <torch/torch.h>          // uses LibTorch
>
> auto model = torch::nn::Sequential(torch::nn::Linear(4, 8), torch::nn::ReLU());
> ```
>
> ```python
> # Python — swap this import to switch underlying framework
> import nnstudio.torch_compat as torch   # uses NNStudio engine
> # import torch                           # uses real PyTorch
>
> layer = torch.nn.Linear(4, 8)
> x = layer(torch.zeros(2, 4))
> ```
>
> Any use of an **NNStudio extension** (beyond the torch/keras standard surface) triggers
> a UI/CLI warning: *"This project uses NNStudio-only features: [list]. Export to PyTorch
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

### C++ PyTorch-compatible shim (`include/nnstudio/torch_compat.h`)

- [x] `namespace torch` → alias block mapping to `nnstudio::*`:
  - `torch::Tensor` → `nnstudio::core::Tensor`
  - `torch::zeros / torch::ones / torch::rand` → our factory functions
  - `torch::nn::Module` → `nnstudio::core::Layer`
  - `torch::nn::Linear` → `nnstudio::builtin::layers::Dense`
  - `torch::nn::Conv2d` → `nnstudio::builtin::layers::Conv2D`
  - `torch::nn::Embedding` → `nnstudio::builtin::layers::Embedding`
  - `torch::nn::MultiheadAttention` → `nnstudio::builtin::layers::MultiHeadAttention`
  - `torch::nn::BatchNorm1d / LayerNorm / Dropout` → our NormLayers
  - `torch::nn::ReLU / Sigmoid / Tanh / GELU / Softmax` → our activations (wrapped in `ActivationsFnLayer`)
  - `torch::nn::MSELoss / CrossEntropyLoss / BCELoss` → our losses
  - `torch::optim::SGD / Adam / AdamW / RMSProp` → our optimizers
  - `torch::nn::functional::relu / sigmoid / softmax / ...` → our standalone activation callables
- [x] `torch::nn::Sequential` — thin wrapper building `ComputeGraph` from initializer list
- [x] The shim is **header-only** — zero link-time cost; only aliases, no new compiled symbols
- [x] Unit test: compile a simple 3-layer MLP using only `torch::` names against NNStudio

### Python pybind11 bindings — torch-compatible naming (design constraint, not retrofit)

> ⚠️ These bindings do not exist yet — but they **must be designed with torch naming** from day one.
> Do not expose `nnstudio.core.Dense`; expose `nnstudio.nn.Linear` with `Dense` as an alias.

- [x] Top-level module: `import nnstudio` → available sub-namespaces: `nnstudio.nn`, `nnstudio.optim`, `nnstudio.nn.functional`
- [x] `nnstudio.Tensor` matches `torch.Tensor` public surface: `.shape`, `.dtype`, `.device`, `.item()`, `.numpy()` (CPU copy), `__add__/__mul__/...`
- [x] `nnstudio.nn.Linear(in, out)` — default torch constructor signature
- [x] `nnstudio.nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)` — torch signature
- [x] `nnstudio.nn.Embedding(num_embeddings, embedding_dim)` — torch signature
- [x] `nnstudio.nn.MultiheadAttention(embed_dim, num_heads)` — torch signature
- [x] `nnstudio.nn.Sequential(*layers)` — torch constructor
- [x] `nnstudio.nn.ReLU / Sigmoid / Tanh / GELU / Softmax / Dropout / BatchNorm1d / LayerNorm` — torch naming
- [x] `nnstudio.nn.MSELoss / CrossEntropyLoss / BCEWithLogitsLoss` — torch naming
- [x] `nnstudio.optim.SGD / Adam / AdamW / RMSProp` — torch constructor signatures (params, lr, weight_decay, ...)
- [x] `nnstudio.nn.functional.relu / sigmoid / softmax / gelu / dropout` — functional API
- [x] `nnstudio.torch_compat` re-export: `import nnstudio.torch_compat as torch` works as drop-in

### Python Keras-compatible aliases (additive, thin wrapper)

- [x] `nnstudio.keras.layers.Dense / Conv2D / Embedding / LSTM / MultiHeadAttention / BatchNormalization / Dropout / LayerNormalization`
- [x] `nnstudio.keras.losses.MeanSquaredError / CategoricalCrossentropy / BinaryCrossentropy`
- [x] `nnstudio.keras.optimizers.SGD / Adam / AdamW / RMSprop`
- [x] `nnstudio.keras.Model.compile(optimizer, loss, metrics)` — wraps training loop
- [x] `nnstudio.keras.Model.fit(x, y, epochs, batch_size, validation_data, callbacks)` — pure-Python training loop
- [x] `nnstudio.keras.Model.predict(x)` — single forward pass, no grad
- [x] `nnstudio.keras.callbacks.EarlyStopping / ModelCheckpoint` — pure-Python callback system

### Compatibility warning system

- [x] `CompatibilityChecker` — static analysis pass over `ComputeGraph` nodes:
  - Classifies each node as: `standard_torch | standard_keras | nnstudio_extension`
  - Reports list of non-standard ops used in the project
- [ ] CLI: `nnstudio-sign verify --compat=torch` — exits non-zero if any extension ops present
- [ ] UI: yellow warning banner in Layer Inspector when an NNStudio-extension layer is used
- [ ] `.nns` project file: `"compat_level": "torch_standard" | "keras_standard" | "nnstudio_extended"` field
- [ ] Export dialog: blocks ONNX-standard export if extension ops present; offers "export with custom ops sidecar" alternative

### ONNX export alignment

- [x] Map every torch-compatible layer to a standard ONNX op (no custom domain required):
  - `Linear` → `Gemm`, `Conv2d` → `Conv`, `Embedding` → `Gather`, `LayerNorm` → `LayerNormalization`,
    `BatchNorm1d` → `BatchNormalization`, `ReLU/Sigmoid/Tanh/GELU/Softmax` → standard ONNX ops
- [x] NNStudio extension layers: register under `com.nnstudio.*` custom op domain; export with sidecar `.onnx_ops.dll`
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

- [x] `nnstudio::layers::Dense` → `nnstudio::builtin::layers::Dense`
- [x] `nnstudio::activations::*` (ReLU, Sigmoid, Tanh, Softmax, GELU, …) → `nnstudio::builtin::layers::*` *(activations are layers)*
- [x] `nnstudio::losses::*` (MSE, CrossEntropy, …) → `nnstudio::builtin::losses::*`
- [x] `nnstudio::optimizers::*` (SGD, Adam, AdamW) → `nnstudio::builtin::optimizers::*`
- [x] `CpuBackend` in `nnstudio::` → `nnstudio::builtin::backends::CpuBackend`
- [x] `Trainer` internals → `nnstudio::internal::training::*`

### Folder renames (to match namespaces)

- [ ] `nnstudio/core/include/nnstudio/core/` → split into `api/` (Tier 1 public) and `internal/` (Tier 2)
  <!-- DEFERRED: 60+ #include <core/...> references across src, tests, python-bridge. Scope > 1 day.
       Do as Phase 2 SDK prep step 0 (before nnstudio_plugin.h is written), so plugin authors
       see the correct include paths from day one. Namespace migration is already done. -->
- [x] `nnstudio/core/` layers/activations/losses/optimizers/backends → `nnstudio/builtin/`
- [x] Update all `#include` paths in source files and tests

### Verification

- [x] `cmake --build && ctest` — all 16 tests still pass after migration
- [x] No `nnstudio::layers::`, `nnstudio::activations::`, `nnstudio::losses::`, `nnstudio::optimizers::` appear outside of a `using` alias (grep check)
- [x] Only `nnstudio::` (Tier 1 interfaces + data types) and `nnstudio::builtin::` remain after rename

---

## Phase 2 — Plugin SDK

### C ABI contract (`nnstudio/plugin-api/`)
- [x] `nnstudio_plugin.h` — C-linkage structs/function pointers; no C++ in public ABI
- [x] `PluginDescriptor` — name, version, type, factory functions, capabilities
- [x] Plugin types: `LAYER | TOKENIZER | OPTIMIZER | BACKEND | UI_PANEL | TRUST_UPDATE`
- [x] `LayerPlugin` interface — custom `forward()`/`backward()`, operator registration
- [x] `TokenizerPlugin` interface — `encode()`, `decode()`, vocab management
- [x] `UIPlugin` interface — QML component path, property declarations

### Plugin manifests
- [x] `plugin.manifest.json` schema — name, version, author, license, type, capabilities, binary hash, detached signature path
- [x] `TUP` (Trust Update Package) manifest schema — type: `TRUST_UPDATE`, timestamp, cert add/revoke lists
- [x] Schema validation library (C++ + Python)

### Trust system (`nnstudio/plugin-api/trust/`)
- [x] `TrustStore` class — manages `<app_data>/truststore/` (roots/, intermediates/, crls/, history/)
- [x] First-run seed from embedded `seed_roots/root_ca.pem`
- [x] Append-only `history/` audit log
- [x] Atomic TUP application (write temp → verify → rename)
- [x] `TrustVerifier` class — X.509 chain verification, CRL/OCSP check, offline CRL cache fallback
- [x] `TrustUpdateHandler` — validates and applies TUP; compiled into core, NOT a loadable plugin
- [x] `PluginLoader` — wraps `QPluginLoader` + Python importer; runs `TrustVerifier` before any code executes
- [x] Root CA public key embedded as `seed_roots/root_ca.pem` (published openly)
- [x] TUP validation rules: signature valid, timestamp not replayed, no self-referential removal, user confirmation modal required

### `nnstudio-sign` CLI tool
- [x] `keygen` subcommand — generate plugin key pair + CSR
- [x] `sign` subcommand — sign plugin binary + manifest using issued cert
- [x] `verify` subcommand — verify plugin signature offline
- [ ] `submit` subcommand — submit CSR to registry for community/commercial cert
- [x] `create-tup` subcommand — build and sign a Trust Update Package
- [x] `issue-enterprise-ca` subcommand — project owner issues Enterprise Intermediate CA cert (admin only)

### pybind11 bridge (`nnstudio/python-bridge/`)
- [x] Python module `nnstudio` exposing: `Tensor`, all `Layer` subclasses, `ComputeGraph`, `Trainer`, `BackendRegistry`
- [x] `runners/` sub-package: Python-side runner clients mirroring `nnstudio/deployment/`
- [x] `pyproject.toml` for installable Python package
- [ ] Wheel build in CI

### Plugin scaffolds (`nnstudio/plugin-api/templates/`)
- [x] `cpp/` — CMakeLists.txt + `.h`/`.cpp` template (layer type; other types follow same pattern)
- [x] `python/` — `pyproject.toml` + `.py` module template (layer type; other types follow same pattern)
- [x] Both templates include `plugin.manifest.json` generator script (`generate_manifest.py`)
- [x] `README.md` per template explaining the plugin type

### Built-in reference plugins (`nnstudio/plugins/`)
- [x] BPE tokenizer plugin (C++ + Python, from KB A06) — C++ ✓ (319-token vocab, full encode/decode C ABI); Python binding deferred to Phase 3
- [ ] FAISS vector index plugin (C++ + Python) — deferred to Phase 3.5
- [x] Example custom activation plugin (demonstrates both C++ and Python plugin path) — C++ ✓ (Swish/SiLU forward+backward); Python binding deferred to Phase 3

### Phase 2 milestone
- [x] `PluginLoader` successfully loads BPE tokenizer plugin at runtime — validated via 17 whitebox C ABI tests (170/170 green)
- [ ] Unverified plugin triggers modal warning every session — requires Phase 3 UI
- [ ] TUP can be applied and is logged as immutable audit entry — trust disabled (`NN_ENABLE_TRUST=OFF`); integration test deferred

---

## Phase 3 — Qt 6 QML Studio UI

> ⚠️ **DESIGN PRINCIPLE — resolve this before writing a single line of UI code**
>
> ### UI Reentrant Idempotency
>
> The NNStudio UI state machine must be designed around a dominant **`ReadyToEdit`**
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
> **(b) Native NNStudio / torch-compat source code** — the source file
> (`model.py` or `model.cpp`) *is* the canonical representation.  The visual canvas
> is a live rendering of the parsed AST.  Export to other frameworks is a well-defined
> code-to-code translation.
>
> **(c) Multi-framework native editing** — the user can choose which framework language
> is "live" (NNStudio engine, PyTorch, Keras …) and the editor hides/shows features
> that are not expressible in the currently selected framework.  Switching framework
> re-parses the same file under a different grammar.
>
> **Recommendation: (b) — native code as source of truth.**
>
> Rationale: NNStudio already has a `torch_compat.h` C++ shim and a
> `nnstudio.torch_compat` Python module whose public API is *syntactically identical*
> to PyTorch.  This means "NNStudio engine code" and "torch-compatible code" are the
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
> **Option II — Structured comments** — `# @nnstudio {"x":120,"y":80}` comments
> embedded directly above each layer construction call in the source file.
>
> **Recommendation: Option I — sidecar inside the `.nnsx` bundle.**
>
> Rationale: source code stays clean and runnable outside NNStudio without any
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
>    named query patterns covering all recognized NNStudio/torch constructs.
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
>   (`nnstudio/app/grammar/nnstudio_torch.scm` is the suggested path)
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
> | 1 | Add CPU-only `KANLayer` + `SIRENLayer`; mark `nnstudio_extension` in `CompatibilityChecker`; expose `torch::nn::KANLinear` alias in `torch_compat.h` | Phase 4 | Low — isolated, off by default |
> | 2 | Add optional `IBackend::applyFunctor1D` (backend minor-version bump); GPU backends may fuse custom kernels | Phase 5 | Medium — vtable change, all backend plugins must update |
>
> This preserves the **didactic aim**: `ParametricLayer<LinearFn>` reads as a textbook
> definition of a parametric affine layer. KAN/SIREN variants then demonstrate
> exactly what changes when you swap the synapse function — one template parameter,
> everything else identical. NNStudio becomes both technically state-of-the-art *and*
> maximally transparent about what the difference actually is.
>
> **Decision gate items:**
> - [ ] Confirm Phase 4 step 0: refactor `Dense` → `ParametricLayer<LinearFn>` (pure,
>   no behaviour change) — record decision in `ARCHITECTURE.md §ADR-034`
> - [ ] Confirm Phase 4 step 1: add experimental `KANLayer` (CPU-only, `nnstudio_extension`
>   scope; `torch::nn::KANLinear` alias in `torch_compat.h`)
> - [ ] Confirm backend vtable extension strategy for GPU-accelerated functor dispatch
>   (step 2; requires IBackend minor-version bump)
> - [ ] Design `NNSC` `FN` section format for non-linear synapse hyperparameter
>   persistence (spline order, knot vector, frequency initialisation)
> - [ ] Track research: re-evaluate KAN FLOP-efficiency claim on standard ML tasks
>   at Phase 4 start; adjust step 1 priority accordingly

### App shell (`nnstudio/app/`)
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

#### Parser subsystem (`nnstudio/app/parser/`)
- [ ] Integrate **tree-sitter** as a static C library (`third-party-deps/tree-sitter/`)
- [ ] Bundle `tree-sitter-python` grammar (primary) and `tree-sitter-cpp` grammar (secondary)
- [ ] Implement `nnstudio_torch.scm` query file — named captures for all recognized
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
- [ ] NNStudio launches on Windows/macOS/Linux
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
      `gdb` (from MSYS2 MinGW); `cwd` = `nnstudio/`; env: `PATH` includes Qt bin + MSYS2 bin;
      `stopAtEntry=false`
- [ ] `Engine tests — filter (GDB/MinGW)` — same as above with `"args": ["--gtest_filter=${input:gtestFilter}"]`
      and a `${input}` prompt so the user types a filter expression without editing the file
- [ ] `NNStudio App (GDB/MinGW)` — launch the Qt app executable; pre-launch task = build; working dir = project root
- [ ] `NNStudio App (no GPU)` — same with `"env": {"NN_BACKEND": "cpu"}` override
- [ ] `NNStudio App (CUDA backend)` — same with `"env": {"NN_BACKEND": "cuda"}`
- [ ] `Engine tests (LLDB/Clang)` — alternate clang/LLDB profile using `CodeLLDB` extension
- [ ] `Engine tests (MSVC / cppvsdbg)` — Windows-only; requires `ms-vscode.cpptools` debugger;
      preset `engine-vs` (MSVC x64 generator)
- [ ] `NNStudio App (MSVC / cppvsdbg)` — app launch via MSVC debug adapter

**CMake presets**
- [ ] `engine-vs` preset — Visual Studio 17 generator, MSVC x64 (mirrors `engine-ninja` logic, different generator)
- [ ] `engine-clang-ninja` preset — Ninja + clang++ (MSYS2 LLVM package), same flags minus `-fno-exceptions` for
      libcxx compatibility check
- [ ] `app-ninja` preset — includes `nnstudio/app/` target; `CMAKE_PREFIX_PATH` pointing to Qt installation
- [ ] `app-vs` preset — same with Visual Studio generator

**Extensions and toolchain validation guide**
- [ ] `docs/VSCODE-DEV-SETUP.md` written and kept current — full guided tour:
  compiler selection, kit switching in CMake Tools, IntelliSense modes,
  debugging GDB vs LLDB vs cppvsdbg, `.env` file for Qt `PATH`, GTest adapter
  integration, Doxygen task, remote SSH target for GPU build/test (Phase 5+)

### Visual Studio (Windows, secondary)

- [ ] `engine-vs` CMake preset (above) enables "Open Folder" workflow in VS 2022 — no `.sln` needed
- [ ] `nnstudio.natvis` — NatVis visualiser file so VS debugger pretty-prints `Tensor`, `Shape`,
      `Result<T>`, `Parameter` in the Autos/Watch/Locals pane instead of raw byte dumps
- [ ] `nnstudio.props` — shared property sheet: Qt install path, MSYS2 path, include dirs;
      developers override via `nnstudio.user.props` (gitignored) for local paths
- [ ] Verify: right-click `CMakeLists.txt` → "Add to View" → CMake Targets appear; build + test
      via VS Test Explorer (GTest adapter)
- [ ] Document the one known friction point: MSYS2 DLLs not in VS `PATH` → workaround in
      `docs/VSCODE-DEV-SETUP.md` §Visual Studio

### Qt Creator / Qt Design Studio

- [ ] Confirm `CMakeLists.txt` opens cleanly in Qt Creator 13+ via "Open as CMake project"
- [ ] Kit configuration: MinGW 15.2.0 (MSYS2) + Qt 6.10.1; document kit JSON in `docs/VSCODE-DEV-SETUP.md`
- [ ] `.qmlproject` stub for `nnstudio/app/ui/` — allows pure QML/UI work in **Qt Design Studio**
      without building the C++ engine; mock `ModelController` QML singleton provides stub data
- [ ] Verify QML live-preview works in Qt Design Studio 4+ for all `.qml` files in `ui/`
- [ ] Document: Qt Creator run configurations for `test-core` and `NNStudio app`

### Phase 3.5 milestone
- [ ] A contributor on a fresh Windows machine can clone → open in VS Code → press F5 → debugger
      hits a breakpoint in `Dense::forward()` in under 15 minutes, following `docs/VSCODE-DEV-SETUP.md`
- [ ] Same contributor can run and filter GTest cases directly from the VS Code Testing sidebar
- [ ] `nnstudio.natvis` makes `Tensor` readable in the VS 2022 debugger Locals pane
- [ ] Qt Design Studio opens the `ui/` QML project and live-previews the model editor panel stub

---

## Phase 4 — Full Pipeline

### Input adapters (`nnstudio/pipeline/input/`)
- [ ] Text (UTF-8 string)
- [ ] Image (via Qt image loading → tensor)
- [ ] Audio (WAV/FLAC → waveform tensor via libsndfile or similar)
- [ ] Structured data (CSV → tensor, Apache Arrow pass-through)
- [ ] MCP message format adapter
- [ ] Python + C++ implementation for each

### Tokenization chain (`nnstudio/pipeline/tokenize/`)
- [ ] `TokenizationChain` — ordered list of `TokenizerPlugin`s
- [ ] Produces: token-ID tensor, attention mask, position IDs
- [ ] Cache layer for repeated vocabulary lookups
- [ ] Python + C++ implementation

### Context / RAG stage (`nnstudio/pipeline/context/`)
- [ ] `ContextStage` — optional; takes query tensor, returns retrieved chunk tensors
- [ ] Embedded FAISS connector (no server needed)
- [ ] Qdrant HTTP connector
- [ ] Chroma HTTP connector
- [ ] Reranker interface (cross-encoder plugin slot)
- [ ] Python + C++ implementation

### Execution stage (`nnstudio/pipeline/execution/`)
- [ ] `ExecutionStage` — runs one `ComputeGraph` or a chain of chained graphs
- [ ] Output of graph A → input of graph B wiring
- [ ] Streaming output support (token-by-token)
- [ ] Python + C++ implementation

### Output adapters (`nnstudio/pipeline/output/`)
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

### Export wizards (`nnstudio/deployment/export/`)
- [ ] ONNX export wizard (uses `OnnxIO::export()`)
- [ ] TorchScript export (via Python bridge + `torch.jit.script`)
- [ ] TFLite export stub
- [ ] `.nnsr` runner bundle builder: zip of weights + `manifest.json` + `runner.py` + `runner.cpp` + precompiled `.so`/`.dll`
- [ ] **C++ library export** — generates self-contained, NNStudio-free compilable package:
  `model.h` + `model.cpp` + `CMakeLists.txt` + `README.md`; output uses only stdlib +
  Eigen (header-only); no `nnstudio-core` link dependency in the exported build
- [ ] **Python package export** — generates pip-installable package: `pyproject.toml` +
  `model.py` (uses real PyTorch or raw NumPy, no `nnstudio` import); `model.py` is the
  round-trip of the canonical source with the `nnstudio.torch_compat` import replaced by
  `import torch`
- [ ] **"Compile to binary" one-click** — runs CMake + Ninja on the exported C++ package;
  output is a `.dll`/`.so` or standalone executable; build stdout/stderr shown in the
  integrated terminal panel; artifact path opened in OS file manager on success
- [ ] **Multi-language code preview** — all generated code (export wizard, architecture
  presets, sandbox snippets) shown with a `Python | C++` tab switcher; chosen language
  persisted per-user in settings; both tabs always kept in sync with the model graph

### Runner connectors (`nnstudio/deployment/runners/`) — C++ + Python each
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

### `nnstudio-cli` command-line tool (`nnstudio/app/cli/`)

A headless CLI companion to the Qt app. Same C++ engine, no Qt dependency in
the CLI binary itself — links only `nnstudio-core` + `nnstudio-builtin`.
Useful for CI pipelines, scripting, and SSH sessions on remote pods.

- [ ] `nnstudio-cli train <project.nnsp> [--preset <name>] [--epochs N] [--remote <service>]`
  — run training locally or submit to a remote service; streams loss to stdout
- [ ] `nnstudio-cli export <project.nnsp> --format <onnx|nns|nnsr> -o <out>`
  — export model; no UI required
- [ ] `nnstudio-cli run <model.nnsx> --input <file|stdin> --output <file|stdout>`
  — single inference pass; reads NMID for I/O format
- [ ] `nnstudio-cli remote list` — list active pods on configured services
- [ ] `nnstudio-cli remote launch --service vast.ai --gpu RTX4090`
- [ ] `nnstudio-cli remote stop <pod-id>`
- [ ] Common flags: `--backend cpu|cuda|remote`, `--config <settings.json>`,
      `--api-key <env-var-name>` (never accept key as positional arg — shell history)
- [ ] Machine-readable output: `--json` flag on all commands for scripting
- [ ] Built as a separate CMake target `nnstudio-cli`; included in `app-ninja` and
      `app-vs` presets; single static binary goal on Linux/macOS (musl / libc++)

### Plugin Registry server spec (`nnstudio/deployment/registry/`)
- [ ] REST API spec (OpenAPI 3.1): `POST /plugin/register`, `GET /plugin/{id}`, `POST /enterprise/ca/request`, `GET /crl`
- [ ] Reference server implementation (Python FastAPI)
- [ ] `nnstudio-sign submit` integration

### `.nnsp` project file spec (finalise)
- [ ] Adopt ZIP container format (ECMA OPC principle — like .docx/.xlsx)
- [ ] `project.json`: project metadata, NNStudio version, author, signature/encryption header
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
> NNStudio models built in the model editor appear as first-class components in the
> ecosystem canvas — the NN is one node among others in the full AI system picture.

### Component library (`nnstudio/ecosystem/components/`)

Each component type has: a C++ descriptor (name, category, ports, config schema),
a QML node card, a KB reference, and a configuration panel.

#### LLM Runners
- [ ] **Ollama** — local; config: model tag, host:port, context length, GPU layers, NUMA pinning
- [ ] **llama.cpp server** — local; config: GGUF path, threads, context, quantisation level
- [ ] **vLLM** — local / pod; config: model HF ID, tensor-parallel degree, max batch
- [ ] **LiteLLM proxy** — config: upstream provider list, routing policy, rate limits
- [ ] **HuggingFace TGI** — local/container; config: HF model ID, dtype, sharding
- [ ] **OpenAI-compatible endpoint** — generic; config: base URL, API key env-var, model name
- [ ] **NNStudio model (local runner)** — inline reference to any model built in Phase 3;
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
- [ ] **Custom embedding** — NNStudio model from the model editor functioning as an encoder

### Ecosystem canvas (`ui/EcosystemCanvas.qml`)
- [ ] Separate canvas from the model editor — full panel occupying the `Ecosystem` layout preset
- [ ] Component palette (toolbox) docked to left; grouped by category (Runners / Stores /
  Agents / Ingestion / Connectors / NNStudio Models)
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
  - NNStudio model as a component (Studio-trained NN acts as a classifier inside the pipeline)
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
  NNStudio-model components mapped to a `nnstudio-runner` container
- [ ] **Kubernetes Helm values** — `values.yaml` for a generic AI-stack chart; one entry per
  component; secret refs replace inline key values
- [ ] **Ollama Modelfile + run script** — `Modelfile` + `run_ollama.sh`/`.ps1` for the
  configured model tag and parameters; one file per Ollama node in the design
- [ ] **`.env` template** — all required environment variables listed with placeholder values
  and comments; sensitive vars marked `# REQUIRED — set in CI/keychain, do not commit`
- [ ] **Python orchestration script** — `run_pipeline.py` matching the designed architecture;
  uses real library imports (LangChain, requests, qdrant-client, etc.);  no NNStudio import;
  comments explain each section; NNStudio model nodes emitted as ONNX Runtime inference calls
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

## Phase 6 — Quantum Backend

### Quantum backend (`nnstudio/backends/quantum/`)
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
- [ ] Plugin signing step in release pipeline using `nnstudio-sign`

### Performance & optimization
- [ ] CUDA backend benchmarks (matmul, conv) vs CPU baseline
- [ ] Memory layout profiling (cache-friendly tensor strides)
- [ ] Training throughput: tokens/sec, samples/sec metrics exposed via `Trainer` callbacks
- [ ] ONNX model profiling panel in UI

### Security
- [ ] TrustStore permissions hardened on all three platforms
- [ ] `nnstudio-runner` sidecar sandboxing (process isolation, limited syscalls)
- [ ] No eval / dynamic code execution in the Qt app process without sandbox
- [ ] OpenSSL kept up to date; SBOM generated for each release

---

## External Compute Services

NNStudio must support offloading computation to remote GPU services at multiple architectural levels.
This is critical during development (no local CUDA GPU available) and for production scale-out.

### Where external compute surfaces in the NNStudio UI and CLI

| Integration point | Where it appears | Phase |
|---|---|---|
| **Backend selector** (CPU / CUDA / Remote-gRPC) | Training Dashboard panel — backend dropdown | Phase 3 |
| **Remote training panel** (pod launch, job monitor, cost estimate, weight import) | Training Dashboard panel — shown when Remote mode active | Phase 3 |
| **Remote pod manager** (Vast.ai / RunPod / LambdaLabs pod lifecycle) | Deployment panel | Phase 5 |
| **Managed job submitter** (SageMaker / Vertex AI / Azure ML job dispatch) | Deployment panel | Phase 5 |
| **`nnstudio-cli remote *`** (pod list / launch / stop from terminal) | `nnstudio-cli` command-line tool | Phase 5 |
| **`nnstudio-cli train --remote`** (headless training submission) | `nnstudio-cli` command-line tool | Phase 5 |
| **Runner health monitor** (Triton / KServe / TF Serving inference endpoints) | Deployment panel — health/latency/throughput | Phase 5 |

### Integration points in NNStudio architecture

| NNStudio level | What integrates | What it enables |
|---|---|---|
| **Backend level** | New `IBackend` implementation (`RemoteBackend`) | Individual tensor ops (`matmul`, etc.) forwarded via gRPC to a remote CUDA worker; training loop runs locally |
| **Trainer level** | `RemoteTrainer` / job dispatch | Entire training job serialised and sent to a remote service; local machine only submits + receives trained weights |
| **Inference level** | Existing Triton / KServe / TF Serving connectors | Already in TODO Phase 5; model deployed remotely, NNStudio queries it |

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
5. **CoreWeave / AWS** — only once NNStudio's remote training pipeline is proven at smaller scale.

### TODOs — architecture items to add
- [ ] Design `RemoteBackend : IBackend` — forwards individual tensor ops to a remote CUDA worker via gRPC (Phase 1 extension)
- [ ] Design `RemoteTrainer` / job-dispatch adapter — serialises model + dataset, submits to SageMaker/Vertex/RunPod job API, polls for completion, imports weights (Phase 5 extension)
- [ ] `DEPLOYMENT.md` section: external compute service integration guide (which service maps to which NNStudio integration point)
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
  NNStudio becomes a transparent didactic bridge from standard linear layers
  to cutting-edge research architectures with a single template parameter change.

- [ ] **Primitivistic educational `nnprimitive` engine** — a small, fully isolated
  sub-project: its own CMake target, include tree (`include/nnprimitive/`), source
  tree (`src/nnprimitive/`), and C++ namespace `nnprimitive`. No dependency on
  `nnstudio-core`, no `Tensor`, no `IBackend`, no BLAS, no template metaprogramming.
  Implements the identical NN concepts (neuron, layer, forward pass, backward pass,
  gradient descent) as explicit `for`-loops over `std::vector<float>` — exactly
  how a motivated beginner encountering NNs for the first time would write them.
  **Purpose: purely educational.** Acts as the missing stepping-stone between
  "NN mathematical theory" and the matrix-optimised production implementation in
  `nnstudio`: a learner who finds the leap from textbook equations to `Dense::forward()`
  too large can read `nnprimitive` first, then see the same computation generalised
  and optimised in `nnstudio`. A dedicated blueprints.md chapter (to be written)
  will explain every generalisation step and why it was made.
  **Low priority** — implement only once the main engine and Phase 3 UI are stable.
