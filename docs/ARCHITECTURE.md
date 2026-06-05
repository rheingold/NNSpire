# NNSpire — Architecture Whitepaper

**Version**: 0.1 (Phase 0 — design)  
**Date**: 2026-03-31  
**Status**: DRAFT — pre-implementation baseline

---

## 1. Overview

NNSpire is a multiplatform (Qt 6) neural-network design, training, deployment, and learning workbench. It is built "inside out": the core mathematical engine is implemented as a standalone C++ library (`NNSpire-core`), the plugin system and trust chain are layered on top of it, and the Qt/QML user interface surfaces the engine's capabilities without owning any computation logic.

The guiding constraint throughout is **dual-language**: every computable artefact (plugin, export, runner client, sample) exists in both C++17 and Python 3.10+ forms, with identical behaviour. The C++ API is defined first; pybind11 provides the Python mirror.

---

## 2. Component diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  NNSpire Qt 6 application  (GPL v3)                                        │
│  ┌────────────────┐  ┌──────────────────┐  ┌───────────────────────────┐   │
│  │  QML UI panels │  │  Qt controllers  │  │  Dependency Manager / Wiz │   │
│  │  (ModelEditor, │  │  (ModelCtrl,     │  │  (first-run, plugin mgmt, │   │
│  │   Training,    │◄─┤   TrainingCtrl,  │  │   KB help, wizard mode)   │   │
│  │   Pipeline,    │  │   BackendCtrl,   │  └───────────────────────────┘   │
│  │   Deployment,  │  │   PluginCtrl,    │                                   │
│  │   Quantum)     │  │   HelpCtrl)      │                                   │
│  └────────────────┘  └───────┬──────────┘                                   │
└──────────────────────────────┼──────────────────────────────────────────────┘
                                │  C++ API calls
┌──────────────────────────────▼──────────────────────────────────────────────┐
│  Plugin SDK  (LGPL v3 headers — NNSpire-plugin-api)                        │
│  ┌─────────────────────┐  ┌────────────────────────────────────────────┐   │
│  │  PluginLoader       │  │  Trust subsystem                           │   │
│  │  (QPluginLoader +   │  │  TrustStore / TrustVerifier /              │   │
│  │   Python importer)  │  │  TrustUpdateHandler (compiled-in)          │   │
│  └─────────┬───────────┘  └────────────────────────────────────────────┘   │
└────────────┼────────────────────────────────────────────────────────────────┘
             │  dynamic load (.dll / .so / .pyd)
┌────────────▼────────────────────────────────────────────────────────────────┐
│  NNSpire-core  (LGPL v3 static library)                                    │
│                                                                              │
│  tensor/ ──── layers/ ──── activations/ ──── losses/ ──── optimizers/      │
│  graph/ (ComputeGraph, autograd) ──── training/ (Trainer, callbacks)        │
│  formats/ (OnnxIO, NNSFormat) ──── features/ (FeatureFlags)                 │
└────────────┬────────────────────────────────────────────────────────────────┘
             │  IBackend dispatch
┌────────────▼────────────────────────────────────────────────────────────────┐
│  Backends (each a separately loaded shared library)                          │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐                    │
│  │ CpuBackend │  │ CudaBackend  │  │ QuantumBackend   │                    │
│  │ (Eigen)    │  │ (cuBLAS/DNN) │  │ (Qiskit bridge)  │                    │
│  └────────────┘  └──────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────────────────┘
             │  pybind11
┌────────────▼────────────────────────────────────────────────────────────────┐
│  Python bridge  (NNSpire Python package)                                    │
│  Tensor, Layer, ComputeGraph, Trainer, BackendRegistry, runners/            │
└─────────────────────────────────────────────────────────────────────────────┘
             │  plugins (C++ .dll/.so  +  Python .pyd/.so)
┌────────────▼────────────────────────────────────────────────────────────────┐
│  Plugins: BPE tokenizer | FAISS index | custom activations | UI panels …    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core engine (`NNSpire-core`)

### 3.1 Tensor

`Tensor<T>` is the fundamental data container. It owns a type-erased backing buffer (shared pointer to heap allocation) and carries:
- `shape` — `std::vector<int64_t>`
- `strides` — row-major by default; can be overridden for column-major or non-contiguous views
- `dtype` — `DType { FLOAT32, FLOAT16, INT8, INT32, BOOL }`
- `device` — `Device { CPU, CUDA, QUANTUM }` (tag only; actual memory managed by the active backend)
- `requires_grad` — bool; when true, the ComputeGraph records this tensor's creation op

`Tensor` operations are **dispatched through `IBackend`** — no compute code lives in the Tensor class itself. This is what makes backend swapping transparent.

```cpp
// C++ example — @kb: ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
Tensor<float> a({2, 3}, Device::CPU);
Tensor<float> b({3, 4}, Device::CPU);
auto c = BackendRegistry::active().matmul(a, b);  // shape {2,4}
```

```python
# Python equivalent
import NNSpire as nn
a = nn.Tensor([2, 3], dtype=nn.float32, device='cpu')
b = nn.Tensor([3, 4], dtype=nn.float32, device='cpu')
c = nn.matmul(a, b)  # shape [2, 4]
```

### 3.2 Layer

`Layer` is the abstract base for all network building blocks.

```cpp
class Layer {
public:
    virtual Tensor<float> forward(const Tensor<float>& input) = 0;
    virtual Tensor<float> backward(const Tensor<float>& grad_output) = 0;
    virtual std::vector<Tensor<float>*> parameters() = 0;
    virtual void serialize(NNSWriter&) const = 0;
    virtual std::string_view docRef() const = 0;   // KB path#anchor
    virtual ~Layer() = default;
};
```

Every layer carries a `docRef()` returning the KB anchor for its mathematical definition. The UI shows a "?" button that opens the KB at that anchor.

### 3.3 ComputeGraph

`ComputeGraph` records the forward pass as a directed acyclic graph of `OpNode`s. Each `OpNode` holds:
- The operation type (enum matching ONNX operator names where applicable)
- Input tensor references
- Output tensor reference
- A `backward_fn` — the gradient function for autograd

During `backward()`, the graph is traversed in reverse topological order, accumulating gradients. The graph structure is also exported as JSON for the UI layer editor to render a topology diagram.

### 3.4 Trainer

`Trainer` runs the training loop:

```
for epoch in 0..num_epochs:
    for batch in dataset:
        optimizer.zero_grad()
        output = graph.forward(batch.input)
        loss = loss_fn(output, batch.target)
        graph.backward(loss)
        optimizer.step()
        callbacks.on_batch_end(loss, metrics)
    callbacks.on_epoch_end(epoch, loss, metrics)
    if early_stopping.should_stop(): break
checkpoint.save(graph, optimizer, epoch)
```

Callbacks relay training progress to the Qt controller via a signal queue (thread-safe). The training loop runs on a `QThread` worker; the UI thread receives progress signals.

### 3.5 Formats

#### The two-format family

NNSpire uses two distinct ZIP-based formats, both following the Open Packaging Convention
(same principle as `.docx`/`.xlsx`/`.odt` — ECMA-376). All large parts in either format
can be **embedded** (inside the ZIP) or **referenced as sidecars** (relative or absolute
external paths). Neither format ever forces you to pack everything in.

```
Parts reference scheme (in any manifest.json):
  "graph":     "embedded:graph.onnx"         ← inside the ZIP
  "optimizer": "file:../weights/adam.bin"     ← external relative path
  "vocab":     "file:/data/models/vocab.bin"  ← absolute path (local large store)
```

---

#### `.nnsp` — NNSpire Project (always small-to-medium, the "source code")

The project file is the primary unit of work in the Studio. It contains everything
necessary to *describe and recreate* networks, but the actual trained weights and
optimizer state are **optional** parts — they may be embedded, sidecar-referenced, or
absent entirely ("blueprint-only" / homebrew shipping mode — see below).

A single project can contain **multiple network variants** (different architectures,
hyperparameter settings, scale comparisons) and a **pipeline interconnect** describing
how those networks wire together (e.g. an LLM pipeline with encoder, decoder, retriever).

```
project.nnsp  (ZIP — always the small/shippable artefact)
│
├── project.json              ← project metadata, NNSpire version, author,
│                               signature/encryption header, creation date
│
├── networks/                 ← one subfolder per network variant
│   ├── variant_a/
│   │   ├── blueprint.json    ← layer stack, sizes, activations, loss fn,
│   │   │                       optimizer config, hyperparameters
│   │   ├── manifest.json     ← parts refs: graph, optimizer (embedded or sidecar)
│   │   ├── graph.onnx        ← OPTIONAL: embedded weights (absent = "untrained")
│   │   └── training/         ← OPTIONAL: optimizer state and progress
│   ├── variant_b/            ← different architecture for A/B comparison
│   │   └── blueprint.json
│   └── llm_pipeline/         ← full LLM — references sub-networks by name
│       └── blueprint.json
│
├── pipeline/
│   └── interconnect.json     ← how multiple networks wire together:
│                               nodes = network variants, edges = tensor flows,
│                               plugin slots, context/RAG stage positions
│
├── plugins/
│   └── refs.json             ← required plugins: name, version, hash, signature
│
├── analysis/              ← OPTIONAL semi-large data (see below)
│   ├── loss_curves.bin       ← metric histories from training runs
│   ├── cluster_viz.json      ← cluster analysis config, camera state, colour maps
│   └── weight_clusters.bin   ← OPTIONAL: actual cluster data (large, omit when shipping)
│
├── ui/
│   └── layout.json           ← panel arrangement, open tabs, zoom levels, split ratios
│
└── model_card.json           ← intended use, limits, bias disclosures (KB doc 04)
```

**"Instant homebrew NN" shipping mode**

When distributing a project *without* weights (e.g. designer ships to customer who trains
on their own data):

- All `blueprint.json` files are present — complete layer config, hyperparameters, pipeline
- `plugins/refs.json` present — customer knows what to install
- `model_card.json` present — intended use documented
- `project.json` carries designer's signature and optional encryption
- `graph.onnx` and `training/` are **absent from every variant**

NNSpire opens this and presents: *"Weights not present — run training to generate, or
import existing weights."* The large data is generated locally by the customer and stays
local. This is the source-code-vs-binaries analogy: `.nnsp` is the source, trained weights
are the compiled output.

**Semi-large optional data (`analysis/`)**

The `analysis/` folder holds data that is useful for inspection and visualisation but
not required for inference. It is typically only shipped when the full trained model is
also included:
- `loss_curves.bin` — per-epoch/per-step loss and metric histories (used by Training Dashboard)
- `cluster_viz.json` — cluster analysis configuration (which weights to cluster, display settings)
- `weight_clusters.bin` — actual cluster results (can be tens of MB for large nets; omit when shipping)

---

#### `.nnsx` — NNSpire Model Exchange (one network, can be large)

A self-contained single-network artefact. Can exist standalone or be embedded inside a
`.nnsp` project. An `.nnsx` file can reference its large parts as sidecars.

```
model.nnsx  (ZIP — rename to .zip to inspect freely)
│
├── manifest.json         ← parts directory, NNSpire_version, model_name, author
├── graph.onnx            ← valid standalone ONNX blob (embedded or sidecar ref)
├── training/
│   ├── adam.bin            ← optimizer state (embedded or sidecar ref)
│   └── progress.json       ← epoch, batch_within_epoch, last_loss, rng_state
├── tokenizer/config.json
├── plugins/refs.json
├── model_card.json
└── ui/layout.json        ← ignored by runners
```

The `.onnx` blob inside is spec-compliant and extractable as a standalone file:
`unzip model.nnsx graph.onnx`

---

#### Relationship between `.nnsp` and `.nnsx`

The relationship is intentionally **bidirectional** — neither strictly contains the other:

```
.nnsp  contains  .nnsx-equivalent data per variant   (project is the outer container)
.nnsx  can be referenced from .nnsp manifest          (exchange file is standalone)
.nnsx  can be imported into a .nnsp project           (data flows into project)
.nnsp  can export a variant as standalone .nnsx       (project produces exchange file)
```

**Up-references — `.nnsp` embedded inside `.nnsx`**

When a `.nnsp` is stored *inside* a `.nnsx` (e.g. as `project.nnsp` in the ZIP root),
the `.nnsp` is the UI/metadata helper and the weights already exist in the outer `.nnsx`.
In this case the `.nnsp` manifest entries must be able to point *upward* into the container
that holds them — they must NOT duplicate the large data:

```json
// networks/variant_a/manifest.json  (inside the .nnsp, which is inside the .nnsx)
{
  "parts": {
    "graph":     "container:../graph.onnx",        ← refers to graph.onnx in the outer .nnsx
    "optimizer": "container:../training/adam.bin"  ← refers to adam.bin in the outer .nnsx
  }
}
```

The `container:` scheme means: *"resolve this path relative to the root of the ZIP container
that directly holds this manifest, not relative to the .nnsp sub-archive."*

Resolution rules for part references:
| Scheme | Resolves relative to |
|---|---|
| `embedded:file.onnx` | Same ZIP archive as the manifest |
| `file:rel/path` | Directory of the outer ZIP file on disk |
| `file:/abs/path` | Absolute filesystem path |
| `container:../file` | Root of the immediately enclosing ZIP (up-reference) |

This means a `.nnsp` embedded in a `.nnsx` carries only `blueprint.json`, `ui/`, `plugins/`,
`model_card.json` — the human-readable and structural parts — while all large binary data
lives once in the outer `.nnsx` and is referenced, never duplicated.

---

#### Extension aliases

| Extension | Meaning |
|---|---|
| `.nnsp` | Project file (always ZIP, multiple networks, UI state, pipeline) |
| `.nnsx` | Model exchange file (single network, weights, optimizer state) |
| `.nns` | Context-dependent alias: Studio opens as `.nnsp`; runner treats as `.nnsx` |

---

#### Training checkpoint fields (inside `training/`)

`training/adam.bin` stores optimizer state. **Required for true training resume** —
without it, Adam loses accumulated momentum and must rebuild from scratch:

```
training/adam.bin  (binary, one record per named parameter):
  param_name[]     ← ties back to a tensor in graph.onnx by name
  m[]              ← running mean of gradients, same shape as W
                      (Adam's "which direction have we been going?")
  v[]              ← running mean of squared gradients, same shape as W
                      (Adam's "how volatile has this weight been?")
  t                ← global step counter (bias-correction denominator)

training/progress.json:
  epoch, batch_within_epoch, last_loss
  rng_state        ← optional; reproduces exact batch shuffle on resume
```

Raw gradients (`W.grad_`) are **intentionally not stored**. They are zeroed at the start
of every step — recomputing them costs one forward+backward pass on resume.

### 3.6 EvalTrace — step-by-step evaluation visualisation

`EvalTrace` is an **opt-in, zero-cost-when-off** instrumentation layer that lets the UI display what is happening inside each layer during a training step.

#### Data structures

```cpp
// NNSpire/core/include/NNSpire/core/EvalTrace.h

struct LayerTrace {
    std::string    layerId;       // matches Layer::name()
    Tensor<float>  input;         // snapshot of input tensor at forward time
    Tensor<float>  output;        // snapshot of output tensor at forward time
    Tensor<float>  grad_input;    // snapshot of dX after backward() (empty until backward)
    Tensor<float>  grad_output;   // snapshot of dL/dy received by this layer
    Tensor<float>  weight_delta;  // dW = grad of weights, useful for vanishing-gradient display
};

struct EvalTrace {
    std::vector<LayerTrace> layers;  // one entry per layer, in forward order
    uint64_t stepIndex;              // which training step produced this trace
};
```

#### How it hooks into the engine

```cpp
// ILayer interface (pseudo-code showing the optional parameter)
class ILayer {
public:
    virtual Tensor<float> forward (const Tensor<float>& x,
                                   EvalTrace* trace = nullptr) = 0;
    virtual Tensor<float> backward(const Tensor<float>& grad_out,
                                   EvalTrace* trace = nullptr) = 0;
};
```

When `trace == nullptr` (the normal case during bulk training), no snapshots are taken and no branches are entered. The parameter adds zero overhead beyond a pointer comparison.

`Trainer` exposes:

```cpp
void Trainer::setTraceMode(bool enabled);           // toggles trace allocation
void Trainer::setTraceCallback(
    std::function<void(const EvalTrace&)> cb);       // called after each traced step
```

The trace callback is invoked from the training worker thread; the Qt `TrainingController` wraps it in a `Qt::QueuedConnection` signal to safely deliver snapshot copies to the UI thread.

#### UI consumer responsibilities

| Panel | What it reads from `EvalTrace` |
|---|---|
| `WeightViewer.qml` | `LayerTrace::weight_delta` → heatmap flash; `LayerTrace::grad_input` magnitude → gradient bar |
| `NeuronViewer.qml` | `LayerTrace::output` values → node colour; `LayerTrace::input` → connection thickness |
| Step-by-step mode | Pauses between forward and backward by splitting `EvalTrace` delivery into two callbacks |

#### Performance contract

- Trace mode is **off by default** and only enabled when the user opens the NeuronViewer or requests step-by-step mode.
- Each snapshot is a **deep copy** of the tensor at that moment. For small demonstration networks (≤ ~100 neurons) this is negligible. The maximum network size guard in `NeuronViewer` (Phase 3) enforces that trace mode cannot be accidentally enabled on production-scale networks.
- `EvalTrace` objects are owned by the training worker and moved (not copied) into the signal queue; the UI thread owns them after delivery.

---

## 4. Backend abstraction

```cpp
struct IBackend {
    virtual Tensor<float> matmul(const Tensor<float>&, const Tensor<float>&) = 0;
    virtual Tensor<float> elementWise(Op, const Tensor<float>&, const Tensor<float>&) = 0;
    virtual void* alloc(size_t bytes, Device) = 0;
    virtual void free(void*, Device) = 0;
    virtual void sync() = 0;
    virtual std::string_view name() const = 0;
};
```

`BackendRegistry` holds named backends and a selected-active pointer. Backends are loaded as shared libraries at runtime:
- If CUDA is unavailable, `CudaBackend` simply is not loaded — no stub code runs.
- `QuantumBackend` always loads (it is a stub) but `execute()` raises a structured error until the Qiskit bridge is installed.

---

## 5. Threading model

| Thread | Owns |
|---|---|
| **Main (Qt event loop)** | All UI state, QML engine, controller objects |
| **Training worker** (`QThread`) | Trainer::run(), ComputeGraph forward/backward, IBackend compute |
| **Plugin loader** (one-shot worker) | TrustVerifier, `dlopen`/`QPluginLoader` — runs at startup and on plugin install |
| **Dependency checker** (one-shot worker) | Network checks, download + extraction of embeddable Python / MinGW |
| **Runner connector** (async worker per request) | gRPC/REST calls to Triton / TF Serving / KServe |

Synchronisation between training worker and UI thread is done exclusively through `Qt::QueuedConnection` signals — no shared mutable state is accessed without the queue. `Trainer::Callback` implementations post signals; they do not touch QML objects directly.

---

## 6. Process architecture

- **One main process** for the entire Studio.
- **`NNSpire-runner` sidecar** (opt-in): a separate lightweight process for crash-isolated inference of untrusted plugins. Communicates via local Unix socket / named pipe. Disabled by default; enabled via `Settings → Advanced → Sandbox untrusted plugins`.
- **No background services** — nothing survives app exit.
- **External runners** (Triton, TF Serving, KServe): Studio connects to them over gRPC/REST; Studio never launches or manages these processes, except the optional "launch local Triton Docker" helper which uses the Docker CLI.

---

## 7. Portability strategy

| Concern | Approach |
|---|---|
| Platform file paths | `QStandardPaths` throughout; no hardcoded paths |
| Settings in portable mode | `<exe_dir>/settings/` if writable; else `QStandardPaths::AppDataLocation` |
| DLL loading | `QPluginLoader` (cross-platform dlopen abstraction) |
| Qt runtime distribution | `windeployqt` / `macdeployqt` / `linuxdeployqt` |
| Linux distribution | AppImage (glibc ≥ 2.31 baseline) |
| Qt version shim | `NNSpire/app/qt_version_helpers.h` wraps any Qt 6-only APIs |
| Old macOS / Linux | Qt 5.15 LTS CMake config (`-DNN_QT_VERSION=5`) |
| Compiler portability | C++17 only; no compiler-specific extensions except `__has_builtin` guards |

### 7.1 UI frontend portability (desktop-first, web/mobile-ready)

> **Planned guard — Phase 5.7 / ADR-043.** NNSpire ships as a desktop app, but the
> **UI must stay convertible** to Android / iOS / Web (Qt for WebAssembly, `QtWebEngine`)
> without a rewrite. **This applies to the UI only — runners are always a HW/VM/host
> concern and stay native.** A phone or browser front-end would drive a *remote* engine
> through the existing `RemoteBackend` / `ServiceConnector` transports, never run CUDA or
> load native plugins locally.

The enabling design rules (already mostly satisfied by the inside-out architecture):

| Rule | Why it preserves portability |
|---|---|
| QML for the UI (ADR-018) | Qt Quick already targets WebAssembly, Android, and iOS |
| Strict UI ↔ engine façade | QML/controllers call the engine through a transport-abstractable façade (in-process today; gRPC/WebSocket-capable tomorrow) so a WASM/mobile UI can drive a remote engine unchanged |
| No desktop-only assumptions in QML | filesystem access via a `FileService` abstraction (no raw paths), no blocking calls on the UI thread, no hard dependency on multi-window/dock features WASM/mobile lack |
| Native-only work behind the engine | plugin loading, trust store, CUDA, embedded Python live in the engine — exactly the parts a WASM build drops; surfaced to the UI via capability flags so it degrades gracefully |
| `app-wasm` CMake preset stub | a standing (not-yet-built) reminder the target must keep compiling in principle; revisited post-v1.0 |

---

## 8. FeatureFlags

All Studio features are registered with a tier:

```cpp
// NNSpire/core/features/FeatureFlags.h
enum class Tier { FREE, PRO, ENTERPRISE };

struct FeatureFlag {
    std::string_view id;
    Tier tier;
    bool enabled;  // runtime override (for future license check integration)
};

// Current declarations — ALL FREE
constexpr FeatureFlag FF_LEARNING     {"learning",      Tier::FREE, true};
constexpr FeatureFlag FF_VISUALIZATION{"visualization", Tier::FREE, true};
constexpr FeatureFlag FF_PIPELINE     {"pipeline",      Tier::FREE, true};
constexpr FeatureFlag FF_RUNNERS      {"runners",       Tier::FREE, true};
constexpr FeatureFlag FF_KB_HELP      {"kb_help",       Tier::FREE, true};
constexpr FeatureFlag FF_EXPORT_ONNX  {"export_onnx",   Tier::FREE, true};
constexpr FeatureFlag FF_EXPORT_NNSR  {"export_nnsr",   Tier::FREE, true};
```

The `isEnabled(FeatureFlag)` function currently returns `flag.tier == Tier::FREE || flag.enabled`. Adding a license check later is a one-line change in that function, without touching any feature code.

---

## 9. AI Standards KB integration

The `ai-standards-kb/` folder is:

1. **Mounted as a Qt resource** (`qrc`) for in-app rendering, OR accessed as installed files if the folder is present on disk (preferred — allows user to update the KB without reinstalling the app).
2. **Deep-linked** from every layer, activation, and loss implementation via `docRef()` returning a path like `ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#backpropagation`.
3. **Referenced in wizard steps** via `kbRef` QML property — the HelpPanel opens automatically to the correct section when the user enters a wizard step.

Source code `@kb:` comments (in-source cross-references) use the same path format and are harvested by a documentation tool to generate a "math behind this" index.

---

## 10. Dependency management details

### Embeddable Python
Python.org publishes an officially redistribution-approved embeddable ZIP package for Windows; `.tar.gz` equivalents exist for other platforms. NNSpire downloads and extracts this to `<install>/runtime/python/` on first use of any Python plugin. `pip` is bootstrapped into this isolated environment. The system Python is never modified.

### MinGW-w64 (Windows only)
The MinGW-w64 project (GCC runtime exception — legal to bundle with non-GPL software) is offered as an optional download to `<install>/runtime/mingw64/`. It is used to compile C++ plugin sources within the Studio. MSVC is used if available; MinGW is the fallback for users without Visual Studio.

### OpenSSL
OpenSSL 3.x is required for `TrustVerifier`. On Windows it is bundled (Apache 2.0). On Linux/macOS it is linked against the system OpenSSL if version ≥ 3.0; otherwise a bundled static build is used.

---

## 11. Module size budget

| Component | Target size |
|---|---|
| `NNSpire-core` (static, strip+LTO) | < 8 MB |
| Qt runtime (bundled) | ~80 MB |
| Embeddable Python | ~15 MB |
| Core app binary | < 20 MB |
| CUDA backend plugin | ~10 MB (user installed CUDA separately) |
| **Total base install** | **< 200 MB** |

---

## 12. Semantic composition layer (macro / micro architectures)

> **Planned — Phase 5.7.** Full task breakdown and inline decision records (ADR-035…040)
> live in [`TODO.md`](../TODO.md#phase-57--semantic-composition--macro-architecture-studio).
> Conceptual grounding is in [`modern_ai_systems_ontology.md`](modern_ai_systems_ontology.md).

NNSpire edits **one model graph at two zoom levels**:

- **Micro** (Phase 3 model editor) — the graph of *primitive* layers (`Dense`, `ReLU`,
  `GELU`, `Conv2D`, `MultiHeadAttention`). This is the `ILayer` / `IActivation` world
  described in §3.2 and in `blueprints.md`.
- **Macro** (Phase 5.7) — the graph of *semantic* blocks and model roles
  ("token recognition", "positional awareness", "transformer block", "denoiser",
  "text-embedding model", "inference-assistant LLM", "latent diffuser"). Double-clicking
  a macro node drills down to its micro definition.

The organising claim (see the ontology document, §0 and chapters 1–2): **an LLM is not a
distinct kind of object from a diffuser — both are typed semantic message-passing graphs
over tensors.** The autoregressive decode loop and the diffusion denoise loop are the same
"learned transform inside a controller loop" pattern with different typed messages. Hence
NNSpire represents every architecture family as a **preset** of one node-graph editor,
the way ComfyUI represents diffusion workflows — generalised across all families and
across wrapping formats (`.onnx`, `.safetensors`, GGUF, `.nnsx`).

The generalisation goes further: **every semantic concept is a *canvas template*, and
canvases nest recursively.** A concept that is a single network renders as a one-slot
canvas; a composite renders as a many-slot graph; and any canvas can be wrapped by an
outer **`landscape`** canvas (a running/deployment scene) to arbitrary depth. This
recursive nesting has no external-format equivalent, so it is persisted **only** in
NNSpire project files (`.nnsp`/`.nnsx`); export to ONNX/GGUF flattens or extracts the
runnable leaf. See ADR-041 in `TODO.md`.

### 12.1 Four composition tiers

| Tier | Name | Example members | Status |
|---|---|---|---|
| **A** | Primitives | `Dense`, `ReLU`, `GELU`, `Conv2D`, `LayerNorm`, `MultiHeadAttention` | Implemented (Phase 1) |
| **B** | Layer groups / semantic blocks | `TokenEmbeddingBlock`, `PositionalEncodingBlock`, `SelfAttentionBlock`, `FeedForwardBlock`, `TransformerEncoderBlock`, `DenoiserBlock`, `LMHead` | Planned (Phase 5.7) |
| **C** | Model roles | `TextEmbeddingModel`, `InferenceAssistantLLM`, `QueryRewriteModel`, `Reranker`, `LatentDiffuser`, `MultimodalPerceptionModel`, `VAE`, `CLIPEncoder` | Planned (Phase 5.7) |
| **D** | Macro-architectures / pipelines | `LLMPipeline`, `DiffusionPipeline`, `RAGPipeline`, `AgenticPipeline` | Planned (Phase 5.7 / 4 / 5.5) |

Each Tier B/C/D member is a typed, versioned, signable **template** with semantic input
and output ports drawn from ontology Level 5 (`TEXT`, `TOKENS`, `EMBEDDING[dim]`,
`LATENT`, `CONDITIONING`, `IMAGE`, `TAGS`, `DOCS`, …). The same `CompatibilityChecker`
that gates the cross-framework layer classifies each template as `torch_standard`,
`keras_standard`, or `NNSpire_extended`.

The `NN_bricks_overview.md` inventory maps onto these tiers (ADR-042): its **20 `Prim.X`
primitives are Tier A**, each realised through one of three channels — **builtin**
(`NNSpire::builtin::*`), a **native compiled plugin**, or a **script** (Python plugin,
including deliberate builtin-duplicates that exercise the scripting engine). Its **21
`Beh.X` behaviours are emergent functions, not single ops, and are realised as Tier B/C
templates** (e.g. "Token Recognition" → `TokenEmbeddingBlock`).

### 12.2 Backing models and federated execution

A Tier B/C node is backed by exactly one of: **(a)** a native NNSpire subgraph (drills
down to Tier A primitives, trainable in-engine), **(b)** imported weights
(`.safetensors` / ONNX / GGUF / `.nnsx`, run or fine-tune where the format allows), or
**(c)** a remote endpoint via a `ServiceConnector` plugin.

Every node additionally carries an **`ExecutionBinding`**: `local-cpu`, `local-cuda`,
`remote-grpc` (`RemoteBackend`, §4), `remote-job` (`RemoteTrainer`), or `api-service`
(`ServiceConnector`). A single project may therefore split a workload — train one head
locally, offload a fine-tune to a rented GPU job, run a diffusion step on a hosted API —
with all endpoints and key-references (never secrets) recorded in the `.nnsp` project
file. The `ExecutionPlan` scheduler walks the macro graph and dispatches each node to its
bound runner. See the **External Compute Services** annex in `TODO.md` and
[`DEPLOYMENT.md`](DEPLOYMENT.md).

### 12.3 Universal client

The encoder/decoder/dataset nodes that feed Tier C/D models (the role ComfyUI gives its
text-input, CLIP-encode, and VAE-decode nodes) are instances of **one** `UniversalClient`
component whose editor UI is chosen by its declared port type — a `TEXT` port renders a
text editor, a `TAGS` port a chip editor, an `IMAGE` port a picker, a dataset port a
CSV/Arrow/image-folder loader. It wraps the Phase 4 pipeline input/output adapters and
the dataset loaders behind a single draggable node. See [`PIPELINE.md`](PIPELINE.md).

### 12.4 One navigable tree, three orthogonal axes

The entire composition is **one tree** the user navigates from the outermost running
landscape down to the lowest primitive op / functoid. The namespaces (path = namespace,
ADR-021) and the `.nnsp` internal layout mirror that tree (ADR-047). Three concerns that
*look* like layers are deliberately modelled as **orthogonal attributes** of a node or
subtree rather than levels of the tree:

| Axis | Ontology | What it decorates | Values |
|---|---|---|---|
| **Execution** | L0 / L1 | any node | `local-cpu` · `local-cuda` · `local-qpu` (Phase 6) · `remote-grpc` · `api-service` |
| **Packaging** | L2 | any subtree | `.onnx` · GGUF · `.safetensors` · `.nnsx` (export wrapper) |
| **Process** | — | a model scaffold | `train` · `fine-tune` · `infer` · `evaluate` · `distil` · `rlhf` (ADR-046) |

This resolves the "is packaging above or below orchestration?" question: **semantic
orchestration (L3 / Tier D) is the parent of a packaged model (L2)** — an orchestration
wires *many* packaged models — but the packaging *format* is not a tree level at all; it
is an export attribute (ADR-045). The structural containment tree is only **landscape →
pipeline (D) → model role (C) → block (B) → primitive (A) → functoid**.

In the UI the tree is oriented **root = outermost canvas, leaves = primitive ops /
functoids** (hardware / runtime is an attribute, never a leaf) — the inverse of the
ontology's L0→L6 numbering, which is retained only in this conceptual documentation. Each
axis may optionally be *surfaced* as a navigable node without ceasing to be an attribute:
a **packaging-typed canvas** is a virtual lens that greys the parts of a subtree a chosen
format cannot represent (constrain-early rather than resolve-at-export, ADR-045), and a
**procedural canvas** instantiates a structural canvas to run it under one procedure
(ADR-046). Execution bindings reference entries in a **project-scoped Runner Repository**
(seedable from the app's global runner repo; URLs / driver paths / key-refs are project
parameters, secrets in the OS keychain only, ADR-038).

Canvas kinds form **four families**: `generic` (recursive container / folder),
`structural` (NN structure — sub-kind `orchestration` wires multiple models: diffusion /
LLM / RAG), `procedural` (instantiates a structural / orchestration canvas to run it under
one procedure), and `packaging` (the virtual format lens). All visible surfaces are
themeable via a declarative `ThemeDescriptor` using Qt-native styling only (ADR-049).

Two **cross-cutting views** re-slice the same graph without owning new data: a
**Repository view** (definitions/classes vs. instances/references — multi-use across
canvases shares one definition, ADR-047) and a **Message inventory** (the Level-5 typed
messages flowing on edges, per-canvas or suite-wide, ADR-048). The user-facing
"Client / Foundry / Agents" tier is ontology **L6**, lives in the separate `nnagent`
project *outside* this tree, and appears only as a typed orchestration-boundary node
(ADR-044).
