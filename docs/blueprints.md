# NNStudio — Blueprints

A guided tour through the source code, written for humans.  
Each chapter links directly to the relevant file. Open the file alongside this doc in VS Code (`Ctrl+\` to split editor).

> **Reading order:** follow the chapters top-to-bottom on a first pass.  
> Later, use this as a reference/sitemap — every concept links back to the implementation.

---

## The Story — How All Pieces Plug Together

Each piece in this engine answers one question left open by the piece before it.

```
"I need data to flow."
    └── Tensor                  the noun — a smart flat float array
        shape + strides + grad  describes any data: weights, inputs, outputs, gradients

"But who does the math on that data?"
    └── IBackend                the blueprint (Strategy interface)
        CpuBackend              one answer: Eigen on CPU
        CudaBackend             another answer: cuBLAS on GPU  (future)
        BackendRegistry         the switchboard — backend() gives you whichever is active

"But who organises a meaningful calculation?"
    └── ILayer / Dense          one processing step
        owns:  W_, b_           learnable Tensors (requires_grad = true)
        does:  forward(x)  →  y = x @ W^T + b   (calls backend)
               backward(dY) →  dW, db, dX        (calls backend, fills W_.grad_)

"But who chains layers into a network?"
    └── Sequential              ordered list: output of layer N → input of layer N+1
        ComputeGraph            DAG version: records ops for full autograd  (Chapter 7)

"But who judges if the output is correct?"
    └── Loss (MSE, BCE, ...)    compares prediction vs target → scalar float
                                also produces seed gradient dLoss/dPrediction

"But who fixes the weights?"
    └── Optimizer (SGD, Adam)   reads W_.grad_ for every parameter
                                nudges W_ downhill on the loss surface

"But who runs the loop?"
    └── Trainer                 zeroGrad → forward → loss → backward → step
                                repeats for every batch, every epoch
                                fires callbacks so the UI can watch live

"But how do we know it all works?"
    └── XOR test                4 samples, simplest non-linear problem
                                passes when loss < 0.05 and all 4 predictions correct
                                = proof the whole chain collaborates correctly
```

Or as a class diagram:

```
  The three tools that OPERATE ON the model — hold references, not ownership:

  ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
  │       Trainer       │  │      Loss fns        │  │     Optimizer       │
  │   (orchestrator)    │  │      (judge)         │  │   (fix-it agent)    │
  │                     │  │                      │  │                     │
  │  ILayer&    model   │  │  MSE · BCE           │  │  reads  W_.grad_    │
  │  ILoss&     loss    │  │  CrossEntropy        │  │  writes W_.data_    │
  │  IOptimizer& optim  │  │  Huber · …           │  │  SGD · Adam · AdamW │
  │                     │  │                      │  │                     │
  │  zeroGrad()         │  │  compute(p, target)  │  │  step(params)       │
  │  model.forward()    │  │  → scalar float      │  │                     │
  │  loss.compute()     │  │                      │  │                     │
  │  loss.gradient()    │  │  gradient(p, target) │  │                     │
  │  model.backward()   │  │  → seed grad Tensor  │  │                     │
  │  optimizer.step()   │  │                      │  │                     │
  └──────────┬──────────┘  └──────────┬───────────┘  └──────────┬──────────┘
             └─────────────────────────┼──────────────────────────┘
                           ILayer& reference to ↓

  ┌ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
                     ILayer  (pure abstract interface)                       │
  │  build(Shape)  ·  forward(Tensor)  ·  backward(Tensor)  ·  parameters()
       ▲  implemented by BOTH of the following:                              │
  └ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
        │                                      │
        │ IS-A ILayer                          │ IS-A ILayer
        │                                      │
  ╔═════╧══════════════════════════════╗    ┌──┴───────────────────────────┐
  ║  Sequential      (≈ "the Model")  ║    │  Dense  /  ReLU  /  Sigmoid  │
  ║  IS-A ILayer                      ║    │  /  Conv2D  /  Embedding  /… │
  ║                                   ║    │  IS-A ILayer                 │
  ║  std::vector<LayerPtr> layers_    ║    │                              │
  ║  (LayerPtr = shared_ptr<ILayer>)  ║    │  Tensor W_          ← weights│
  ║                                   ║    │  Tensor b_          ← bias   │
  ║  ┌───────────┐  ┌───────────┐    ║    │  Tensor lastInput_  ← cache  │
  ║  │   Dense   │→ │   ReLU    │→ … ║    │   (activation layers:        │
  ║  │  IS-A     │  │  IS-A     │    ║    │    no W_, no b_)             │
  ║  │  ILayer   │  │  ILayer   │    ║    │                              │
  ║  │           │  │           │    ║    │  forward():  y = x @ W^T + b │
  ║  │ Tensor W_ │  │(no params)│    ║    │  backward(): dW, db, dX      │
  ║  │ Tensor b_ │  │           │    ║    └──────────────┬───────────────┘
  ║  └─────┬─────┘  └─────┬─────┘    ║                  │
  ╚════════╪══════════════╪═══════════╝                  │
           └──────────────┴──────────────────────────────┘
                          │  all leaf ILayer impls call backend()
                          ▼
           ┌──────────────────────────────┐
           │  IBackend  +  BackendRegistry│  ← Strategy: one setActive("cuda")
           │  CpuBackend (Eigen)          │    switches the whole engine
           │  CudaBackend (future)        │
           └──────────────┬───────────────┘
                          │
                          ▼
           ┌──────────────────────────────┐
           │           Tensor             │  ← the noun; flows everywhere
           │  float*   data_              │    weights, inputs, outputs,
           │  Shape    shape_             │    gradients — all same type
           │  Shape    strides_           │
           │  int64_t  numel_             │
           │  Tensor*  grad_              │
           └──────────────────────────────┘
```

---

## Architectural Invariants

> **How to use this list.**  
> Whenever a conversation shifts toward designing or reviewing a *published interface*
> — anything a plugin author will implement, anything crossing a phase boundary, anything
> going into PLUGIN-SDK.md or an `IXxx.h` header — stop and verify each item below.  
> The list exists because the dangerous moment is when everything *looks* fine.

| # | Invariant | Rationale |
|---|---|---|
| I-1 | Every published interface (ILayer, IActivation, IBackend, Plugin SDK) must be **reentrant** — safe to call concurrently on the same object with distinct caller-owned state | Phase 2 users will run models in web servers and GPU backends; silent wrong-gradient failures under concurrency are unacceptable |
| I-2 | No published interface shall hold **mutable state between a call-pair** (e.g. forward/backward) inside the implementing object | Same reason as I-1; caller-owned context (ADR-020 Option C) is the approved pattern |
| I-3 | **Silent failure modes** (wrong result, no crash, no assertion) are never acceptable in a published interface; prefer `Result<T>` + assert over undefined behaviour | Wrong gradients that only appear under load are the hardest possible bugs to diagnose |
| I-4 | The **Phase 2 boundary** is the highest-risk design gate: any interface a plugin author will implement must be reviewed against I-1 through I-3 before the first line of SDK documentation is written | Post-publication API breaks have downstream cost; pre-publication review has near-zero cost |
| I-5 | `Tensor` copy = **reference-count increment only** — no float data copied unless `.clone()` is called explicitly | Performance assumption underlying all interface designs; if violated, every interface signature discussion changes |

*This list is a living document. Add an entry whenever a design review reveals a property that must hold across the entire codebase.*

---

## Chapter 1 — The Memory Model (`Tensor`)

**File:** [`core/include/nnstudio/core/Tensor.h`](nnstudio/core/include/nnstudio/core/Tensor.h)

### Overview (plain language)

Physically it is a flat, one-dimensional array of floats in memory (a `std::vector<float>` under the hood).  
Logically it can represent any multi-dimensional structure — a vector, a matrix, a 3D cube — by using `shape_` and `strides_` as a navigation map on top of that flat array. It is, in the most literal sense, a one-row matrix that happens to describe itself as multi-dimensional through those navigation parameters.

In addition to the data itself, `Tensor` carries two important companions:
- `grad_` — the **gradient**: a sibling array of the same shape that accumulates, during training, the answer to the question *"by how much would the loss change if this particular float were nudged slightly?"* It is the engine's memory of blame, built up step by step during each backward pass.
- `numel_` — a **cached element count**: purely a computational shortcut. Multiplying a large shape every time it is needed would be expensive; `numel_` stores the result once and saves those cycles.

A `Tensor` instance is intentionally role-neutral — the instances (instance variables) of the very same class may represent weights, biases, layer inputs, layer outputs, and gradients. Which role it plays is determined entirely by context (eg. class who owns it (typically some ILayer implementation, see below), how it is named, whether `requires_grad` is set).

### What Tensor is (and isn't)

`Tensor` is a **generic N-dimensional float array**.  
It is used for *everything* — weights, biases, inputs, outputs, and gradients all live in separate `Tensor` instances. Role is determined by context, not by type.

It is **not** a compute engine. It holds data. All arithmetic is delegated to `IBackend` (see Chapter 2).

### The four key private members

```
data_    — flat float[] on the heap, owned via shared_ptr
shape_   — e.g. {3, 2} = 3 rows, 2 columns
strides_ — how many floats to skip per dimension to navigate data_
numel_   — cached product of shape (= total element count)
grad_    — optional sibling Tensor holding the gradient (null until needed)
```

**What are strides?**  
Strides are the step sizes you add to your current array index to move one unit in each dimension.  
For a 2D matrix encoded as one flat vector, there are exactly two stride numbers — think of them as questions:

- `strides_[0]` — *"how many elements do I skip in `data_[]` to move down by one **row**?"*
- `strides_[1]` — *"how many elements do I skip in `data_[]` to move right by one **column**?"*

The formula for any element at logical index `[row, col]`:  
`offset = row × strides_[0] + col × strides_[1]`

Example — shape `{3, 2}`, strides `{2, 1}`:
```
data_ = [ 10,  20,  30,  40,  50,  60 ]
index    [0]  [1]  [2]  [3]  [4]  [5]

[row=1, col=0] → 1×2 + 0×1 = 2 → data_[2] = 30  ✓
[row=1, col=1] → 1×2 + 1×1 = 3 → data_[3] = 40  ✓
```

The rule for row-major (C-order) strides: `strides_[k] = product of all dims after k`.  
Shape `{3,2}` → strides `{2,1}`. Shape `{4,3,2}` → strides `{6,2,1}`.  
Strides have **nothing to do with computation** — they are purely a navigation map: "given a logical index, which float in `data_[]` is that?" Every computation result is always a brand-new `Tensor` with its own `data_[]`.

### How that lives in memory — Dense 2→3 example

A `Dense(2→3)` layer has 2 inputs and 3 neurons.

**Input** `x`, shape `{1, 2}` — one sample, two features:
```
data_ = [ 0.6,  0.4 ]
          x₁    x₂
strides_ = {2, 1}  — skip 2 floats per row, 1 float per column
```

**Weight matrix** `W`, shape `{3, 2}` — 3 neurons, each with 2 weights:
```
data_ = [ w₀₀, w₀₁,   ← neuron 0's weights (its "sensitivity" to each input)
          w₁₀, w₁₁,   ← neuron 1's weights
          w₂₀, w₂₁ ]  ← neuron 2's weights
strides_ = {2, 1}
```
Each row = one neuron.

**Output** `y = x @ W^T + b`, shape `{1, 3}`:
```
y[0] = 0.6·w₀₀ + 0.4·w₀₁ + b₀   ← neuron 0's output
y[1] = 0.6·w₁₀ + 0.4·w₁₁ + b₁   ← neuron 1's output
y[2] = 0.6·w₂₀ + 0.4·w₂₁ + b₂   ← neuron 2's output
```
Each output is exactly the "weighted sum of all inputs plus bias" that defines a neuron.

> **Implementation:**  
> [`Dense.cpp — forward()`](nnstudio/core/src/layers/Dense.cpp)  
> [`BackendRegistry.h — backend()`](nnstudio/core/include/nnstudio/core/BackendRegistry.h)

### Batching — why the first dimension exists

Running one sample at a time is slow. Instead we stack `B` samples into one tensor, shape `{B, inF}`:
```
data_ = [ x₁⁽⁰⁾, x₂⁽⁰⁾,   ← sample 0
          x₁⁽¹⁾, x₂⁽¹⁾,   ← sample 1
          ...             ]
```
The matmul `x @ W^T` produces `{B, outF}` — all `B` outputs in one Eigen (or CUDA) call.  
That is the entire reason layers are written as matrix operations rather than loops over neurons.

In the XOR test batch size is always 1, so shapes flow as: `{1,2} → {1,4} → {1,1}`.

### Transpose is free — strides without copying

`Dense::forward` needs `W^T`. Instead of copying the data, we just swap `strides_`:
```
W   shape={3,2}  strides_={2,1}   ← read data as rows of 2
W^T shape={2,3}  strides_={1,2}   ← same data_, different navigation
```
`shared_ptr<float[]>` means both views share the same heap allocation. Not a single float moves.  
This is the entire purpose of storing `strides_` separately from `shape_`.

### The gradient tensor

After backpropagation, `W.grad_` has shape `{3, 2}` — identical to `W`.  
Each element `grad_[i][j]` answers: *"if I increase `W[i][j]` by a tiny amount, how much does the loss change?"*

The optimizer (`Adam`, `SGD`) then subtracts a scaled version of `grad_` from `W.data_`.  
Because both have the same shape, Adam can do this as a single flat loop over `numel_` floats.

> See also: [Chapter 7 — The Training Loop](#chapter-7--the-training-loop-trainer) for where zeroGrad/accumulateGrad/step fit together.

---

### §1.x — Foundation types: `Device`, `DType`, `Result<T>`

Three small enums/types serve as the vocabulary shared by every other file in the engine. They live in `core/include/nnstudio/core/` and are included transitively by almost everything.

**`Device`** ([Device.h](nnstudio/core/include/nnstudio/core/Device.h)) — a `uint8_t` enum tag stored on every Tensor declaring *where its data lives*: `CPU` (main memory, Phase 1 default), `CUDA` (GPU device memory, Phase 2+), `Quantum` (Phase 6 stub). Device is a property of **data**, not computation — a `DeviceMismatch` error fires before any arithmetic if the active Backend and the Tensor disagree.

**`DType`** ([DType.h](nnstudio/core/include/nnstudio/core/DType.h)) — the element type of a Tensor: `Float32` (training, always supported), `Float16` (inference compression, GPU), `Int8` (quantized export), `Int32` (index/token-ID tensors), `Bool` (masks). Phase 1 computes in `Float32` only; the others are reserved for CUDA and quantization phases. `dtypeBytes(d)` drives all stride and memory-size calculations; `dtypeName(d)` produces the stable string used in ONNX and `.nns` serialization.

**`Result<T>`** ([Result.h](nnstudio/core/include/nnstudio/core/Result.h)) — the error-handling contract for the entire engine, replacing exceptions (see ADR-011). The engine is compiled with `-fno-exceptions` and `-fno-rtti` because it will be linked into plugin shared libraries; C++ exceptions crossing shared-library boundaries are undefined behaviour in the C ABI. `Result<T>` holds either a `T` value (success) or an `Error` (code + message string). `Result<void>` / `VoidResult` carries success-or-error with no payload. The `NN_TRY(expr)` macro propagates an error upward immediately — analogous to Rust's `?` — and is the standard idiom in any function that calls multiple fallible operations in sequence.

---

## Chapter 2 — Who Does the Math? (`IBackend` + `CpuBackend`)

**Files:**  
[`core/include/nnstudio/core/IBackend.h`](nnstudio/core/include/nnstudio/core/IBackend.h)  
[`core/include/nnstudio/core/BackendRegistry.h`](nnstudio/core/include/nnstudio/core/BackendRegistry.h)  
[`backends/cpu/CpuBackend.cpp`](nnstudio/backends/cpu/CpuBackend.cpp)

### Overview (plain language)

Then we have the `Backend` — the interface to our pocket-calculator chip.  
It is a wrapper providing the fundamental mathematical operations needed when computing neuron functions in higher-level classes. The two core operations are **matrix multiplication** and **transposition**; it also provides elementwise arithmetic, reductions (`sum`, `mean`), and reshaping.

The reason this abstraction exists is portability: the same operations can be routed to a CPU implementation today, a CUDA GPU tomorrow, or a quantum accelerator the day after — without any layer ever needing to know which one is active. The currently active implementation is managed by `BackendRegistry`, a singleton registry of named backend instances. A single `setActive("cuda")` call is all that is needed to switch the entire engine from CPU to GPU math.

Both `Tensor` and `Backend`/`BackendRegistry` are deliberately ignorant of neural network concepts. They have no notion of neurons, forward propagation, or training. Their design is strictly NN-utilitarian: they provide the data and operations foundation that higher-level NN classes can be built on top of, independent of the underlying hardware.

`Tensor` holds data. `IBackend` computes on it.  
`BackendRegistry` is a singleton that holds the active backend.  
`backend()` (free function in `BackendRegistry.h`) always returns the active one — so no layer ever mentions `CpuBackend` directly. Switching to CUDA is a single `setActive("cuda")` call.

### The Eigen row/col-major trick

Eigen stores matrices column-major (memory goes down columns first).  
Our tensors are row-major (memory goes across rows first, like C arrays).

Key identity used throughout `CpuBackend::matmul`:
```
(A_rowmajor @ B_rowmajor)  ≡  interpreted col-major  =  (B^T @ A^T)
```
So for `C[M,N] = A[M,K] @ B[K,N]` we map to Eigen as:
```
Eigen sees A as K×M col-major  =  A^T
Eigen sees B as N×K col-major  =  B^T
Eigen writes C as N×M col-major  =  C^T
C^T = B^T @ A^T   ✓  (mathematically identical to C = A @ B)
```
`noalias()` tells Eigen the output buffer doesn't overlap any input — skips a defensive internal copy.

---

## Chapter 3 — A Layer's Contract (`ILayer` + `Dense`)

**Files:**  
[`core/include/nnstudio/core/Layer.h`](nnstudio/core/include/nnstudio/core/Layer.h)  
[`core/include/nnstudio/core/layers/Dense.h`](nnstudio/core/include/nnstudio/core/layers/Dense.h)  
[`core/src/layers/Dense.cpp`](nnstudio/core/src/layers/Dense.cpp)

### Overview (plain language)

`ILayer` is the first interface that is genuinely about neural networks.  
It represents a single computational step in a network — a class that owns a collection of learnable parameter variables of the type `Tensor` (weights `W` and bias `b`) and knows how to command the `BackendRegistry` to perform calculations on them.

When looking at the functions of such a class, apart from initialization (`construct`, `build`) and a housekeeping reset (`zeroGrad`), it exposes two key calculation functions:
- `forward(x)` — the forward propagation step: takes the input activation vector (the outputs of the previous layer), runs the layer's mathematical formula, and returns the output activation vector for the next layer.
- `backward(gradOut)` — the reverse flow used during training: receives the error gradient arriving from the layer above, uses it to compute how wrong each weight and bias was (`dW`, `db`), accumulates those into the parameter `grad_` fields, and returns the gradient signal `dX` to pass further back to the layer below. Structurally it mirrors `forward()` but runs in the opposite direction.

Architectural notes on `ILayer` and its concrete types:
- `ILayer` is only the interface definition for a NN layer. However in a NN, there might be multiple types of layers, differing by the architecture of the "interconnections" and calculations (which inputs/previous layer neuron outputs are used and how, and which mathematical functions are applied to them during forward/backward propagation operations).
- `Dense` is the most fundamental concrete implementation of `ILayer` — the "no structure assumed" baseline. It is a **fully-connected** layer: every single input value is connected to every single output neuron with its own independent learned weight. All connections are always computed, even those whose weights are near zero; skipping connections with insignificant weights requires explicit post-training pruning, which is a separate step not performed here.
- There are also other implementations of `ILayer` apart from `Dense`: `ReLU` / `Sigmoid` / `Tanh` (parameterless activations — a fixed mathematical gate applied elementwise, no learned weights), `Conv2D` (local spatial filter — each output connects only to a small neighbourhood of the input, not all of it; key for image/signal data), `BatchNorm` (normalisation — 2 learned scale factors per feature; keeps activations in a well-behaved range during training), `Dropout` (randomly zeroes some activations at train time — a regularisation technique), `Embedding` (integer token index → learned float vector via a direct lookup table — typical first layer in text/language networks). A full comparison table is in §3.7.

**A note on neurons:** there is no `Neuron` class, no `Neuron` object, no abstract `INeuron` interface anywhere in this engine. This is correct and intentional. A neuron in a `Dense` layer has no individual identity, no functionality of its own, and no biological meaning beyond a convenient metaphor. What we call "neuron k" is simply a set of related indices in the weight matrix `W`: row `k` of `W` holds that neuron's weights, element `k` of `b` holds its bias, and element `k` of the output vector holds its activation. The neuron exists only as a virtual address (an index into the `Tensor`-typed variables in an `ILayer` implementing class instance such as `Dense`) — an implicit `[row, *]` slice of a matrix — never as a stored object.

### Layer vs Backend — what is each responsible for?

| | Responsibility |
|---|---|
| **Backend** | Raw tensor math: `matmul`, `add`, `exp`, `sum`… pure arithmetic, no concept of neurons |
| **Layer** | A meaningful NN operation: holds `W` and `b`, uses Backend to run its formula, computes gradients |
| **Optimizer** | Reads those gradients and **updates** the weight values — the Layer never touches its own weights |

The Layer is the *"who is to blame?"* accountant. The Optimizer is the *"fix it"* agent.

---

### 3.1 Lifecycle — the contract every layer must honour

```
construct  →  build(inputShape)  →  forward(x)  →  backward(gradOut)  →  zeroGrad()
```

| Step | Method | What happens |
|---|---|---|
| 1 | **construct** | Store hyper-params (e.g. `outFeatures_`). Nothing allocated — `inFeatures` not yet known. |
| 2 | **build(inputShape)** | Now we know `inFeatures` → allocate `W` and `b`. Idempotent: same shape twice = no-op. |
| 3 | **forward(x)** | Run the NN formula. Save `lastInput_` — backward will need it. Return output tensor. |
| 4 | **backward(gradOut)** | Receive `dL/dY` from the layer above. Accumulate `dW`, `db` into `grad_`. Return `dX` downward. |
| 5 | **zeroGrad()** | Reset all `grad_` fields to 0. Trainer calls this before every training step. |

> The layer accumulates gradients but does **not** update weights — that is the Optimizer's job.

---

### 3.2 What are `B`, `inF`, `outF`?

Start with **one sample**: say a 28×28 grayscale image flattened to 784 pixel values.

```
inF  (inFeatures) = 784   ← how many numbers describe ONE sample entering this layer
                           • Layer 0: the raw input size you provide
                           • Any other layer: equals the previous layer's outF

outF (outFeatures) = 128  ← you chose this; Dense(128) means "produce 128 numbers per sample"
                           • Becomes inF of the next layer
                           • One number per neuron in this layer

B (batch size) = 32       ← samples processed simultaneously
                           • Each sample is one ROW in the input matrix
                           • The matmul handles all rows in one call
                           • Per-sample math is identical; batch just amortises the call cost
```

So the tensor flowing between layers is:
```
x: [32, 784]   ← 32 rows (samples), 784 columns (features per sample)
    ^B      ^inF

y: [32, 128]   ← 32 rows of output, one per input sample
    ^B      ^outF
```

---

### 3.3 What is Dense? — from one neuron to a matrix

**One neuron** has `inF` incoming connections, each with its own weight, plus a bias:

```
output = w₁·x₁ + w₂·x₂ + … + w_inF·x_inF + b
       = dot(w_row, x) + b
       = one scalar
```

**Dense = `outF` such neurons, all reading the same input:**

```
neuron 1:    y₁     = dot(w_row₁,    x) + b₁
neuron 2:    y₂     = dot(w_row₂,    x) + b₂
             …
neuron outF: y_outF = dot(w_row_outF, x) + b_outF
```

Stack all weight rows into a matrix `W` of shape `[outF, inF]`:

```
ONE sample:   y = W @ x + b          [outF]   = [outF,inF] @ [inF]   + [outF]
BATCH:        Y = X @ W^T + b        [B,outF] = [B,inF]   @ [inF,outF] + [outF]
```

(The transpose `W^T` makes shapes align: `[B,inF] @ [inF,outF] = [B,outF]`.)

**"Fully connected" / "Dense"**: every input $x_i$ connects to every neuron $j$ with its own independent weight $w_{ji}$.  
Total parameters: `outF × inF` weights + `outF` biases = `outF × (inF + 1)`.

> **Near-zero weights are still computed.** Dense always runs all `outF×inF` multiply-adds.  
> A weight of ~0 means "this connection barely matters" — but the hardware still multiplies by it.  
> Skipping such connections requires post-training **weight pruning / sparsification** (not implemented here).

---

### 3.4 Forward pass: `Y = X @ W^T + b`

```
X        [B, inF]        raw data or previous layer's output — each row = one sample
W        [outF, inF]     weight matrix — each ROW = one neuron's weight vector
W^T      [inF, outF]     transposed so matmul shapes align with the batch input
Y        [B, outF]       one output row per sample → feeds into the next layer
b        [outF]          one bias per output neuron, broadcast over the batch
```

`lastInput_ = X` is saved here — the backward pass needs it to compute `dW`.  
Without storing it, we'd have to re-run forward, doubling the work.

---

### 3.5 Backward pass: three gradients from one incoming signal

`gradOut` = `dL/dY` arrives from the layer **above** — shape `[B, outF]`.  
We must produce:

| Gradient | Formula | Shape | Meaning | Destination |
|---|---|---|---|---|
| `dW` | `gradOut^T @ X` | `[outF, inF]` | how wrong was each weight? | stored in `W_.grad_` |
| `db` | `sum(gradOut, dim=0)` | `[outF]` | how wrong was each bias? | stored in `b_.grad_` |
| `dX` | `gradOut @ W` | `[B, inF]` | what do I blame the layer below for? | **returned** downward |

Shape check for `dW`: `gradOut^T` is `[outF, B]` × `X` is `[B, inF]` = `[outF, inF]`. Same shape as `W`. ✓

`accumulateGrad()` **adds** into `grad_` rather than replacing it — supports running multiple  
forward/backward passes before calling `optimizer.step()` once (gradient accumulation for memory saving).

`dX` is the signal the layer below uses to compute *its own* `dW`.

---

### 3.6 How Dense plugs into the Tensor → Backend → Layer system

```
User code
  └─ Dense::forward(X)                   ← concrete ILayer subclass
       └─ B().transpose(W_.tensor)        ← fetches active Backend via free function B()
       └─ B().matmul(X, Wt)              ← CpuBackend → Eigen GEMM on raw float* pointers
            └─ Tensor::data()             ← returns pointer into flat std::vector<float>
```

The layer gives **NN meaning** to the numbers. The Backend does the math. The Tensor holds the bytes.  
Nothing below the Layer level knows what "weights" or "neurons" are.

**Memory note:** `transpose()` only swaps `strides_[]` — no data is copied.  
The same `float*` block is read by Eigen with a different stride interpretation.

---

### 3.7 Dense is one of many ILayer types

| Class | Namespace | Has parameters? | Connectivity | Mental box |
|---|---|---|---|---|
| **Dense** | `nnstudio::layers` | Yes: `W_`, `b_` | Every input → every output | Most general; most expensive |
| **ReLU** | `nnstudio::activations` | No | 1-to-1 elementwise | `y = max(0,x)` — non-linearity, zero params |
| **LeakyReLU** | `nnstudio::activations` | No (but has `alpha_` hyper-param) | 1-to-1 elementwise | Like ReLU but negative side scales by α |
| **Sigmoid** | `nnstudio::activations` | No | 1-to-1 elementwise | Squash to (0,1); output = probability |
| **TanhAct** | `nnstudio::activations` | No | 1-to-1 elementwise | Squash to (−1,1); zero-centred variant |
| **Softmax** | `nnstudio::activations` | No | All inputs → all outputs | Normalise to probability distribution; NOT elementwise |
| **GELU** | `nnstudio::activations` | No | 1-to-1 elementwise | Smooth probabilistic gate; used in Transformers |
| **Conv2D** | `nnstudio::layers` | Yes: small kernel | Local patch → one output | Fewer weights; shared spatially |
| **BatchNorm** | `nnstudio::layers` | Yes: γ, β (2 per feature) | 1-to-1 with statistics | Keeps activations well-scaled |
| **Dropout** | `nnstudio::layers` | No | Randomly zeros activations | Regularisation; train/eval differ |
| **Embedding** | `nnstudio::layers` | Yes: lookup table | Integer index → float vector | Token ID → dense representation |

`Dense` lives in `nnstudio::builtin::layers::`. Activations also live in `nnstudio::builtin::layers::`. Both extend the same `nnstudio::ILayer`. See §3.8 for the inheritance hierarchy.

---

### 3.8 Activation functions — `ActivationBase` and its subclasses

**Files:**  
[`builtin/include/nnstudio/builtin/layers/Activations.h`](nnstudio/builtin/include/nnstudio/builtin/layers/Activations.h)  
[`builtin/layers/Activations.cpp`](nnstudio/builtin/layers/Activations.cpp)  
[`tests/core/test_activations.cpp`](nnstudio/tests/core/test_activations.cpp)

#### The three-level inheritance chain

The full chain for, say, `ReLU`:

```
nnstudio::ILayer                              ← the contract (pure virtual)
  └─ nnstudio::builtin::layers::ActivationBase   ← partial impl (two no-ops factored out)
       └─ nnstudio::builtin::layers::ReLU        ← concrete: adds typeName/forward/backward
```

`Dense` skips `ActivationBase` entirely — it extends `ILayer` directly because it does have parameters and a non-trivial `build()`.

#### What `ActivationBase` contributes

`ActivationBase` is a partial `ILayer` implementation. It provides exactly two methods inline, factoring out boilerplate that is identical for every parameterless activation:

```cpp
// Activations have no learnable parameters — always return empty list
std::vector<Parameter*>       parameters()       override { return {}; }
std::vector<const Parameter*> parameters() const override { return {}; }

// Output shape == input shape — no allocation needed, just mark built
Result<Shape> build(const Shape& inputShape) override {
    markBuilt();
    return Result<Shape>(inputShape);   // pass shape through unchanged
}
```

What it does **not** provide — these remain pure virtual, forcing each concrete class to implement them:
- `typeName()` — each activation has its own name
- `forward()` — each activation has its own math
- `backward()` — each activation has its own derivative

#### Two different save strategies

Every activation must save something in `forward()` so that `backward()` can compute the derivative without re-running forward. Which tensor to save depends on the math:

| Group | Who | What is saved | Why |
|---|---|---|---|
| **Save input** | `ReLU`, `LeakyReLU`, `GELU` | `lastInput_` | Derivative depends on the *sign* of the input: was it positive or negative? |
| **Save output** | `Sigmoid`, `TanhAct`, `Softmax` | `lastOutput_` | Derivative simplifies to an expression in the *output* value itself — cheaper to reuse than to recompute from input |

`lastInput_` and `lastOutput_` are both inherited `Tensor` fields from `ILayer` (see [Layer.h](nnstudio/core/include/nnstudio/core/Layer.h)).

#### Per-activation math

**ReLU** — `f(x) = max(0, x)`

```
forward:   y = clamp(x, 0, ∞)           saves lastInput_
backward:  dX = gradOut ⊙ (lastInput_ > 0 ? 1 : 0)
           (the ⊙ mask zeros out gradient for every input that was ≤ 0)
```

`B().clamp()` dispatches to the active backend. The backward mask is a direct float loop — no backend call needed, just compare and write.

**LeakyReLU** — `f(x) = x if x > 0 else α·x`, default `α = 0.01`

```
forward:   y_i = x_i > 0 ? x_i : α·x_i     saves lastInput_
backward:  dX_i = gradOut_i · (lastInput_i > 0 ? 1 : α)
```

`α` is a constructor parameter (`LeakyReLU(0.01f)`). It is stored in `alpha_` and persisted via `config()`. Unlike a `Parameter`, it is **not** learned — it is fixed after construction.

**Sigmoid** — `f(x) = σ(x) = 1 / (1 + e^{−x})`

```
forward:   y = 1 / (1 + exp(−x))        saves lastOutput_ = y
backward:  dX = gradOut ⊙ y·(1−y)
           (the elegant identity: dσ/dx = σ(1−σ))
```

The backward pass never looks at the original input — it only needs the saved output. This is why `lastOutput_` is used here: `y·(1−y)` is cheaper to compute than expanding `σ(x)·(1−σ(x))` from `x`.

**TanhAct** — `f(x) = tanh(x)`

```
forward:   y = tanh(x)                  saves lastOutput_ = y
backward:  dX = gradOut ⊙ (1 − y²)
           (identity: d/dx tanh(x) = 1 − tanh²(x))
```

Same pattern as Sigmoid: the derivative in terms of the output is simpler.  
Note: the class is named `TanhAct` (not `Tanh`) to avoid collision with `std::tanh`.

**Softmax** — `f(x)_i = exp(x_i − max) / Σ_j exp(x_j − max)` (row-wise)

```
forward:   Per row: subtract max, exp, divide by row sum.   saves lastOutput_
           The max subtraction is numerical stability only — mathematically neutral.
backward:  dX_i = s_i · (gradOut_i − Σ_j gradOut_j·s_j)
           where s = lastOutput_  (the softmax output vector)
```

**Softmax backward is NOT element-wise.** Every output `s_i` depends on every input (because of the normalisation denominator), which means `∂s_i / ∂x_j ≠ 0` for `i ≠ j`. The full Jacobian is a dense matrix — the implementation computes it using the dot-product form above, which is `O(C)` per sample rather than `O(C²)`.

**GELU** — `f(x) = 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (Hendrycks & Gimpel 2016)

```
forward:   element-wise; saves lastInput_
backward:  chain rule through the tanh approximation (see Activations.cpp for derivation)
```

GELU is the standard activation in Transformer and LLM architectures (GPT, BERT). It is a smooth, probabilistic gate: unlike ReLU it is never exactly zero for negative inputs — it has a small negative region, which helps gradients flow.

#### How activations sit in the XOR pipeline

The XOR model is literally four `ILayer` objects in a `Sequential`:

```
Dense   (2→4)   ← nnstudio::builtin::layers::Dense
ReLU            ← nnstudio::builtin::layers::ReLU
Dense   (4→1)   ← nnstudio::builtin::layers::Dense
Sigmoid         ← nnstudio::builtin::layers::Sigmoid
```

`Sequential` does not know or care which namespace each child comes from — they are all `ILayer*`. Forward and backward dispatch through the vtable. An activation layer costs essentially zero in parameters: `parameters()` returns `{}`, `build()` just marks itself built, and the math is a simple element-wise loop or backend call.

### 3.9 Why activations exist — the non-linearity requirement

The sandwich pattern (Dense → ReLU → Dense → Sigmoid) is not decoration. It is mathematically necessary.

**A Dense layer is a linear function:** `y = Wx + b`.
The composition of two linear functions is still a linear function:

```
layer1: y = W1·x + b1
layer2: z = W2·y + b2
           = W2·(W1·x + b1) + b2
           = (W2·W1)·x + (W2·b1 + b2)
           = W'·x + b'          ← still just one linear transformation
```

This is not an approximation — it is exact algebra. **No matter how many Dense layers you stack without activations, the entire network is mathematically equivalent to a single Dense layer.** You could always collapse them all into one. All those weights, all that training time, for zero additional representational power.

The consequence for XOR: XOR is **not linearly separable**. You cannot draw a single straight line through the four XOR samples that correctly separates class 0 from class 1. A single Dense layer (= a linear classifier) cannot solve XOR. It is a mathematical impossibility, not an engineering limitation.

What ReLU adds: `max(0, x)` is a *non-linear* function. Composing it with a Dense layer creates a **piecewise linear** function — one slope for the active region, zero for the inactive region. Stack several of these and the network can approximate functions that bend, curve, and fold the input space. The universal approximation theorem (Cybenko 1989, Hornik 1991) states that a network with at least one hidden layer and a non-linear activation can approximate any continuous function to arbitrary accuracy, given enough neurons.

**The sandwich in concrete terms for XOR:**

```
Input space:  four points — (0,0), (0,1), (1,0), (1,1)
Target:       0,           1,     1,     0

Dense(2→4):   applies 4 different linear cuts through the 2D input space
ReLU:         gates each cut — "is this linear feature active or not?"
Dense(4→1):   combines the 4 gated features into one scalar vote
Sigmoid:      squashes the vote into a probability in (0,1)
```

The ReLU layer is the pivot. Without it, Dense→Dense→Sigmoid is still just one linear classifier — which is proven to fail on XOR. With ReLU in between, the network can carve out the non-convex region that XOR requires.

**Why Sigmoid at the end and not ReLU?** Sigmoid squashes to (0,1), which matches the probability interpretation of the BCE loss. ReLU output is unbounded — `BCE::compute()` would take `log(p)` on a value that might be 5.0, which is valid math but meaningless as a probability. The activation at the final layer is chosen to match the loss function's expectation, not for non-linearity.

**The sandwich in one sentence:** Dense layers are the thinkers — the places where learned knowledge lives as weight values after training. Activations are the mandatory non-linear glue that prevents all the thinkers from mathematically merging into one, and gives the whole structure the ability to model problems that aren't straight lines.

### 3.10 Architecture templates — how far is stacking constrained?

There is **one hard mathematical constraint** and everything else is engineering choice:

> Every pair of consecutive Dense layers must have at least one non-linear layer between them.

That is it. The constraint says nothing about which non-linearity, how many Dense layers, in what width or depth, or whether the graph is even linear. Everything else is a template — a pattern that has been proven useful, not a rule.

The major templates in common use:

| Template | Characteristic stacking | What it models |
|---|---|---|
| **MLP** (what XOR is) | Dense → ReLU → Dense → Sigmoid | Tabular data, classification, regression |
| **CNN** | Conv2D → ReLU → Pool → Dense | Spatial data — images, audio spectrograms |
| **RNN / LSTM** | recurrent cell looped over sequence steps | Sequential data — text, time series |
| **Transformer** | Attention → LayerNorm → FFN (Dense→GELU→Dense) | Long-range dependencies — language models, vision |
| **ResNet** | Dense → ReLU → Dense + skip connection back to input | Very deep networks — the skip connection stops vanishing gradients |
| **Autoencoder** | Encoder (shrinking Dense stack) → Decoder (growing Dense stack) | Compression, anomaly detection, unsupervised |

None of these is the "only" valid architecture. They are solutions that happened to work well enough that the community converged on them. New architectures appear regularly — Mixture of Experts (MoE), State Space Models (Mamba, 2023), Kolmogorov-Arnold Networks (KAN, 2024, replacing Dense layers entirely with learnable spline functions on edges rather than fixed activations on nodes).

**What NNStudio's design allows:**

- `Sequential`: any ordering of any `ILayer` descendants. You can put Sigmoid before ReLU, repeat Dense ten times, or put a custom plugin layer anywhere. Nothing in the engine enforces a template.
- `ComputeGraph` (Chapter 9 / Phase 3): DAG topology. Skip connections, multi-head branches, encoder-decoder splits — anything that is a directed acyclic graph.
- Plugin SDK (Phase 2): any `ILayer` (or future `IActivation`) implementation, including layer types that don't exist yet.

**What NNStudio's design does NOT currently include:**

- `Conv2D` — not implemented yet (Phase 1 TODO). The interface (`ILayer`) supports it; the implementation doesn't exist.
- Recurrent / stateful layers — `Sequential::backward()` assumes one forward = one backward. Recurrent layers require unrolling across time steps; this needs a different `ILayer` contract or explicit loop in the training loop. Not blocked, but not designed for yet.
- Attention mechanism — requires `ComputeGraph` (non-sequential topology) plus `MultiHeadAttention` layer implementation. Both are Phase 3+ items.

So: NNStudio is currently a full MLP workbench, with the architecture to become a Transformer workbench once Phase 2/3 are complete. The stacking variability goal is structurally intact — the engine imposes no template. The limitation is which layer *types* are implemented, not how they can be combined.

---

## Chapter 4 — The Ordered Pipeline (`Sequential`)

**Files:**  
[`core/include/nnstudio/core/Layer.h`](nnstudio/core/include/nnstudio/core/Layer.h) — `ILayer` + `Sequential` class definition  
[`core/src/layers/Layer.cpp`](nnstudio/core/src/layers/Layer.cpp) — `Sequential` implementation

> **Call-chain trace:** [`docs/examples/sequential-call-chain.cpp`](docs/examples/sequential-call-chain.cpp)  
> A single annotated file that walks through every virtual call in one forward pass and one backward pass for the XOR model — readable without a debugger.

### Overview (plain language)

`Sequential` is the simplest answer to *"how do we wire layers into a network?"*.  
It owns an ordered list of `ILayer` instances and threads data through them one by one — left-to-right on `forward()`, right-to-left on `backward()`. The remarkable design choice is that `Sequential` itself **implements `ILayer`**: it has the same `forward()`, `backward()`, `build()`, and `parameters()` as any single layer. This means you can nest a `Sequential` inside another one, or hand it directly to `Trainer` — the Trainer never needs to know whether it received one `Dense` or a deep nested pipeline.

Ownership is via `std::vector<LayerPtr>` (`shared_ptr<ILayer>`). Each `model.add<Dense>(...)` call constructs the layer on the heap and pushes ownership into that vector. The layers live exactly as long as the `Sequential` does.

`build(inputShape)` chains shapes through the layer list: the output shape of layer N becomes the input shape for layer N+1, so each layer allocates its own weight matrices exactly sized for whatever arrives before it. `parameters()` flattens all sub-layer parameters into one `std::vector<Parameter*>` — the single list the Optimizer reads when updating weights.

### Sequential *is* an ILayer

`Sequential` does not float above the layer system — it **implements `ILayer`** directly.  
This single decision has large consequences:

- You hand a `Sequential` to `Trainer` via an `ILayer&` reference — same as a bare `Dense`.
- You can nest a `Sequential` inside another `Sequential` (a sub-network is itself a layer).
- `ComputeGraph` (the DAG variant) also implements `ILayer` — the Trainer never needs to know which it received.

### Ownership: `std::vector<LayerPtr>`

```cpp
private:
    std::vector<LayerPtr> layers_;   // LayerPtr = std::shared_ptr<ILayer>
```

Layer memory is heap-allocated and owned by `Sequential` exclusively. Adding a layer:

```cpp
template<typename L, typename... Args>
Sequential& add(Args&&... args) {
    return add(std::make_shared<L>(std::forward<Args>(args)...));  // construct on heap
}                                                                   // push to layers_

// Returns *this — that is the only reason the fluent chain compiles:
model.add<Dense>(4, true, WeightInit::GlorotUniform)
     .add<ReLU>()
     .add<Dense>(1, true, WeightInit::GlorotUniform)
     .add<Sigmoid>();
```

All four layers are destroyed automatically when `model` goes out of scope.

### `build()` — the shape relay

No weight matrices exist until `build()` is called. `Sequential::build(inputShape)` chains shapes through the list:

```cpp
Shape current = inputShape;
for (auto& layer : layers_) {
    auto r = layer->build(current);  // "your input will be shape X"
    current = r.value();             // "my output will be shape Y" → next layer's input
}
```

Output shape of layer N becomes input shape of layer N+1. Each `Dense` inside allocates `W_[outF × inF]` and `b_[outF]` at this point — sized precisely for what arrives before it. Shape mismatch at any step propagates the error immediately; nothing after that failed layer is built.

### `forward()` — data flows left to right

```cpp
Tensor current = x;
for (auto& layer : layers_) {
    current = layer->forward(current).value();
}
return current;
```

Each layer transforms the tensor and the result becomes the next layer's input. The individual layers save `lastInput_` and `lastOutput_` quietly during their own `forward()` — `Sequential::forward()` itself stores nothing. The pipeline is stateless at the container level; state accumulates inside the leaves.

### `backward()` — the same pipe, reversed

```cpp
Tensor grad = gradOut;
for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    grad = (*it)->backward(grad).value();
}
return grad;
```

`rbegin()` to `rend()` — the gradient seed enters at the last layer and travels back through the list in reverse. Each layer accumulates `dW` / `db` into its own parameters during this pass, then returns `dX` to become the gradient input for the layer before it. `Sequential` is a routing relay only — no accumulation here.

**Critical ordering constraint:** `backward()` must be called after `forward()` on the same data. Each layer needs its `lastInput_` (saved in `forward()`) to compute `dW = gradOut^T @ lastInput_`. Wrong order → garbage gradients.

### `parameters()` — the flat collector

```cpp
for (auto& layer : layers_) {
    auto ps = layer->parameters();
    all.insert(all.end(), ps.begin(), ps.end());
}
```

The Optimizer calls `model.parameters()` once and receives a flat `std::vector<Parameter*>` — direct mutable pointers to every weight and bias in the network, across all layers at every nesting depth. Adam applies the same update rule uniformly to every element of that list. It never needs to know which layer a parameter belongs to.

### `setTraining()` — mode propagation

```cpp
for (auto& l : layers_) l->setTraining(t);
```

Layers such as `Dropout` and `BatchNorm` behave differently at train vs eval time. One call to `model.eval()` propagates the flag to every child simultaneously. `model.train()` reverses it before resuming training.

### ComputeGraph — the DAG alternative (planned)

`Sequential` covers the linear pipeline: layer N feeds exactly layer N+1.  
`ComputeGraph` handles skip connections, residual blocks, and branching architectures (encoder-decoders, multi-head attention) — the general directed acyclic graph. Both implement `ILayer`. For XOR, `Sequential` is all that is needed, and it is sufficient to understand the full forward/backward chain end-to-end before introducing graph complexity.

---

## Chapter 5 — Judging the Output (`Loss`)

**Files:**  
[`core/include/nnstudio/core/Losses.h`](nnstudio/core/include/nnstudio/core/Losses.h)  
[`core/src/losses/Losses.cpp`](nnstudio/core/src/losses/Losses.cpp)

### Overview (plain language)

> *Loss compares prediction vs target → scalar float.  
> Also produces the seed gradient dLoss/dPrediction.*

That two-liner is the complete contract. Loss is the only component in the system that can see both the network's answer and the correct answer at the same time. From that comparison it does two things: it produces a single scalar that tells you *how wrong* the network was, and it produces a gradient Tensor of the same shape as the prediction that tells `Sequential::backward()` *which direction is wrong*.

`ILoss` is a pure abstract interface with exactly two methods:

```cpp
Result<Tensor> compute (const Tensor& pred, const Tensor& target);  // → scalar Tensor{1}
Result<Tensor> gradient(const Tensor& pred, const Tensor& target);  // → dL/dPred, same shape as pred
```

`compute()` and `gradient()` are **separate calls** — `Trainer` calls them one after the other (steps 3 and 4 of `trainStep()`). There is no internal state between them. Both receive the same `pred` and `target`, so no caching is needed.

### The four loss functions

#### MSE — Mean Squared Error

Used for regression (predicting a continuous value). Penalises large errors more than small ones (quadratic).

$$L = \frac{1}{N} \sum_i (p_i - y_i)^2 \qquad \frac{\partial L}{\partial p_i} = \frac{2(p_i - y_i)}{N}$$

```cpp
// compute():
Tensor diff = sub(pred, target);         // pred − target, elementwise
Tensor sq   = mul(diff, diff);           // square elementwise
return mean(sq, -1, false);             // global mean → scalar

// gradient():
const float invN = 2.0f / pred.numel();
return mulScalar(sub(pred, target), invN);
```

#### BCE — Binary Cross-Entropy

Used when the output is a single probability (a Sigmoid output). The standard loss for binary classification — XOR uses this.

$$L = -\frac{1}{N}\sum_i \bigl[y_i \log p_i + (1-y_i)\log(1-p_i)\bigr]$$

$$\frac{\partial L}{\partial p_i} = \frac{1}{N}\left(-\frac{y_i}{p_i} + \frac{1-y_i}{1-p_i}\right)$$

`kEps = 1e-7f` clips `pred` away from exact 0 and 1 before taking `log` — without this, `log(0) = -∞` would produce `NaN` gradients and training would silently collapse.

```cpp
float p = clamp(pred.flat(i), kEps, 1.0f - kEps);  // guard against log(0)
loss += -(y * log(p) + (1-y) * log(1-p));
```

#### CrossEntropy — Categorical Cross-Entropy

Used when the output is a probability distribution over multiple classes (a Softmax output). Each target value is a probability (one-hot or soft label).

$$L = -\frac{1}{B}\sum_i y_i \log p_i \qquad \frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i \cdot B}$$

Note: normalised by **batch size** (`pred.shape()[0]`), not total element count. This keeps the gradient magnitude independent of how many classes there are.

#### HuberLoss — Smooth L1

Hybrid: behaves like MSE when the error is small (quadratic, smooth gradient), like MAE when error is large (linear, bounded gradient). Controlled by `delta_` (default 1.0).

$$L_i = \begin{cases} \tfrac{1}{2}r_i^2 & |r_i| \le \delta \\ \delta(|r_i| - \tfrac{1}{2}\delta) & |r_i| > \delta \end{cases} \qquad r_i = p_i - y_i$$

The transition at `delta_` prevents the exploding gradients that large outliers cause with pure MSE, while preserving smooth curvature (and therefore well-behaved Adam moments) in the normal range.

### Why Loss has no state

Loss functions hold no mutable members between calls (except `HuberLoss::delta_` which is a constructor parameter, not call state). This means the same `ILoss` instance can be shared across threads, called in any order, or used for both training loss and validation metrics simultaneously — without any synchronisation.

---

## Chapter 6 — Fixing the Weights (`Optimizer`)

**Files:**  
[`core/include/nnstudio/core/Optimizers.h`](nnstudio/core/include/nnstudio/core/Optimizers.h)  
[`core/src/optimizers/Optimizers.cpp`](nnstudio/core/src/optimizers/Optimizers.cpp)

### Overview (plain language)

> *Optimizer reads W_.grad_ for every parameter.  
> Nudges W_ downhill on the loss surface.*

After a backward pass every `Parameter.tensor` has `grad_` populated — a Tensor of the same shape recording `dLoss/dWeight` for every single float. The optimizer's job is to read those numbers and make a small update to `data_[]` that moves the weight in the downhill direction. It then steps aside until the next backward pass.

`IOptimizer` has one method that matters:

```cpp
void step(std::vector<Parameter*>& params);   // reads grad(), updates data_[]
```

And one convenience that delegates to `Tensor::zeroGrad()`:

```cpp
void zeroGrad(std::vector<Parameter*>& params);  // called by Trainer BEFORE forward
```

Both methods work off the same flat `vector<Parameter*>` that `Sequential::parameters()` returns — every weight and bias in the whole model, in layer order, in one list.

### SGD with momentum

The simplest update rule: move each weight by `learning_rate × gradient`.  
With momentum, a velocity buffer accumulates past gradients so the update has inertia — it resists sharp direction changes and speeds up in consistent directions.

$$v_t = \mu\, v_{t-1} + g_t \qquad \theta_t = \theta_{t-1} - \alpha\, v_t$$

```cpp
// velocity buffer keyed by parameter's data pointer — allocated lazily on first step
vel[i] = momentum_ * vel[i] + g;   // accumulate
p->tensor.flat(i) -= lr_ * vel[i]; // update weight
```

`velocities_` is an `unordered_map<const float*, vector<float>>`, keyed by the raw data pointer of each parameter. The pointer is stable for the lifetime of the `Tensor` object, so this is a safe identity key. The vector is allocated lazily (zero-initialised) on the first `step()` call for each parameter.

### Adam — Adaptive Moment Estimation

Adam maintains two moment buffers **per parameter, per weight float**:
- `m` — exponential moving average of the gradient (1st moment: direction)
- `v` — exponential moving average of the squared gradient (2nd moment: variance)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$

At the start of training both `m` and `v` are zero, so their estimates are biased low. Bias correction hats remove this:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

The update divides by $\sqrt{\hat{v}} + \varepsilon$ — this makes the effective learning rate **per-weight adaptive**: a weight that has been receiving large, consistent gradients gets a smaller step; a weight that rarely activates gets a proportionally larger step.

$$\theta_t = \theta_{t-1} - \alpha\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$

```cpp
const float bc1 = 1.0f - pow(beta1_, step_);   // bias correction denominators
const float bc2 = 1.0f - pow(beta2_, step_);

mBuf[i] = beta1_ * mBuf[i] + (1.0f - beta1_) * g;
vBuf[i] = beta2_ * vBuf[i] + (1.0f - beta2_) * g * g;

float mHat = mBuf[i] / bc1;
float vHat = vBuf[i] / bc2;
p->tensor.flat(i) -= lr_ * mHat / (sqrt(vHat) + eps_);
```

Default hyperparameters (`β₁=0.9`, `β₂=0.999`, `ε=1e-8`, `lr=1e-3`) work well across a wide range of problems without tuning — which is why Adam is the default choice for new models.

#### One neuron, one weight, one step — concrete numbers

The simplest possible Dense: one input `x`, one weight `w`, one bias `b`, one output.  
Using the XOR values from the backward-pass trace:

```
Forward:   x=0.8,  w=0.3,  b=0.1  →  y = 0.3×0.8 + 0.1 = 0.34
Seed grad arriving from Sigmoid+BCE:  dX3 = -0.5

Dense::backward computes:
  dW = dX3 × lastInput_ = -0.5 × 0.8 = -0.4   ← stored in w.grad_
  db = dX3             = -0.5               ← stored in b.grad_
  dX = dX3 × w        = -0.5 × 0.3 = -0.15  ← returned to previous layer
```

Adam::step now reads `w.grad_ = -0.4`.  Starting from `m=0`, `v=0` (first step):

```
m = 0.9×0  + 0.1×(-0.4) = -0.04
v = 0.999×0 + 0.001×(-0.4)² = 0.00016

Bias correction (step 1 — bc1=0.1, bc2=0.001):
  mHat = -0.04  / 0.1   = -0.4
  vHat =  0.00016 / 0.001 =  0.16

Update:
  w_new = 0.3 − 0.001 × (−0.4) / (√0.16 + 1e-8)
        = 0.3 − 0.001 × (−0.4) / 0.4
        = 0.3 + 0.001
        = 0.301
```

The weight moved from `0.3` to `0.301`. The negative `dW` (-0.4) means "loss decreases if `w` increases" — so Adam nudged it upward. The step size is exactly `lr=0.001` on step 1 because the bias-corrected gradient and its RMS both simplify cleanly.

Step 2 with `dW=-0.3` (weaker signal next sample):
```
m = 0.9×(-0.04) + 0.1×(-0.3) = -0.036 − 0.03 = -0.066
```
The previous gradient is still influencing direction — that is the momentum. If the next sample happened to produce `dW=+0.1` (opposite direction), `m` would still be negative and the update would still go upward, just more weakly. Adam resists reversals.

Standard Adam applies weight decay by adding `λ·θ` to the gradient before computing moments. This couples weight decay to the gradient scale, making its effect inconsistent across parameters.

AdamW applies weight decay **directly to the weight, before the Adam update**, so it is decoupled from the gradient magnitude:

```cpp
p->tensor.flat(i) *= (1.0f - lr_ * decoupledDecay_);   // shrink the weight first
// ... then normal Adam update on the gradient
```

This restores the correct regularisation behaviour and is the recommended default for Transformer-based models.

### The `frozen` flag

Every `Parameter` has a `bool frozen` field. Both SGD and Adam check it:

```cpp
if (p->frozen || !p->tensor.hasGrad()) continue;
```

A frozen parameter is never updated. This is the mechanism for transfer learning: load a pretrained model, freeze all layers except the final one, train only the head. No separate parameter list needed — the same `parameters()` flat vector is used, with `frozen = true` on the layers you don't want to change.

### LR Schedulers

`ILRScheduler::onStep(opt, globalStep)` is called by the training loop once per epoch (or step). The only concrete implementation is `StepDecayScheduler`: multiply `lr` by `gamma` every `stepSize` steps.

```cpp
if (globalStep % stepSize_ == 0)
    opt.setLR(opt.lr() * gamma_);   // e.g. lr × 0.1 every 100 epochs
```

`CosineAnnealingLR` is defined in the header but not yet implemented (stub in TODO).

---

## Chapter 7 — Running the Loop (`Trainer`)

**Files:**  
[`core/include/nnstudio/core/Trainer.h`](nnstudio/core/include/nnstudio/core/Trainer.h)  
[`core/src/training/Trainer.cpp`](nnstudio/core/src/training/Trainer.cpp)

### Overview (plain language)

`Trainer` is the conductor. It owns nothing — meaning it holds **references** (`&`) to a model, a loss function, and an optimizer, but it does not create, copy, or destroy any of them.

The actual objects and their ownership in practice (as seen in the XOR test):
- **`Sequential model`** — a local variable that owns the layers. Each `model.add<Dense>(...)` call creates the `Dense` object on the heap and stores it via `std::shared_ptr<ILayer>` inside `Sequential::layers_`. The layers live exactly as long as `model` does.
- **`BCE bce`** and **`Adam adam`** — further local variables in the same scope.
- **`Trainer trainer(model, bce, adam)`** — receives `ILayer&`, `ILoss&`, `IOptimizer&`. When `trainer` goes out of scope it drops references only; the real objects are unaffected.

This separation is intentional: you can train the same model with two different optimizers, swap the loss function, or run evaluation after training — all without moving or copying the model. The scope of "I want a model" and the scope of "I want to train it right now" are legitimately different.

One counter-intuitive subtlety: the `Trainer` does not touch weights directly — it only calls `optimizer.step(params)`. The optimizer owns the update arithmetic. The layer owns the parameters. The trainer only orchestrates the sequence.

### `trainStep()` — the six-step atomic unit of learning

`trainStep()` is the atomic unit of learning — one forward + backward + weight update.  
Everything else (`trainEpoch`, `train`) is a counted loop calling it.

```
1. zeroGrad        — clear all grad_ fields from the previous step
2. forward         — X flows through all layers; each saves lastInput_
3. loss.compute    — compare prediction to target → scalar float L
4. loss.gradient   — dL/dPrediction — the seed gradient
5. backward        — gradient flows in reverse: last layer → first layer
                     each layer accumulates dW/db into grad_, returns dX downward
6. optimizer.step  — Adam/SGD reads grad_ from every Parameter, updates data_
```

### Adam update rule (Step 6)

$$m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha\,\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\varepsilon}$$

`m` = running mean of gradients (momentum).  
`v` = running mean of squared gradients (adaptive per-weight learning rate).  
Bias-correction hats (^) stop the estimates from being near-zero at the start of training.

> **Implementation:** [`core/src/optimizers/Optimizers.cpp`](nnstudio/core/src/optimizers/Optimizers.cpp)

---

## Chapter 8 — The XOR Milestone

**File:** [`tests/core/test_trainer_xor.cpp`](nnstudio/tests/core/test_trainer_xor.cpp)

XOR is the classic "hello world" for neural networks:  
the simplest problem that is *not* linearly separable — no single line divides the 0s from the 1s.  
It proves the engine works end-to-end: forward, backward, optimizer, convergence.

```
[0,0]→0   [0,1]→1   [1,0]→1   [1,1]→0
```

**Architecture:** `Dense(2→4) → ReLU → Dense(4→1) → Sigmoid`

| Layer | Purpose |
|---|---|
| `Dense(2→4)` | Projects 2D input into 4D space where XOR *becomes* linearly separable |
| `ReLU` | Non-linearity — without it, two Dense layers collapse into one linear function |
| `Dense(4→1)` | Collapses 4D features into one output logit |
| `Sigmoid` | Squashes logit to (0,1) — interpretable as probability |

Test passes when: `finalLoss < 0.05` *and* all 4 predictions round correctly.  
Runs in ~69ms for 3000 epochs on 4 samples (~170k weight updates/second, single CPU thread).

---

## Chapter 9 — Step-by-Step Visualization (Planned — Phase 3 UI)

> This chapter describes a **planned feature**, not yet implemented.

### The problem

A beginner looking at the network needs to see:
- the actual float values arriving at each neuron
- what the weighted sum produces before and after the activation
- how the gradient signal changes as it flows backwards

None of this is visible from the outside once training is running at full speed.

### The solution: `EvalTrace`

We will add an opt-in **trace mode** to the engine.  
When active, `forward()` records every layer's input and output into an `EvalTrace` object instead of discarding them:

```cpp
// Planned API (not yet implemented — see TODO.md Phase 3)
struct LayerTrace {
    std::string   layerName;
    Tensor        input;    // what arrived
    Tensor        output;   // what was produced
    Tensor        gradIn;   // gradient that arrived in backward (optional)
    Tensor        gradOut;  // gradient that was produced
};

struct EvalTrace {
    std::vector<LayerTrace> layers;   // one entry per layer, in forward order
};
```

The `Trainer` (or a new `TracingForward` wrapper) populates this when the UI is in "explain" mode. Normal training ignores it entirely — zero cost.

### What the UI will show

For each layer in the graph (Phase 3 Qt UI):

1. **Neuron view** — a grid of circles. Each circle = one neuron.  
   Colour/size = activation value. Click a neuron to see its individual `w·x+b` breakdown.

2. **Weight heatmap** — the weight matrix `W` as a colour grid.  
   After each step, cells that changed significantly flash.

3. **Gradient flow** — arrows between layers show `dX` magnitude.  
   Thin arrow = small gradient (vanishing gradient problem visible here).

4. **Step-by-step mode** — pause after each of the 6 `trainStep` phases,  
   inspect the state at that exact moment.

> **Engine hook needed before UI:** `ILayer::forward()` needs an optional  
> `EvalTrace*` parameter (defaulting to `nullptr` = no tracing).  
> This is a small, non-breaking addition — all existing code passes `nullptr` implicitly.

---

## Appendix — File Map

```
nnstudio/
├── core/
│   ├── include/nnstudio/core/
│   │   ├── Tensor.h          ← Ch.1  data model, shape, strides, grad
│   │   ├── IBackend.h        ← Ch.2  compute interface
│   │   ├── BackendRegistry.h ← Ch.2  singleton + backend() free function
│   │   ├── Layer.h           ← Ch.3  ILayer lifecycle + Sequential
│   │   ├── layers/Dense.h    ← Ch.3  fully-connected layer API
│   │   ├── Activations.h     ← Ch.3  ReLU, Sigmoid, GELU, ...
│   │   ├── Losses.h          ← Ch.5  MSE, BCE, CrossEntropy
│   │   ├── Optimizers.h      ← Ch.6  SGD, Adam, AdamW
│   │   └── Trainer.h         ← Ch.7  training loop API
│   └── src/
│       ├── tensor/Tensor.cpp
│       ├── layers/Layer.cpp
│       ├── layers/Dense.cpp  ← Ch.3  forward/backward with comments
│       ├── activations/Activations.cpp
│       ├── losses/Losses.cpp
│       ├── optimizers/Optimizers.cpp
│       └── training/Trainer.cpp ← Ch.7  trainStep with comments
├── backends/
│   └── cpu/CpuBackend.cpp    ← Ch.2  Eigen matmul, row/col-major trick
└── tests/
    └── core/
        ├── test_tensor.cpp
        ├── test_activations.cpp
        └── test_trainer_xor.cpp ← Ch.8  XOR milestone with comments
```

---

## Appendix — Namespace Map and Naming Conventions

> **TODO:** Expand this section into a standalone `NAMING-CONVENTIONS.md`.

---

### The three-tier namespace design (target state)

Three visibility tiers separate the stable public contract from engine internals from bundled implementations.

#### Tier 1 — `nnstudio::core::` — the stable, versioned public contract

Contains **only** abstract extension points and core data types.  
**No concrete implementation ever lives here.**  
Plugin authors include `<core/ILayer.h>` etc. and work entirely inside `nnstudio::core::`.  
Breaking changes require a version bump and a transition period.

```
nnstudio::core::
    ILayer          abstract layer contract (build/forward/backward/zeroGrad)
    IBackend        abstract compute contract (matmul, transpose, elementwise...)
    ILoss           abstract loss contract
    IOptimizer      abstract optimizer contract
    ILRScheduler    optional learning-rate schedule contract
    Tensor          core data type (not NN-specific; used by all tiers)
    Parameter       { name, Tensor, frozen } — the unit the optimizer reads
    Result<T>       error-or-value return type
    Shape           vector<int64_t> alias
    Device          CPU | CUDA | QUANTUM enum
    DType           float32 | float16 | int8 enum
    BackendRegistry singleton registry + backend() free function
    PluginDescriptor  C-ABI plugin descriptor (name, version, type tag, factory)
    FeatureFlags    FREE | PRO | ENTERPRISE tier gating
    Trainer         training loop (DataBatch, Dataset, TrainMetrics, TrainCallbacks)
    Sequential      ordered container of ILayer instances
```

#### Tier 2 — `nnstudio::internal::` — engine guts, not for plugin authors

Contains the engine's own machinery.  
Documented as: *"do not use from plugins — may change in any release with no warning."*

```
nnstudio::internal::
    graph::         ComputeGraph, autograd traversal, DAG node types
    training::      Trainer step loop, callback dispatch, EvalTrace capture
    formats::       .nnsp / .nnsx file I/O, ONNX serialisation
    detail::        utility templates, helpers, type traits
```

#### Tier 2b — `nnstudio::ui::` — UI extension points (Phase 3+)

Semi-stable; evolves with Qt version requirements.  
Plugin authors use this only when registering a custom UI panel.

```
nnstudio::ui::
    panels::        QML panel plugin registration interface
    qml::           QML-side property/signal contract types
```

#### Tier 3 — `nnstudio::builtin::` — NNStudio's own implementations

**NNStudio's default implementations are treated identically to third-party plugins.**  
They happen to ship bundled with the installer, but they have no special namespace or access privilege.  
A list of all loaded layer types will show `nnstudio::builtin::layers::Dense` alongside `myplugin::layers::MyLayer` — no VIP.

```
nnstudio::builtin::
    layers::        Dense, ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU,
                    Conv2D, BatchNorm, Dropout, Embedding, ...
    backends::      CpuBackend, CudaBackend, QuantumBackend
    losses::        MSE, CrossEntropy, BinaryCrossEntropy, HuberLoss
    optimizers::    SGD, Adam, AdamW, RMSProp
```

A third-party plugin developer writes:
```cpp
eu::plachy::nnplugins::myplugin::layers::MyLayer : public nnstudio::core::ILayer { ... }
```
NNStudio itself writes:
```cpp
nnstudio::builtin::layers::Dense : public nnstudio::core::ILayer { ... }
```
Both are just implementations. The only allowed difference is that `builtin` ships by default and appears first in UI lists.

---

### Current state vs target state

~~The current code does not yet match the target. This is the known delta:~~

**Migration complete.** All refactors below are applied to the on-disk source; all 16 `ctest` tests pass; `cmake --build` clean; old files deleted.

#### Phase 1 — builtin namespaces

| Symbol | Old namespace (deleted) | New namespace ✅ |
|---|---|---|
| `Dense` | `nnstudio::layers::` | `nnstudio::builtin::layers::` |
| `ReLU`, `Sigmoid`, etc. | `nnstudio::activations::` | `nnstudio::builtin::layers::` (activations are layers) |
| `MSE`, `BCE`, etc. | `nnstudio::losses::` | `nnstudio::builtin::losses::` |
| `SGD`, `Adam`, `AdamW` | `nnstudio::optimizers::` | `nnstudio::builtin::optimizers::` |
| `CpuBackend` | `nnstudio::` | `nnstudio::builtin::backends::` |
| `ILoss` | bundled in `Losses.h` | extracted to `nnstudio::core::ILoss` in `ILoss.h` (Tier 1) |
| `IOptimizer`, `ILRScheduler` | bundled in `Optimizers.h` | extracted to `nnstudio::core::IOptimizer` in `IOptimizer.h` (Tier 1) |

#### Phase 2 — core namespace + Option B folder structure

| Symbol | Old namespace (deleted) | New namespace ✅ |
|---|---|---|
| `Tensor`, `Parameter`, `Shape`, `Device`, `DType` | `nnstudio::` | `nnstudio::core::` |
| `ILayer`, `Sequential`, `LayerPtr` | `nnstudio::` | `nnstudio::core::` |
| `IBackend`, `BackendRegistry` | `nnstudio::` | `nnstudio::core::` |
| `ILoss`, `IOptimizer`, `ILRScheduler` | `nnstudio::` | `nnstudio::core::` |
| `Trainer`, `DataBatch`, `Dataset`, `TrainMetrics` | `nnstudio::` | `nnstudio::core::` |
| `Result<T>`, `FeatureFlags` | `nnstudio::` | `nnstudio::core::` |
| All files | `core/include/nnstudio/*.h` + `builtin/include/nnstudio/**/*.h` | `include/core/*.h` + `include/builtin/**/*.h` (Option B shared root) |

Builtin `include/` and `src/` files gained `using namespace nnstudio::core;` inside (or just before) their namespace block so they reference Tier 1 types without qualification.

---

### The path = namespace rule

Every on-disk path segment translates directly to exactly one C++ namespace segment — with two exceptions that are **visibility boundaries only**, never namespace contributors: `include/` and `src/`.

```
Disk path segment          →   C++ namespace segment
─────────────────────────────────────────────────────────────────
nnstudio/                  →   nnstudio::           (repo root = outermost namespace)
include/  or  src/         →   (SKIP — visibility boundary, not a namespace layer)
core/                      →   core::
builtin/                   →   builtin::
layers/                    →   layers::
backends/                  →   backends::
losses/                    →   losses::
optimizers/                →   optimizers::
Tensor.h                   →   class/namespace Tensor  (contents of the file)
─────────────────────────────────────────────────────────────────
Rule:  EVERY segment except include/ and src/ maps 1-to-1 to a namespace segment.
```

Examples:

```
nnstudio / include / core    / Tensor.h
    ↓       (skip)    ↓            ↓
nnstudio::           core::      Tensor

nnstudio / include / builtin / layers / Dense.h
    ↓       (skip)     ↓          ↓         ↓
nnstudio::            builtin:: layers::  Dense

nnstudio / src     / core    / Tensor.cpp
    ↓       (skip)    ↓            ↓
nnstudio::           core::      (Tensor impl — same namespace, visibility-only boundary)
```

This is the convention enforced by the Option B shared-root layout. See the **Plugin exception** section below for the one deliberate reversal of this rule.

---

### Folder structure (current state)

The folder layout reflects namespace tiers, the one-directional mirror rule, and the colocated-headers-for-implementations principle.

```
nnstudio/
    include/                           ← ONE CMake search root (all targets: core, builtin, …)
        core/                          → namespace nnstudio::core::
            Tensor.h                   → nnstudio::core::Tensor
            Layer.h                    → nnstudio::core::ILayer, Sequential, LayerPtr, Parameter
            IBackend.h                 → nnstudio::core::IBackend
            BackendRegistry.h          → nnstudio::core::BackendRegistry
            ILoss.h                    → nnstudio::core::ILoss
            IOptimizer.h               → nnstudio::core::IOptimizer, ILRScheduler
            Trainer.h                  → nnstudio::core::Trainer, DataBatch, Dataset, TrainMetrics
            Result.h                   → nnstudio::core::Result<T>
            Device.h  DType.h          → nnstudio::core::Device, DType
            FeatureFlags.h             → nnstudio::core::FeatureFlags
        builtin/                       → namespace nnstudio::builtin::
            layers/                    → nnstudio::builtin::layers::
                Dense.h
                Activations.h
            backends/                  → nnstudio::builtin::backends::
                CpuBackend.h
            losses/                    → nnstudio::builtin::losses::
                Losses.h
            optimizers/                → nnstudio::builtin::optimizers::
                Optimizers.h

    src/                               ← mirrors include/ exactly — build-only, never installed
        core/
            Tensor.cpp  BackendRegistry.cpp  Layer.cpp  Trainer.cpp
        builtin/
            layers/     Dense.cpp  Activations.cpp
            backends/   CpuBackend.cpp
            losses/     Losses.cpp
            optimizers/ Optimizers.cpp

    core/                              ← CMake target definition only
        CMakeLists.txt                 ← target nnstudio-core; sources from ../src/core/
        CONTRIBUTING.md

    builtin/                           ← CMake target definition only
        CMakeLists.txt                 ← targets nnstudio-builtin + nnstudio-backend-cpu
        CONTRIBUTING.md

    plugins/                           ← third-party and first-party plugin slots
        README.md
        CONTRIBUTING.md
        (see Plugin exception below for layout rules inside this folder)

    tests/
        core/
            test_tensor.cpp
            test_activations.cpp
            test_trainer_xor.cpp
```

**The mirror rule (one-directional):**  
Every subfolder present in `include/` **must** have a corresponding subfolder in `src/`.  
The reverse does not hold — `src/` may gain private subfolders (e.g. `src/graph/`, `src/training/`) that have no counterpart in `include/` when they are build-internal only.

Plugin authors add `${nnstudio_SOURCE_DIR}/include` to their include search path. Nothing else.

**Per-folder documentation files:**  
Every folder that contains primarily subfolders (rather than files) carries two Markdown files:
- `README.md` — what this folder contains in terms of *software concept* (what it does at runtime)
- `CONTRIBUTING.md` — why this folder exists *here*, why it is structured this way, architectural decisions (the CUDA-linking argument, the curated-install argument, the intended reading order of subfolders as a numbered list under a `## Reading order` heading)

---

### On `interfaces/` subfolders and why collocation with plural names is better

The C++ community argument against a separate `interfaces/` folder: it mechanically separates every `IFoo.h` from every `FooImpl.h` into parallel trees — a separation that adds navigation work but communicates nothing the `I` prefix doesn't already convey.

The `include/` tree above is different: it is the **publicly installed plugin SDK surface** — a boundary between what is stable and what is internal. The folder name is `include/` (universal tooling convention), not `api/` (which implies a REST or binding layer to most readers). Inside `include/`, plural subfolder names (`layers/`, `backends/`, …) mirror `src/` subfolders, making navigation predictable. This is what the Android NDK, LLVM, Qt, and virtually every major C++ SDK does: installed headers in a predictable include tree vs implementation files that never leave the build.

---

### Naming rules summary

| Element | Convention | Example |
|---|---|---|
| Interface / abstract base | `I` prefix | `ILayer`, `IBackend` |
| Concrete implementation | Descriptive name only | `Dense`, `Adam`, `CpuBackend` |
| Path → namespace | Every path segment (except `include/`/`src/`) maps 1-to-1 to a namespace segment | `include/core/Tensor.h` → `nnstudio::core::Tensor` |
| Namespace for plugin implementations | Author's reverse-domain identity (**not** derived from `plugins/` folder position) | `eu::plachy::nnplugins::myplugin::layers::MyLayer` |
| Plugin distribution ID (manifest) | reverse-domain dot-notation | `"id": "eu.plachy.nnplugins.myplugin"` |
| Filename | PascalCase, no prefix/suffix | `Dense.h`, `CpuBackend.cpp` |
| `_` prefix in filenames | **Do not use** — reserved/confusing in C++ | — |
| Folder names (domain collections) | Plural by convention | `layers/`, `backends/`, `losses/` |
| Folder names (single-concept roots) | Singular or plural as natural language dictates | `plugins/`, `formats/` |
| `include/` subfolder → `src/` subfolder | One-directional mirror: every `include/X/` implies `src/X/` exists | `include/core/` ↔ `src/core/` |
| `src/` private subfolder | May exist with no `include/` counterpart | `src/graph/`, `src/training/` |
| `using namespace nnstudio::core;` in builtin | Placed after innermost namespace opening in `.h`; at file scope before namespace block in `.cpp` | `namespace nnstudio::builtin::layers { using namespace nnstudio::core; … }` |
| Per-folder documentation | `README.md` (software concept) + `CONTRIBUTING.md` (repo structure rationale + `## Reading order` list) | — |

The `I` prefix on the filename is the complete interface signal — no separate `interfaces/` subfolder is needed. The `include/` vs `src/` split already expresses the contract-vs-implementation distinction at the directory level.

---

### Plugin exception — the rule is deliberately reversed for `plugins/`

All rules above derive one direction: **folder path → C++ namespace**. Inside `plugins/` this direction is **inverted**: the author's namespace identity is the primary artifact, and the on-disk folder layout follows from it.

**Why?**  
The namespace of a plugin is owned by its author, not by NNStudio. It is based on the author's reverse-domain identity and must not change when the plugin moves between repositories, organizations, or deployment targets. Deriving it from the folder position inside `nnstudio/plugins/` would couple it to NNStudio's internal directory structure — the exact coupling the plugin system exists to avoid.

**Layout inside `plugins/`:**

```
nnstudio/plugins/
    <plugin-slug>/                    ← reverse-domain slug, e.g. eu.plachy.nnplugins.myplugin
        include/                      ← visibility boundary (same semantics as everywhere else)
            layers/                   → author-controlled namespace segment
                MyLayer.h
            backends/
                MyBackend.h
            ui/
                MyPanel.h
        src/                          ← build-only mirror of include/
            layers/
                MyLayer.cpp
            backends/
                MyBackend.cpp
            ui/
                MyPanel.cpp
        CMakeLists.txt
        README.md
        CONTRIBUTING.md
```

**Namespace ownership:**  
The `<plugin-slug>` folder name does **not** contribute a namespace segment. The plugin author decides the full namespace independently:

```
plugins / eu.plachy.nnplugins.myplugin / include / layers / MyLayer.h
 (skip)          (slug — skip)           (skip)      ↓          ↓
                                               layers::      MyLayer
```

The segments above `layers/` are entirely the author's choice, e.g.:

```cpp
// Author chooses their own top-level namespace:
namespace eu::plachy::nnplugins::myplugin::layers {
    class MyLayer : public nnstudio::core::ILayer { … };
}
```

**Matching table — core/builtin vs plugins:**

| Aspect | `core/` and `builtin/` | `plugins/<slug>/` |
|---|---|---|
| Namespace derived from | folder path (forward rule) | author identity (reverse rule) |
| Folder slug contributes namespace segment? | Yes — `core/` → `::core::`, `builtin/` → `::builtin::` | No — slug is an artifact identifier, not a namespace |
| `include/` / `src/` are visibility boundaries? | Yes | Yes (same) |
| Domain subfolders (`layers/`, `backends/`, …) contribute namespace? | Yes | Yes — same sub-segment rule applies *within* the plugin |
| Inherits from | `nnstudio::core::ILayer`, etc. | `nnstudio::core::ILayer`, etc. (same) |

> **Current status (documentation only):** The `plugins/` folder exists as a placeholder. Moving `builtin/` content into `plugins/builtin/` is a future option that requires no namespace changes — only file relocation and CMakeLists updates.


---

## Appendix — Architecture Templates

> 🥪 These are the "sandwich presets" in the UI. Each is a proven starting point, not a law.
> The engine imposes no template; the only hard rule is: **no two consecutive Dense layers without a non-linear layer between them**.
> See `blueprints.md §3.9` for the mathematical proof of why.

---

### Template 1 — MLP (Multi-Layer Perceptron)

**Use for:** tabular data, binary/multi-class classification, regression.

```
Input → Dense(n_hidden) → ReLU → [Dense(n_hidden) → ReLU]... → Dense(n_out) → output activation
```

Output activation by task:
- Binary classification → Sigmoid + BCE loss
- Multi-class → Softmax + CrossEntropy loss
- Regression → none (linear output) + MSE loss

**Typical sizes:**

| Network | Layer widths | Params |
|---|---|---|
| XOR (our test) | 2→4→1 | 13 |
| Small classifier (MNIST) | 784→512→256→10 | ~670 K |
| Medium tabular | 64→256→256→128→1 | ~130 K |

**Status in NNStudio:** fully operational. XOR is the proof.

---

### Template 2 — Autoencoder

**Use for:** compression, anomaly detection, denoising, unsupervised feature learning.

```
Input → Dense(512)→ReLU → Dense(256)→ReLU → Dense(bottleneck)→ReLU   ← encoder
      → Dense(256)→ReLU → Dense(512)→ReLU → Dense(input_size)→Sigmoid  ← decoder
```

The loss is reconstruction: MSE or BCE between output and original input.
`target = input` — self-supervised, no external labels needed.

**Status in NNStudio:** engine supports it (all required layers and losses exist); ILoss contract uses an external `target` argument so you pass the input batch as both `inputs` and `targets` in `trainStep()`.

---

### Template 3 — CNN (Convolutional Neural Network)

**Use for:** images, audio spectrograms, any data with spatial locality.

```
Input → [Conv2D → ReLU → MaxPool]... → Flatten → Dense(256) → ReLU → Dense(n_classes) → Softmax
```

Conv2D layers detect local patterns (edges, textures) independent of position.
MaxPool reduces spatial size. Flatten converts 2D feature maps to 1D for the Dense head.

**Typical sizes:**

| Network | Architecture | Params |
|---|---|---|
| LeNet-5 (1998) | 2×Conv + 2×Dense | ~60 K |
| AlexNet (2012) | 5×Conv + 3×Dense | ~60 M |
| ResNet-50 (2015) | 50 layers, skip connections | ~25 M |

**Status in NNStudio:** `Conv2D` not yet implemented. `ILayer` interface supports it; Phase 1 TODO.

---

### Template 4 — Transformer block (encoder or decoder)

**Use for:** text, language understanding, generation, reasoning, long-range dependencies.

One Transformer block (stacked N times):

```
x → LayerNorm → MultiHeadSelfAttention → + x   (residual / skip connection)
  → LayerNorm → Dense(4×d_model)→GELU→Dense(d_model) → + x   (FFN block + residual)
```

The full GPT-style model:
```
Token embedding + positional encoding
→ N × Transformer block
→ LayerNorm
→ Dense(d_model → vocab_size)   (the "language model head")
```

---

#### Layer-by-layer breakdown

> All four layer types below are **not yet in NNStudio**. They are explained here so the design intent is clear when they are implemented.

---

##### 4a. Embedding

**What it is:** a learnable lookup table.

Words (tokens) arrive as integers — indices into a vocabulary of size `V` (e.g. 50 257 for GPT-2). The network cannot process raw integers directly; it needs real-valued vectors it can do linear algebra on. An embedding layer is simply a weight matrix `E` of shape `[V, d_model]`. A forward pass for token index `i` returns row `i` of `E` — that is it.

```
token id = 42  →  E[42]  →  vector of shape [d_model]
```

No sigmoid, no dot product. Just a lookup. But because `E` is a weight matrix, backprop tunes it: after training, tokens with similar meaning end up with nearby vectors (this is where "word2vec-style geometry" comes from: `king − man + woman ≈ queen`).

**Parameters:** `V × d_model` weights. For GPT-2 small: `50 257 × 768 ≈ 38.6 M` parameters — the embedding table alone is 33% of the model.

**Status in NNStudio:** not implemented. `ILayer::build(Shape)` would need to accept a scalar integer index as input shape, which breaks the current Tensor contract (which expects float tensors). This needs a design decision before implementing.

---

##### 4b. Positional encoding

**Why it is needed:** attention (see §4c) treats the input as a *set* — it has no idea which token came first. Position must be injected explicitly.

Two approaches exist:

**Learned (GPT-2, BERT):** another embedding table `PE` of shape `[max_sequence_length, d_model]`. Position `p` maps to `PE[p]`. Added to the token embedding: `x = E[token] + PE[position]`.

**Sinusoidal (original 2017 Transformer):** a fixed formula, no learned parameters:

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

Different frequency sine/cosine waves fill alternating dimensions. The result: each position has a unique vector, and nearby positions have similar vectors. Modern models (Llama) use **RoPE** (Rotary Position Embedding) instead — applied inside the attention computation rather than at input.

**Status in NNStudio:** not implemented. Sinusoidal PE is parameter-free (just an `apply()` function on the input tensor) and could be implemented before the embedding layer.

---

##### 4c. MultiHeadSelfAttention

This is the core invention of the Transformer. It replaces recurrence entirely.

**The problem attention solves:** in a sentence like "The animal didn't cross the street because *it* was too tired", deciding what "it" refers to requires comparing every token to every other token simultaneously. An RNN has to thread that information through a bottleneck hidden state. Attention makes all token-to-token comparisons in one step.

**Single-head attention:**

Each token vector `x_i` is projected into three roles through three learned weight matrices `W_Q`, `W_K`, `W_V` (all `[d_model, d_k]`):

- **Query** Q: "what am I looking for?" — `Q = x W_Q`
- **Key** K: "what do I contain?" — `K = x W_K`
- **Value** V: "what do I contribute?" — `V = x W_V`

The attention score between position `i` (query) and position `j` (key) is their dot product, scaled:

$$A_{ij} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}$$

The $\sqrt{d_k}$ scaling prevents the dot products from growing large and pushing softmax into its flat saturation region (vanishing gradients). Softmax across all `j` makes the scores sum to 1 — they become mixing weights. The output for token `i` is a weighted sum of all value vectors:

$$\text{out}_i = \sum_j \text{softmax}(A_{ij}) \cdot V_j$$

In matrix form (all tokens at once):

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

**Multi-head:** instead of one set of (Q, K, V) matrices, run `n_heads` of them in parallel, each with `d_k = d_model / n_heads`. Each head can attend to different aspects (syntax, coreference, topic). Outputs are concatenated and projected back:

$$\text{MultiHead}(x) = \text{concat}(\text{head}_1, \ldots, \text{head}_h) \; W_O$$

**Parameters per attention block:** `4 × d_model²` — the four matrices W_Q, W_K, W_V, W_O. For `d_model=768`: `4 × 768² = 2.36 M` per block.

**Causal mask (decoder only):** in a GPT-style model, position `i` must not attend to position `j > i` (future tokens). This is enforced by adding `-∞` to the attention scores before softmax for all future positions — they become 0 after softmax. This is a single mask operation, not a structural change.

**Status in NNStudio:** not implemented. Requires:
- The QKV projection (three Dense layers per head, or one fused Dense of `3×d_model`)
- Scaled dot product (Tensor batch-matmul operation, not yet in the Tensor API)
- Softmax over a variable-length dimension
- Output projection Dense
- Optional causal mask (Phase 3)

---

##### 4d. LayerNorm

**What it is:** normalization applied independently to each token's vector.

After attention and FFN operations, activation magnitudes can drift — some vectors become very large, some very small. This destabilizes subsequent layers. Normalization is the fix.

**Formula:** for a single vector `x` of length `d_model`:

$$\hat{x}_i = \frac{x_i - \mu}{\sigma + \varepsilon}, \quad \mu = \frac{1}{d}\sum_i x_i, \quad \sigma = \sqrt{\frac{1}{d}\sum_i (x_i - \mu)^2}$$

Then apply learned scale `γ` and shift `β` (both vectors of length `d_model`):

$$y_i = \gamma_i \hat{x}_i + \beta_i$$

`γ` and `β` are initialized to 1 and 0, meaning "do nothing initially". They are learned weights — the optimizer adjusts them so the normalization doesn't remove useful structure.

**Why not BatchNorm?** BatchNorm normalizes across the batch dimension (per-feature statistics). For sequences this is unstable: batch size varies, sequences have different lengths, and during inference you might process one token at a time. LayerNorm normalizes within a single vector — no dependency on the rest of the batch.

**Parameters:** `2 × d_model` (one `γ` and one `β` vector). Tiny — for `d_model=768` that is 1 536 floats.

**Status in NNStudio:** not implemented. Forward pass is a pure tensor operation (mean, std, elementwise scale+shift). Backward requires the gradient through mean and variance — slightly involved but closed-form. A good early implementation target because it has no exotic topology requirements.

---

##### 4e. GELU (activation)

**What it is:** Gaussian Error Linear Unit — the standard activation in the FFN block of modern Transformers.

$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the standard Gaussian cumulative distribution function. Because computing $\Phi$ exactly requires numerical integration, a fast approximation is used in practice:

$$\text{GELU}(x) \approx 0.5 \cdot x \cdot \left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)$$

**Compared to ReLU:**
- ReLU: hard zero for `x < 0`, linear for `x > 0`. The sharp kink at 0 can cause neurons to "die" (stuck at zero gradient permanently).
- GELU: smoothly tapers near zero instead of clipping. Slightly negative values still pass a fractional signal through. Gradient exists everywhere.

Practically the difference is small in shallow networks. In very deep Transformers with 96+ layers (GPT-3) the accumulated smoothness advantage compounds.

**Status in NNStudio:** GELU can be implemented as an `IActivation` right now — it is a pure elementwise operation, same contract as ReLU/Sigmoid. The only thing needed is the math in `apply()` and the derivative `Φ(x) + x·φ(x)` (where `φ` is the Gaussian PDF) in `gradient()`.

---

#### Full data flow, annotated

```
Input: sequence of T token IDs  [T]  (integers)

[Embedding]      x = E[token_ids]                    shape: [T, d_model]
[+PE]            x = x + PE[0:T]                     shape: [T, d_model]

For each of N blocks:
  [LayerNorm]    x_norm = LN(x)                      shape: [T, d_model]
  [Attention]    attn = MultiHead(Q=x_norm, K=x_norm, V=x_norm)  shape: [T, d_model]
  [+ residual]   x = x + attn                        shape: [T, d_model]

  [LayerNorm]    x_norm = LN(x)                      shape: [T, d_model]
  [Dense→GELU]   h = GELU(x_norm @ W1 + b1)          shape: [T, 4×d_model]
  [Dense]        ffn = h @ W2 + b2                   shape: [T, d_model]
  [+ residual]   x = x + ffn                         shape: [T, d_model]

[LayerNorm]      x = LN(x)                           shape: [T, d_model]
[Dense head]     logits = x @ W_vocab + b_vocab       shape: [T, vocab_size]
[Softmax]        probs = softmax(logits[-1])          shape: [vocab_size]
                 → sample or argmax next token
```

---

**Key architectural choices:**
- `d_model` = embedding dimension (width of every token vector throughout the network)
- `n_heads` = number of parallel attention heads (each sees a `d_model/n_heads` subspace)
- FFN hidden is always `4×d_model` wide (convention from the 2017 paper; still used unchanged)
- GELU standard activation in FFN; ReLU also works
- Every sub-block has a residual/skip connection — this is why stacking 96 blocks is stable

**Typical sizes for text reasoning / QA:**

| Model | d_model | Heads | Layers | Params | Context window |
|---|---|---|---|---|---|
| GPT-2 small | 768 | 12 | 12 | 117 M | 1 024 tokens |
| GPT-2 medium | 1 024 | 16 | 24 | 345 M | 1 024 tokens |
| GPT-2 XL | 1 600 | 25 | 48 | 1.5 B | 1 024 tokens |
| GPT-3 | 12 288 | 96 | 96 | 175 B | 2 048 tokens |
| Llama 2 7B | 4 096 | 32 | 32 | 7 B | 4 096 tokens |
| Llama 2 70B | 8 192 | 64 | 80 | 70 B | 4 096 tokens |
| GPT-4 (est.) | ~16 384 (MoE) | ~128 | ~120 | ~1.8 T (MoE) | 8 192+ tokens |

**For NNStudio's text parsing / reasoning / QA use case:**  
A Llama 2 7B-class model is the practical minimum for reliable open-domain reasoning. That is 7 billion floats × 4 bytes = **28 GB** just for weights in fp32; 14 GB in fp16. Training from scratch is not realistic — fine-tuning a pre-trained checkpoint (LoRA or full fine-tune) is the standard approach. NNStudio would load an existing `.onnx` or `.nns` checkpoint and run inference or fine-tuning on specific layers (`frozen = true` on base weights, `frozen = false` on LoRA adapters).

**Status in NNStudio:**

| Layer | Status | Blocker |
|---|---|---|
| `Embedding` | ❌ not implemented | Tensor API accepts float only; integer index input needs design decision |
| `LayerNorm` | ❌ not implemented | No blocker — good early target |
| `GELU` | ❌ not implemented as `IActivation` | No blocker — easy after ADR-020 migration |
| `MultiHeadAttention` | ❌ not implemented | Needs batch-matmul in Tensor API + scaled dot-product |
| `ComputeGraph` (skip connections) | ❌ Phase 3 | Without this, residual connections cannot be expressed |
| FFN (Dense→GELU→Dense) | ✅ Dense ready, GELU missing | GELU is the only missing piece |

Overall: **Phase 3–4 target.** LayerNorm and GELU are the two easiest to add first.

---

### Template 5 — ResNet block (skip connection)

**Use for:** deep networks where vanilla MLP would suffer vanishing gradients.

```
x → Dense(n) → ReLU → Dense(n) → + x  ← skip: add original x back
                                     → ReLU
```

The skip connection (residual connection) routes the gradient directly around the block. Even if the block adds near-zero contribution, the gradient highway through the skip ensures early layers still receive a clean signal. This is why networks with hundreds of layers became trainable.

**Status in NNStudio:** requires `ComputeGraph` for the branching topology. `Sequential` cannot express a skip connection (it is strictly linear). Phase 3.

---

### On extending `ILayer` for genuinely new layer types

Plugging in a new activation: straightforward (Phase 2 `IActivation`, ADR-020).

Plugging in a new *structural* layer type (Conv2D, MultiHeadAttention, custom graph node): the `ILayer` interface itself is the extension point, but the downstream impact is non-trivial:

| Component affected | Impact |
|---|---||
| `ILayer::forward/backward` | Must be implemented correctly including `lastInput_` / context management |
| `ILayer::build(Shape)` | Must return correct output shape so the shape-relay chain works |
| `ILayer::parameters()` | Must expose all learnable parameters for the Optimizer |
| `ComputeGraph` | New layer must register its ops as graph nodes (Phase 3) |
| `ONNX export` | Must map to an ONNX op-node or a custom op (Phase 4) |
| Studio UI | Need a property form, a topology diagram node, KB help entry |
| Plugin SDK | Must be distributable as a signed `.nnsp` plugin (Phase 2 ABI freeze) |

This is not blocked — it is the intentional design. But it means adding a genuinely new structural layer type is a multi-phase commitment, not a weekend task. The engine is correctly designed for it; the scope is simply larger than adding an activation.

---

## Appendix — Vocabulary

All short-hand identifiers used in this document, grouped by category.

---

### A — C++ field names (engine source code)

Trailing `_` is the project convention for private member variables.

| Symbol | Type | Lives in | Meaning |
|---|---|---|---|
| `data_` | `std::shared_ptr<float[]>` | `Tensor` | The flat heap array of floats that stores all values |
| `shape_` | `std::vector<int64_t>` | `Tensor` | Logical dimensions, e.g. `{3, 2}` = 3 rows × 2 columns |
| `strides_` | `std::vector<int64_t>` | `Tensor` | Navigation map: `strides_[k]` = how many floats to skip to move +1 in dimension k |
| `numel_` | `int64_t` | `Tensor` | Cached product of all `shape_` dimensions = total element count |
| `grad_` | `std::optional<Tensor>` | `Tensor` | Sibling tensor holding accumulated gradients; absent until first backward pass |
| `W_` | `Parameter` | `Dense` | Weight matrix parameter; `W_.tensor` has shape `{outF, inF}` |
| `b_` | `Parameter` | `Dense` | Bias parameter; `b_.tensor` has shape `{outF}` |
| `lastInput_` | `Tensor` | `Dense`, `ReLU`, `LeakyReLU`, `GELU` | Input saved during `forward()` so `backward()` can compute `dW = gradOut^T @ lastInput_` (Dense), or the sign-mask derivative (ReLU group) |
| `lastOutput_` | `Tensor` | `Sigmoid`, `TanhAct`, `Softmax` | Output saved during `forward()` because the derivative of these functions is expressible purely in terms of the output value — cheaper than recomputing from input |
| `layers_` | `std::vector<LayerPtr>` | `Sequential` | Ordered list of owned child layers |
| `training_` | `bool` | `ILayer` | True = train mode (Dropout active, BatchNorm uses batch stats); false = eval mode |
| `built_` | `bool` | `ILayer` | True after `build()` has allocated weight matrices; guards against double-allocation |
| `m_` | `std::unordered_map<…, Tensor>` | `Adam` | First-moment (mean gradient) running average per parameter |
| `v_` | `std::unordered_map<…, Tensor>` | `Adam` | Second-moment (mean squared gradient) running average per parameter |
| `t_` | `int64_t` | `Adam` | Step counter; used in bias-correction exponents |

---

### B — Math notation (standard ML convention)

| Symbol | Reads as | Meaning |
|---|---|---|
| `W` | "weights" | The weight matrix of a Dense layer; shape `[outF, inF]` |
| `b` | "bias" | The bias vector of a Dense layer; shape `[outF]` |
| `x` / `X` | "input" | Input to a layer; uppercase = batch `[B, inF]`, lowercase = single sample `[inF]` |
| `y` / `Y` | "output" | Output of a layer; uppercase = batch `[B, outF]`, lowercase = single sample `[outF]` |
| `@` | "matmul" | Matrix multiplication operator (NumPy/Python notation used in this document) |
| `^T` | "transposed" | Matrix transpose; swaps rows and columns |
| `L` | "loss" | Scalar loss value — a single float measuring total prediction error for one step |
| `dW` | "delta W" | Shorthand for the gradient of W (dL/dW); same shape as W |
| `db` | "delta b" | Shorthand for the gradient of b (dL/db); same shape as b |
| `dX` | "delta X" | Gradient passed downward to the previous layer; same shape as X |
| `g_t` | "gradient at step t" | Raw gradient value at Adam step t; input to the moment update equations |
| `m_t` | "first moment" | Adam's running mean of gradients (momentum); smooths noisy gradient updates |
| `v_t` | "second moment" | Adam's running mean of squared gradients; enables per-weight adaptive learning rates |
| `beta_1` | "beta one" | Adam decay rate for first moment; typically 0.9 |
| `beta_2` | "beta two" | Adam decay rate for second moment; typically 0.999 |
| `alpha` | "alpha" / "learning rate" | Step size scalar; scales how far each update moves in parameter space |
| `epsilon` | "epsilon" | Small constant (e.g. 1e-8) added to denominator to prevent division by zero |
| `theta` | "theta" | Generic parameter symbol; stands for any W or b in the optimizer update rule |

---

### C — Dimension shorthands (used in examples and diagrams)

| Symbol | Expands to | Meaning |
|---|---|---|
| `B` | batch size | Number of samples processed simultaneously in one forward/backward call |
| `inF` | `inFeatures` | Number of input values arriving at a layer per sample |
| `outF` | `outFeatures` | Number of output values produced by a layer per sample |
| `M`, `N`, `K` | matrix dims | Generic matrix dimension names used in GEMM descriptions: `C[M,N] = A[M,K] @ B[K,N]` |

---

### D — Didactic shorthands (invented for this document)

These identifiers appear only in explanatory examples, not in production source code.

| Symbol | Context | Meaning |
|---|---|---|
| `h1` | ONNX node name example | Hypothetical node `"h1"` in an ONNX graph — represents a hidden-layer output |
| `h1_biased` | ONNX node name example | Same node after the bias Add op; shows how `y = W*x + b` becomes two ONNX nodes |
| `x1`, `x2` | Dense 2->3 example | First and second features of a 2-feature input sample |
| `w00`, `w10`, ... | Dense 2->3 example | Individual weight entries: w_{neuron, input} |
| `y0`, `y1`, `y2` | Dense 2->3 example | Individual neuron pre-activation outputs |
| `b0`, `b1`, `b2` | Dense 2->3 example | Bias values for each neuron |
| `x(0)`, `x(1)` | Batching example | Superscript = sample index within a batch |

---

### E — Standard library and industry terms

| Term | Meaning |
|---|---|
| `GEMM` | General Matrix Multiply — the fundamental operation underlying Dense, Conv2D, and attention. Dispatched via `IBackend::matmul`. |
| `noalias()` | Eigen hint: output buffer does not overlap any input — skips defensive internal allocation |
| Row-major / C-order | Layout where the last index changes fastest; elements of one row are contiguous. Used by NNStudio `Tensor`. |
| Col-major / Fortran-order | Layout where the first index changes fastest. Used internally by Eigen. |
| `Result<T>` | Engine error-or-value return type. Holds either a `T` value or an error. Methods: `.value()`, `.error()`, `.hasValue()`. |
| `LayerPtr` | `std::shared_ptr<ILayer>` — ownership handle for a layer inside `Sequential::layers_` |
| `Shape` | `std::vector<int64_t>` — alias for tensor dimension lists throughout the engine |
| `Parameter` | `struct { std::string name; Tensor tensor; bool frozen; }` — the unit the Optimizer reads and updates |
| Bias-correction | Adam technique: dividing `m_t`/`v_t` by `(1 - beta^t)` counteracts near-zero initialisation at training start |
| Weight pruning | Post-training zeroing or removal of near-zero weights to reduce inference cost — not yet implemented |
| Transfer learning | Using a pre-trained model's weights as starting point, training only certain layers (the "head") |
| `frozen` | Flag on a `Parameter`: Optimizer skips its update; layer still computes and backprops its gradient |
| ONNX | Open Neural Network Exchange — standard format for exporting models as a flat graph of named op-nodes (MatMul, Add, Relu, ...) |
| DAG | Directed Acyclic Graph — general model topology enabling skip connections and multi-head architectures (cf. `Sequential` which is linear) |
| BCE | Binary Cross-Entropy loss. For one sample: `L = -[y*log(p) + (1-y)*log(1-p)]` |
