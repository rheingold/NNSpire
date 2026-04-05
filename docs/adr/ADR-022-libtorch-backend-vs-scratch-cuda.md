# ADR-022 — LibTorch as Production Backend vs. Scratch CUDA Implementation

> **Foundational note:** This ADR records the most fundamental architectural question of the entire project — arguably it should have been ADR-000. It is placed here because it surfaced organically during Phase 1 implementation rather than at project inception. That is how most real foundational decisions actually happen.

**Status:** ACCEPTED  
**Date:** 2026-04-06  
**Author:** NNStudio project  

---

## Context / Problem Statement

NNStudio implements its own compute engine (Tensor, Dense, Adam, ComputeGraph, CpuBackend via Eigen) for didactical reasons — to make every internal NN operation visible and inspectable. During Phase 1 implementation it became clear that:

1. The compute engine (Tensor, layers, optimizers) is a reinvented wheel. PyTorch/LibTorch does this better, with 10 years of CUDA optimization and a dedicated team.
2. Everything *around* the compute engine (plugin trust/PKI, plugin ABI, project format `.nns`, UI, workflow, export pipeline) is genuinely novel — no existing NN framework has it.
3. The performance concern is **scoped exclusively to Studio-side training** (UI, CLI, embedded server). Exported models run on their own independent runtimes (PyTorch, ONNX Runtime, TensorRT) regardless of which backend NNStudio used during training — so the backend choice has zero effect on the performance of shipped models.

| Runtime context | Backend affects performance? |
|---|---|
| NNStudio training (UI / CLI) | ✅ Yes — this is the scope of this ADR |
| NNStudio embedded server (Ollama-like) | ✅ Yes |
| Exported model on PyTorch runner | ❌ No — independent runtime |
| Exported model on ONNX Runtime / TensorRT | ❌ No — independent runtime |

The Linus Torvalds analogy: Torvalds wrote a kernel because MINIX couldn't do what he needed — there was a genuine gap. The gap NNStudio fills is the *studio* (trust, plugin ecosystem, visual workflow, project format), not the compute engine itself.

---

## Decision

**Option B accepted: Add `LibTorchBackend : IBackend` as an optional production backend alongside the existing Eigen `CpuBackend`.**

- `CpuBackend` (Eigen) is retained permanently as the **reference and didactical backend**. It makes every operation transparent, is self-contained, requires no external dependencies, and is sufficient for teaching, prototyping, and interactive experimentation on small models.
- `LibTorchBackend` is added as an **opt-in production backend** (`NN_ENABLE_LIBTORCH=ON` CMake flag). It delegates all compute operations to LibTorch's optimized CUDA/cuDNN kernels via the existing `IBackend` interface (~8 virtual methods to implement).
- The scratch CUDA backend originally planned for Phase 2 is **cancelled**. Its didactical value is covered by `CpuBackend`; its performance value is covered by `LibTorchBackend`.
- The `IBackend` interface is **not changed**. `LibTorchBackend` is a pure addition — no existing code is modified.

**Prerequisite:** The Tensor internal buffer must be refactored from `shared_ptr<float[]>` to `shared_ptr<void>` + `DType` dispatch before `LibTorchBackend` can be implemented. This is already recorded as a blocker in the Cross-Framework Compatibility TODO section.

---

## Consequences

**Positive:**
- CUDA, cuDNN, and multi-GPU training become available to NNStudio users at production quality without a multi-year kernel implementation effort.
- The `torch_compat.h` alignment story becomes trivially true for the production path: "NNStudio literally runs on LibTorch under the hood when you choose it."
- `CpuBackend` remains available for learning, debugging, and zero-dependency builds.
- Plugin authors using `IBackend` get LibTorch acceleration transparently with no API changes.
- Shipped/exported models are completely unaffected — they run on their own runtime.

**Negative / costs:**
- LibTorch is a large dependency (~500 MB). Must be optional (`NN_ENABLE_LIBTORCH`), not required.
- Tensor buffer refactor (float[] → void* + dtype) required before implementation.
- One-time integration work estimated at 2-3 weeks.
- Tests must cover both backends; CI gains a separate LibTorch-enabled preset.

**Deferred:**
- Whether `LibTorchBackend` becomes the *default* when available, or always opt-in via CLI/UI — deferred to Phase 3 UI work.
- Whether the Tensor internal buffer actually *is* a `torch::Tensor` when LibTorch is active (zero-copy path), or whether it copies in/out — deferred to implementation; zero-copy is preferred but not required for correctness.

---

## Conversation Record

The following exchange is reproduced verbatim as the basis for this ADR.

---

### User prompt

> A small sidenote, a part of our dialogue worth noting somewhere: I know, that what we are doing here is something of the kind of that known story how Linux began: Linus Torvalds simply wanted to test the x386 protected mode and while he fiddled around with that, he has laid the foundations of the nowadays Linux kernel. But I'm afraid I'm neither Linus Torvalds neither proficient enough in this, so I'm afraid we are on the other extreme end of the chain: re-inventing a wheel, in this case building an entire NN framework (eco)system in contrary to those, that already exist. Any thoughts about that? Its something of the wedding ceremony paraphrase "If anyone knows a reason why this is not the way to go (and we should simply adhere to create a UI controller of existing frameworks), so shall he speak up now or remain silent forever."

---

### Assistant response

Speaking up now, since you asked directly.

**The honest diagnosis:**

The "reinvented wheel" concern is partially valid, but it's aimed at the wrong layer. Here's the precise breakdown:

| What we built | Reinvented wheel? | Verdict |
|---|---|---|
| Plugin trust system (PKI, TUPs, TrustStore) | No — nothing like this exists in any NN framework | ✅ Genuinely novel |
| Plugin ABI + sandbox architecture | No — PyTorch has no such thing | ✅ Genuinely novel |
| Project format (`.nns`), UI layer, workflow | No | ✅ Genuinely novel |
| `Tensor`, `Dense`, `Adam`, `ComputeGraph` | Yes — PyTorch does this better, with 10 years of CUDA optimization | ⚠️ Wheel |

The NN compute engine is the wheel. The studio around it is not.

**The uncomfortable truth about our compute engine:**

Our `CpuBackend` will never match cuDNN-optimized LibTorch on real workloads. A user training a transformer on our engine vs. PyTorch would see 10-100x slower performance. We couldn't realistically fix that without essentially reimplementing CUDA kernels, which is a multi-year effort by a dedicated team.

**What Linus actually did, precisely:**

He wrote his own kernel *because* he couldn't use MINIX for what he needed. There was a genuine gap. The gap here is the *studio* — the UI, the plugin ecosystem, the trust chain for distributing model components. There is no PyTorch Studio with sandboxed signed plugins, visual training inspection, and a project format. **That** is the MINIX gap we're filling.

**The architecture that resolves this:**

The `LibTorchBackend : IBackend` path is not a compromise — it's *the correct answer*:

```
NNStudio (our work — genuinely novel)
├── UI, project format, plugin trust, workflow
├── Layer/Trainer/ComputeGraph (our API, torch-compatible naming)
└── IBackend
    ├── CpuBackend (ours, Eigen — stays as reference + didactical)
    └── LibTorchBackend (production — delegates ALL compute to LibTorch)
```

With `LibTorchBackend`: CUDA support is free, ONNX export is trivial (`torch.onnx.export`), the `torch_compat` story becomes "we literally ARE torch under the hood for production use", and the didactical backend remains available for learning purposes.

**Verdict: don't abandon what we've built. Retrofit the compute, keep the studio.**

The work isn't wasted — the architecture, the plugin SDK, the trust system, the test suite, the API design — none of that goes away. Adding `LibTorchBackend` as the production backend would take perhaps 2-3 weeks and would make NNStudio genuinely competitive on performance without abandoning a single line of the novel work.

---

### Follow-up exchange (same session)

**User:**

> please simply copy and paste the above prompt and your response as a new ADR (we will fill it with decision and problem statement few prompts later)
>
> so your question hints me to the fact, that you were intending to implement the backend for CUDA "from scratch" (simpler, lower perf, didactical rather for understanding of mine/users learning NNs), however now we are discussing the option to — at some higher costs — add another backend, the same CUDA, YET not being called directly, but using ("plugging in into our reinvented-wheel solution a few more already existing spokes") the NVIDIA LibTorch implementation, of course the costs being some rewrites to accommodate it, tests, and maybe architectural limitations if that high-end backend should be used.
>
> Nevertheless, if I understand clearly from the architecture so far, we are NOT compromising the performance of the RESULTING networks, as they will be run on independent runners anyway, so the performance issue is only aiming at NNStudio runners (UI, CLI, some embedded like-to-be-Ollama server...), not on the resulting networks published and run on Torch-only based runners, right? (of course, the plugins and extensibility may be an issue, but that is ok, that we shall solve as it comes, that's understandable — anyway, we do not aim to reconstruct the entire AI NN industry here :-)

**Assistant:**

Yes — your understanding is exactly right, and this is an important clarification worth recording precisely.

**Performance scope of the backend question:**

| Runtime context | Backend used | Performance concern? |
|---|---|---|
| NNStudio training (UI/CLI) | Our `CpuBackend` or `LibTorchBackend` | ✅ Yes — this is what the ADR is about |
| NNStudio embedded server (Ollama-like) | Our backends | ✅ Yes — same concern |
| **Exported model running on PyTorch runner** | Pure PyTorch / LibTorch | ❌ **Not our concern** — it's their runtime |
| **Exported model on ONNX Runtime** | ONNX Runtime kernels | ❌ **Not our concern** |
| **Exported model on TensorRT** | TensorRT | ❌ **Not our concern** |

The backend performance question is entirely scoped to *Studio-side training and inference* — the process of building and iterating on the model. Once exported (`.onnx`, `.pt`, `.nns` → runner), the model runs on whatever runtime the user deploys to, which is completely independent of NNStudio's backends.

This means the stakes of the backend decision are bounded:
- High-volume production training (GPT-scale, days of A100 time): LibTorch backend matters a lot
- Interactive experimentation (small models, teaching, prototyping): our Eigen CpuBackend is perfectly adequate — response times are human-scale
- Exported model performance: not affected by this decision at all

**What `LibTorchBackend` actually costs:**

It is not a rewrite of the engine. `IBackend` already defines the contract:
```cpp
virtual Tensor matmul(const Tensor&, const Tensor&) = 0;
virtual Tensor elementWise(Op, const Tensor&, const Tensor&) = 0;
// ... ~8 more operations
```

`LibTorchBackend` implements those ~8 methods by calling `torch::` equivalents and wrapping the result back into our `Tensor`. The Tensor dtype-generic buffer change (ADR blocker item) is what enables this — the buffer becomes a `torch::Tensor` internally when the LibTorch backend is active.

That is a well-bounded, testable piece of work. It does not touch the plugin system, the trust chain, the UI, the project format, or any of the novel parts.

---

## Options Considered

### Option A — Keep only our Eigen CpuBackend + scratch CUDA (original plan)
- Didactically complete; students see every computation
- CUDA performance gap of 10-100x vs LibTorch on real models
- CUDA implementation is a multi-year effort to match cuDNN quality

### Option B — Add `LibTorchBackend : IBackend` as production backend (proposed)
- CpuBackend stays for didactical use and testing
- LibTorch backend delegates compute to optimized NVIDIA kernels
- CUDA, cuDNN, TensorRT acceleration are free
- `torch_compat` alignment becomes trivially true for the production path
- Cost: ~2-3 weeks implementation + Tensor buffer refactor (already planned)

### Option C — Abandon our compute engine, wrap LibTorch entirely
- Lowest implementation cost
- Loses the didactical CpuBackend and its pedagogical value
- Not proposed — the whole-engine replacement is unnecessary given Option B

---

## Notes for Decision Filling

Key questions to resolve:
1. Does `LibTorchBackend` become the *default* backend or opt-in (`--backend=libtorch`)?
2. Does the LibTorch dependency become required or optional (CMake `NN_ENABLE_LIBTORCH`)?
3. Does the Tensor internal buffer unify with `torch::Tensor` when LibTorch is active, or does it copy?
4. Plugin authors who use our `IBackend` API — do they get LibTorch acceleration transparently?
