# ADR-022 — LibTorch as Production Backend vs. Scratch CUDA Implementation

**Status:** DRAFT — decision pending  
**Date:** 2026-04-06  
**Author:** NNStudio project  

---

## Context / Problem Statement

*(To be filled in from session dialogue — see Conversation Record below.)*

---

## Decision

*(To be filled in.)*

---

## Consequences

*(To be filled in.)*

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
