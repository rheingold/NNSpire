# ADR-020 — `IActivation` Functional Contract: Stateless Activation with Caller-Owned Context

**Date**: 2026-04-03  
**Status**: Accepted  
**Decided by**: Project founder (design review session)

---

## Context

NNSpire currently implements activation functions (`ReLU`, `LeakyReLU`, `Sigmoid`,
`TanhAct`, `Softmax`, `GELU`) as full `ILayer` descendants via the intermediate
class `ActivationBase`.  Each activation stores the input or output of its last
`forward()` call as a private member (`lastInput_` or `lastOutput_`) so that
`backward()` can use it to compute the derivative.

This design works correctly for the current single-threaded, single-use-site XOR
workbench.  However, when evaluated against likely future scenarios it carries
hard-wired limitations:

| Future scenario | Impact of stateful activation |
|---|---|
| Multi-threaded inference (web server, GPU backend) | Thread A's `forward()` overwrites `lastInput_` while Thread B's `backward()` reads it → silent wrong gradients, no crash |
| Shared activation object in branching graphs (ResNet, attention) | Second call to `forward()` clobbers state from first call before its `backward()` runs → same silent failure |
| ONNX export / `EvalTrace` graph walking | Saved state is an invisible side-channel; the tracer cannot observe it without instrumenting internals |
| Gradient checkpointing (drop saved tensors, recompute on demand) | Activation always saves; checkpointing requires dropping `lastInput_` → `backward()` crashes |

These are not hypothetical: they are precisely the scenarios that arise once Phase 2
(Plugin SDK) is published and third parties write their own activations following the
`ActivationBase` pattern.

Three options were evaluated:

**Option A — Keep stateful `ActivationBase` as the published plugin extension point.**  
Simple for plugin authors.  Fails under concurrency and shared instances.  Cannot
support checkpointing.  ONNX tracing requires internal instrumentation.

**Option B — Caller-managed state: activation is stateless, receives saved tensor as parameter.**  
```cpp
Tensor apply(const Tensor& x);
Tensor gradient(const Tensor& saved, const Tensor& gradOut);
```
Reentrant.  Activation declares which tensor to save (`input` or `output`) via a
companion enum.  Caller (`FunctionLayer`) stores it.  Checkpointing possible.
One extra parameter on `gradient()`.

**Option C — Functional pair: `apply()` returns both output and the context needed by `backward()`.**  
```cpp
struct ForwardResult { Tensor output; Tensor ctx; };
ForwardResult      apply(const Tensor& x);
Tensor             gradient(const Tensor& ctx, const Tensor& gradOut);
```
Activation decides on its own what to store as `ctx` (`x` for ReLU, `output` for
Sigmoid).  Caller holds `ctx` until `backward()`.  Analogous to PyTorch's
`torch.autograd.Function.save_for_backward()`.  Fully reentrant, no hidden
side-channels, checkpointing-ready, ONNX-traceable.

---

## Decision

The **published `IActivation` interface** (Phase 2 Plugin SDK) shall use **Option C**
— the functional pair contract:

```cpp
namespace NNSpire {

struct ActivationForward {
    Tensor output;   // result of apply(x), passed to the next layer
    Tensor ctx;      // whatever backward() will need: x for ReLU, output for Sigmoid
                     // shared_ptr semantics — no float data is copied
};

class IActivation {
public:
    virtual ~IActivation() = default;

    /// Human-readable name, e.g. "relu", "sigmoid".
    virtual std::string name() const = 0;

    /// Forward pass.  Returns the activated output AND the context tensor
    /// that will be passed back to gradient().  The activation chooses
    /// what to store as ctx (input, output, or any derived tensor).
    virtual ActivationForward apply(const Tensor& x) = 0;

    /// Backward pass.  ctx is exactly the Tensor returned in the forward
    /// ActivationForward::ctx.  Returns dX = dLoss/dInput.
    virtual Tensor gradient(const Tensor& ctx, const Tensor& gradOut) = 0;
};

} // namespace NNSpire
```

A companion `FunctionLayer` adapter wraps any `IActivation` into a full `ILayer`
suitable for placement in a `Sequential`:

```cpp
class FunctionLayer : public ILayer {
    std::shared_ptr<IActivation> activation_;
    Tensor                       ctx_;          // held between forward and backward
public:
    explicit FunctionLayer(std::shared_ptr<IActivation> fn);

    Result<Tensor> forward(const Tensor& x) override {
        auto r = activation_->apply(x);
        ctx_ = r.ctx;           // store, awaiting backward()
        return Result<Tensor>(std::move(r.output));
    }

    Result<Tensor> backward(const Tensor& gradOut) override {
        return Result<Tensor>(activation_->gradient(ctx_, gradOut));
    }
    // parameters(), build(), setTraining(), save(), load() — trivial no-ops
};
```

### What this means for current internal activations

The six existing `ActivationBase` descendants are **not changed** in this ADR.
They remain as `ILayer`-based internal implementations for the current engine.
In Phase 2 they will be refactored to implement `IActivation` by:

1. Removing `lastInput_` / `lastOutput_` member fields.
2. Moving the saved value into the return of `apply()` as `ctx`.
3. Accepting `ctx` as the first parameter of `gradient()`.

This is a mechanical, testable refactor of ~5 lines per activation.

### What plugin authors write

A minimal custom activation implementing `IActivation`:

```cpp
// ReLU written to the new interface — no state, no ILayer boilerplate
class MyReLU : public NNSpire::IActivation {
public:
    std::string name() const override { return "my_relu"; }

    ActivationForward apply(const Tensor& x) override {
        Tensor out = clamp(x, 0.0f, std::numeric_limits<float>::max());
        return { out, x };   // ctx = input (needed by gradient())
    }

    Tensor gradient(const Tensor& ctx, const Tensor& gradOut) override {
        // ctx is the original input x
        Tensor mask = /* ctx > 0 ? 1 : 0 */ ...;
        return backend().mul(gradOut, mask);
    }
};
```

Compare to the same author trying to implement `ActivationBase` (Option A): they must
also implement `build()`, `parameters()`, `save()`, `load()`, `setTraining()` —
none of which are relevant to an activation.  The cognitive mismatch is real and would
appear in every piece of plugin documentation and every forum question.

---

## Consequences

### Positive

- `IActivation::apply()` is a pure function from the caller's perspective — no hidden
  mutable state, fully reentrant, safe under concurrent calls with distinct `ctx` owners.
- Shared activation objects are safe: each call site holds its own `ctx`.
- Gradient checkpointing is possible: discard `ctx`, store only original input,
  recompute `ctx` on demand during backward.
- ONNX / `EvalTrace` graph walking: the `(x → {output, ctx})` signature is an
  explicit, inspectable contract; no instrumentation of internals needed.
- The `ctx` field is a `Tensor` (shared-ptr semantics) — no float data is copied on
  return; overhead is two atomic reference-count increments.
- Plugin authors implement exactly 3 methods for a new activation.

### Negative / constraints

- `FunctionLayer` retains one stateful member (`ctx_`) between `forward()` and
  `backward()`.  It is the activation adapter's responsibility, not the activation
  itself.  The activation remains stateless; `FunctionLayer` is a thin, auditable
  wrapper.  This is acceptable.
- Existing internal `ActivationBase` classes must be migrated before Phase 2.  This
  is planned, mechanical work, not design risk.
- If a future activation genuinely needs multiple saved tensors, `ctx` may need to
  become a `std::vector<Tensor>` or a small named struct.  This is deferred; a single
  `Tensor` covers all six current activations and the vast majority of practical cases.

### Follow-on work

- [ ] `core/include/NNSpire/core/IActivation.h` — add new header
- [ ] `FunctionLayer.h / .cpp` — adapter implementation  
- [ ] Migrate `ReLU`, `LeakyReLU`, `Sigmoid`, `TanhAct`, `Softmax`, `GELU` to the new interface
- [ ] Update `PLUGIN-SDK.md` to document `IActivation` as the extension point for activations
- [ ] Update `blueprints.md` §3.8 to reflect the new interface
- [ ] Write unit tests: reentrant call, shared instance with two concurrent callers

---

## Alternatives not chosen

**Option A (stateful `ActivationBase` as SDK contract)** — rejected: threading and
shared-instance failures are silent (wrong gradients, no crash), would appear only in
production under load, and would be extremely hard to diagnose.  Publishing an API
with a known silent-failure mode is not acceptable.

**Option B (caller-managed state with explicit `WhatToSave` enum)** — rejected:
requires the activation to declare what it wants saved via an enum, then the caller
stores it and passes it back.  Option C is strictly simpler: the activation *chooses
what to return as `ctx`* without any protocol negotiation.  The only advantage of B
was familiarity to authors who know the field-parameter pattern; C is cleaner.
