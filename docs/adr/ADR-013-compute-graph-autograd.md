# ADR-013 — Compute Graph DAG for Automatic Differentiation

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

Training a neural network requires computing gradients of the loss with respect to every
learnable parameter — backpropagation. There are two broad implementation strategies:

| Strategy | Description | Notable users |
|---|---|---|
| **Define-by-run (dynamic graph)** | Record the forward pass into a DAG on the fly; `backward()` traverses it | PyTorch, raw autograd |
| Define-and-run (static graph) | Fully specify the graph before any data flows through it | Early TensorFlow, ONNX Runtime |

NNSpire is simultaneously a **learning tool** (users must be able to inspect the graph,
pause at any node, visualise gradients) and a **design tool** (users build architectures
interactively). Static graphs would make interactive construction and live inspection very
difficult.

---

## Decision

`ComputeGraph` is a **define-by-run directed acyclic graph** (dynamic autograd).

During the forward pass, each operation creates an `OpNode` that records:
- The operation type (enum aligned with ONNX operator names where applicable).
- References to input tensors.
- A reference to the output tensor.
- A `backward_fn` — the gradient function for autograd.

`Tensor::requires_grad = true` opts a tensor into graph recording.
Calling `graph.backward()` traverses the DAG in reverse topological order,
accumulating gradients into each participating tensor's `.grad` field.

### Learning / inspection extension: `EvalTrace`

An opt-in `EvalTrace*` parameter is threaded through `ILayer::forward()` and
`ILayer::backward()`. When non-null, the layer captures its inputs, outputs, and
gradients into the trace struct. `Trainer::setTraceMode(true)` enables this globally.  
When `nullptr` (normal training), the branch is a zero-cost no-op — no allocations,
no copies.

This mechanism powers the Studio's neuron visualiser and the gradient inspector panel.

### Graph serialisation

The DAG is serialisable to JSON for:
- UI topology display (the graph editor panel reads this JSON).
- Debugging / logging checkpoints.
- Diff-able storage in `.nns` files alongside the ONNX weight blob.

---

## Consequences

**Positive**
- Users can build and modify architectures interactively; the graph adapts at runtime.
- `EvalTrace` provides deep inspection without penalising production training speed.
- Graph topology JSON enables the UI panel to render the architecture without coupling
  the UI to the C++ graph object directly.

**Negative / constraints**
- Dynamic graphs use more memory during the forward pass than pre-optimised static graphs
  (all intermediate tensors must be retained for `backward()`).
- Graph optimisation (operator fusion, constant folding) must be applied explicitly
  at export time (ONNX export runs an optimisation pass); it is not applied during training.
- `backward_fn` closures must not capture raw pointers to tensors that may be freed —
  use `shared_ptr` reference counting.

**Follow-on**
- `ComputeGraph`, `OpNode`, `backward_fn` in `NNSpire/core/graph/`.
- `EvalTrace` struct in `NNSpire/core/graph/eval_trace.h`.
- Graph JSON schema documented in `NNSpire/core/graph/README.md`.
- Python binding: `NNSpire.graph.ComputeGraph` via pybind11.
