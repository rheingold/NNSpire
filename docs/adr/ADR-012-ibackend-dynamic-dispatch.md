# ADR-012 — `IBackend` Abstraction with Dynamically Loaded Backend Libraries

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio targets multiple compute backends:
- **CPU** — Eigen-based BLAS, for development, testing, and machines without a GPU.
- **CUDA** — cuBLAS / cuDNN-accelerated operations for NVIDIA GPUs.
- **Quantum** — Hybrid classical-quantum execution via Qiskit (Phase 6).
- (Future) Custom FPGA, Apple MPS, RISC-V accelerators via plugin backends.

If backend-specific code (CUDA headers, cuBLAS calls) is compiled into `nnstudio-core`,
the core library cannot be distributed to machines that lack the CUDA toolkit.
Static linking of backends also prevents hot-swapping (switching CPU↔CUDA at runtime).

---

## Decision

All compute operations in `nnstudio-core` are **dispatched through `IBackend`**.
No backend-specific code lives in the core library itself.

```cpp
class IBackend {
public:
    virtual ~IBackend() = default;

    virtual Tensor<float> matmul(const Tensor<float>&, const Tensor<float>&) = 0;
    virtual Tensor<float> elementwise_add(const Tensor<float>&, const Tensor<float>&) = 0;
    // ... all primitive operations

    virtual std::string_view name() const = 0;
    virtual DeviceType deviceType() const = 0;
};
```

Each backend is a **separately compiled shared library** (`nnstudio-backend-cpu.so`,
`nnstudio-backend-cuda.so`, etc.) loaded at runtime by `BackendRegistry`.

`BackendRegistry::active()` returns the currently selected backend.
Switching backends is done by calling `BackendRegistry::select(name)` before running
any compute — not mid-graph.

Backend shared libraries use the same **C ABI boundary** convention as plugins (ADR-003):
they export a `nnstudio_backend_descriptor()` C function returning a descriptor struct.

---

## Consequences

**Positive**
- Core library ships and compiles on any machine.
- CUDA backend is only required on machines that actually have a CUDA-capable GPU.
- New backends (Apple MPS, FPGA) can be added as plugins without modifying core.
- Backend selection is testable independently.

**Negative / constraints**
- Every new primitive operation must be added to `IBackend` interface (a single point of change).
- Backend dispatch introduces one level of virtual indirection per operation.
  Mitigated by batching: operations are issued in bulk tensors, not element-by-element.
- Mid-graph backend switches not supported; models must be designed for a single backend.

**Follow-on**
- `IBackend` defined in `nnstudio/core/include/nnstudio/core/backend/IBackend.h`.
- `BackendRegistry` singleton in `nnstudio/core/backend/`.
- CPU backend in `nnstudio/backends/cpu/` (Eigen dependency only, always available).
- CUDA backend in `nnstudio/backends/cuda/` (guards for CUDA toolkit presence in CMake).
- Quantum backend stub in `nnstudio/backends/quantum/` (Phase 6, Qiskit bridge).
- See ADR-015 for the quantum backend decision.
