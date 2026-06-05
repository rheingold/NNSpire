# ADR-015 — Quantum Backend via Qiskit Python Bridge

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNSpire's long-term goal (Phase 6) is to support hybrid classical-quantum neural network
execution, where selected layers or graph sub-trees are compiled to quantum circuits and
executed on quantum hardware or simulators.

This requires:
1. A quantum circuit representation compatible with one or more quantum SDKs.
2. Execution on real quantum hardware (IBM Quantum, IonQ, etc.) or simulators (AerSimulator).
3. Integration with the existing `IBackend` dispatch system (ADR-012).

The quantum computing SDK landscape as of 2026:

| SDK | Language | Hardware access | Notes |
|---|---|---|---|
| **Qiskit** | Python | IBM Quantum, many via plugins | Largest community; most mature |
| Cirq | Python | Google, IonQ | Google-focused |
| PennyLane | Python | Many backends | Strong ML integration |
| Q# | C# / hybrid | Azure Quantum | Microsoft-specific |

---

## Decision

The quantum backend is implemented as a `QuantumBackend` shared library that bridges
to **Qiskit** via the NNSpire Python bridge (pybind11, ADR-005).

Architecture:
- `QuantumBackend` implements `IBackend` (ADR-012).
- Selected operations on tensors with `device = Device::QUANTUM` are translated to
  Qiskit `QuantumCircuit` objects by a **circuit compiler** in `NNSpire/backends/quantum/`.
- Circuit execution is dispatched to Qiskit's backend registry
  (`AerSimulator` for local simulation; IBM Quantum provider for real hardware).
- Results (probability distributions / expectation values) are mapped back to tensors.

This is intentionally Phase 6 — a stub `QuantumBackend` that returns `ErrorCode::NOT_IMPLEMENTED`
is created in Phase 0/1 to reserve the interface, but real circuit compilation and
execution is deferred.

PennyLane is noted as a potential alternative / complement (strong classical-quantum ML
integration) and should be re-evaluated when Phase 6 begins.

---

## Consequences

**Positive**
- Qiskit's large ecosystem and IBM Quantum access give the widest hardware reach.
- The Python bridge (already required for ADR-004) means no additional FFI layer.
- Stub backend from day one ensures the `Device::QUANTUM` tag and `IBackend` interface
  never need to be retrofitted.

**Negative / constraints**
- Qiskit is Python-only — the C++ quantum backend is a thin wrapper delegating to Python.
  The dual-language principle (ADR-004) is technically satisfied (C++ calls Python),
  but there is no pure-C++ quantum circuit implementation.
- Quantum hardware access requires external API keys (IBM Quantum account) —
  stored in `ai_priv/ai_priv.md`, never in source.
- Quantum computing is evolving rapidly; SDK choice may need revision at Phase 6.

**Follow-on**
- `QuantumBackend` stub in `NNSpire/backends/quantum/` — created in Phase 1 alongside CPU backend.
- Circuit compiler design deferred to Phase 6.
- Re-evaluate PennyLane vs Qiskit at Phase 6 kickoff.
- IBM Quantum API key path recorded in `ai_priv/ai_priv.md`.
