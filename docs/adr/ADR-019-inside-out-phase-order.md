# ADR-019 — Inside-Out Phase Ordering: Core → Plugin → UI → Pipeline → Deploy → Quantum

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio is a large system with many interdependent subsystems. The order in which
subsystems are built determines how quickly working software is available, how easy
it is to validate decisions, and how much rework occurs when upstream APIs change.

Two broad strategies were considered:

| Strategy | Description | Risk |
|---|---|---|
| **Outside-in** | Start with the UI shell; fill in engine later | UI is built against undefined engine APIs; placeholder stubs everywhere; hard integration later |
| **Inside-out** | Start with the engine maths; build outward | No UI demo early; but every layer is built on a tested, real foundation |

NNStudio is first and foremost a learning tool about neural network **mathematics**.
If the mathematical engine is solid, everything else is I/O and presentation.
If the UI is solid but the maths are wrong, nothing else matters.

---

## Decision

NNStudio is built **inside-out** in six phases:

| Phase | Scope | Key deliverable |
|---|---|---|
| **0** | Project scaffolding: docs, directory layout, `.gitignore`, CMake skeleton | Compilable empty project; all design documents |
| **1** | NN engine core C++ library: `Tensor`, `Layer`, `ComputeGraph`, `Trainer`, formats | `nnstudio-core` static library; unit tests; CPU backend with Eigen |
| **2** | Plugin SDK: C ABI, `TrustVerifier`, `TrustStore`, `PluginLoader`, pybind11 bridge | Loadable signed plugins; Python `nnstudio` package |
| **3** | Qt 6 QML Studio UI: layer editor, training dashboard, weight viewer, KB help | First runnable GUI backed by a real engine |
| **4** | Full pipeline: input adapters, tokenisation, context DB, chained execution, output | End-to-end inference from raw input to raw output |
| **5** | Deployment & runners: export wizards, runner connectors, `.nnsr` bundle, Registry API | Production-deployable model bundles |
| **6** | Quantum backend: hybrid classical-quantum graph, Qiskit bridge, circuit compiler | Quantum-executable subgraphs |

### Phase dependencies

```
Phase 0 (scaffolding)
    └── Phase 1 (engine) ─────────────────────────────────────────┐
            └── Phase 2 (plugin SDK + Python bridge)               │
                    └── Phase 3 (GUI) ──────────────────────────┐  │
                            └── Phase 4 (pipeline)              │  │
                                    └── Phase 5 (deployment) ───┘  │
                                                                    │
                    Phase 6 (quantum) ──────────────────────────────┘
                    (depends on Phase 1 IBackend; otherwise independent)
```

Phase 6 depends only on Phase 1 (`IBackend` interface stub) and can begin independently
of Phase 3–5 once Phase 1 is complete.

---

## Consequences

**Positive**
- Every phase has a testable deliverable before the next begins.
- Engine API is stable before the UI is built against it — no UI rework from API churn.
- Python bridge (Phase 2) provides an interactive testing and scripting surface before
  the GUI exists.
- Quantum backend (Phase 6) is naturally decoupled and can be worked on in parallel
  with Phase 3–5 by a separate contributor.

**Negative / constraints**
- No runnable GUI until Phase 3 — early progress is command-line / unit test only.
- Phase ordering must be respected: starting Phase 3 before Phase 1 is complete is
  explicitly disallowed (tracked in TODO.md phase gates).

**Follow-on**
- Each phase gate in `TODO.md` must be fully checked before the next phase begins.
- Phase 0 scaffolding is complete when all root `.md` documents exist and the CMake
  skeleton configures and compiles (even if it produces only empty libraries).
