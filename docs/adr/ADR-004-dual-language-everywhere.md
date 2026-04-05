# ADR-004 — Dual-Language Parity: C++ and Python for Every Artefact

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio serves two distinct audiences:
1. **C++ practitioners** who want maximum performance and ABI-level control.
2. **Python ML practitioners** who expect a `pip`-installable, NumPy-compatible workflow.

Without a dual-language commitment, the project risks becoming C++-only (alienating the
larger ML community) or Python-first (sacrificing performance and embedding capability).

Additionally, the learning and educational mandate of NNStudio requires that every concept
be demonstrable in the language the learner already knows.

---

## Decision

Every *computable artefact* in NNStudio ships **both** a C++ implementation and a Python
implementation exhibiting **identical external behaviour**:

| Artefact type | C++ form | Python form |
|---|---|---|
| Plugin | `.dll`/`.so` with C ABI descriptor | `.pyd`/`.so` via pybind11 bridge |
| Runner client | `IRunnerClient` C++ class | Python class with same interface |
| Export script | `OnnxIO::export()` method | `nnstudio.export.onnx()` function |
| Training sample | `examples/cpp/` | `examples/python/` |
| Layer implementation | `Layer` subclass in `core/layers/` | Exposed via pybind11 + Python subclassing |

**C++ is defined first.** pybind11 provides the Python mirror of the public C++ API exactly.
No Python-only public API path exists at the engine level.

The community/commercial plugin compliance verification checks that both forms are present
and behaviourally equivalent before a signing certificate is issued.

---

## Consequences

**Positive**
- Studio docs, KB help, and wizard walkthroughs can always show both language forms side by side.
- C++ and Python communities can both contribute plugins and samples.
- The Python bridge doubles as an integration testing surface for the C++ engine.

**Negative / constraints**
- Every new engine feature requires two implementations (C++ + pybind11 binding).
- Behavioural equivalence must be tested, not assumed.
- Plugin signing registry enforces dual-language compliance — single-language plugins will not receive a signing certificate.

**Follow-on**
- pybind11 binding files live in `nnstudio/python-bridge/`.
- Test suite (`nnstudio/tests/`) must include cross-language equivalence tests
  (run the same operation in C++ and Python, compare results numerically).
- See ADR-005 for the choice of pybind11 over alternatives.
