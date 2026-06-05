# ADR-005 — pybind11 as the Python Bridge

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

ADR-004 mandates Python parity for all engine artefacts. A binding mechanism must expose
the `NNSpire-core` C++ API to Python without rewriting it.

Candidates evaluated:

| Library | Notes |
|---|---|
| **pybind11** | Header-only, C++11/17, mature, minimal overhead, NumPy array integration |
| Cython | Requires `.pyx` files; adds a build step; less transparent for C++ API mirroring |
| SWIG | Generates large, hard-to-debug code; coarser type system mapping |
| Nanobind | Newer, smaller, faster at module load time, but less mature ecosystem as of 2026 |
| ctypes / cffi | No type safety; manual; suitable only for thin C ABI wrappers |

---

## Decision

**pybind11** is the Python bridge for NNSpire.

- All pybind11 binding code lives in `NNSpire/python-bridge/`.
- The Python package name is `NNSpire` (top-level).
- Python minimum version: **3.10** (matches embeddable Python distribution shipped with the app).
- Python packaged with the app lives in `<install>/runtime/python/` — it never touches the user's system Python.
- `pyproject.toml` (PEP 517/518) governs the Python package build.
- NumPy array ↔ `Tensor<float>` shared-memory bridge is provided via `pybind11::array_t<float>`.

---

## Consequences

**Positive**
- pybind11's `array_t<float>` allows zero-copy NumPy ↔ Tensor interop.
- The binding layer is thin, readable C++ rather than generated code.
- Widely used in the ML community (used by PyTorch itself), so contributors will know it.

**Negative / constraints**
- pybind11 binding files must be kept in sync with C++ API changes.
- Module load time is non-trivial for very large binding sets; may need to be split into sub-modules.
- pybind11 header version must be pinned in `third-party/pybind11/`.

**Follow-on**
- Add pybind11 as a git submodule in `NNSpire/third-party/pybind11/`.
- Equivalence tests compare C++ and Python tensor shapes, dtypes, and numeric results.
- If nanobind matures sufficiently, evaluate migration in Phase 3+.
