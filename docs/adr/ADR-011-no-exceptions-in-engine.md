# ADR-011 — No C++ Exceptions in the Engine Core

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

`NNSpire-core` is a C++ library that will be:
- Consumed by the Qt application layer.
- Exposed to Python via pybind11.
- Potentially embedded into a standalone runner sidecar.
- Eventually loaded into plugin shared libraries with mixed compiler provenance.

C++ exception propagation across shared library (`.dll`/`.so`) boundaries is implementation-defined
and is not guaranteed to work correctly when the plugin and the host are compiled with
different compilers or different CRTs (common on Windows with MSVC vs MinGW, for example).

Additionally, with RTTI disabled (see ADR-002), `dynamic_cast` and `std::bad_cast` are unavailable,
and exception type information is internally RTTI-dependent in some implementations.

Disabling exceptions also enables `-fno-exceptions` / `/EHs-c-` compiler flags which
reduce binary size and eliminate hidden control flow paths.

---

## Decision

**C++ exceptions are disabled in `NNSpire-core`, `NNSpire-backends/`, and `NNSpire-plugin-api/`.**

Error handling uses **`Result<T, Error>`** as the universal return type.

```cpp
// Core error type
enum class ErrorCode : uint32_t {
    OK = 0,
    SHAPE_MISMATCH,
    DTYPE_MISMATCH,
    DEVICE_MISMATCH,
    OUT_OF_MEMORY,
    PLUGIN_API_VERSION_MISMATCH,
    // ...
};

struct Error {
    ErrorCode code;
    const char* message;   // static string — no allocation
};

template<typename T>
using Result = std::variant<T, Error>;
```

The Qt application layer (`NNSpire/app/`) **may** use exceptions freely — it is not a
shared library and operates within a single compiler environment.

Python exception translation is handled at the pybind11 boundary: `Error` → `RuntimeError`
(or a typed NNSpire Python exception subclass).

---

## Consequences

**Positive**
- No hidden control flow paths in the engine; profiling and code review are clearer.
- Safe to load across mixed-compiler plugin boundaries.
- Smaller binary size; predictable code-gen.
- `Result<T, Error>` forces callers to handle errors explicitly.

**Negative / constraints**
- More verbose call-site error handling compared to try/catch.
- C++ STL operations that throw (e.g., `std::vector` allocations) must be wrapped or avoided in critical paths.
- pybind11 exception translation layer must be maintained and tested.

**Follow-on**
- Define `Result<T, Error>` in `NNSpire/core/include/NNSpire/core/result.h`.
- All public engine API functions must return `Result<T, Error>` or `void` (no-fail operations).
- pybind11 bridge must install an exception translator via `pybind11::register_exception_translator`.
