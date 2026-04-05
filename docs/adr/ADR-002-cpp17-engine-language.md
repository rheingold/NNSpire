# ADR-002 — C++17 as the Core Engine Language

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

The NNStudio engine core (`nnstudio-core`) must be:
- High-performance (SIMD, cache-friendly tensor operations).
- Portable across Windows / macOS / Linux with MSVC, Clang, and GCC.
- Readily bridgeable to Python via pybind11 (see ADR-005).
- Usable as a stable shared library by third-party plugins written in C, C++, Rust, or Zig.

C++20 offers improvements (concepts, ranges, coroutines) but toolchain support in
Qt 6.5's minimum supported compilers is still uneven as of 2026.

---

## Decision

C++17 is the minimum language standard for all code in `nnstudio-core`,
`nnstudio-backends/`, and `nnstudio-plugin-api/`.

- C++20 features may be used **only** behind `#if __cplusplus >= 202002L` guards,
  providing a C++17 fallback wherever used.
- RTTI is **disabled** in the engine core (`-fno-rtti` / `/GR-`).
  Dynamic dispatch uses explicit vtable structs (see ADR-003), not `dynamic_cast`.
- Exceptions are **disabled** in the engine core (see ADR-011).
- The Qt application layer (`nnstudio/app/`) may use whatever C++ standard Qt 6 requires (currently C++17, may move to C++20 at a Qt upgrade).

### Naming conventions

| Category | Convention |
|---|---|
| Classes, structs, enums | `PascalCase` |
| Member functions, free functions | `camelCase` |
| Local variables, parameters | `snake_case` |
| Constants, macros | `SCREAMING_SNAKE_CASE` |
| Header files | `.h` extension, `#pragma once` |
| Public API headers | `include/nnstudio/<module>/` — no implementation in public headers except templates |

---

## Consequences

**Positive**
- Maximum compiler compatibility.
- Clean ABI across DLL/SO boundaries (further ensured by ADR-003).
- pybind11 works without restrictions.

**Negative / constraints**
- Some ergonomic C++20 features (concepts, `std::span` over ranges, coroutines) unavailable without guards.
- `dynamic_cast` / RTTI being disabled means all polymorphism must be via virtual functions or explicit vtables.

**Follow-on**
- CI matrix must test all three compilers (MSVC 2022, GCC 12+, Clang 15+) at C++17.
- Add a C++20 optional CI lane for future migration tracking.
