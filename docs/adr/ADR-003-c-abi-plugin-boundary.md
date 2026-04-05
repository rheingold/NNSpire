# ADR-003 — C ABI at the Plugin Boundary

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio loads third-party plugins at runtime as shared libraries (`.dll` on Windows,
`.so` on Linux, `.dylib` on macOS).

C++ does not have a stable ABI: name mangling, vtable layout, and exception propagation
differ between compiler versions and even between debug and release builds of the same
version. Shipping a C++ virtual interface in `nnstudio-plugin-api/` would mean:

- Plugins compiled with MSVC 2019 could silently corrupt vtables when loaded into a
  Studio compiled with MSVC 2022.
- Plugins written in Rust, Zig, or D would require language-specific C++ ABI shims.
- A single C++ ABI break in a Studio update would silently invalidate all existing plugins.

---

## Decision

The public plugin interface (`nnstudio_plugin.h`) uses **only C linkage**:

- `extern "C"` on all exported symbols.
- Only C-compatible types in all structs: plain integers, pointers, and function pointers.
  No `std::string`, no `std::vector`, no C++ exceptions.
- Polymorphism is expressed as explicit vtable structs (plain C function pointer tables),
  not C++ virtual dispatch.
- The single required export symbol is:
  ```c
  const NNPluginDescriptor* nnstudio_plugin_descriptor(void);
  ```
- `NNPluginDescriptor` carries a `void* vtable` pointer cast per plugin type.

Python plugins are wrapped via the pybind11 bridge and go through the same `NNPluginDescriptor`
pathway — they are logically C ABI compliant even though the bridge is C++.

---

## Consequences

**Positive**
- Plugins are compiler-agnostic: any toolchain that can produce a C-compatible shared library works.
- ABI is stable across Studio versions as long as the `api_version` field is respected.
  Plugins can be binary-compatible across multiple Studio releases.
- Enables Rust/Zig/D plugin authorship without a C++ shim.

**Negative / constraints**
- More verbose plugin authoring than a pure C++ virtual interface.
- Error propagation must use C integer return codes + error string pointers, not exceptions.
- Passing complex data (tensors, strings) requires agreed struct layouts (`NNTensorView`, etc.).

**Follow-on**
- `nnstudio_plugin.h` must include a `api_version` field in `NNPluginDescriptor`;
  `PluginLoader` must reject plugins with incompatible versions.
- Scaffold generators (`nnstudio-sign keygen --scaffold`) produce compliant boilerplate
  so authors do not need to understand the raw ABI.
- See PLUGIN-SDK.md for the full header definition.
