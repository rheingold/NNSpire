# plugins/ — Repository Architecture Notes

> This folder is intentionally nearly empty at Phase 1.
> For the Plugin SDK design, read [../../blueprints.md](../../blueprints.md)
> and `TODO.md § Phase 2 — Plugin SDK`.

---

## Why this folder exists now (when it is empty)

The folder reserves the slot in the repository layout so that:
1. The CMake target graph can reference it without structural refactoring later.
2. Contributors reading the top-level `CONTRIBUTING.md` reading order can see it in context
   before Phase 2 work begins.
3. It documents the *intent* — third-party plugin implementations go here, not in `core/` or `builtin/`.

---

## What will live here (Phase 2+)

Third-party plugins that are bundled with the NNSpire installer for convenience,
but that have **no special access** to engine internals beyond `core/include/NNSpire/core/`.

Each plugin is a subdirectory with its own `CMakeLists.txt` and `PluginDescriptor` registration.
Plugin authors outside this repository follow the same structure in their own repos.

```
plugins/
    acme-fast-conv/
        CMakeLists.txt
        AcmeFastConv.h
        AcmeFastConv.cpp
        manifest.json          ← { "id": "com.acme.FastConv", "type": "LAYER", ... }
    eu-plachy-quantum-backend/
        ...
```

---

## The no-VIP-treatment rule

NNSpire's own built-in layers and backends live in `../builtin/` and are compiled into
`NNSpire-builtin` — a separate CMake target that links `NNSpire-core` exactly as any
third-party plugin would. They have no access to `core/src/` private headers.

A plugin in `plugins/` is treated identically. The Studio runtime sees both as implementations
of `ILayer` / `IBackend` loaded via the `PluginDescriptor` C-ABI.
