# ADR-021 — Path = Namespace Layout and Three-Tier Namespace Design

**Date**: 2026-04-05  
**Status**: Accepted  
**Decided by**: Project founder (design session — namespace/folder restructure)

---

## Context

As the codebase grew from a flat prototype into a multi-target library, two
independent layout questions arose:

**Question A — Which path segments should correspond to which namespace segments?**  
The early code had `core/include/NNSpire/` containing headers that declared
`namespace NNSpire { ... }`. The `include/NNSpire/` prefix disappeared at the
include path boundary but was not reflected in the C++ namespace, creating a
mismatch: `#include <NNSpire/Tensor.h>` implied `NNSpire::Tensor`, but nothing
in the directory tree encoded that mapping formally. Adding sub-domains (layers,
backends) required a choice: make them namespace segments, folder names, or both.

**Question B — One shared `include/` root vs. per-module include roots?**  
Option A kept per-module roots (`core/include/`, `builtin/include/`) so that CMake
include paths were scoped to each target. Option B used a single `NNSpire/include/`
root for all targets. The tie-breaking argument: `NNSpire-builtin` has a permanent,
legitimate dependency on `NNSpire-core` (builtin layers inherit from `ILayer`, etc.).
Per-module include paths would either block those includes or require cross-module
search-path leakage — neither is acceptable. The dependency boundary is enforced by
`target_link_libraries`, not by include paths.

**Question C — What namespace should third-party plugin authors use?**  
If the plugin's namespace were derived from its position under `NNSpire/plugins/`,
moving a plugin between repositories would silently rename its public API — exactly
the coupling the plugin system exists to eliminate. Plugin namespaces must be
author-owned identity, not repository-position-derived.

---

## Decision

### Rule: every path segment except `include/` and `src/` maps 1-to-1 to a namespace segment

`include/` and `src/` are **visibility boundaries only** — they separate the installed
public surface from the build-only implementation. They contribute no namespace segment.
Every other directory level maps directly to one C++ namespace segment.

```
Disk path segment          →   C++ namespace segment
───────────────────────────────────────────────────────────────────
NNSpire/                  →   NNSpire::
include/  or  src/         →   (SKIP — visibility boundary)
core/                      →   core::
builtin/                   →   builtin::
layers/                    →   layers::
backends/                  →   backends::
losses/                    →   losses::
optimizers/                →   optimizers::
───────────────────────────────────────────────────────────────────
```

Examples:

```
NNSpire/include/core/Tensor.h        →   NNSpire::core::Tensor
NNSpire/include/builtin/layers/Dense.h  →   NNSpire::builtin::layers::Dense
NNSpire/src/core/Tensor.cpp          →   NNSpire::core:: (same namespace, build-only)
```

### Three-tier namespace design

| Tier | Namespace | Content | Stability |
|---|---|---|---|
| 1 — public contract | `NNSpire::core::` | Abstract interfaces (`ILayer`, `IBackend`, `ILoss`, `IOptimizer`), data types (`Tensor`, `Result<T>`, `Parameter`), engine services (`Trainer`, `BackendRegistry`) | Versioned; breaking changes require migration period |
| 2 — engine internals | `NNSpire::internal::` | Autograd graph, training loop, format I/O — not for plugin authors | May change between minor releases |
| 2b — UI extension | `NNSpire::ui::` | QML panel registration, property/signal contracts | Evolves with Qt version |
| 3 — bundled implementations | `NNSpire::builtin::` | NNSpire's own layers, backends, losses, optimizers — treated identically to third-party plugins | No special access over any plugin |

No concrete implementation lives in `NNSpire::core::`. Tier 3 (`builtin`) uses
`using namespace NNSpire::core;` inside its namespace blocks so it can reference Tier 1
types without full qualification; this does not add any include-path coupling beyond
what `target_link_libraries(NNSpire-builtin PUBLIC NNSpire-core)` already expresses.

### Option B — single shared `include/` and `src/` roots

```
NNSpire/
    include/           ← ONE CMake search root (all targets add this)
        core/          → NNSpire::core::
        builtin/       → NNSpire::builtin::
    src/               ← build-only; mirrors include/ structure exactly
        core/
        builtin/
    plugin-api/        ← CMake target definitions and C ABI header
    plugins/           ← plugin slots — see Plugin exception below
    tests/
    CMakeLists.txt     ← all NNSpire-core / NNSpire-builtin targets defined here
```

The one-directional mirror rule: every subfolder in `include/` **must** have a
matching subfolder in `src/`. The reverse does not hold — `src/` may gain private
subfolders (e.g. `src/internal/graph/`) that have no `include/` counterpart.

### Plugin exception — namespace direction is reversed for `plugins/`

The namespace of a plugin is owned by its author based on their reverse-domain identity.
It must not change when the plugin moves repositories. The `<plugin-slug>` folder name
under `plugins/` therefore does **not** contribute a namespace segment.

```
plugins/<slug>/include/layers/MyLayer.h
  (skip)  (slug — skip)  (skip)   ↓          ↓
                               layers::    MyLayer
```

The author decides the segments above `layers/`:

```cpp
// Author's namespace — entirely their choice:
namespace eu::plachy::nnplugins::myplugin::layers {
    class MyLayer : public NNSpire::core::ILayer { … };
}
```

Within the plugin's own subtree, the same path=namespace rule applies: domain
subfolders (`layers/`, `backends/`, …) continue to map to namespace segments.

---

## Consequences

**Positive**
- Any developer can determine the exact namespace of any public symbol by reading the
  file path alone, with no IDE necessary.
- Cross-module references (`builtin` → `core`) use the same stable include root; no
  relative `../../` path hacks.
- `cmake --install` can install the entire `include/` tree as the Plugin SDK with a
  single `install(DIRECTORY include/ ...)` rule — no per-file lists to maintain.
- Plugin authors have full namespace ownership; their types survive repository moves
  without ABI changes.

**Negative / constraints**
- The `plugin-api/include/NNSpire/` subtree is a deliberate exception: it uses the
  pre-existing `<NNSpire/NNSpire_plugin.h>` include path for the C ABI header because
  that path is already in external documentation and plugin templates. It does not
  declare a `namespace NNSpire` — it is pure C, so the path=namespace rule does not
  apply.
- All `cmake --build` targets must add `${CMAKE_SOURCE_DIR}/include` to their include
  search path. Forgetting this yields a clean error (header not found), not silent
  wrong behaviour.

**Follow-on**
- `src/internal/` subtree (graph, training, formats) will follow the same rule when
  implemented: `src/internal/graph/` → `NNSpire::internal::graph::`.
- `blueprints.md § Appendix — Namespace Map` is the canonical prose documentation;
  this ADR records the decision rationale.
