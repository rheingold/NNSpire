# ADR-051 — Scoped Implementation Language for NNSpire Agent

**Date:** 2026-07-16
**Status:** Accepted
**Decided by:** Project founder (architectural discussion session)
**Related:** [`ADR-002`](ADR-002-cpp17-engine-language.md), [`ADR-004`](ADR-004-dual-language-everywhere.md), [`ADR-050`](ADR-050-nnagent-ui-framework-tauri.md)

---

## Context

ADR-004 mandates dual-language parity (C++ + Python) for every computable artefact. This was designed for the Engine pillar (L1-L2 numeric computation). NNSpire Agent is L6 ontology (application task flow) — its artefacts are chat routers, MCP clients, orchestration engines, not neural network layers.

The founder's "no Python" directive applies to **NNAgent core implementation language only**, NOT to:
- NNSpire Engine/Studio (ADR-004 remains fully in force)
- Automation script blocks (Python scripts are user content, not framework code)
- Python interfaces/bindings between Agent and Engine (Runner API)

---

## Decision

**Scoped language assignment per NNAgent component:**

| Component | Primary Language | Python? | Rationale |
|-----------|-----------------|---------|-----------|
| `nnagent-core` (C++ lib) | **C++17** | ❌ No | Business logic — not numeric computation |
| └── LLM Router | C++17 | ❌ | No Python ML equivalent |
| └── MCP Client | C++17 | ❌ | Protocol client |
| └── Chat Engine | C++17 | ❌ | State machine |
| └── Orchestration Engine | C++17 | ❌ | Graph executor |
| └── Storage Backend | C++17 | ❌ | Data access |
| Plugin Interfaces | C++17 + Python | ✅ Yes | ADR-004 signing compliance |
| └── IModelProvider | C++ + Python | ✅ | ML-adjacent |
| └── IFileParser | C++ + Python | ✅ | Data processing |
| └── IVectorDB | C++ + Python | ✅ | RAG-adjacent |
| Tauri Rust Shell | **Rust** | ❌ N/A | Tauri requirement (ADR-050) |
| React Frontend | **TypeScript** | ❌ N/A | UI layer (ADR-050) |
| Automation Script Runtime | C++ + Python | ✅ Yes | User scripts require Python execution |
| CLI | C++17 | ❌ | Thin wrapper |
| Studio Embed (QML) | QML | ❌ N/A | WebView host only |

### What "Scoped" Means

1. **NNAgent core is C++17-only** — no pybind11 bindings for internal classes
2. **Plugin interfaces get Python bindings** — required for ADR-004 signing compliance
3. **Automation scripts can use Python** — user content, not framework code
4. **Engine communication uses Runner API** — Python bindings exist on Engine side, not Agent side

### Impact on ADR-004

ADR-004 is **NOT superseded**. It remains fully in force for:
- NNSpire Engine (`nnspire/` excl. `app/`)
- NNSpire Studio (`nnspire/app/`)
- All plugin interfaces (both Engine and Agent plugins)

ADR-004 is **scoped-exempt** for:
- NNAgent internal core classes (non-plugin, non-interface C++ code)
- NNAgent UI layer (TypeScript/Rust/QML)

---

## Consequences

### Positive
- ~60% reduction in NNAgent implementation effort (no dual-language for core)
- Clear language boundaries per component
- ADR-004 preserved for Engine pillar (no precedent erosion)
- Python present only where functionally required (scripts, plugin interfaces)

### Negative / Constraints
- NNAgent internal classes cannot be directly tested from Python
- Plugin developers must still provide dual-language forms
- Documentation must clarify scope boundary

### Follow-on
- NNAgent CMake configuration must exclude pybind11 for core targets
- Plugin interface headers must include pybind11 binding templates
- Automation runtime must include Python subprocess launcher
