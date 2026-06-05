# ADR-006 — LGPL v3 Core / GPL v3 App License Split

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNSpire must satisfy two competing goals simultaneously:

1. **Commercial plugin ecosystem**: third parties must be able to write proprietary,
   commercially licensed plugins that link against the NNSpire engine without being
   forced to open-source their plugin code.
2. **Copyleft protection**: modifications to the NNSpire engine and Studio application
   itself — including any attempt to fork the Studio with the trust chain disabled — must
   be disclosed under an open-source license.

A single license applied uniformly across the project cannot satisfy both goals.

---

## Decision

The project uses a **dual-tier license split**:

| Component | License | Rationale |
|---|---|---|
| `NNSpire-core` | **GNU LGPL v3** | Proprietary plugins may dynamically link without GPL infection |
| `NNSpire/plugin-api/` | **GNU LGPL v3** | Plugin authors link against these headers |
| `NNSpire/python-bridge/` | **GNU LGPL v3** | Bridge to core; same terms |
| `NNSpire/backends/` | **GNU LGPL v3** | Backend plugins are linkable libraries |
| `NNSpire/app/` | **GNU GPL v3** | App is not a library; GPL prevents proprietary Studio forks |
| `NNSpire/plugins/` (built-in reference plugins) | **MIT** | Widest use for learning/scaffolding |
| `NNSpire-sign` CLI | **GNU GPL v3** | Standalone executable |
| AI Standards KB | **CC BY 4.0** | Documentation / knowledge base |

### Key implications

- A third party **may** build a closed-source commercial plugin that *dynamically links*
  against `NNSpire-core` — explicitly permitted by LGPL v3 §4.
- A third party **must not** statically link `NNSpire-core` into a closed-source plugin
  without publishing core source modifications.
- Anyone distributing a fork of the Studio application (`NNSpire/app/`) that disables
  the trust verification system **must** publish that fork's source under GPL v3.
  Disabling the trust chain and distributing the result is a GPL violation.

---

## Consequences

**Positive**
- Commercial plugin ecosystem is legally unblocked from day one.
- The codebase remains fully open, auditable, and forkable.
- The trust chain is legally protected against silent removal.

**Negative / constraints**
- Contributors must sign a CLA (or equivalent) if the project owner wishes to offer
  commercial licensing in future.
- License headers must be present in every source file; tooling needed to enforce this.
- Full license texts must be in `LICENSES/` at project root.

**Follow-on**
- Create `LICENSES/` directory with GPL-3.0.txt, LGPL-3.0.txt, MIT.txt, CC-BY-4.0.txt.
- Add SPDX license identifiers to all source file headers.
- See LICENSING.md for the full plugin charter and Pro-tier policy.
