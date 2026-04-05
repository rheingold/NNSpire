# ADR-016 — Core Features Permanently Free: No Paid Tier for Learning, Visualization, or Pipeline

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNStudio has a dual mandate: it is both a **learning workbench** and a
**professional deployment tool**. These two goals are in tension when considering
a commercial sustainability model.

Common monetisation patterns in developer tooling:

| Pattern | Risk |
|---|---|
| Free core, paid Pro tier gating key features | Users distrust the tool; learning features disappear behind paywalls |
| Open-core with proprietary plugins | Acceptable if the open core is genuinely useful |
| Plugin marketplace revenue | Sustainable without restricting the core app |
| Donations / sponsorship | Unpredictable; insufficient for sustained development |

The project's primary revenue path is the **commercial plugin signing certificate
economy** (ADR-007): authors pay for Vendor Signing Certificates; the core app stays free.

---

## Decision

The following categories of NNStudio functionality are declared **permanently FREE**
and encoded as `Tier::FREE` in `nnstudio/core/features/FeatureFlags.h`:

| Category | Scope |
|---|---|
| **Learning & wizard mode** | All guided walkthroughs, wizard steps, educational visualisations |
| **Visualization** | Weight heatmaps, neuron viewer, loss charts, activation histograms, topology diagram |
| **Full pipeline** | All input adapters, tokenization, context/RAG, execution, output adapters |
| **Standard runner connectors** | Triton, TF Serving, KServe, ONNX Runtime embedded, OpenAI-compatible REST |
| **KB help system** | In-app help, context-sensitive KB popups, wizard KB integration |
| **All standard export formats** | ONNX, TorchScript, `.nns`, `.nnsr` |
| **Plugin SDK** | C ABI headers, Python bridge, scaffolding tools, `nnstudio-sign` CLI |
| **Trust verification** | The signing/verification mechanism is open; only the *certificate issuance service* is commercial |

### Rules for any future paid-tier consideration

1. Learning, visualisation, and full pipeline features may **never** be gated.
2. Any proposed paid feature requires a public issue, community comment period, and explicit rationale.
3. A `Tier::FREE` → `Tier::PRO` change in `FeatureFlags.h` is a **documented,
   version-controlled architectural decision** — it cannot be made silently.
4. If a Pro tier is ever introduced, a perpetual free legacy version must be maintained
   for all features that were previously free.

This policy is a **public commitment** enforceable by the GPL/LGPL license terms
(any feature-flag change in distributed source must be disclosed).

---

## Consequences

**Positive**
- Builds community trust — users know the learning/visualization features will not disappear.
- Plugin signing certificate economy can be the sustainable revenue path without
  compromising the open workbench vision.
- GPL enforcement of the commitment is automatic.

**Negative / constraints**
- Reduces monetisation flexibility.
- `FeatureFlags.h` must be carefully designed so that accidental `Tier::PRO` assignment
  is caught by code review and CI.

**Follow-on**
- `FeatureFlags.h` in `nnstudio/core/features/` with `enum class Tier { FREE, PRO }`.
- CI lint rule: any PR changing a flag from `FREE` to `PRO` requires a linked ADR.
- See LICENSING.md §2 and §3 for the full plugin charter and Pro-tier policy.
