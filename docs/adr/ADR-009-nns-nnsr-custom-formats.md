# ADR-009 — `.nns` and `.nnsr` Custom Formats for ONNX Gaps

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

ONNX (ADR-008) is the canonical model format, but it does not cover:
- UI layout state (panel arrangement, editor zoom, annotation layers).
- Training configuration (optimizer hyper-parameters, LR schedule, dataset binding).
- Optimizer state needed to resume training (Adam `m`/`v` accumulators, step counter).
- Plugin references and tokenizer configuration.
- Model Card metadata (intended use, limitations, evaluation results).
- Hardware hints (minimum VRAM, preferred backend).
- Multi-model pipeline topology (how `N` graphs are chained together).

Separately, a deployment artefact is needed that bundles everything required to run a model
on a machine that does not have the Studio installed (runners, tokeniser vocabularies, config).

---

## Decision

Two custom formats are defined. ONNX is stored inside them — never reimplemented by them.

### `.nns` — NNSpire Native Project File (design-time)

A **MessagePack** (binary) envelope with a JSON-only inspection mode (for diffs/version control).

Contents:
- **Embedded ONNX blob** — canonical weight and graph storage.
- **Extension JSON** — covers all fields ONNX does not: plugin refs, tokeniser config,
  training config, optimizer state, UI layout, Model Card, hardware hints.

> Rule: anything ONNX covers is stored in the ONNX blob only — never duplicated in the extension JSON.

The embedded ONNX blob is extractable as a standalone `.onnx` via a single Studio menu action.

### `.nnsr` — NNSpire Runner Bundle (deployment-time)

A standard **ZIP archive** containing everything needed for offline inference:

```
model.nnsr  (zip)
├── manifest.json
├── model.onnx
├── vocab.json             (if text model)
├── plugins/               (required signed plugin binaries)
├── runner.py              (Python entry point)
├── runner.cpp             (C++ source, compilable standalone)
└── runner_<platform>.so   (precompiled C++ runner, optional)
```

---

## Consequences

**Positive**
- ONNX remains the authoritative format; `.nns` adds exactly the gaps.
- `.nnsr` bundles are self-contained and xcopy-deployable.
- MessagePack gives compact binary for large weight files; JSON mode enables `git diff`.

**Negative / constraints**
- Two custom format parsers must be maintained (C++ + Python, per ADR-004).
- `.nns` schema versioning must be carefully managed to allow forward-compatible reads.
- Runner bundles must be signed (plugin binaries inside are trust-verified on load).

**Follow-on**
- Define `.nns` JSON schema with JSON Schema or protobuf (decision deferred to Phase 1).
- `NNSWriter` / `NNSReader` classes in `NNSpire-core/formats/`.
- `BundleBuilder` and `BundleRunner` in `NNSpire/deployment/`.
- See DEPLOYMENT.md for the complete format specifications and manifest schemas.
