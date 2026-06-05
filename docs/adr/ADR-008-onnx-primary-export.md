# ADR-008 — ONNX as the Primary Model Export Format

**Date**: 2026-03-31  
**Status**: Accepted  
**Decided by**: Project founder (initial design session)

---

## Context

NNSpire must export trained models in a format that:
- Is accepted by all major inference runtimes (Triton, TF Serving, ONNX Runtime, TensorRT).
- Is hardware-agnostic at the description level.
- Is supported by standard tools for validation, optimisation, and profiling.
- Minimises lock-in to any single framework.

Candidate export formats:

| Format | Supported runtimes | Notes |
|---|---|---|
| **ONNX** | ONNX Runtime, Triton, TF, TensorRT, many others | Cross-vendor; community standard |
| TorchScript (`.pt`) | PyTorch-based runtimes | Framework-specific; not portable |
| TFLite (`.tflite`) | TF Lite runtime, Edge TPU | Mobile/edge focus; limited ops |
| CoreML (`.mlpackage`) | Apple platforms only | Platform-specific |
| Custom binary | None — must build own runtime | Maximum work; zero interop |

---

## Decision

**ONNX is the primary model export format** for NNSpire.

The export procedure (implemented in `NNSpire-core/formats/OnnxIO`):
1. Trace the `ComputeGraph` with a sample input to resolve all dynamic shapes.
2. Map each `Layer` to the closest ONNX standard operator.
3. Register custom layers that have no ONNX equivalent under the `com.NNSpire` custom operator domain.
4. Serialise to `ModelProto` protobuf.
5. Validate with the ONNX checker.
6. Write the `.onnx` file.

NNSpire ships an ONNX Runtime custom-op shared library for `com.NNSpire` operators,
enabling the custom ops to run on any ONNX Runtime deployment.

Secondary export formats (TorchScript, TFLite) are provided in later phases via the
Python bridge, and are **optional** — ONNX remains the canonical interchange format.

---

## Consequences

**Positive**
- One export integrates with the widest possible set of inference runtimes and cloud platforms.
- ONNX protobuf schema is the canonical weight storage in the `.nns` project file (see ADR-009).
- ONNX checker provides validation for free.
- Community KB (`ai-standards-kb/standards/02-ONNX.md`) can directly inform implementation.

**Negative / constraints**
- ONNX opset versions evolve; `OnnxIO` must track opset compatibility.
- Custom operators (`com.NNSpire.*`) require the custom-op library to be bundled with every deployment.
- Dynamic shapes require a representative sample input at export time.

**Follow-on**
- `NNSpire-core/formats/OnnxIO` (C++) + `NNSpire.formats.onnx` (Python) per ADR-004.
- Custom-op library built as a separate CMake target: `NNSpire-onnxrt-custom-ops`.
- TorchScript and TFLite stubs planned for Phase 5.
