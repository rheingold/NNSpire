# NNSpire — Deployment & Runners Whitepaper

**Version**: 0.1 (Phase 0 — design)  
**Date**: 2026-03-31  
**Status**: DRAFT — pre-implementation baseline

> Standards references: KB 02 (ONNX), KB 05 (TF Serving), KB 08 (gRPC), KB 09 (KServe), KB 10 (Triton), KB 03 (OpenAPI)

---

## 1. Principles

- **Standards first**: use existing interchange formats (ONNX, TorchScript) wherever they are sufficient. Custom formats only cover the gaps.
- **Dual-language**: every runner connector ships both a C++ implementation and a Python implementation with identical behaviour.
- **No server bundling**: NNSpire is a *client* of inference servers. It does not bundle Triton, TF Serving, or KServe. It can assist launching a local Docker container, but does not own the server process.
- **Portable bundles**: the `.nnsr` runner bundle is a self-contained zip that can be loaded anywhere that has the NNSpire runtime — no internet needed for inference.

---

## 2. Export formats

### 2.1 ONNX (`.onnx`)

Primary standard export. Uses `OnnxIO::export()` from `NNSpire-core/formats/`.

Procedure:
1. Trace the `ComputeGraph` with a sample input to resolve all dynamic shapes
2. Map each `Layer` to the closest ONNX operator (or `com.NNSpire.<layer>` custom op if no standard op covers it)
3. Serialise to `ModelProto` protobuf
4. Validate with ONNX runtime checker
5. Write `.onnx` file

Custom operators that have no ONNX equivalent are registered under the `com.NNSpire` domain. The Studio ships an ONNX Runtime custom op shared library for these.

Reference: KB doc 02 (ONNX), especially §Operator specs and §Custom ops.

### 2.2 TorchScript (`.pt`)

Via the Python bridge: `torch.jit.script()` or `torch.jit.trace()` is called on the Python-mirror pipeline. The resulting serialised TorchScript is optionally validated with `torch.jit.load()` + a sample forward pass.

Only available when PyTorch is installed in the embedded Python environment (facilitated by Dependency Manager).

### 2.3 TFLite (`.tflite`) — stub

Planned Phase 5. Conversion via the `tf.lite.TFLiteConverter` Python API. Full quantization (INT8, float16) supported through the converter's existing tooling.

### 2.4 `.nns` (NNSpire native project file)

The native design-time format. Contains:
- Embedded ONNX blob (model graph + weights — canonical weight storage)
- Extension JSON envelope covering gaps:

```json
{
  "NNSpireVersion": "1.0.0",
  "modelId": "uuid-v4",
  "created": "2026-03-31T00:00:00Z",
  "pluginRefs": ["com.example.bpe-tokenizer@1.2.0"],
  "tokenizerConfig": { "type": "BPE", "vocabPath": "vocab.json" },
  "trainingConfig": {
    "optimizer": "AdamW",
    "lr": 3e-4,
    "epochs": 10,
    "checkpointInterval": 1
  },
  "uiLayout": { "panels": [ ... ] },
  "modelCard": {
    "name": "My Model",
    "task": "text-classification",
    "license": "MIT",
    "intendedUse": "...",
    "limitations": "...",
    "evaluationResults": { ... }
  },
  "inputModalities": ["text"],
  "outputModalities": ["text"],
  "hardwareHints": { "minVRAM_MB": 0, "preferCUDA": false }
}
```

Anything ONNX covers is stored in the embedded ONNX blob only — never duplicated in the extension JSON. The embedded ONNX blob is extractable as a standalone `.onnx` with a single Studio menu action.

Serialisation: MessagePack for binary efficiency; a JSON-only mode available for human inspection and version control diffs.

### 2.5 `.nnsr` (NNSpire Runner Bundle)

A deployment-time zip archive containing everything needed to run the model without the Studio:

```
model.nnsr  (zip)
├── manifest.json          ← bundle metadata (see schema below)
├── model.onnx             ← extracted ONNX model
├── vocab.json             ← tokenizer vocabulary (if text model)
├── plugins/               ← required plugin binaries (licensed/signed)
│   └── com.example.bpe-tokenizer/
│       ├── bpe-tokenizer.dll   (or .so)
│       └── plugin.manifest.json
├── runner.py              ← Python runner entry point
├── runner.cpp             ← C++ runner source (compilable standalone)
├── runner_<platform>.so   ← precompiled C++ runner shared lib (if bundled)
└── CMakeLists.txt         ← build file for runner.cpp
```

#### `manifest.json` schema

```json
{
  "bundleVersion": 1,
  "modelId": "uuid-v4",
  "name": "My Model",
  "version": "1.0.0",
  "created": "2026-03-31T00:00:00Z",
  "inputModalities": ["text"],
  "outputModalities": ["text"],
  "tokenizerConfig": { "type": "BPE", "vocabPath": "vocab.json" },
  "requiredPlugins": ["com.example.bpe-tokenizer@1.2.0"],
  "hardwareHints": {
    "minVRAM_MB": 0,
    "preferCUDA": false,
    "quantization": "float32"
  },
  "minStudioRuntime": "1.0.0",
  "license": "MIT",
  "modelCard": { ... },
  "onnxModel": "model.onnx",
  "runners": {
    "python": "runner.py",
    "cpp_source": "runner.cpp",
    "cpp_prebuilt": {
      "windows_x64": "runner_windows_x64.dll",
      "linux_x64":   "runner_linux_x64.so",
      "macos_arm64": "runner_macos_arm64.dylib"
    }
  }
}
```

---

## 3. Runner connectors

Each connector is implemented in **both C++ and Python** with identical method signatures.

### 3.1 Triton Inference Server

Protocol: HTTP/2 gRPC (primary) + HTTP/1.1 REST (fallback).  
Reference: KB docs 10 (Triton) and 08 (gRPC).

```cpp
// C++
class TritonRunnerClient : public IRunnerClient {
public:
    TritonRunnerClient(const std::string& url);
    void loadModel(const std::string& name, int version = -1) override;
    Tensor<float> infer(const std::string& model, const Tensor<float>& input) override;
    HealthStatus health() override;
    ModelMetrics metrics() override;  // scraped from Prometheus endpoint
};
```

```python
# Python (python-bridge/runners/triton.py)
class TritonRunnerClient(RunnerClient):
    def __init__(self, url: str): ...
    def load_model(self, name: str, version: int = -1) -> None: ...
    def infer(self, model: str, input: Tensor) -> Tensor: ...
    def health(self) -> HealthStatus: ...
    def metrics(self) -> ModelMetrics: ...
```

### 3.2 TensorFlow Serving

Protocol: REST (primary) + gRPC (secondary).  
Reference: KB doc 05.

```
POST /v1/models/{model_name}:predict
GET  /v1/models/{model_name}                  (status)
GET  /v1/models/{model_name}/versions/{ver}   (version status)
```

### 3.3 KServe (V2 Inference Protocol)

Protocol: KServe V2 HTTP + gRPC.  
Reference: KB doc 09.

```
POST /v2/models/{model_name}/infer
GET  /v2/models/{model_name}/ready
GET  /v2/models/{model_name}/metadata
```

### 3.4 ONNX Runtime (embedded)

In-process inference. No server needed.

```cpp
#include <onnxruntime/onnxruntime_cxx_api.h>
class OnnxRuntimeClient : public IRunnerClient {
    Ort::Session session_;
    // ...
};
```

```python
import onnxruntime as ort
class OnnxRuntimeClient(RunnerClient):
    def __init__(self, model_path: str): ...
    def infer(self, model: str, input: Tensor) -> Tensor: ...
```

### 3.5 OpenAI-compatible REST

For connecting to any server implementing the OpenAI Chat Completions API (local Ollama, LM Studio, OpenAI, Azure OpenAI, Mistral, etc.).  
Reference: KB annex A05.

```
POST /v1/chat/completions
POST /v1/embeddings
GET  /v1/models
```

---

## 4. Deployment panel (UI)

The Studio Deployment panel provides:

- **Export wizard**: step-by-step guided export to any format. Shows a validation summary (output tensor shape match verified by running a sample through the exported model).
- **Runner configuration**: URL, auth type (none / API key / mTLS), TLS certificate path.
- **Deploy action**: pushes model to runner (where supported — Triton model repository upload, HuggingFace Hub push).
- **Health monitor**: polls runner health endpoint (configurable interval); shows latency histogram and throughput gauge (requests/sec).
- **Launch local Docker**: optional helper — runs `docker pull nvcr.io/nvidia/tritonserver:latest` and `docker run ...` via Docker CLI. Studio does not own the process; tracks it via container name.

---

## 5. Plugin Registry API

The NNSpire Plugin Registry is a lightweight REST server (reference implementation: Python FastAPI).

### Endpoints

```
POST   /v1/plugin/register
         Body: { manifest, csr, sourceUrl, pluginType }
         Response: { certPem, communityPluginId }

GET    /v1/plugin/{id}
         Response: { manifest, certPem, downloadUrl }

GET    /v1/plugins?type=LAYER&page=1&limit=20
         Response: { plugins: [...], total }

POST   /v1/enterprise/ca/request
         Body: { csr, orgName, orgUrl, contactEmail }
         Response: { status: "pending" | "approved", tup?: <base64 TUP> }

GET    /v1/crl
         Response: DER-encoded Certificate Revocation List (for offline cache refresh)

GET    /v1/health
```

### Installation within Studio

`Tools → Plugin Manager → Browse Registry` — fetches plugin listing, shows capabilities, trust level, author, license. Install button downloads + verifies manifest + extracts to user plugin directory.

---

## 6. Dual-language export verification

All code exports are verified for behavioural equivalence before the export wizard shows "Export successful":

1. Studio loads the just-exported model in ONNX Runtime (C++ path)
2. Studio runs the just-exported `runner.py` in embedded Python
3. Both receive the same sample input tensor
4. Output tensors compared — must match within configured tolerance (default: `|diff| < 1e-5` for float32)
5. If mismatch: export is flagged as "behaviour mismatch — review custom ops"

This is the deployment-time enforcement of the dual-language principle.
