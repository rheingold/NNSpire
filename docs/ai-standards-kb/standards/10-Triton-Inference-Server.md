# Triton Inference Server

**Document Version:** 1.0  
**Generated:** December 4, 2025  
**Standard:** NVIDIA Triton Model Repository Format  
**Status:** Industry Standard (NVIDIA)

---

## Overview

**Triton Inference Server** (formerly TensorRT Inference Server) is NVIDIA's production inference platform. It optimizes model serving across diverse hardware (GPUs, CPUs, edge devices).

### Key Features

- **Multi-Framework**: TensorFlow, PyTorch, ONNX, TensorRT, Triton Backend
- **Multi-GPU**: Parallel execution across GPUs
- **Model Repository**: Standardized directory layout
- **Dynamic Batching**: Automatic batching for throughput
- **Ensemble Models**: Chained model pipelines
- **Metrics API**: Prometheus metrics
- **In-Process Backend**: C++ plugin system

### Primary Use Cases

1. **High-Throughput Inference**: GPU-optimized serving
2. **Multi-Model Management**: Serving many models simultaneously
3. **Ensemble Pipelines**: Chained model inference
4. **Real-Time Optimization**: Dynamic batching
5. **Edge Deployment**: NVIDIA Jetson deployment

---

## Authoritative References

- **Triton Docs**: https://docs.nvidia.com/deeplearning/triton
- **GitHub Repository**: https://github.com/triton-inference-server/server
- **Model Repository Format**: https://docs.nvidia.com/deeplearning/triton/user-guide/docs/user_guide/model_repository.html
- **Backend Documentation**: https://docs.nvidia.com/deeplearning/triton/user-guide/docs/reference/backend_apis.html

---

## Format Structure

### Model Repository Layout

**Real example**:
```
model_repository/
├── bert_classifier/
│   ├── 1/                               # Model version
│   │   ├── model.pt                    # PyTorch model
│   │   └── tokenizer.json
│   ├── config.pbtxt                    # Model configuration
│   └── labels.txt
│
├── image_classifier/
│   ├── 1/
│   │   ├── model.onnx                  # ONNX model
│   │   └── resnet50_labels.txt
│   └── config.pbtxt
│
├── preprocessing_ensemble/
│   ├── 1/
│   │   └── preprocessing.pt
│   ├── config.pbtxt
│   └── ensemble_model.onnx             # Ensemble definition
│
└── ensemble_pipeline/
    ├── 1/
    └── config.pbtxt                    # Ensemble orchestration
```

### Model Configuration (config.pbtxt)

**Abstract pseudo-schema** — config.pbtxt field taxonomy (required vs optional):
```
# ── config.pbtxt Generalized Schema ──────────────────────────────────
#
# REQUIRED fields
#   name:             "<model_name>"          # Must match repository directory name
#   platform:         "<backend_identifier>"  # e.g. "pytorch_libtorch", "tensorflow_savedmodel",
#                                             #      "onnxruntime_onnx", "tensorrt_plan", "ensemble"
#   max_batch_size:   <int>                   # 0 = batching disabled; >0 = max requests per batch
#
#   input [ ... ]                             # One or more input tensors
#     name:       "<tensor_name>"             #   Logical name for the input
#     data_type:  <TRITON_TYPE>               #   TYPE_FP32, TYPE_INT64, TYPE_STRING, etc.
#     dims:       [ <shape> ]                 #   Tensor dimensions; -1 = variable length
#
#   output [ ... ]                            # One or more output tensors
#     name:       "<tensor_name>"
#     data_type:  <TRITON_TYPE>
#     dims:       [ <shape> ]
#
# OPTIONAL fields
#   reshape { shape: [ <dims> ] }             # Reshape tensor before/after inference
#   label_filename:    "<file>"               # Maps output indices → human-readable labels
#   default_model_filename: "<file>"          # Override default model file name
#   version_policy:    { <policy> }           # "all", "latest", or { specific: { versions: [N] } }
#   dynamic_batching:  { ... }                # Enable automatic request batching (see below)
#   instance_group:    [ ... ]                # GPU/CPU placement & replica count
#   optimization:      { ... }               # Graph-level accelerator settings
#   response_cache:    { enable: <bool> }     # Cache repeated inference results
#   rate_limiter:      { ... }               # Throttle incoming requests
#   ensemble_scheduling: { step [ ... ] }     # Only for platform: "ensemble"
# ─────────────────────────────────────────────────────────────────────
```

**Real example** (BERT PyTorch model):
```protobuf
name: "bert_classifier"
platform: "pytorch_libtorch"
max_batch_size: 32

# Model input specification
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    reshape { shape: [ -1, 128 ] }
    optional: false
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    reshape { shape: [ -1, 128 ] }
    optional: false
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    reshape { shape: [ -1, 128 ] }
    optional: false
  }
]

# Model output specification
output [
  {
    name: "logits"
    data_type: TYPE_FP32
    reshape { shape: [ -1, 2 ] }
    label_filename: "labels.txt"
  }
]

# Dynamic batching
dynamic_batching {
  preferred_batch_size: [ 16, 32 ]
  max_queue_delay_microseconds: 100000
  preserve_ordering: false
}

# Optimization settings
optimization {
  graph { execution_accelerators { 
    gpu_execution_accelerator { 
      gpu_device: 0
    } 
  } }
}

# Instance groups (GPU placement)
instance_group [
  {
    name: "bert_gpu_0"
    kind: KIND_GPU
    gpus: [ 0 ]
    count: 2
  },
  {
    name: "bert_gpu_1"
    kind: KIND_GPU
    gpus: [ 1 ]
    count: 2
  }
]

# Resource limits
default_model_filename: "model.pt"
version_policy {
  policy: "latest"
}

# Response cache (optional)
response_cache {
  enable: true
}
```

#### Dynamic Batching — Algorithm Pseudocode

The `dynamic_batching` block above controls how Triton groups individual
requests into batches.  The abstract algorithm is:

```
// ── Dynamic Batching Algorithm (abstract) ─────────────────────────
//
// Configuration knobs
//   preferred_batch_sizes  = [B1, B2, …]   // e.g. [16, 32]
//   max_queue_delay        = D µs           // e.g. 100 000 µs
//   max_batch_size         = M              // from top-level config
//
// State
//   request_queue  ← empty FIFO
//
// ── Main scheduling loop ──────────────────────────────────────────
LOOP forever:
  1.  WAIT until  request_queue is non-empty

  2.  // ── Batch formation phase ──
      candidate_batch ← []
      WHILE |candidate_batch| < max_batch_size
            AND request_queue is non-empty:
          next_request ← request_queue.peek()
          IF  next_request can be padded/shaped to match candidate_batch:
              candidate_batch.append( request_queue.dequeue() )
          ELSE:
              BREAK   // shape-incompatible request; start new batch later

  3.  // ── Deadline check ──
      IF |candidate_batch| ∈ preferred_batch_sizes
         OR oldest request in candidate_batch has waited ≥ max_queue_delay:
          GOTO step 4            // dispatch immediately
      ELSE:
          SLEEP briefly, GOTO step 2   // wait for more requests

  4.  // ── Dispatch phase ──
      SEND candidate_batch to next available model instance
      (instance selected from instance_group round-robin / least-loaded)

  5.  COLLECT per-request results; RETURN each response to its caller
END LOOP
// ──────────────────────────────────────────────────────────────────
```

### Model Configuration for TensorFlow

```protobuf
name: "tf_image_classifier"
platform: "tensorflow_savedmodel"
max_batch_size: 64

input [
  {
    name: "input_images"
    data_type: TYPE_UINT8
    dims: [ -1, 224, 224, 3 ]
    reshape { shape: [ -1, 224, 224, 3 ] }
  }
]

output [
  {
    name: "predictions"
    data_type: TYPE_FP32
    dims: [ -1, 1000 ]
    label_filename: "imagenet_labels.txt"
  }
]

dynamic_batching {
  preferred_batch_size: [ 32, 64 ]
  max_queue_delay_microseconds: 50000
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
```

### Ensemble Model Configuration

**Abstract pseudocode** — generalized ensemble DAG pattern:
```
// ── Ensemble Model: Generalized DAG Pattern ──────────────────────
//
// An ensemble in Triton is a directed acyclic graph (DAG) of models.
// Each "step" invokes one sub-model, mapping its inputs/outputs to
// named tensors that flow between steps.
//
// Schema:
//   platform: "ensemble"
//   input  → the tensors exposed to external callers
//   output → the tensors returned to external callers
//
//   ensemble_scheduling {
//     step [
//       { model_name, model_version,
//         input_map  { model_input_name  → dag_tensor_name },
//         output_map { model_output_name → dag_tensor_name } },
//       …
//     ]
//   }
//
// Execution order is inferred from data dependencies:
//
//   external_input
//       │
//       ▼
//   ┌──────────────┐
//   │  Step 1       │  model: preprocessor
//   │  in:  external_input  → out: intermediate_A
//   └──────┬───────┘
//          │  intermediate_A
//          ▼
//   ┌──────────────┐
//   │  Step 2       │  model: core_model
//   │  in:  intermediate_A  → out: intermediate_B
//   └──────┬───────┘
//          │  intermediate_B
//          ▼
//   ┌──────────────┐
//   │  Step N       │  model: postprocessor
//   │  in:  intermediate_B  → out: external_output
//   └──────────────┘
//       │
//       ▼
//   external_output  (returned to caller)
//
// Steps with independent inputs may run IN PARALLEL automatically.
// ─────────────────────────────────────────────────────────────────
```

```protobuf
name: "preprocessing_inference_postprocessing"
platform: "ensemble"

# Input specification (ensemble input)
input [
  {
    name: "image_b64"
    data_type: TYPE_STRING
  }
]

# Output specification (final output)
output [
  {
    name: "class_names"
    data_type: TYPE_STRING
  }
]

# Ensemble composition
ensemble_scheduling {
  step [
    {
      model_name: "image_preprocessor"
      model_version: -1
      input_map {
        key: "image_input"
        value: "image_b64"
      }
      output_map {
        key: "preprocessed_image"
        value: "preprocessed"
      }
    },
    {
      model_name: "image_classifier"
      model_version: -1
      input_map {
        key: "images"
        value: "preprocessed"
      }
      output_map {
        key: "logits"
        value: "classification_output"
      }
    },
    {
      model_name: "postprocessor"
      model_version: -1
      input_map {
        key: "logits_input"
        value: "classification_output"
      }
      output_map {
        key: "class_names"
        value: "class_names"
      }
    }
  ]
}
```

---

## Procedural Use-Cases

### Use-Case 1: PyTorch Model Deployment

**Step 1: Create Model Directory**:
```bash
mkdir -p model_repository/bert_classifier/1
```

**Step 2: Save Model**:
```python
import torch
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model_repository/bert_classifier/1/model.pt")

# Save tokenizer
import json
with open("model_repository/bert_classifier/1/tokenizer.json", "w") as f:
    json.dump(tokenizer.get_vocab(), f)
```

**Step 3: Create config.pbtxt**:
```bash
cat > model_repository/bert_classifier/config.pbtxt << 'EOF'
name: "bert_classifier"
platform: "pytorch_libtorch"
max_batch_size: 32

input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    reshape { shape: [ -1, 128 ] }
  }
]

output [
  {
    name: "logits"
    data_type: TYPE_FP32
    reshape { shape: [ -1, 2 ] }
  }
]

dynamic_batching {
  preferred_batch_size: [ 16, 32 ]
  max_queue_delay_microseconds: 100000
}

instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
EOF
```

**Step 4: Launch Triton**:
```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.04-py3 \
  tritonserver --model-repository=/models
```

### Use-Case 2: Client Inference

**Abstract pseudocode** — HTTP/REST inference protocol flow:
```
// ── Triton HTTP Inference Protocol (KServe v2 compatible) ────────
//
// Endpoint:  POST  /v2/models/<model_name>/versions/<ver>/infer
//
// ── Client-side workflow (abstract) ─────────────────────────────
//
// 1. CONNECT to Triton HTTP endpoint (host:8000)
//
// 2. (Optional) HEALTH CHECK
//       GET  /v2/health/ready          → 200 OK
//       GET  /v2/models/<name>/ready   → 200 OK
//
// 3. (Optional) QUERY model metadata
//       GET  /v2/models/<name>
//       → { name, versions, platform, inputs[ ], outputs[ ] }
//
// 4. BUILD inference request
//       FOR EACH input tensor:
//           a. Create InferInput(name, shape, datatype)
//           b. Serialize numpy array → raw bytes or JSON
//       FOR EACH desired output:
//           a. Create InferRequestedOutput(name)
//
// 5. SEND request
//       POST /v2/models/<name>/infer
//       Body: { inputs: [ {name, shape, datatype, data}, … ],
//              outputs: [ {name}, … ] }
//
// 6. RECEIVE response
//       → { model_name, model_version, id,
//           outputs: [ {name, shape, datatype, data}, … ] }
//
// 7. DESERIALIZE each output tensor → numpy array / native type
//
// 8. RETURN structured result to caller
// ─────────────────────────────────────────────────────────────────
```

**Python Triton Client**:
```python
import tritonclient.http as httpclient
import numpy as np
from tritonclient.utils import np_to_triton_dtype

class TritonInferenceClient:
    def __init__(self, url: str = "localhost:8000"):
        self.client = httpclient.InferenceServerClient(url)
    
    def predict(
        self,
        model_name: str,
        inputs: dict,
        model_version: str = "1"
    ) -> dict:
        """Perform inference on Triton server"""
        
        # Prepare inputs
        triton_inputs = []
        for name, data in inputs.items():
            input_data = httpclient.InferInput(
                name,
                data.shape,
                np_to_triton_dtype(data.dtype)
            )
            input_data.set_data_from_numpy(data)
            triton_inputs.append(input_data)
        
        # Prepare outputs
        triton_outputs = [
            httpclient.InferRequestedOutput(
                "logits",
                binary_data=False
            )
        ]
        
        # Perform inference
        response = self.client.infer(
            model_name=model_name,
            inputs=triton_inputs,
            outputs=triton_outputs,
            model_version=model_version
        )
        
        # Extract outputs
        logits = response.as_numpy("logits")
        
        return {
            "logits": logits,
            "model_name": response.get_response()["model_name"],
            "model_version": response.get_response()["model_version"]
        }

# Usage
client = TritonInferenceClient()

# Prepare input
input_ids = np.array([[101, 2054, 2003, 2009, 102]], dtype=np.int64)

# Run inference
result = client.predict("bert_classifier", {"input_ids": input_ids})
print(f"Logits shape: {result['logits'].shape}")
print(f"Predictions: {result['logits']}")
```

**gRPC Client** (C++):
```cpp
#include <grpc++/grpc++.h>
#include "grpc_service.pb.h"

class TritonGrpcClient {
public:
    TritonGrpcClient(const std::string& url) {
        channel_ = grpc::CreateChannel(
            url,
            grpc::InsecureChannelCredentials()
        );
        stub_ = inference::GRPCInferenceService::NewStub(channel_);
    }
    
    void Infer(const std::string& model_name) {
        inference::ModelInferRequest request;
        request.set_model_name(model_name);
        request.set_model_version("1");
        
        // Add input
        auto* input = request.add_inputs();
        input->set_name("input_ids");
        input->set_datatype("INT64");
        input->add_shape(1);
        input->add_shape(128);
        
        // Set input data
        std::vector<int64_t> data = {101, 2054, 2003, 2009, 102};
        input->set_raw_contents(
            std::string((char*)data.data(), data.size() * sizeof(int64_t))
        );
        
        // Execute
        inference::ModelInferResponse response;
        grpc::ClientContext context;
        
        grpc::Status status = stub_->ModelInfer(
            &context,
            request,
            &response
        );
        
        if (!status.ok()) {
            std::cerr << "RPC failed: " << status.error_message() << std::endl;
            return;
        }
        
        std::cout << "Model: " << response.model_name() << std::endl;
    }

private:
    std::shared_ptr<grpc::Channel> channel_;
    std::unique_ptr<inference::GRPCInferenceService::Stub> stub_;
};
```

### Use-Case 3: Ensemble Pipeline

**Create Preprocessing Model** (Python):
```python
import torch
import torch.nn as nn
from PIL import Image
import base64
from io import BytesIO
import numpy as np

class ImagePreprocessor(nn.Module):
    def forward(self, image_b64_tensor):
        # Decode, resize, normalize
        # This runs in TorchScript
        pass

# Save as TorchScript
preprocessor = ImagePreprocessor()
torch.jit.script(preprocessor).save(
    "model_repository/preprocessing/1/model.pt"
)
```

**Ensemble Configuration**:
```protobuf
name: "image_pipeline"
platform: "ensemble"

input [
  {
    name: "image_b64"
    data_type: TYPE_STRING
  }
]

output [
  {
    name: "class_name"
    data_type: TYPE_STRING
  }
]

ensemble_scheduling {
  step [
    {
      model_name: "image_preprocessor"
      input_map { key: "input" value: "image_b64" }
      output_map { key: "output" value: "preprocessed" }
    },
    {
      model_name: "resnet50_classifier"
      input_map { key: "images" value: "preprocessed" }
      output_map { key: "logits" value: "logits" }
    },
    {
      model_name: "label_converter"
      input_map { key: "logits" value: "logits" }
      output_map { key: "labels" value: "class_name" }
    }
  ]
}
```

---

## Examples

### Example 1: Docker Compose Setup

```yaml
version: '3.8'

services:
  triton:
    image: nvcr.io/nvidia/tritonserver:23.04-py3
    ports:
      - "8000:8000"  # HTTP
      - "8001:8001"  # gRPC
      - "8002:8002"  # Metrics
    volumes:
      - ./model_repository:/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - TRITON_LOG_LEVEL=2
    command: tritonserver --model-repository=/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

### Example 2: Metrics Collection

**Access Prometheus Metrics**:
```bash
curl http://localhost:8002/metrics
```

**Real Metrics Output**:
```
# HELP nv_inference_request_total Total number of inference requests
# TYPE nv_inference_request_total counter
nv_inference_request_total{model="bert_classifier",version="1"} 1523.0

# HELP nv_inference_exec_time_ms Inference execution time in ms
# TYPE nv_inference_exec_time_ms histogram
nv_inference_exec_time_ms_bucket{le="50.0",model="bert_classifier"} 1200.0
nv_inference_exec_time_ms_bucket{le="100.0",model="bert_classifier"} 1450.0

# HELP nv_gpu_utilization GPU utilization percentage
# TYPE nv_gpu_utilization gauge
nv_gpu_utilization{gpu_id="0"} 87.5
nv_gpu_utilization{gpu_id="1"} 92.3
```

### Example 3: Custom Backend

**C++ Backend Plugin**:
```cpp
#include "triton/core/tritonserver.h"

TRITONSERVER_Error* CustomBackendInitialize(TRITONSERVER_Backend* backend) {
    // Initialize backend
    return nullptr;
}

TRITONSERVER_Error* CustomBackendModelInstanceExecute(
    TRITONSERVER_ModelInstance* instance,
    TRITONSERVER_InferenceRequest* request) {
    
    // Extract inputs
    // Run custom logic
    // Set outputs
    
    return nullptr;
}
```

---

## Performance Tuning

| Parameter | Impact | Use Case |
|-----------|--------|----------|
| **max_batch_size** | Throughput vs latency | High-throughput scenarios |
| **dynamic_batching** | Automatic batching | Variable input rate |
| **instance_group** | GPU placement | Multi-GPU scaling |
| **cache** | Response caching | Repeated requests |
| **rate_limiter** | Request throttling | Resource protection |

---

## Tools & Ecosystem

| Tool | Description | URL |
|------|-------------|-----|
| **Triton Server** | Inference server | https://github.com/triton-inference-server |
| **TensorRT** | Model optimization | https://developer.nvidia.com/tensorrt |
| **NVIDIA NGC** | Model hub | https://catalog.ngc.nvidia.com |
| **Triton Metrics** | Prometheus integration | https://docs.nvidia.com/deeplearning/triton/user-guide |
| **Model Analyzer** | Performance profiling | https://github.com/triton-inference-server/model_analyzer |

---

**Navigation**: [Back to Index](../INDEX.md) | [Previous: KServe](./09-KServe.md) | [Next: Vector Embeddings Standards](../annexes/A01-Vector-Embeddings-Standards.md)
