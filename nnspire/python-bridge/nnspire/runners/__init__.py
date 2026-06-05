"""
NNSpire.runners — Inference runner connector clients.

Each connector class implements the RunnerClient protocol to provide
a uniform interface for connecting to inference servers.

RUNNERS
───────
    triton        — NVIDIA Triton Inference Server (gRPC + REST)
    tf_serving    — TensorFlow Serving (REST + gRPC)
    kserve        — KServe V2 Inference Protocol
    onnx_runtime  — ONNX Runtime (embedded, in-process)
    openai        — OpenAI-compatible REST API (Ollama, LM Studio, etc.)

USAGE
─────
    from NNSpire.runners import TritonRunnerClient, OpenAIRunnerClient

    # Triton (requires tritonclient: pip install tritonclient[all])
    client = TritonRunnerClient("localhost:8001")
    client.load_model("my_model")
    output = client.infer("my_model", input_tensor)

    # OpenAI-compatible (requires httpx: pip install httpx)
    client = OpenAIRunnerClient("http://localhost:11434")  # Ollama
    result = client.chat("llama3", messages=[{"role": "user", "content": "hi"}])

@kb: DEPLOYMENT.md §3  |  ai-standards-kb/standards/10-Triton-Inference-Server.md
"""

from .base import (
    RunnerClient,
    HealthStatus,
    ModelInfo,
    ModelMetrics,
    RunnerError,
    RunnerConnectionError,
    RunnerInferenceError,
    RunnerModelNotFoundError,
)
from .triton import TritonRunnerClient
from .tf_serving import TfServingRunnerClient
from .kserve import KServeRunnerClient
from .onnx_runtime import OnnxRuntimeRunnerClient
from .openai import OpenAIRunnerClient

__all__ = [
    "RunnerClient",
    "HealthStatus",
    "ModelInfo",
    "ModelMetrics",
    "RunnerError",
    "RunnerConnectionError",
    "RunnerInferenceError",
    "RunnerModelNotFoundError",
    "TritonRunnerClient",
    "TfServingRunnerClient",
    "KServeRunnerClient",
    "OnnxRuntimeRunnerClient",
    "OpenAIRunnerClient",
]
