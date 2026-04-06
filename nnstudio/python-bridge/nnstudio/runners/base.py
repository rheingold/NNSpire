"""
nnstudio.runners.base — Runner client base classes and data types.

All runner connectors inherit from RunnerClient and return the
types defined here.

@kb: DEPLOYMENT.md §3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ─── Exceptions ──────────────────────────────────────────────────────────────

class RunnerError(Exception):
    """Base exception for runner connector errors."""
    pass


class RunnerConnectionError(RunnerError):
    """Raised when a connection to the runner server cannot be established."""
    pass


class RunnerModelNotFoundError(RunnerError):
    """Raised when the requested model is not loaded on the server."""
    pass


class RunnerInferenceError(RunnerError):
    """Raised when an inference call returns an error response."""
    pass


# ─── Data types ──────────────────────────────────────────────────────────────

class HealthStatus(Enum):
    """Runner health state."""
    HEALTHY   = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN   = "unknown"


@dataclass
class ModelMetrics:
    """Performance metrics for a loaded model."""
    request_count:       int   = 0
    success_count:       int   = 0
    error_count:         int   = 0
    avg_latency_ms:      float = 0.0
    p99_latency_ms:      float = 0.0
    throughput_rps:      float = 0.0   # requests per second
    gpu_utilization_pct: Optional[float] = None
    gpu_memory_used_mb:  Optional[float] = None


@dataclass
class ModelInfo:
    """Metadata about a model loaded on the runner."""
    name:          str
    version:       int            = -1   # -1 = latest
    state:         str            = "READY"
    input_shapes:  List[List[int]] = field(default_factory=list)
    output_shapes: List[List[int]] = field(default_factory=list)
    platform:      str             = ""
    backend:       str             = ""


# ─── RunnerClient ABC ────────────────────────────────────────────────────────

class RunnerClient(ABC):
    """
    Abstract base class for inference runner connectors.

    Each concrete subclass provides a uniform interface over a specific
    inference server protocol (Triton, TF Serving, KServe, etc.).

    All methods that communicate with the server may raise RunnerError or
    its subclasses on network/protocol errors.

    @kb: DEPLOYMENT.md §3
    """

    @abstractmethod
    def connect(self, endpoint: str) -> None:
        """
        Connect to the inference server at `endpoint`.

        Args:
            endpoint: URL or host:port string.

        Raises:
            RunnerConnectionError: if the connection cannot be established.
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Close the connection and release resources."""

    @abstractmethod
    def health(self) -> HealthStatus:
        """
        Query the server health status.

        Returns:
            HealthStatus.HEALTHY if the server is ready to accept inference
            requests, otherwise HealthStatus.UNHEALTHY or HealthStatus.UNKNOWN.
        """

    @abstractmethod
    def load_model(self, name: str, version: int = -1) -> None:
        """
        Request the server to load a model.  On servers that manage their own
        model repository (Triton, TF Serving) this triggers model loading.
        On embedded runners (ONNX Runtime) it loads the model file.

        Args:
            name:    Model name as registered on the server.
            version: Model version, or -1 for the latest version.

        Raises:
            RunnerModelNotFoundError: if the model is not available.
        """

    @abstractmethod
    def infer(self, model: str, inputs: Dict[str, Any],
              version: int = -1) -> Dict[str, Any]:
        """
        Run inference on the server.

        Args:
            model:   Model name.
            inputs:  Dictionary of input tensor name → numpy array / Tensor.
            version: Model version, or -1 for the latest.

        Returns:
            Dictionary of output tensor name → numpy array.

        Raises:
            RunnerModelNotFoundError: if the model is not loaded.
            RunnerInferenceError: if the server returns an error.
        """

    @abstractmethod
    def model_info(self, name: str, version: int = -1) -> ModelInfo:
        """
        Return metadata for a loaded model.

        Raises:
            RunnerModelNotFoundError: if the model is not loaded.
        """

    @abstractmethod
    def metrics(self, model: str) -> ModelMetrics:
        """
        Return performance metrics for a model.  Returns a zeroed
        ModelMetrics if the server does not support metrics.
        """

    # ── Context manager support ───────────────────────────────────────────────

    def __enter__(self) -> "RunnerClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()
