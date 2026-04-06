"""
nnstudio.runners.triton — Triton Inference Server runner connector.

Uses the gRPC protocol by default, falling back to HTTP REST when
``tritonclient.grpc`` is unavailable.  Install the optional dependency
with::

    pip install tritonclient[grpc]   # gRPC (preferred)
    pip install tritonclient[http]   # HTTP REST fallback

If neither is available the client operates in *stub* mode: connect(),
load_model(), and health() succeed synchronously while infer() raises
RunnerError with a helpful message.

@kb: DEPLOYMENT.md §3.1
"""

from __future__ import annotations

from typing import Any, Dict

from .base import (
    HealthStatus,
    ModelInfo,
    ModelMetrics,
    RunnerClient,
    RunnerConnectionError,
    RunnerError,
    RunnerInferenceError,
    RunnerModelNotFoundError,
)

# ── Optional tritonclient detection ───────────────────────────────────────────

_grpc_available = False
_http_available = False

try:
    import tritonclient.grpc as _triton_grpc  # type: ignore
    _grpc_available = True
except ModuleNotFoundError:
    pass

if not _grpc_available:
    try:
        import tritonclient.http as _triton_http  # type: ignore
        _http_available = True
    except ModuleNotFoundError:
        pass


# ── TritonRunnerClient ─────────────────────────────────────────────────────────

class TritonRunnerClient(RunnerClient):
    """
    Runner client for NVIDIA Triton Inference Server.

    Triton exposes the KServe V2 Inference Protocol over both gRPC and
    HTTP.  This connector prefers gRPC (lower latency) and falls back to
    the HTTP endpoint.

    Args:
        endpoint: ``host:port`` string, default ``"localhost:8001"`` for gRPC
                  or ``"localhost:8000"`` for HTTP.
        prefer_grpc: Force gRPC if True, HTTP if False.  When unspecified the
                     connector picks whichever backend is available.

    @kb: DEPLOYMENT.md §3.1
    """

    def __init__(self, endpoint: str = "localhost:8001",
                 prefer_grpc: bool = True) -> None:
        self._endpoint    = endpoint
        self._prefer_grpc = prefer_grpc
        self._client: Any = None
        self._use_grpc    = False

    # ── RunnerClient interface ─────────────────────────────────────────────────

    def connect(self, endpoint: str = "") -> None:
        if endpoint:
            self._endpoint = endpoint
        if self._prefer_grpc and _grpc_available:
            try:
                self._client   = _triton_grpc.InferenceServerClient(
                    url=self._endpoint, verbose=False)
                self._use_grpc = True
                return
            except Exception as exc:
                raise RunnerConnectionError(
                    f"Triton gRPC connect failed ({self._endpoint}): {exc}"
                ) from exc
        if _http_available:
            try:
                self._client   = _triton_http.InferenceServerClient(
                    url=self._endpoint, verbose=False)
                self._use_grpc = False
                return
            except Exception as exc:
                raise RunnerConnectionError(
                    f"Triton HTTP connect failed ({self._endpoint}): {exc}"
                ) from exc
        # Stub mode — no real client
        self._client = None

    def disconnect(self) -> None:
        self._client = None

    def health(self) -> HealthStatus:
        if self._client is None:
            return HealthStatus.UNKNOWN
        try:
            ok = self._client.is_server_live()
            return HealthStatus.HEALTHY if ok else HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    def load_model(self, name: str, version: int = -1) -> None:
        if self._client is None:
            return  # stub mode — silently succeed
        try:
            self._client.load_model(name)
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"Triton: could not load model '{name}': {exc}"
            ) from exc

    def infer(self, model: str, inputs: Dict[str, Any],
              version: int = -1) -> Dict[str, Any]:
        if self._client is None:
            raise RunnerError(
                "TritonRunnerClient is in stub mode. "
                "Install 'tritonclient[grpc]' or 'tritonclient[http]' and "
                "call connect() before infer()."
            )
        try:
            import numpy as np  # type: ignore

            if self._use_grpc:
                triton_inputs = []
                for name_, data in inputs.items():
                    arr = np.asarray(data)
                    inp = _triton_grpc.InferInput(
                        name_, arr.shape,
                        _numpy_dtype_to_triton(arr.dtype))
                    inp.set_data_from_numpy(arr)
                    triton_inputs.append(inp)
                response = self._client.infer(model_name=model,
                                              inputs=triton_inputs)
                return {out.name: response.as_numpy(out.name)
                        for out in response.get_output_names_and_dtypes()}
            else:
                # HTTP client — same API surface
                triton_inputs = []
                for name_, data in inputs.items():
                    arr = np.asarray(data)
                    inp = _triton_http.InferInput(
                        name_, arr.shape,
                        _numpy_dtype_to_triton(arr.dtype))
                    inp.set_data_from_numpy(arr)
                    triton_inputs.append(inp)
                response = self._client.infer(model_name=model,
                                              inputs=triton_inputs)
                return {out: response.as_numpy(out)
                        for out in [o["name"] for o in
                                    response.get_output()]}
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerInferenceError(
                f"Triton infer failed for model '{model}': {exc}") from exc

    def model_info(self, name: str, version: int = -1) -> ModelInfo:
        if self._client is None:
            return ModelInfo(name=name, version=version)
        try:
            meta = self._client.get_model_metadata(name)
            return ModelInfo(
                name=name,
                version=version,
                state="READY",
                platform=getattr(meta, "platform", ""),
            )
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"Triton: model '{name}' not found: {exc}") from exc

    def metrics(self, model: str) -> ModelMetrics:
        # Triton exposes Prometheus metrics; parse on best-effort basis
        return ModelMetrics()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _numpy_dtype_to_triton(dtype: Any) -> str:
    """Convert a numpy dtype to the Triton type string."""
    import numpy as np  # type: ignore
    mapping = {
        np.float32: "FP32",
        np.float64: "FP64",
        np.float16: "FP16",
        np.int8:    "INT8",
        np.int16:   "INT16",
        np.int32:   "INT32",
        np.int64:   "INT64",
        np.uint8:   "UINT8",
        np.uint16:  "UINT16",
        np.uint32:  "UINT32",
        np.uint64:  "UINT64",
        np.bool_:   "BOOL",
    }
    return mapping.get(dtype.type, "FP32")
