"""
NNSpire.runners.kserve — KServe / Open Model Service runner connector.

Uses the KServe V2 Inference Protocol, which is compatible with Triton,
Seldon, OVMS, and other servers that implement the same standard::

    POST /v2/models/{name}/infer
    GET  /v2/models/{name}
    GET  /v2/health/live

Optional dependency::

    pip install httpx

Without ``httpx`` the client operates in *stub* mode.

@kb: DEPLOYMENT.md §3.3
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

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

# ── Optional httpx detection ──────────────────────────────────────────────────

_httpx_available = False
try:
    import httpx as _httpx  # type: ignore
    _httpx_available = True
except ModuleNotFoundError:
    pass


# ── KServeRunnerClient ────────────────────────────────────────────────────────

class KServeRunnerClient(RunnerClient):
    """
    Runner client for KServe (V2 Inference Protocol).

    Targets the HTTP endpoint of any server that implements the
    Open Model Service V2 API (KServe, Triton HTTP, OVMS, …).

    Args:
        endpoint: Base URL, e.g. ``"http://localhost:8080"``.
        timeout:  HTTP request timeout in seconds (default 30).
        namespace: Kubernetes namespace for model URIs (optional).

    @kb: DEPLOYMENT.md §3.3
    """

    def __init__(self, endpoint: str = "http://localhost:8080",
                 timeout: float = 30.0,
                 namespace: str = "") -> None:
        self._base_url  = endpoint.rstrip("/")
        self._timeout   = timeout
        self._namespace = namespace
        self._connected = False
        self._http: Optional[Any] = None

    # ── RunnerClient interface ─────────────────────────────────────────────────

    def connect(self, endpoint: str = "") -> None:
        if endpoint:
            self._base_url = endpoint.rstrip("/")
        if not _httpx_available:
            self._connected = True
            return
        try:
            self._http = _httpx.Client(
                base_url=self._base_url, timeout=self._timeout)
            self._connected = True
        except Exception as exc:
            raise RunnerConnectionError(
                f"KServe connect failed ({self._base_url}): {exc}"
            ) from exc

    def disconnect(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None
        self._connected = False

    def health(self) -> HealthStatus:
        if not self._connected:
            return HealthStatus.UNKNOWN
        if self._http is None:
            return HealthStatus.UNKNOWN
        try:
            r = self._http.get("/v2/health/live")
            return HealthStatus.HEALTHY if r.status_code == 200 \
                else HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    def load_model(self, name: str, version: int = -1) -> None:
        if not self._connected or self._http is None:
            return
        path = f"/v2/models/{name}"
        try:
            r = self._http.get(path)
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"KServe: model '{name}' not found at {self._base_url}")
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"KServe: could not query model '{name}': {exc}") from exc

    def infer(self, model: str, inputs: Dict[str, Any],
              version: int = -1) -> Dict[str, Any]:
        if self._http is None:
            raise RunnerError(
                "KServeRunnerClient is in stub mode. "
                "Install 'httpx' and call connect() before infer()."
            )
        ver_segment = f"/versions/{version}" if version > 0 else ""
        url = f"/v2/models/{model}{ver_segment}/infer"
        try:
            body = _build_v2_infer_request(inputs)
            r = self._http.post(url,
                                content=json.dumps(body),
                                headers={"Content-Type": "application/json"})
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"KServe: model '{model}' not found")
            if r.status_code != 200:
                raise RunnerInferenceError(
                    f"KServe: infer returned HTTP {r.status_code}: "
                    f"{r.text[:200]}")
            resp = r.json()
            # V2 protocol: outputs[i].data
            outputs: Dict[str, Any] = {}
            for out in resp.get("outputs", []):
                outputs[out["name"]] = out.get("data")
            return outputs
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerInferenceError(
                f"KServe infer failed for model '{model}': {exc}") from exc

    def model_info(self, name: str, version: int = -1) -> ModelInfo:
        if self._http is None:
            return ModelInfo(name=name, version=version)
        try:
            r = self._http.get(f"/v2/models/{name}")
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"KServe: model '{name}' not found")
            meta = r.json()
            return ModelInfo(
                name=name,
                version=version,
                state="READY",
                platform=meta.get("platform", ""),
            )
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"KServe: model info for '{name}' failed: {exc}") from exc

    def metrics(self, model: str) -> ModelMetrics:
        return ModelMetrics()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_v2_infer_request(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a name→array dict to a KServe V2 infer request body.

    Ref: https://kserve.github.io/website/modelserving/data_plane/v2_protocol/
    """
    v2_inputs: List[Dict[str, Any]] = []
    for name, data in inputs.items():
        arr = data
        if hasattr(arr, "tolist"):
            flat = arr.reshape(-1).tolist()
            shape: List[int] = list(arr.shape)
        elif isinstance(arr, (list, tuple)):
            import numpy as np  # type: ignore
            arr = np.asarray(arr)
            flat = arr.reshape(-1).tolist()
            shape = list(arr.shape)
        else:
            flat = [arr]
            shape = [1]
        v2_inputs.append({
            "name":     name,
            "shape":    shape,
            "datatype": "FP32",
            "data":     flat,
        })
    return {"inputs": v2_inputs}
