"""
NNSpire.runners.tf_serving — TensorFlow Serving runner connector.

Communicates via the TF Serving REST prediction API::

    POST /v1/models/{name}:predict
    GET  /v1/models/{name}

Optional dependency::

    pip install httpx

Without ``httpx`` the client operates in *stub* mode.

@kb: DEPLOYMENT.md §3.2
"""

from __future__ import annotations

from typing import Any, Dict, Optional

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


# ── TfServingRunnerClient ─────────────────────────────────────────────────────

class TfServingRunnerClient(RunnerClient):
    """
    Runner client for TensorFlow Serving.

    TF Serving exposes a REST API at ``http://host:8501`` and optionally a
    gRPC API at ``host:8500``.  This connector uses the REST endpoint.

    Args:
        endpoint: Base URL, e.g. ``"http://localhost:8501"``.
        timeout:  HTTP request timeout in seconds (default 30).

    @kb: DEPLOYMENT.md §3.2
    """

    def __init__(self, endpoint: str = "http://localhost:8501",
                 timeout: float = 30.0) -> None:
        self._base_url = endpoint.rstrip("/")
        self._timeout  = timeout
        self._connected = False
        self._http: Optional[Any] = None   # httpx.Client

    # ── RunnerClient interface ─────────────────────────────────────────────────

    def connect(self, endpoint: str = "") -> None:
        if endpoint:
            self._base_url = endpoint.rstrip("/")
        if not _httpx_available:
            self._connected = True
            return  # stub mode
        try:
            self._http = _httpx.Client(
                base_url=self._base_url, timeout=self._timeout)
            self._connected = True
        except Exception as exc:
            raise RunnerConnectionError(
                f"TF Serving connect failed ({self._base_url}): {exc}"
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
            r = self._http.get("/")
            return HealthStatus.HEALTHY if r.status_code < 500 \
                else HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    def load_model(self, name: str, version: int = -1) -> None:
        # TF Serving loads models from its model config; just verify existence
        if not self._connected or self._http is None:
            return
        path = f"/v1/models/{name}"
        try:
            r = self._http.get(path)
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"TF Serving: model '{name}' not found")
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"TF Serving: could not query model '{name}': {exc}"
            ) from exc

    def infer(self, model: str, inputs: Dict[str, Any],
              version: int = -1) -> Dict[str, Any]:
        if self._http is None:
            raise RunnerError(
                "TfServingRunnerClient is in stub mode. "
                "Install 'httpx' and call connect() before infer()."
            )
        ver_segment = f"/versions/{version}" if version > 0 else ""
        url = f"/v1/models/{model}{ver_segment}:predict"
        try:
            import json
            body = {"instances": _to_json_serialisable(inputs)}
            r = self._http.post(url, content=json.dumps(body),
                                headers={"Content-Type": "application/json"})
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"TF Serving: model '{model}' not found")
            if r.status_code != 200:
                raise RunnerInferenceError(
                    f"TF Serving: infer returned HTTP {r.status_code}: "
                    f"{r.text[:200]}")
            data = r.json()
            # TF Serving returns {"predictions": [...]}
            return {"predictions": data.get("predictions", data)}
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerInferenceError(
                f"TF Serving infer failed for model '{model}': {exc}"
            ) from exc

    def model_info(self, name: str, version: int = -1) -> ModelInfo:
        if self._http is None:
            return ModelInfo(name=name, version=version)
        try:
            r = self._http.get(f"/v1/models/{name}/metadata")
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"TF Serving: model '{name}' not found")
            meta = r.json()
            return ModelInfo(
                name=name,
                version=version,
                state=meta.get("model_spec", {}).get("version", "READY"),
                platform="tensorflow",
            )
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"TF Serving: model info for '{name}' failed: {exc}"
            ) from exc

    def metrics(self, model: str) -> ModelMetrics:
        return ModelMetrics()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_json_serialisable(obj: Any) -> Any:
    """Recursively convert numpy arrays to nested lists."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_json_serialisable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_serialisable(v) for v in obj]
    return obj
