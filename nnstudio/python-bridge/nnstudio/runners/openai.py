"""
nnstudio.runners.openai — OpenAI-compatible REST runner connector.

Targets any server that implements the OpenAI chat/embeddings endpoints,
including local llama.cpp servers, Ollama, vLLM, Mistral AI, and the
official OpenAI API::

    POST /v1/chat/completions
    POST /v1/embeddings
    GET  /v1/models

Optional dependency::

    pip install httpx

Without ``httpx`` the client operates in *stub* mode.

@kb: DEPLOYMENT.md §3.5
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


# ── OpenAIRunnerClient ────────────────────────────────────────────────────────

class OpenAIRunnerClient(RunnerClient):
    """
    Runner client for OpenAI-compatible REST APIs.

    Supports both cloud (``api.openai.com``) and local servers.
    The ``infer()`` method calls ``/v1/chat/completions``.  For embedding
    models use the ``embed()`` convenience method instead.

    Args:
        endpoint: Base URL, e.g. ``"https://api.openai.com"`` or
                  ``"http://localhost:11434"`` for Ollama.
        api_key:  Bearer token for the ``Authorization`` header.
                  Leave empty for local servers that don't require auth.
        timeout:  HTTP request timeout in seconds (default 60).
        org_id:   OpenAI organisation ID (optional).

    @kb: DEPLOYMENT.md §3.5
    """

    def __init__(self, endpoint: str = "https://api.openai.com",
                 api_key: str = "",
                 timeout: float = 60.0,
                 org_id: str = "") -> None:
        self._base_url = endpoint.rstrip("/")
        self._api_key  = api_key
        self._timeout  = timeout
        self._org_id   = org_id
        self._http: Optional[Any] = None

    # ── RunnerClient interface ─────────────────────────────────────────────────

    def connect(self, endpoint: str = "") -> None:
        if endpoint:
            self._base_url = endpoint.rstrip("/")
        if not _httpx_available:
            return  # stub mode — no real connection
        headers: Dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._org_id:
            headers["OpenAI-Organization"] = self._org_id
        try:
            self._http = _httpx.Client(
                base_url=self._base_url,
                headers=headers,
                timeout=self._timeout,
            )
        except Exception as exc:
            raise RunnerConnectionError(
                f"OpenAI connect failed ({self._base_url}): {exc}") from exc

    def disconnect(self) -> None:
        if self._http is not None:
            self._http.close()
            self._http = None

    def health(self) -> HealthStatus:
        if self._http is None:
            return HealthStatus.UNKNOWN
        try:
            r = self._http.get("/v1/models")
            return HealthStatus.HEALTHY if r.status_code == 200 \
                else HealthStatus.UNHEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY

    def load_model(self, name: str, version: int = -1) -> None:
        """
        For OpenAI-compat servers, models are always resident.
        This checks that the model is listed by the server.
        """
        if self._http is None:
            return
        try:
            r = self._http.get("/v1/models")
            if r.status_code != 200:
                return  # may not implement /v1/models — not fatal
            ids = {m["id"] for m in r.json().get("data", [])}
            if ids and name not in ids:
                raise RunnerModelNotFoundError(
                    f"OpenAI server: model '{name}' is not available. "
                    f"Available: {sorted(ids)}")
        except RunnerError:
            raise
        except Exception:
            pass   # non-standard server — ignore

    def infer(self, model: str, inputs: Dict[str, Any],
              version: int = -1) -> Dict[str, Any]:
        """
        Call ``/v1/chat/completions`` with the provided inputs.

        ``inputs`` should contain at least a ``"messages"`` key with the
        OpenAI messages list.  Any additional keys are forwarded as-is
        (e.g. ``"temperature"``, ``"max_tokens"``).

        Returns the raw ``choices`` list from the response.
        """
        if self._http is None:
            raise RunnerError(
                "OpenAIRunnerClient is in stub mode. "
                "Install 'httpx' and call connect() before infer()."
            )
        if "messages" not in inputs:
            raise RunnerInferenceError(
                "OpenAI infer: 'messages' key required in inputs dict.")
        body: Dict[str, Any] = {"model": model, **inputs}
        try:
            r = self._http.post(
                "/v1/chat/completions",
                content=json.dumps(body),
                headers={"Content-Type": "application/json"})
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"OpenAI server: model '{model}' not found")
            if r.status_code != 200:
                raise RunnerInferenceError(
                    f"OpenAI chat/completions returned HTTP {r.status_code}: "
                    f"{r.text[:200]}")
            data = r.json()
            return {"choices": data.get("choices", []), "usage": data.get("usage")}
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerInferenceError(
                f"OpenAI infer failed for model '{model}': {exc}") from exc

    def chat(self, model: str, messages: List[Dict[str, str]],
             **kwargs: Any) -> Dict[str, Any]:
        """
        Convenience wrapper around infer() for chat completion.

        Args:
            model:    Model ID (e.g. ``"gpt-4o"`` or ``"llama3"``).
            messages: OpenAI messages list: ``[{"role": "user", "content": "…"}]``
            **kwargs: Optional OpenAI parameters (temperature, max_tokens, …).

        Returns:
            ``{"choices": [...], "usage": {...}}``
        """
        return self.infer(model, {"messages": messages, **kwargs})

    def embed(self, model: str, input: Any,
              **kwargs: Any) -> Dict[str, Any]:
        """
        Call ``/v1/embeddings``.

        Args:
            model: Embedding model ID (e.g. ``"text-embedding-3-small"``).
            input: String or list of strings to embed.
            **kwargs: Optional parameters forwarded to the API.

        Returns:
            ``{"data": [{"embedding": [...], "index": 0}], "usage": {...}}``
        """
        if self._http is None:
            raise RunnerError(
                "OpenAIRunnerClient is in stub mode. "
                "Install 'httpx' and call connect() before embed()."
            )
        body: Dict[str, Any] = {"model": model, "input": input, **kwargs}
        try:
            r = self._http.post(
                "/v1/embeddings",
                content=json.dumps(body),
                headers={"Content-Type": "application/json"})
            if r.status_code != 200:
                raise RunnerInferenceError(
                    f"OpenAI embeddings returned HTTP {r.status_code}: "
                    f"{r.text[:200]}")
            return r.json()
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerInferenceError(
                f"OpenAI embed failed for model '{model}': {exc}") from exc

    def model_info(self, name: str, version: int = -1) -> ModelInfo:
        if self._http is None:
            return ModelInfo(name=name, version=version)
        try:
            r = self._http.get(f"/v1/models/{name}")
            if r.status_code == 404:
                raise RunnerModelNotFoundError(
                    f"OpenAI server: model '{name}' not found")
            meta = r.json()
            return ModelInfo(
                name=name,
                version=version,
                state="READY",
                platform=meta.get("owned_by", "openai"),
            )
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerModelNotFoundError(
                f"OpenAI: model info for '{name}' failed: {exc}") from exc

    def metrics(self, model: str) -> ModelMetrics:
        return ModelMetrics()
