"""
nnstudio.runners.onnx_runtime — ONNX Runtime in-process runner connector.

Loads ``.onnx`` model files directly into the current process using
``onnxruntime``::

    pip install onnxruntime        # CPU
    pip install onnxruntime-gpu    # CUDA / TensorRT

Without ``onnxruntime`` the client operates in *stub* mode.

@kb: DEPLOYMENT.md §3.4
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from .base import (
    HealthStatus,
    ModelInfo,
    ModelMetrics,
    RunnerClient,
    RunnerError,
    RunnerInferenceError,
    RunnerModelNotFoundError,
)

# ── Optional onnxruntime detection ────────────────────────────────────────────

_ort_available = False
try:
    import onnxruntime as _ort  # type: ignore
    _ort_available = True
except ModuleNotFoundError:
    pass


# ── OnnxRuntimeRunnerClient ───────────────────────────────────────────────────

class OnnxRuntimeRunnerClient(RunnerClient):
    """
    Fully in-process runner backed by ONNX Runtime.

    Unlike the server-based connectors, ONNX Runtime runs inside the same
    process.  ``connect()`` sets the model search path; ``load_model()``
    creates an ``ort.InferenceSession`` from a ``.onnx`` file.

    Args:
        model_dir:   Directory containing ``.onnx`` files.  A model named
                     ``"bert"`` is looked up as ``model_dir/bert.onnx``.
        providers:   ONNX Runtime execution providers in priority order.
                     Default: ``["CUDAExecutionProvider",
                     "CPUExecutionProvider"]``.
        inter_threads: Number of threads between operators.
        intra_threads: Number of threads within operators.

    @kb: DEPLOYMENT.md §3.4
    """

    def __init__(self, model_dir: str = ".",
                 providers: Optional[List[str]] = None,
                 inter_threads: int = 0,
                 intra_threads: int = 0) -> None:
        self._model_dir     = model_dir
        self._providers     = providers or [
            "CUDAExecutionProvider", "CPUExecutionProvider"]
        self._inter_threads = inter_threads
        self._intra_threads = intra_threads
        self._sessions: Dict[str, Any] = {}  # name → InferenceSession
        self._connected = False

    # ── RunnerClient interface ─────────────────────────────────────────────────

    def connect(self, endpoint: str = "") -> None:
        """
        For ONNX Runtime, ``endpoint`` is the model directory path.
        If omitted the model_dir supplied at construction is used.
        """
        if endpoint:
            self._model_dir = endpoint
        self._connected = True

    def disconnect(self) -> None:
        self._sessions.clear()
        self._connected = False

    def health(self) -> HealthStatus:
        if not self._connected:
            return HealthStatus.UNKNOWN
        if not _ort_available:
            return HealthStatus.UNKNOWN
        return HealthStatus.HEALTHY

    def load_model(self, name: str, version: int = -1) -> None:
        if not _ort_available:
            return  # stub mode — silently succeed
        path = self._resolve_model_path(name, version)
        if not os.path.isfile(path):
            raise RunnerModelNotFoundError(
                f"ONNX Runtime: model file not found: {path}")
        try:
            opts = _ort.SessionOptions()
            if self._inter_threads:
                opts.inter_op_num_threads = self._inter_threads
            if self._intra_threads:
                opts.intra_op_num_threads = self._intra_threads

            # Filter to providers that are actually available
            available = {p for p in _ort.get_available_providers()}
            providers  = [p for p in self._providers if p in available]
            if not providers:
                providers = ["CPUExecutionProvider"]

            self._sessions[name] = _ort.InferenceSession(
                path, sess_options=opts, providers=providers)
        except Exception as exc:
            raise RunnerError(
                f"ONNX Runtime: could not load '{name}' from {path}: {exc}"
            ) from exc

    def infer(self, model: str, inputs: Dict[str, Any],
              version: int = -1) -> Dict[str, Any]:
        if not _ort_available:
            raise RunnerError(
                "OnnxRuntimeRunnerClient is in stub mode. "
                "Install 'onnxruntime' or 'onnxruntime-gpu' and call "
                "load_model() before infer()."
            )
        if model not in self._sessions:
            # Auto-load on first use
            self.load_model(model, version)
        session = self._sessions[model]
        try:
            import numpy as np  # type: ignore
            feed = {name: np.asarray(val) for name, val in inputs.items()}
            outputs = session.run(None, feed)
            out_names = [o.name for o in session.get_outputs()]
            return dict(zip(out_names, outputs))
        except RunnerError:
            raise
        except Exception as exc:
            raise RunnerInferenceError(
                f"ONNX Runtime infer failed for model '{model}': {exc}"
            ) from exc

    def model_info(self, name: str, version: int = -1) -> ModelInfo:
        if name in self._sessions:
            session = self._sessions[name]
            input_shapes  = [list(i.shape or []) for i in
                             session.get_inputs()]
            output_shapes = [list(o.shape or []) for o in
                             session.get_outputs()]
            return ModelInfo(
                name=name,
                version=version,
                state="READY",
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                platform="onnxruntime",
            )
        path = self._resolve_model_path(name, version)
        if not os.path.isfile(path):
            raise RunnerModelNotFoundError(
                f"ONNX Runtime: model '{name}' not loaded and path "
                f"not found: {path}")
        return ModelInfo(name=name, version=version, platform="onnxruntime")

    def metrics(self, model: str) -> ModelMetrics:
        # ONNX Runtime does not natively expose server-side metrics
        return ModelMetrics()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resolve_model_path(self, name: str, version: int) -> str:
        """
        Resolve a model name to a file path.

        Lookup order:
        1. ``{model_dir}/{name}/{version}/{name}.onnx``
        2. ``{model_dir}/{name}/{name}.onnx``
        3. ``{model_dir}/{name}.onnx``
        """
        candidates: List[str] = []
        if version > 0:
            candidates.append(
                os.path.join(self._model_dir, name, str(version),
                             f"{name}.onnx"))
        candidates += [
            os.path.join(self._model_dir, name, f"{name}.onnx"),
            os.path.join(self._model_dir, f"{name}.onnx"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return candidates[-1]   # fall through — caller checks existence
