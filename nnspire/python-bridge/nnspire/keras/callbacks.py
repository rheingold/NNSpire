"""
NNSpire.keras.callbacks — Keras-compatible training callbacks.

Supported callbacks
-------------------
* ``EarlyStopping``   — stop training when a metric stops improving.
* ``ModelCheckpoint`` — (Phase 1 stub) records filepath intent; actual
  binary save/load via ``Checkpoint`` C++ class is wired in a future patch.

Usage::

    from NNSpire.keras.callbacks import EarlyStopping, ModelCheckpoint

    es = EarlyStopping(monitor="loss", patience=5)
    history = model.fit(x, y, epochs=200, callbacks=[es])
"""

from __future__ import annotations

import math as _math
from typing import Any, Dict


class Callback:
    """Abstract base class for Keras-style training callbacks."""

    def on_train_begin(self, logs: Dict[str, Any]) -> None:
        """Called at the start of training."""

    def on_train_end(self, logs: Dict[str, Any]) -> None:
        """Called at the end of training."""

    def on_epoch_begin(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the start of each epoch."""

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each epoch with the epoch metrics."""

    def on_batch_begin(self, batch: int, logs: Dict[str, Any]) -> None:
        """Called at the start of each batch."""

    def on_batch_end(self, batch: int, logs: Dict[str, Any]) -> None:
        """Called at the end of each batch."""

    @property
    def should_stop(self) -> bool:
        """Return ``True`` to request early termination of training."""
        return False


class EarlyStopping(Callback):
    """Stop training when a monitored metric stops improving.

    Mirrors ``tf.keras.callbacks.EarlyStopping``.

    Parameters
    ----------
    monitor : str
        Name of the metric to watch.  Defaults to ``'loss'``.
    patience : int
        Number of epochs with no improvement before training is stopped.
    min_delta : float
        Minimum absolute change in the monitored value to qualify as an
        improvement.
    restore_best_weights : bool
        **Phase 1 stub** — accepted for API compatibility but not yet
        implemented (weights are not restored on stop).
    verbose : int
        If non-zero, print a message when training is stopped.
    """

    def __init__(
        self,
        monitor: str = "loss",
        patience: int = 5,
        min_delta: float = 1e-4,
        restore_best_weights: bool = False,
        verbose: int = 1,
    ) -> None:
        self.monitor               = monitor
        self.patience              = patience
        self.min_delta             = min_delta
        self.restore_best_weights  = restore_best_weights
        self.verbose               = verbose
        self._best: float          = _math.inf
        self._counter: int         = 0
        self._stop: bool           = False

    # ------------------------------------------------------------------
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        value: float = float(logs.get(self.monitor, _math.inf))
        if value < self._best - self.min_delta:
            self._best    = value
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                self._stop = True
                if self.verbose:
                    print(
                        f"EarlyStopping: '{self.monitor}' did not improve "
                        f"for {self.patience} epochs. Stopping at epoch {epoch + 1}."
                    )

    @property
    def should_stop(self) -> bool:
        return self._stop

    def reset(self) -> None:
        """Reset state — call before re-using the callback in a new fit()."""
        self._best    = _math.inf
        self._counter = 0
        self._stop    = False


class ModelCheckpoint(Callback):
    """Save the model after every epoch (or only the best epoch).

    Phase 1 implementation: the filepath is stored and reported but
    actual binary I/O (via ``NNSpire.core.Checkpoint``) is wired in a
    future patch once the Python checkpoint bindings land.

    Parameters
    ----------
    filepath : str
        Path pattern for the saved checkpoint, e.g. ``"best_model.nns"``.
    monitor : str
        Metric used to decide whether to save.
    save_best_only : bool
        If ``True``, save only when the metric improves.
    verbose : int
        If non-zero, print a message when a checkpoint is (would be) saved.
    """

    def __init__(
        self,
        filepath: str,
        monitor: str = "loss",
        save_best_only: bool = True,
        verbose: int = 0,
    ) -> None:
        self.filepath       = filepath
        self.monitor        = monitor
        self.save_best_only = save_best_only
        self.verbose        = verbose
        self._best: float   = _math.inf

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]) -> None:
        value = float(logs.get(self.monitor, _math.inf))
        if self.save_best_only and value >= self._best:
            return
        self._best = value
        if self.verbose:
            print(
                f"ModelCheckpoint: epoch {epoch + 1} — "
                f"would save to '{self.filepath}' (Checkpoint bindings pending)."
            )
        # TODO: Checkpoint::save once Python bindings are exposed.


__all__ = ["Callback", "EarlyStopping", "ModelCheckpoint"]
