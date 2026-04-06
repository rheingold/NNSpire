"""
nnstudio.keras._model — Keras-compatible Model class.

The ``Model`` class wraps any ``nnstudio.nn.Module`` (e.g. Sequential)
and exposes the Keras ``compile / fit / predict / evaluate`` API.

Training loop
-------------
``fit()`` performs full-batch gradient descent using the standard 6-step
cycle for each epoch:

  1. ``model.zeroGrad()``           — clear accumulated gradients
  2. ``model.forward(x)``           — forward pass → predictions
  3. ``loss_fn.gradient(pred, y)``  — seed gradient for backprop
  4. ``model.backward(grad)``       — propagate gradients to weights
  5. ``optimizer.step(params)``     — update weights
  6. ``loss_fn.forward(pred, y)``   — compute monitoring loss scalar

Mini-batching is not yet supported in Phase 1 (requires Tensor slicing).
The entire dataset is treated as a single batch.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


class Model:
    """Keras-compatible training wrapper for an ``nnstudio.nn.Module``.

    Parameters
    ----------
    model : nnstudio.nn.Module
        The neural network to train (e.g. an ``nnstudio.nn.Sequential``).

    Examples
    --------
    ::

        import nnstudio
        from nnstudio.keras import Model
        from nnstudio.keras import layers, losses, optimizers

        seq = nnstudio.nn.Sequential()
        seq.add(nnstudio.nn.Linear(2, 4))
        seq.add(nnstudio.nn.ReLU())
        seq.add(nnstudio.nn.Linear(4, 1))

        m = Model(seq)
        m.compile(optimizer=optimizers.Adam(lr=1e-3),
                  loss=losses.MeanSquaredError())

        x = nnstudio.ones([100, 2])
        y = nnstudio.zeros([100, 1])
        history = m.fit(x, y, epochs=50)
        print(history["loss"][-1])
    """

    def __init__(self, model: Any) -> None:
        self._model:     Any              = model
        self._optimizer: Any              = None
        self._loss_fn:   Any              = None
        self._metrics:   List[str]        = []

    # ------------------------------------------------------------------
    # compile
    # ------------------------------------------------------------------
    def compile(
        self,
        optimizer: Any,
        loss: Any,
        metrics: Optional[List[str]] = None,
    ) -> None:
        """Configure the model for training.

        Parameters
        ----------
        optimizer
            An nnstudio optimizer, e.g. ``optimizers.Adam(lr=1e-3)``.
        loss
            An nnstudio loss function, e.g. ``losses.MeanSquaredError()``.
        metrics : list of str, optional
            Metric names — accepted for API compatibility, currently ignored.
        """
        self._optimizer = optimizer
        self._loss_fn   = loss
        self._metrics   = list(metrics) if metrics else []

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    def fit(
        self,
        x: Any,
        y: Any,
        epochs: int = 1,
        batch_size: Optional[int] = None,  # noqa: ARG002  (Phase 1: full-batch only)
        validation_data: Optional[Tuple[Any, Any]] = None,
        callbacks: Optional[List[Any]] = None,
        verbose: int = 1,
    ) -> Dict[str, List[float]]:
        """Train the model for a fixed number of epochs.

        Parameters
        ----------
        x, y
            Input and target tensors (full dataset treated as one batch).
        epochs : int
            Number of training epochs.
        batch_size : int, optional
            Ignored in Phase 1 — mini-batching requires Tensor slicing support.
        validation_data : (x_val, y_val), optional
            Compute validation loss after each epoch (no gradient computation).
        callbacks : list, optional
            Keras-style callback objects. ``on_epoch_end(epoch, logs)`` and the
            ``should_stop`` property are honoured.
        verbose : int
            0 = silent; 1 = print loss per epoch.

        Returns
        -------
        dict
            ``{"loss": [...], "val_loss": [...]}`` — one value per epoch.
        """
        if self._optimizer is None or self._loss_fn is None:
            raise RuntimeError(
                "Call Model.compile(optimizer=..., loss=...) before Model.fit()."
            )

        history: Dict[str, List[float]] = {"loss": []}
        if validation_data is not None:
            history["val_loss"] = []

        cbs = callbacks or []
        logs: Dict[str, Any] = {}

        # on_train_begin
        for cb in cbs:
            if callable(getattr(cb, "on_train_begin", None)):
                cb.on_train_begin(dict(logs))

        for epoch in range(epochs):

            # ── 1. zero gradients ────────────────────────────────────────────
            self._model.zeroGrad()

            # ── 2. forward pass ──────────────────────────────────────────────
            pred = self._model.forward(x)

            # ── 3. loss gradient (seeds backward) ───────────────────────────
            grad = self._loss_fn.gradient(pred, y)

            # ── 4. backward pass ─────────────────────────────────────────────
            self._model.backward(grad)

            # ── 5. optimizer step ────────────────────────────────────────────
            params = self._model.parameters()
            self._optimizer.step(params)

            # ── 6. monitoring loss ───────────────────────────────────────────
            loss_tensor = self._loss_fn.forward(pred, y)
            epoch_loss  = float(loss_tensor.item())
            history["loss"].append(epoch_loss)
            logs = {"loss": epoch_loss}

            # ── 7. validation loss (no grad) ─────────────────────────────────
            if validation_data is not None:
                x_val, y_val  = validation_data
                pred_val      = self._model.forward(x_val)
                val_t         = self._loss_fn.forward(pred_val, y_val)
                val_loss      = float(val_t.item())
                history["val_loss"].append(val_loss)
                logs["val_loss"] = val_loss

            # ── 8. callbacks ─────────────────────────────────────────────────
            for cb in cbs:
                if callable(getattr(cb, "on_epoch_end", None)):
                    cb.on_epoch_end(epoch, dict(logs))

            # ── 9. early-stopping check ──────────────────────────────────────
            if any(getattr(cb, "should_stop", False) for cb in cbs):
                break

            # ── 10. console logging ──────────────────────────────────────────
            if verbose:
                val_str = (
                    f"  val_loss={logs.get('val_loss', 0.):.4f}"
                    if "val_loss" in logs
                    else ""
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}"
                    f"  loss={epoch_loss:.4f}{val_str}"
                )

        # on_train_end
        for cb in cbs:
            if callable(getattr(cb, "on_train_end", None)):
                cb.on_train_end(dict(logs))

        return history

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------
    def predict(self, x: Any) -> Any:
        """Return predictions for the input — no gradient computation.

        Temporarily switches the model to eval mode.
        """
        self._model.eval()
        result = self._model.forward(x)
        self._model.train()
        return result

    # ------------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------------
    def evaluate(self, x: Any, y: Any) -> Dict[str, float]:
        """Compute loss on the given data without gradient computation."""
        if self._loss_fn is None:
            raise RuntimeError(
                "Call Model.compile() before Model.evaluate()."
            )
        pred   = self.predict(x)
        loss_t = self._loss_fn.forward(pred, y)
        return {"loss": float(loss_t.item())}

    # ------------------------------------------------------------------
    # Properties — read-only access to internals
    # ------------------------------------------------------------------
    @property
    def trainable_variables(self) -> list:
        """All non-frozen Parameter objects (matches Keras naming)."""
        return [p for p in self._model.parameters() if not p.frozen]
