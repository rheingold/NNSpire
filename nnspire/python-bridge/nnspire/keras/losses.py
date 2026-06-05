"""
NNSpire.keras.losses — Keras-compatible loss function aliases.

Usage::

    from NNSpire.keras import losses
    loss_fn = losses.MeanSquaredError()
    loss    = loss_fn.forward(pred, target)
    grad    = loss_fn.gradient(pred, target)
"""

from __future__ import annotations

import NNSpire.nn as _nn

MeanSquaredError        = _nn.MSELoss
CategoricalCrossentropy = _nn.CrossEntropyLoss
BinaryCrossentropy      = _nn.BCELoss

__all__ = [
    "MeanSquaredError",
    "CategoricalCrossentropy",
    "BinaryCrossentropy",
]
