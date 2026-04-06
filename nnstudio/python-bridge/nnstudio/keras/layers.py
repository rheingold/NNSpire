"""
nnstudio.keras.layers — Keras-compatible layer aliases.

All names match ``tf.keras.layers.*`` so that::

    from nnstudio.keras import layers
    model = layers.Sequential()
    model.add(layers.Dense(8, 4))   # Keras: Dense(units, input_dim)

is a valid Keras-style usage.

Note on Dense signature
-----------------------
Keras ``Dense(units)`` vs NNStudio ``Linear(in, out)``.
The NNStudio ``nn.Linear`` binding accepts ``(in_features, out_features)``
where ``in_features`` is informational only (shape is inferred at build time).
For Keras-style code, call ``Dense(units)`` or ``Dense(in_dim, units)``.
"""

from __future__ import annotations

import nnstudio.nn as _nn

# fmt: off
Dense                = _nn.Linear
Conv2D               = _nn.Conv2d
Embedding            = _nn.Embedding
MultiHeadAttention   = _nn.MultiheadAttention
BatchNormalization   = _nn.BatchNorm1d
Dropout              = _nn.Dropout
LayerNormalization   = _nn.LayerNorm
ReLU                 = _nn.ReLU
Sigmoid              = _nn.Sigmoid
Tanh                 = _nn.Tanh
GELU                 = _nn.GELU
Softmax              = _nn.Softmax
LeakyReLU            = _nn.LeakyReLU
Sequential           = _nn.Sequential
# fmt: on


class LSTM:
    """Placeholder — LSTM is not yet implemented in NNStudio Phase 1.

    Raises ``NotImplementedError`` on construction.
    """

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        raise NotImplementedError(
            "nnstudio.keras.layers.LSTM is not yet implemented in Phase 1. "
            "Use nnstudio.nn.MultiheadAttention for transformer-style attention."
        )


__all__ = [
    "Dense", "Conv2D", "Embedding",
    "MultiHeadAttention", "BatchNormalization", "Dropout",
    "LayerNormalization", "ReLU", "Sigmoid", "Tanh", "GELU",
    "Softmax", "LeakyReLU", "Sequential", "LSTM",
]
