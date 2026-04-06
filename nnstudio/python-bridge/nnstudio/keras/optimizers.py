"""
nnstudio.keras.optimizers — Keras-compatible optimizer aliases.

Usage::

    from nnstudio.keras import optimizers
    opt = optimizers.Adam(lr=1e-3)
    opt.step(model.parameters())
"""

from __future__ import annotations

import nnstudio.optim as _optim

SGD     = _optim.SGD
Adam    = _optim.Adam
AdamW   = _optim.AdamW
RMSprop = _optim.RMSprop

__all__ = ["SGD", "Adam", "AdamW", "RMSprop"]
