"""
nnstudio.keras — Keras-compatible API surface.

Usage
-----
::

    from nnstudio.keras import Model
    from nnstudio.keras import layers, losses, optimizers, callbacks

    seq = layers.Sequential()
    seq.add(layers.Dense(4, 1))

    m = Model(seq)
    m.compile(optimizer=optimizers.Adam(), loss=losses.MeanSquaredError())
    history = m.fit(x, y, epochs=10)
"""

from . import layers        # noqa: F401
from . import losses        # noqa: F401
from . import optimizers    # noqa: F401
from . import callbacks     # noqa: F401
from ._model import Model   # noqa: F401

__all__ = [
    "Model",
    "layers",
    "losses",
    "optimizers",
    "callbacks",
]
