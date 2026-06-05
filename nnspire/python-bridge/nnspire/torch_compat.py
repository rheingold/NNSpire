"""
NNSpire.torch_compat — drop-in replacement for ``import torch``.

Usage::

    import NNSpire.torch_compat as torch   # instead of: import torch
    import NNSpire.torch_compat.nn as nn   # instead of: import torch.nn as nn
    import NNSpire.torch_compat.nn.functional as F

Any code that uses only the standard PyTorch API surface should work
unchanged after swapping the import.  NNSpire-extension features that go
beyond the standard PyTorch surface remain accessible via ``import NNSpire``
directly.

Covered aliases
---------------
* ``torch.Tensor`` → ``NNSpire.Tensor``
* ``torch.zeros / ones / full`` → NNSpire factories
* ``torch.float32 / float16 / int8 / int32 / bool`` → ``NNSpire.DType.*``
* ``torch.nn`` → ``NNSpire.nn``  (Linear, Conv2d, ReLU, Sequential, …)
* ``torch.optim`` → ``NNSpire.optim``  (SGD, Adam, AdamW, RMSprop)

See docs/blueprints.md §9 for the full compatibility specification.
"""

from __future__ import annotations

import NNSpire as _ns  # always resolves to the already-loaded package

# ------------------------------------------------------------------
# Top-level tensor factories — match torch.zeros / ones / full
# ------------------------------------------------------------------
Tensor = _ns.Tensor
zeros  = _ns.zeros
ones   = _ns.ones
full   = _ns.full

# ------------------------------------------------------------------
# dtype constants — match torch.float32, torch.int32, etc.
# ------------------------------------------------------------------
float32 = _ns.DType.float32
float16 = _ns.DType.float16
int8    = _ns.DType.int8
int32   = _ns.DType.int32
# torch.bool — stored as 'bool' attribute; does NOT shadow Python builtin
# because this is a module-level name, not a local.
bool    = _ns.DType.bool_   # noqa: A001  (shadowing builtin intentional — matches torch)

# ------------------------------------------------------------------
# Device aliases — match torch.device / torch.device("cpu")
# ------------------------------------------------------------------
cpu  = _ns.Device.cpu
cuda = _ns.Device.cuda

# ------------------------------------------------------------------
# Submodules — NNSpire.nn / optim already carry torch-compatible names.
# Expose them here so ``torch.nn.Linear``, ``torch.optim.Adam`` etc. work.
# ------------------------------------------------------------------
nn    = _ns.nn
optim = _ns.optim

__version__ = _ns.__version__

__all__ = [
    "Tensor", "zeros", "ones", "full",
    "float32", "float16", "int8", "int32", "bool",
    "cpu", "cuda",
    "nn", "optim",
    "__version__",
]
