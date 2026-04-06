"""
nnstudio — NNStudio Python package.

PyTorch-compatible API surface:

    import nnstudio
    import nnstudio.nn as nn
    import nnstudio.nn.functional as F
    import nnstudio.optim as optim

    model = nn.Sequential()
    model.add(nn.Linear(4, 8))
    model.add(nn.ReLU())
    model.add(nn.Linear(8, 1))

    optim = optim.Adam(lr=1e-3)

For a true drop-in replacement of torch.nn in scripts:

    import nnstudio as torch            # ← swap this from 'import torch'
    import nnstudio.nn as nn
    import nnstudio.nn.functional as F
    import nnstudio.optim as optim

See docs/blueprints.md §9 for the full compat specification.
"""

# The compiled extension module is imported by CMake into this package.
# When installed this __init__.py sits alongside nnstudio.<ext>.so/.pyd.
#
# Windows / MinGW note
# --------------------
# The .pyd is built with MinGW GCC and therefore depends on the MinGW runtime
# DLLs (libgcc_s_seh-1.dll, libstdc++-6.dll, libwinpthread-1.dll).
# Python ≥ 3.8 ignores PATH when resolving DLL dependencies of extension
# modules, so the MinGW bin directory must be registered explicitly.
# We try a few well-known MSYS2 locations here; if none is found the user
# must call `os.add_dll_directory(...)` before importing nnstudio, or copy
# the MinGW DLLs alongside the .pyd.

import os as _os
import sys as _sys

if _sys.platform == "win32":
    _MINGW_CANDIDATES = [
        r"C:\msys64\mingw64\bin",
        r"C:\msys2\mingw64\bin",
        _os.path.join(_os.environ.get("MINGW_PREFIX", ""), "bin"),
    ]
    for _d in _MINGW_CANDIDATES:
        if _os.path.isdir(_d):
            try:
                _os.add_dll_directory(_d)
            except (AttributeError, OSError):
                pass  # Python < 3.8 or access denied
            break

from .nnstudio import *          # noqa: F401, F403  — re-export compiled bindings
from .nnstudio import __version__  # noqa: F401

from . import keras              # noqa: F401
from . import torch_compat       # noqa: F401
from . import plugin_manifest    # noqa: F401

__all__ = [
    "nn", "optim", "Tensor", "DType", "Device",
    "zeros", "ones", "full", "__version__",
    "keras", "torch_compat", "plugin_manifest",
]
