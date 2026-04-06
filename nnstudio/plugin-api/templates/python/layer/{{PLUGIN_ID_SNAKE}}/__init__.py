"""
{{PLUGIN_NAME}} — NNStudio Layer Plugin

Public entry point for the nnstudio plugin system.
"""

from .layer import {{PLUGIN_CLASS_NAME}}

__all__ = ["{{PLUGIN_CLASS_NAME}}"]

# ── Plugin registration ───────────────────────────────────────────────────────
# When this package is imported inside the NNStudio Python bridge, the
# pybind11 PluginRegistry wrapper automatically discovers and registers
# classes decorated with @nnstudio.layer_plugin or implementing the
# LayerPlugin protocol.  No explicit registration call is required here.
