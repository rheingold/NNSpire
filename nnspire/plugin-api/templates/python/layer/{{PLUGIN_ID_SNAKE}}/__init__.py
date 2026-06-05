"""
{{PLUGIN_NAME}} — NNSpire Layer Plugin

Public entry point for the NNSpire plugin system.
"""

from .layer import {{PLUGIN_CLASS_NAME}}

__all__ = ["{{PLUGIN_CLASS_NAME}}"]

# ── Plugin registration ───────────────────────────────────────────────────────
# When this package is imported inside the NNSpire Python bridge, the
# pybind11 PluginRegistry wrapper automatically discovers and registers
# classes decorated with @NNSpire.layer_plugin or implementing the
# LayerPlugin protocol.  No explicit registration call is required here.
