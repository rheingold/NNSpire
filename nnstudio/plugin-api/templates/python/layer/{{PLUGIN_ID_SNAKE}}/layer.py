"""
{{PLUGIN_NAME}} — NNStudio Layer Plugin (Python)

Replace every {{PLACEHOLDER}} with your actual values before distribution.

@kb: PLUGIN-SDK.md §7
"""

from __future__ import annotations

from nnstudio import Tensor


class {{PLUGIN_CLASS_NAME}}:
    """
    {{PLUGIN_NAME}} — custom NNStudio layer plugin.

    Implements the LayerPlugin protocol:
      forward(x)         — compute layer output
      backward(grad_out) — backpropagate gradients
      parameters()       — return trainable parameter tensors

    plugin_id      = "{{PLUGIN_REVERSE_DOMAIN_ID}}"
    plugin_version = "{{PLUGIN_VERSION}}"
    plugin_type    = "LAYER"

    @kb: ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
    """

    # ── Plugin metadata (required) ───────────────────────────────────────────

    plugin_id      = "{{PLUGIN_REVERSE_DOMAIN_ID}}"
    plugin_version = "{{PLUGIN_VERSION}}"
    plugin_type    = "LAYER"

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self) -> None:
        """
        Initialise layer state.  Weights should be allocated here if needed.
        """
        # TODO: initialise trainable parameters
        # Example:
        #   import nnstudio
        #   self.weight = nnstudio.Parameter(shape=[input_size, output_size])
        #   self.bias   = nnstudio.Parameter(shape=[output_size])
        pass

    # ── Forward pass ─────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        Args:
            x: Input tensor, shape [batch, in_features].

        Returns:
            Output tensor, shape [batch, out_features].

        Save any intermediate values needed in backward().
        """
        raise NotImplementedError(
            f"{type(self).__name__}.forward() is not implemented"
        )

    # ── Backward pass ────────────────────────────────────────────────────────

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backpropagate gradients.

        Args:
            grad_output: dLoss/dOutput, same shape as forward() output.

        Returns:
            dLoss/dInput, same shape as forward() input.

        Accumulate parameter gradients internally via param.grad_ +=
        so the optimizer can read them in optimizer.step().
        """
        raise NotImplementedError(
            f"{type(self).__name__}.backward() is not implemented"
        )

    # ── Parameters ───────────────────────────────────────────────────────────

    def parameters(self) -> list[Tensor]:
        """
        Return all trainable parameter tensors.

        The optimizer calls this after backward() to update weights.
        Return an empty list if this layer has no learnable weights.
        """
        return []

    # ── Optional ─────────────────────────────────────────────────────────────

    def doc_ref(self) -> str:
        """Return a KB path#anchor documenting this layer type (optional)."""
        return "ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md"
