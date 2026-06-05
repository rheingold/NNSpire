/* ============================================================================
 * {{PLUGIN_NAME}} — NNSpire Layer Plugin
 *
 * Replace every {{PLACEHOLDER}} with your actual values.
 *
 * @kb: PLUGIN-SDK.md §6
 * ============================================================================ */

#pragma once

#include <NNSpire_plugin.h>  // from NNSpire plugin-api

/**
 * {{PLUGIN_NAME}}
 *
 * An NNSpire layer plugin implementing forward() and backward().
 * One instance is created per model layer that uses this plugin type.
 *
 * @kb: ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#training-loop
 */
class {{PLUGIN_CLASS_NAME}} {
public:
    /* ── Construction / initialisation ─────────────────────────────────────── */

    {{PLUGIN_CLASS_NAME}}();
    ~{{PLUGIN_CLASS_NAME}}();

    /* ── ILayer forward pass ────────────────────────────────────────────────── */

    /**
     * Compute the layer output from `in` and write it to `out`.
     * `in`  has shape [batch, input_features].
     * `out` has shape [batch, output_features].
     *
     * Called during both training (forward pass of backprop) and inference.
     * Must save any values needed in backward() — e.g. the input activation.
     */
    void forward(const NNTensorView* in, NNTensorView* out);

    /* ── ILayer backward pass ───────────────────────────────────────────────── */

    /**
     * Backpropagate gradients.
     *
     * `grad_out` contains dLoss/dOutput (upstream gradient), same shape as `out`.
     * Write dLoss/dInput into `grad_in` (same shape as `in`).
     * Accumulate parameter gradients internally; the optimizer reads them later.
     */
    void backward(const NNTensorView* grad_out, NNTensorView* grad_in);

    /* ── Parameters (optional — remove if stateless) ───────────────────────── */

    /**
     * Return views of all trainable parameter tensors.
     * The engine passes these to the optimizer after backward().
     *
     * If your layer has no learnable weights, set vtable.parameters = nullptr.
     */
    void parameters(NNTensorView** params, int32_t* count);

    /* ── Documentation reference ────────────────────────────────────────────── */

    /** Return a KB path#anchor documenting this layer.  May return nullptr. */
    const char* docRef() const noexcept;

private:
    /* Internal state — e.g. saved activations, weight tensors */
    // TODO: add your fields here
};
