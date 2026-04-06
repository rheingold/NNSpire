/* ============================================================================
 * IActivation.h — stateless activation function interface  [nnstudio::core]
 * LGPL v3
 *
 * ADR-020 — motivation
 * ────────────────────
 * The original ActivationBase (builtin/activations/Activations.h) stores
 * lastInput_/lastOutput_ as member variables, making each instance stateful
 * and non-reentrant. That is fine for single-threaded sequential training but
 * will break under:
 *   • Multi-threaded batch evaluation (thread A's forward clobbers thread B's
 *     saved context before thread B calls backward).
 *   • ComputeGraph replay (graph re-runs forward without calling backward —
 *     saved context is stale).
 *   • Quantised/fused kernels (context may not be a plain Tensor at all).
 *
 * THE CONTRACT
 * ────────────
 * IActivation is a pure stateless functor:
 *   forward()  — takes input x, returns ActivationForward{ output, ctx }.
 *                ctx is whatever the backward pass needs, returned to the
 *                caller for storage outside the activation object.
 *   backward() — takes gradOut + the ActivationForward from the matching
 *                forward() call, returns dL/dx.
 *
 * No mutable state. One instance can be shared across threads and graph nodes.
 *
 * MIGRATION PATH
 * ──────────────
 * Phase 1 (now):
 *   IActivation interface + FunctionLayer adapter created.
 *   Existing ActivationBase subclasses left as-is (still ILayer, still tested).
 *   New code should prefer FunctionLayer<ConcreteActivation>.
 *
 * Phase 2 (before ABI freeze):
 *   Migrate all concrete activations to IActivation.
 *   Retire ActivationBase::lastInput_/lastOutput_.
 *   ActivationBase becomes a thin FunctionLayer<IActivation*> alias.
 *
 * CONTEXT STRATEGY
 * ────────────────
 * Different activations need different saved data:
 *   ReLU, LeakyReLU, GELU  → save input  (need x sign / value)
 *   Sigmoid, Tanh           → save output (derivative cheaper via output)
 *   Softmax                 → save output (full jacobian via dot-product form)
 *
 * ActivationForward::ctxIsInput signals which tensor is in ctx.
 *
 * @see docs/blueprints.md §3.8
 * @see nnstudio/include/builtin/activations/FnLayer.h
 * ============================================================================
*/
#pragma once

#include <core/Tensor.h>
#include <core/Result.h>
#include <string_view>

namespace nnstudio::core {

// ─── Context returned by forward() ────────────────────────────────────────────
struct ActivationForward {
    Tensor output;      ///< the activation's output (always present)
    Tensor ctx;         ///< saved tensor for backward; interpretation depends on ctxIsInput
    bool   ctxIsInput;  ///< true → ctx is the input x; false → ctx is the output y
};

// ─── IActivation — pure stateless activation interface ────────────────────────
class IActivation {
public:
    virtual ~IActivation() = default;

    /// Compute activation and save context needed for backward.
    /// Must be const — no mutable state allowed.
    virtual ActivationForward forward(const Tensor& x) const = 0;

    /// Compute input gradient from upstream gradient and saved context.
    /// gradOut is dL/d(output); returns dL/d(input).
    virtual Result<Tensor> backward(const Tensor& gradOut,
                                    const ActivationForward& fwd) const = 0;

    /// Short identifier, e.g. "ReLU", "Sigmoid".
    virtual std::string_view name() const noexcept = 0;
};

} // namespace nnstudio::core
