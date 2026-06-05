/* ============================================================================
 * ILoss.h — abstract loss function interface (Tier 1 public API)
 * LGPL v3
 *
 * ILoss is the stable, versioned public contract for loss computation.
 * It lives in NNSpire:: (Tier 1) so plugin authors can implement custom
 * loss functions without depending on any builtin implementation.
 *
 * Concrete built-in implementations (MSE, BCE, CrossEntropy, HuberLoss)
 * live in NNSpire::builtin::losses:: — see builtin/losses/Losses.h.
 *
 * WHY TWO SEPARATE METHODS?
 * ─────────────────────────
 * compute() and gradient() are split deliberately:
 *   compute()  → scalar monitoring value; call alone for validation loss
 *   gradient() → dL/dPred tensor; only needed during training backward
 *
 * Both receive the same pred and target — no internal state is shared
 * between calls. Loss objects are therefore fully stateless and reentrant.
 *
 * @see docs/blueprints.md — Chapter 5
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#loss-functions
 * ============================================================================ */

#pragma once

#include <core/Tensor.h>
#include <core/Result.h>
#include <string_view>

namespace NNSpire::core {

// ─── Abstract loss interface ──────────────────────────────────────────────────
class ILoss {
public:
    virtual ~ILoss() = default;
    virtual std::string_view name() const noexcept = 0;

    /** Forward: returns mean scalar loss over the batch. */
    virtual Result<Tensor> compute(const Tensor& pred, const Tensor& target) = 0;

    /** Gradient: dL/d(pred) — same shape as pred. */
    virtual Result<Tensor> gradient(const Tensor& pred, const Tensor& target) = 0;
};

} // namespace NNSpire::core
