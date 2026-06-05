/* ============================================================================
 * IOptimizer.h — abstract optimizer interface (Tier 1 public API)
 * LGPL v3
 *
 * IOptimizer is the stable, versioned public contract for parameter update
 * strategies. It lives in NNSpire:: (Tier 1) so plugin authors can provide
 * custom optimizers without depending on any builtin implementation.
 *
 * Concrete built-in implementations (SGD, Adam, AdamW, StepDecayScheduler)
 * live in NNSpire::builtin::optimizers:: — see builtin/optimizers/Optimizers.h.
 *
 * WHAT AN OPTIMIZER DOES
 * ──────────────────────
 * step() receives a flat Parameter* list from ILayer::parameters().
 * It reads each param's grad_ (filled by backward()) and updates
 * param's data_ in-place according to the optimizer's update rule.
 * zeroGrad() resets all grad_ to zero. Trainer calls this BEFORE forward(),
 * not before backward() — grad_ accumulates with += so stale values from
 * the previous step must be cleared at the start of a new step.
 *
 * @see docs/blueprints.md — Chapter 6
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#optimizers
 * ============================================================================ */

#pragma once

#include <core/Layer.h>
#include <cstdint>
#include <iosfwd>
#include <vector>
#include <string_view>

namespace NNSpire::core {

// ─── Abstract optimizer interface ─────────────────────────────────────────────
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    virtual std::string_view name() const noexcept = 0;

    /** Perform one parameter update step using accumulated gradients. */
    virtual void step(std::vector<Parameter*>& params) = 0;

    /** Zero all gradients in the parameter list. */
    void zeroGrad(std::vector<Parameter*>& params) {
        for (auto* p : params) {
            if (p->frozen) continue;
            p->tensor.zeroGrad();  // no-op if grad not yet allocated
        }
    }

    float lr() const noexcept { return lr_; }
    void  setLR(float lr) noexcept { lr_ = lr; }

    uint64_t stepCount() const noexcept { return step_; }
    void     setStepCount(uint64_t t) noexcept { step_ = t; }

    // ── Checkpoint serialization ─────────────────────────────────────────────
    /**
     * Serialize optimizer state (moment vectors, step counter, etc.) to a
     * binary stream.  The `params` list must be the same order as the model's
     * parameters() — state is stored by parameter index, not by pointer.
     *
     * Default no-op: optimizers without persistent state (e.g. vanilla SGD)
     * need not override this.
     */
    virtual void saveState(std::ostream& /*out*/,
                           const std::vector<Parameter*>& /*params*/) const {}

    /**
     * Restore optimizer state from a binary stream produced by saveState().
     * `params` must be the same list (same order, same count) as at save time.
     *
     * Default no-op.
     */
    virtual void loadState(std::istream& /*in*/,
                           const std::vector<Parameter*>& /*params*/) {}

protected:
    float    lr_;
    uint64_t step_{0};

    explicit IOptimizer(float lr) : lr_(lr) {}
};

// ─── LR Scheduler base ───────────────────────────────────────────────────────
// Kept here (Tier 1) because schedulers wrap IOptimizer — any optimizer
// implementation should be schedulable.
class ILRScheduler {
public:
    virtual ~ILRScheduler() = default;
    /** Called once per epoch or step; updates optimizer's lr. */
    virtual void onStep(IOptimizer& opt, uint64_t globalStep) = 0;
};

} // namespace NNSpire::core
