/* ============================================================================
 * Optimizers.h — SGD, Adam, AdamW + StepDecayScheduler  [nnstudio::builtin::optimizers]
 * LGPL v3
 *
 * All concrete optimizer classes inherit from nnstudio::IOptimizer (Tier 1).
 * The interface (incl. zeroGrad and ILRScheduler) is in <nnstudio/IOptimizer.h>.
 *
 * VELOCITY BUFFERS (Adam / AdamW)
 * ───────────────────────────────
 * Adam maintains per-parameter moment vectors (m, v) keyed on raw Parameter*
 * address. That address is stable for the model's lifetime — layers own their
 * Parameters and never move them. Do NOT pass a different Parameter list
 * between steps; the map entries would not match and Adam restarts silently.
 *
 * FROZEN PARAMETERS
 * ─────────────────
 * If Parameter::frozen == true, step() skips that parameter entirely.
 * This is the transfer-learning / LoRA pattern.
 *
 * ADAM BIAS CORRECTION
 * ─────────────────────
 * mHat = m / (1 - β1^t),   vHat = v / (1 - β2^t)
 * After ~10 steps both denominators approach 1 and correction vanishes.
 *
 * ADAMW DISTINCTION
 * ─────────────────
 * AdamW applies weight decay directly (θ *= (1 - λ)) before the Adam update,
 * decoupling regularisation from the adaptive learning rate.
 *
 * @see docs/blueprints.md — Chapter 6
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#optimizers
 * ============================================================================ */

#pragma once

#include <core/IOptimizer.h>
#include <cstdint>
#include <vector>
#include <unordered_map>
#include <string>
#include <memory>

namespace nnstudio {
namespace builtin {
namespace optimizers {
using namespace nnstudio::core;

// ─── SGD with optional momentum ──────────────────────────────────────────────
class SGD : public IOptimizer {
public:
    explicit SGD(float lr = 0.01f, float momentum = 0.0f,
                 float weightDecay = 0.0f)
        : IOptimizer(lr), momentum_(momentum), weightDecay_(weightDecay) {}

    std::string_view name() const noexcept override { return "SGD"; }
    void step(std::vector<Parameter*>& params) override;

private:
    float momentum_;
    float weightDecay_;
    std::unordered_map<const float*, std::vector<float>> velocities_;
};

// ─── Adam: Adaptive Moment Estimation ────────────────────────────────────────
// @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#adam
class Adam : public IOptimizer {
public:
    explicit Adam(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
                  float eps = 1e-8f, float weightDecay = 0.0f)
        : IOptimizer(lr), beta1_(beta1), beta2_(beta2),
          eps_(eps), weightDecay_(weightDecay) {}

    std::string_view name() const noexcept override { return "Adam"; }
    void step(std::vector<Parameter*>& params) override;

protected:
    float beta1_, beta2_, eps_, weightDecay_;
    std::unordered_map<const float*, std::vector<float>> m_, v_;  // 1st/2nd moments
};

// ─── AdamW: Adam with decoupled weight decay ─────────────────────────────────
class AdamW : public Adam {
public:
    explicit AdamW(float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f,
                   float eps = 1e-8f, float weightDecay = 0.01f)
        : Adam(lr, beta1, beta2, eps, 0.0f), decoupledDecay_(weightDecay) {}

    std::string_view name() const noexcept override { return "AdamW"; }
    void step(std::vector<Parameter*>& params) override;

private:
    float decoupledDecay_;
};

// ─── Step Decay LR Scheduler ─────────────────────────────────────────────────
class StepDecayScheduler : public ILRScheduler {
public:
    explicit StepDecayScheduler(uint64_t stepSize, float gamma = 0.1f)
        : stepSize_(stepSize), gamma_(gamma) {}
    void onStep(IOptimizer& opt, uint64_t globalStep) override;
private:
    uint64_t stepSize_;
    float    gamma_;
};

} // namespace optimizers
} // namespace builtin
} // namespace nnstudio
