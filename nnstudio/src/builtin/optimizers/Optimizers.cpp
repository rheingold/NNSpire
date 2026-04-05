/* ============================================================================
 * Optimizers.cpp — SGD, Adam, AdamW implementations
 * LGPL v3
 * ============================================================================ */

#include <builtin/optimizers/Optimizers.h>
#include <cmath>
#include <cassert>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace optimizers {

// ─── SGD ─────────────────────────────────────────────────────────────────────
void SGD::step(std::vector<Parameter*>& params) {
    ++step_;
    for (auto* p : params) {
        if (p->frozen || !p->tensor.hasGrad()) continue;
        const int64_t n   = p->tensor.numel();
        auto          key = p->tensor.data();

        if (velocities_.find(key) == velocities_.end())
            velocities_[key].assign(static_cast<size_t>(n), 0.0f);

        auto& vel = velocities_[key];
        for (int64_t i = 0; i < n; ++i) {
            float g = p->tensor.grad().flat(i);
            if (weightDecay_ != 0.0f) g += weightDecay_ * p->tensor.flat(i);
            if (momentum_ != 0.0f) {
                vel[static_cast<size_t>(i)] =
                    momentum_ * vel[static_cast<size_t>(i)] + g;
                g = vel[static_cast<size_t>(i)];
            }
            p->tensor.flat(i) -= lr_ * g;
        }
    }
}

// ─── Adam ─────────────────────────────────────────────────────────────────────
void Adam::step(std::vector<Parameter*>& params) {
    ++step_;
    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));

    for (auto* p : params) {
        if (p->frozen || !p->tensor.hasGrad()) continue;
        const int64_t n   = p->tensor.numel();
        auto          key = p->tensor.data();

        if (m_.find(key) == m_.end()) m_[key].assign(static_cast<size_t>(n), 0.0f);
        if (v_.find(key) == v_.end()) v_[key].assign(static_cast<size_t>(n), 0.0f);

        auto& mBuf = m_[key];
        auto& vBuf = v_[key];

        for (int64_t i = 0; i < n; ++i) {
            float g = p->tensor.grad().flat(i);
            if (weightDecay_ != 0.0f) g += weightDecay_ * p->tensor.flat(i);

            const size_t si = static_cast<size_t>(i);
            mBuf[si] = beta1_ * mBuf[si] + (1.0f - beta1_) * g;
            vBuf[si] = beta2_ * vBuf[si] + (1.0f - beta2_) * g * g;

            const float mHat = mBuf[si] / bc1;
            const float vHat = vBuf[si] / bc2;
            p->tensor.flat(i) -= lr_ * mHat / (std::sqrt(vHat) + eps_);
        }
    }
}

// ─── AdamW ───────────────────────────────────────────────────────────────────
void AdamW::step(std::vector<Parameter*>& params) {
    ++step_;
    const float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(step_));
    const float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(step_));

    for (auto* p : params) {
        if (p->frozen || !p->tensor.hasGrad()) continue;
        const int64_t n   = p->tensor.numel();
        auto          key = p->tensor.data();

        if (m_.find(key) == m_.end()) m_[key].assign(static_cast<size_t>(n), 0.0f);
        if (v_.find(key) == v_.end()) v_[key].assign(static_cast<size_t>(n), 0.0f);

        auto& mBuf = m_[key];
        auto& vBuf = v_[key];

        for (int64_t i = 0; i < n; ++i) {
            const float g = p->tensor.grad().flat(i);
            // Decoupled weight decay BEFORE Adam update
            p->tensor.flat(i) *= (1.0f - lr_ * decoupledDecay_);

            const size_t si = static_cast<size_t>(i);
            mBuf[si] = beta1_ * mBuf[si] + (1.0f - beta1_) * g;
            vBuf[si] = beta2_ * vBuf[si] + (1.0f - beta2_) * g * g;

            const float mHat = mBuf[si] / bc1;
            const float vHat = vBuf[si] / bc2;
            p->tensor.flat(i) -= lr_ * mHat / (std::sqrt(vHat) + eps_);
        }
    }
}

// ─── StepDecayScheduler ───────────────────────────────────────────────────────
void StepDecayScheduler::onStep(IOptimizer& opt, uint64_t globalStep) {
    if (stepSize_ > 0 && globalStep > 0 && (globalStep % stepSize_) == 0)
        opt.setLR(opt.lr() * gamma_);
}

} // namespace optimizers
} // namespace builtin
} // namespace nnstudio
