/* ============================================================================
 * Optimizers.cpp — SGD, Adam, AdamW implementations
 * LGPL v3
 * ============================================================================ */

#include <builtin/optimizers/Optimizers.h>
#include <cmath>
#include <cassert>
#include <istream>
#include <ostream>

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
// ─── RMSProp ─────────────────────────────────────────────────────────────────
void RMSProp::step(std::vector<Parameter*>& params) {
    ++step_;
    for (auto* p : params) {
        if (p->frozen || !p->tensor.hasGrad()) continue;
        const int64_t n   = p->tensor.numel();
        auto          key = p->tensor.data();

        if (sq_.find(key) == sq_.end())
            sq_[key].assign(static_cast<size_t>(n), 0.0f);

        auto& sq = sq_[key];
        for (int64_t i = 0; i < n; ++i) {
            float g = p->tensor.grad().flat(i);
            if (weightDecay_ != 0.0f) g += weightDecay_ * p->tensor.flat(i);

            const size_t si = static_cast<size_t>(i);
            sq[si] = alpha_ * sq[si] + (1.0f - alpha_) * g * g;
            p->tensor.flat(i) -= lr_ * g / (std::sqrt(sq[si]) + eps_);
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

// ─── Adam::saveState / loadState ─────────────────────────────────────────────
// Binary layout (per parameter):
//   has_state : uint8   (1 = this param has accumulated moments)
//   if has_state:
//     count   : uint64  (== numel of the parameter)
//     m       : float32[count]
//     v       : float32[count]
//
// The step counter is written once at the start (uint64).

namespace nnstudio { namespace builtin { namespace optimizers {

void Adam::saveState(std::ostream& out,
                     const std::vector<Parameter*>& params) const {
    // Write step counter
    out.write(reinterpret_cast<const char*>(&step_), sizeof(step_));

    for (const auto* p : params) {
        const float* key = p->tensor.data();
        auto itM = m_.find(key);
        auto itV = v_.find(key);
        if (itM == m_.end() || itV == v_.end()) {
            uint8_t has = 0;
            out.write(reinterpret_cast<const char*>(&has), sizeof(has));
        } else {
            uint8_t has = 1;
            out.write(reinterpret_cast<const char*>(&has), sizeof(has));
            uint64_t count = static_cast<uint64_t>(itM->second.size());
            out.write(reinterpret_cast<const char*>(&count), sizeof(count));
            out.write(reinterpret_cast<const char*>(itM->second.data()),
                      static_cast<std::streamsize>(count * sizeof(float)));
            out.write(reinterpret_cast<const char*>(itV->second.data()),
                      static_cast<std::streamsize>(count * sizeof(float)));
        }
    }
}

void Adam::loadState(std::istream& in,
                     const std::vector<Parameter*>& params) {
    in.read(reinterpret_cast<char*>(&step_), sizeof(step_));

    for (auto* p : params) {
        uint8_t has = 0;
        in.read(reinterpret_cast<char*>(&has), sizeof(has));
        if (!has) continue;

        uint64_t count = 0;
        in.read(reinterpret_cast<char*>(&count), sizeof(count));

        float* key = p->tensor.data();
        auto& mBuf = m_[key];
        auto& vBuf = v_[key];
        mBuf.resize(static_cast<size_t>(count));
        vBuf.resize(static_cast<size_t>(count));

        in.read(reinterpret_cast<char*>(mBuf.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
        in.read(reinterpret_cast<char*>(vBuf.data()),
                static_cast<std::streamsize>(count * sizeof(float)));
    }
}

} } } // namespace nnstudio::builtin::optimizers
