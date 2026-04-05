/* ============================================================================
 * Losses.h — built-in loss functions  [nnstudio::builtin::losses]
 * LGPL v3
 *
 * All concrete loss classes inherit from nnstudio::ILoss (Tier 1 interface).
 * The interface is defined in <nnstudio/ILoss.h>.
 *
 * WHY TWO SEPARATE METHODS?
 * ─────────────────────────
 * ILoss defines compute() and gradient() as separate calls:
 *   compute(pred, target)  → scalar mean loss  (display/monitoring)
 *   gradient(pred, target) → dL/dPred tensor   (seed gradient for backprop)
 *
 * Both are stateless — they receive the same pred and target. No internal
 * state is shared between calls. All loss objects are fully reentrant.
 *
 * THE kEps TRAP
 * ─────────────
 * BCE and CrossEntropy clamp pred to [kEps, 1-kEps] before log().
 * log(0) = -∞ → NaN loss → silent divergence.
 * If loss is NaN on step 1: check your output activation produces values
 * strictly in (0,1). A missing Sigmoid before BCE is the #1 cause.
 *
 * LOSSES PROVIDED
 * ───────────────
 *   MSE (Mean Squared Error)      — regression; delegates to IBackend
 *   BCE (Binary Cross-Entropy)    — binary classification; CPU loop
 *   CrossEntropy (Categorical CE) — multi-class; expects softmax pred
 *   HuberLoss                     — robust regression; less penalty for outliers
 *
 * Known inconsistency: MSE delegates to the active IBackend (will benefit from
 * CUDA); BCE uses a direct CPU loop. Tracked for remediation before CUDA lands.
 *
 * @see docs/blueprints.md — Chapter 5
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#loss-functions
 * ============================================================================ */

#pragma once

#include <core/ILoss.h>
#include <core/Tensor.h>
#include <core/Result.h>
#include <string_view>

namespace nnstudio {
namespace builtin {
namespace losses {
using namespace nnstudio::core;

// ─── MSE Loss: L = mean( (pred - target)^2 ) ─────────────────────────────────
class MSE : public ILoss {
public:
    std::string_view name() const noexcept override { return "MSE"; }
    Result<Tensor> compute (const Tensor& pred, const Tensor& target) override;
    Result<Tensor> gradient(const Tensor& pred, const Tensor& target) override;
};

// ─── Binary Cross-Entropy: L = -mean(y*log(p) + (1-y)*log(1-p)) ──────────────
// pred should be sigmoid output in (0,1).
class BCE : public ILoss {
public:
    std::string_view name() const noexcept override { return "BCE"; }
    Result<Tensor> compute (const Tensor& pred, const Tensor& target) override;
    Result<Tensor> gradient(const Tensor& pred, const Tensor& target) override;
private:
    static constexpr float kEps = 1e-7f;
};

// ─── Categorical Cross-Entropy: L = -mean(sum(target * log(pred))) ────────────
// pred should be softmax output.
class CrossEntropy : public ILoss {
public:
    std::string_view name() const noexcept override { return "CrossEntropy"; }
    Result<Tensor> compute (const Tensor& pred, const Tensor& target) override;
    Result<Tensor> gradient(const Tensor& pred, const Tensor& target) override;
private:
    static constexpr float kEps = 1e-7f;
};

// ─── Huber Loss (smooth L1): ──────────────────────────────────────────────────
//   L = 0.5*(pred-target)^2           if |pred-target| <= delta
//       delta*(|pred-target|-0.5*delta) otherwise
class HuberLoss : public ILoss {
public:
    explicit HuberLoss(float delta = 1.0f) : delta_(delta) {}
    std::string_view name() const noexcept override { return "HuberLoss"; }
    Result<Tensor> compute (const Tensor& pred, const Tensor& target) override;
    Result<Tensor> gradient(const Tensor& pred, const Tensor& target) override;
private:
    float delta_;
};

} // namespace losses
} // namespace builtin
} // namespace nnstudio
