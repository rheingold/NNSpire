/* ============================================================================
 * Losses.cpp — loss function implementations
 * LGPL v3
 * ============================================================================ */

#include <builtin/losses/Losses.h>
#include <core/BackendRegistry.h>

#include <cmath>
#include <cassert>

using namespace NNSpire::core;

namespace NNSpire {
namespace builtin {
namespace losses {

static inline IBackend& B() { return backend(); }

// ─── MSE ─────────────────────────────────────────────────────────────────────

Result<Tensor> MSE::compute(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    Tensor diff  = B().sub(pred, target);
    Tensor sq    = B().mul(diff, diff);
    return Result<Tensor>(B().mean(sq, -1, false));  // global mean → scalar
}

Result<Tensor> MSE::gradient(const Tensor& pred, const Tensor& target) {
    // dL/d(pred) = 2*(pred-target) / N
    assert(pred.sameShape(target));
    const float invN = 2.0f / static_cast<float>(pred.numel());
    Tensor diff = B().sub(pred, target);
    return Result<Tensor>(B().mulScalar(diff, invN));
}

// ─── BCE ─────────────────────────────────────────────────────────────────────

Result<Tensor> BCE::compute(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    const int64_t n = pred.numel();
    float loss = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float p = std::max(kEps, std::min(1.0f - kEps, pred.flat(i)));
        float y = target.flat(i);
        loss += -(y * std::log(p) + (1.0f - y) * std::log(1.0f - p));
    }
    return Result<Tensor>(Tensor::full({1}, loss / static_cast<float>(n)));
}

Result<Tensor> BCE::gradient(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    const int64_t n = pred.numel();
    Tensor grad(pred.shape());
    for (int64_t i = 0; i < n; ++i) {
        float p = std::max(kEps, std::min(1.0f - kEps, pred.flat(i)));
        float y = target.flat(i);
        grad.flat(i) = ((-y / p) + ((1.0f - y) / (1.0f - p))) / static_cast<float>(n);
    }
    return Result<Tensor>(grad);
}

// ─── CrossEntropy ─────────────────────────────────────────────────────────────

Result<Tensor> CrossEntropy::compute(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    const int64_t n = pred.numel();
    float loss = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float p = std::max(kEps, pred.flat(i));
        loss -= target.flat(i) * std::log(p);
    }
    const float batch = static_cast<float>(pred.ndim() >= 1 ? pred.shape()[0] : 1);
    return Result<Tensor>(Tensor::full({1}, loss / batch));
}

Result<Tensor> CrossEntropy::gradient(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    const float invBatch = 1.0f / static_cast<float>(pred.ndim() >= 1 ? pred.shape()[0] : 1);
    Tensor grad(pred.shape());
    for (int64_t i = 0; i < pred.numel(); ++i) {
        float p = std::max(kEps, pred.flat(i));
        grad.flat(i) = (-target.flat(i) / p) * invBatch;
    }
    return Result<Tensor>(grad);
}

// ─── HuberLoss ────────────────────────────────────────────────────────────────

Result<Tensor> HuberLoss::compute(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    const int64_t n = pred.numel();
    float loss = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float r    = pred.flat(i) - target.flat(i);
        float absR = std::abs(r);
        if (absR <= delta_)
            loss += 0.5f * r * r;
        else
            loss += delta_ * (absR - 0.5f * delta_);
    }
    return Result<Tensor>(Tensor::full({1}, loss / static_cast<float>(n)));
}

Result<Tensor> HuberLoss::gradient(const Tensor& pred, const Tensor& target) {
    assert(pred.sameShape(target));
    const float invN = 1.0f / static_cast<float>(pred.numel());
    Tensor grad(pred.shape());
    for (int64_t i = 0; i < pred.numel(); ++i) {
        float r    = pred.flat(i) - target.flat(i);
        float absR = std::abs(r);
        if (absR <= delta_)
            grad.flat(i) = r * invN;
        else
            grad.flat(i) = delta_ * (r > 0.0f ? 1.0f : -1.0f) * invN;
    }
    return Result<Tensor>(grad);
}

} // namespace losses
} // namespace builtin
} // namespace NNSpire
