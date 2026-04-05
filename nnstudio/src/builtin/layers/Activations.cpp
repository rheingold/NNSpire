/* ============================================================================
 * Activations.cpp — element-wise activation implementations
 * LGPL v3
 * ============================================================================ */

#include <builtin/layers/Activations.h>
#include <core/BackendRegistry.h>

#include <cmath>
#include <cassert>
#include <limits>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace layers {

// Shorthand for active backend
static inline IBackend& B() { return backend(); }

// ─── ReLU ────────────────────────────────────────────────────────────────────

Result<Tensor> ReLU::forward(const Tensor& x) {
    lastInput_ = x;
    return Result<Tensor>(B().clamp(x, 0.0f, std::numeric_limits<float>::infinity()));
}

Result<Tensor> ReLU::backward(const Tensor& gradOut) {
    // d/dx ReLU(x) = 1 if x>0, else 0
    assert(lastInput_.numel() > 0 && "ReLU::backward called before forward");
    Tensor mask(lastInput_.shape());
    for (int64_t i = 0; i < lastInput_.numel(); ++i)
        mask.flat(i) = lastInput_.flat(i) > 0.0f ? 1.0f : 0.0f;
    return Result<Tensor>(B().mul(gradOut, mask));
}

// ─── LeakyReLU ───────────────────────────────────────────────────────────────

Result<Tensor> LeakyReLU::forward(const Tensor& x) {
    lastInput_ = x;
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i)
        out.flat(i) = x.flat(i) > 0.0f ? x.flat(i) : alpha_ * x.flat(i);
    return Result<Tensor>(out);
}

Result<Tensor> LeakyReLU::backward(const Tensor& gradOut) {
    assert(lastInput_.numel() > 0);
    Tensor mask(lastInput_.shape());
    for (int64_t i = 0; i < lastInput_.numel(); ++i)
        mask.flat(i) = lastInput_.flat(i) > 0.0f ? 1.0f : alpha_;
    return Result<Tensor>(B().mul(gradOut, mask));
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

Result<Tensor> Sigmoid::forward(const Tensor& x) {
    // σ(x) = 1/(1+exp(-x))
    Tensor neg_x = B().neg(x);
    Tensor ex    = B().exp(neg_x);
    Tensor denom = B().addScalar(ex, 1.0f);   // 1 + exp(-x)
    Tensor ones  = Tensor::ones(x.shape());
    lastOutput_  = B().div_(ones, denom);
    return Result<Tensor>(lastOutput_);
}

Result<Tensor> Sigmoid::backward(const Tensor& gradOut) {
    // dσ/dx = σ(1-σ)
    assert(lastOutput_.numel() > 0);
    Tensor one_minus = B().subScalar(B().mulScalar(lastOutput_, -1.0f), -1.0f); // 1 - σ
    Tensor deriv     = B().mul(lastOutput_, one_minus);
    return Result<Tensor>(B().mul(gradOut, deriv));
}

// ─── TanhAct ─────────────────────────────────────────────────────────────────

Result<Tensor> TanhAct::forward(const Tensor& x) {
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i)
        out.flat(i) = std::tanh(x.flat(i));
    lastOutput_ = out;
    return Result<Tensor>(out);
}

Result<Tensor> TanhAct::backward(const Tensor& gradOut) {
    assert(lastOutput_.numel() > 0);
    // d/dx tanh(x) = 1 - tanh²(x)
    Tensor sq = B().mul(lastOutput_, lastOutput_);    // tanh²
    Tensor d  = B().subScalar(B().mulScalar(sq, -1.0f), -1.0f); // 1 - tanh²
    return Result<Tensor>(B().mul(gradOut, d));
}

// ─── Softmax ─────────────────────────────────────────────────────────────────
// Numerically stable: subtract max per sample before exp

Result<Tensor> Softmax::forward(const Tensor& x) {
    if (x.ndim() == 1) {
        // Single sample
        float maxVal = x.flat(0);
        for (int64_t i = 1; i < x.numel(); ++i)
            if (x.flat(i) > maxVal) maxVal = x.flat(i);
        Tensor shifted = B().subScalar(x, maxVal);
        Tensor ex      = B().exp(shifted);
        float  s       = 0.0f;
        for (int64_t i = 0; i < ex.numel(); ++i) s += ex.flat(i);
        Tensor out = B().divScalar(ex, s);
        lastOutput_ = out;
        return Result<Tensor>(out);
    }

    // Batched [N, C]
    const int64_t N = x.shape()[0];
    const int64_t C = x.shape()[1];
    Tensor out(x.shape());
    for (int64_t n = 0; n < N; ++n) {
        float maxVal = x.flat(n * C);
        for (int64_t c = 1; c < C; ++c)
            if (x.flat(n*C+c) > maxVal) maxVal = x.flat(n*C+c);
        float s = 0.0f;
        for (int64_t c = 0; c < C; ++c) {
            out.flat(n*C+c) = std::exp(x.flat(n*C+c) - maxVal);
            s += out.flat(n*C+c);
        }
        for (int64_t c = 0; c < C; ++c)
            out.flat(n*C+c) /= s;
    }
    lastOutput_ = out;
    return Result<Tensor>(out);
}

Result<Tensor> Softmax::backward(const Tensor& gradOut) {
    // Full Jacobian: dL/dx_i = s_i*(dL/ds_i - sum_j(dL/ds_j * s_j))
    assert(lastOutput_.numel() > 0);
    if (lastOutput_.ndim() == 1) {
        const int64_t C = lastOutput_.numel();
        float dot = 0.0f;
        for (int64_t c = 0; c < C; ++c)
            dot += gradOut.flat(c) * lastOutput_.flat(c);
        Tensor grad(lastOutput_.shape());
        for (int64_t c = 0; c < C; ++c)
            grad.flat(c) = lastOutput_.flat(c) * (gradOut.flat(c) - dot);
        return Result<Tensor>(grad);
    }
    const int64_t N = lastOutput_.shape()[0];
    const int64_t C = lastOutput_.shape()[1];
    Tensor grad(lastOutput_.shape());
    for (int64_t n = 0; n < N; ++n) {
        float dot = 0.0f;
        for (int64_t c = 0; c < C; ++c)
            dot += gradOut.flat(n*C+c) * lastOutput_.flat(n*C+c);
        for (int64_t c = 0; c < C; ++c)
            grad.flat(n*C+c) = lastOutput_.flat(n*C+c) * (gradOut.flat(n*C+c) - dot);
    }
    return Result<Tensor>(grad);
}

// ─── GELU ─────────────────────────────────────────────────────────────────────
// GELU(x) = 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³)))

static constexpr float kSqrt2OverPi = 0.7978845608f; // √(2/π)
static constexpr float kGELUCoeff   = 0.044715f;

Result<Tensor> GELU::forward(const Tensor& x) {
    lastInput_ = x;
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i) {
        float xi = x.flat(i);
        float inner = kSqrt2OverPi * (xi + kGELUCoeff * xi * xi * xi);
        out.flat(i) = 0.5f * xi * (1.0f + std::tanh(inner));
    }
    return Result<Tensor>(out);
}

Result<Tensor> GELU::backward(const Tensor& gradOut) {
    assert(lastInput_.numel() > 0);
    Tensor grad(lastInput_.shape());
    for (int64_t i = 0; i < lastInput_.numel(); ++i) {
        float xi     = lastInput_.flat(i);
        float inner  = kSqrt2OverPi * (xi + kGELUCoeff * xi * xi * xi);
        float t      = std::tanh(inner);
        float sech2  = 1.0f - t * t;
        float dinnerDx = kSqrt2OverPi * (1.0f + 3.0f * kGELUCoeff * xi * xi);
        float dGELUdx = 0.5f * (1.0f + t) + 0.5f * xi * sech2 * dinnerDx;
        grad.flat(i) = gradOut.flat(i) * dGELUdx;
    }
    return Result<Tensor>(grad);
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
