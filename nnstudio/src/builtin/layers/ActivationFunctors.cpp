/* ============================================================================
 * ActivationFunctors.cpp — IActivation implementations
 * LGPL v3
 * ============================================================================ */

#include <builtin/layers/ActivationFunctors.h>
#include <core/BackendRegistry.h>

#include <cmath>
#include <cassert>
#include <limits>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace layers {

static inline IBackend& B() { return backend(); }

static constexpr float kSqrt2OverPi = 0.7978845608f;
static constexpr float kGELUCoeff   = 0.044715f;

// ─── ReLUFn ──────────────────────────────────────────────────────────────────

ActivationForward ReLUFn::forward(const Tensor& x) const {
    Tensor out = B().clamp(x, 0.0f, std::numeric_limits<float>::infinity());
    return { out, x.clone(), /*ctxIsInput=*/true };
}

Result<Tensor> ReLUFn::backward(const Tensor& gradOut,
                                 const ActivationForward& fwd) const {
    // fwd.ctx = saved input x
    Tensor mask(fwd.ctx.shape());
    for (int64_t i = 0; i < fwd.ctx.numel(); ++i)
        mask.flat(i) = fwd.ctx.flat(i) > 0.0f ? 1.0f : 0.0f;
    return Result<Tensor>(B().mul(gradOut, mask));
}

// ─── LeakyReLUFn ─────────────────────────────────────────────────────────────

ActivationForward LeakyReLUFn::forward(const Tensor& x) const {
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i)
        out.flat(i) = x.flat(i) > 0.0f ? x.flat(i) : alpha_ * x.flat(i);
    return { out, x.clone(), /*ctxIsInput=*/true };
}

Result<Tensor> LeakyReLUFn::backward(const Tensor& gradOut,
                                      const ActivationForward& fwd) const {
    // fwd.ctx = saved input x
    Tensor mask(fwd.ctx.shape());
    for (int64_t i = 0; i < fwd.ctx.numel(); ++i)
        mask.flat(i) = fwd.ctx.flat(i) > 0.0f ? 1.0f : alpha_;
    return Result<Tensor>(B().mul(gradOut, mask));
}

// ─── SigmoidFn ───────────────────────────────────────────────────────────────

ActivationForward SigmoidFn::forward(const Tensor& x) const {
    // σ(x) = 1/(1+exp(-x))
    Tensor out = B().div_(Tensor::ones(x.shape()),
                          B().addScalar(B().exp(B().neg(x)), 1.0f));
    return { out, out.clone(), /*ctxIsInput=*/false };   // save output
}

Result<Tensor> SigmoidFn::backward(const Tensor& gradOut,
                                    const ActivationForward& fwd) const {
    // dσ/dx = σ(1 - σ)     fwd.ctx = saved output σ(x)
    Tensor one_minus = B().subScalar(B().mulScalar(fwd.ctx, -1.0f), -1.0f);
    return Result<Tensor>(B().mul(gradOut, B().mul(fwd.ctx, one_minus)));
}

// ─── TanhActFn ───────────────────────────────────────────────────────────────

ActivationForward TanhActFn::forward(const Tensor& x) const {
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i)
        out.flat(i) = std::tanh(x.flat(i));
    return { out, out.clone(), /*ctxIsInput=*/false };   // save output
}

Result<Tensor> TanhActFn::backward(const Tensor& gradOut,
                                    const ActivationForward& fwd) const {
    // d/dx tanh(x) = 1 - tanh²(x)     fwd.ctx = saved output tanh(x)
    Tensor sq = B().mul(fwd.ctx, fwd.ctx);                        // tanh²
    Tensor d  = B().subScalar(B().mulScalar(sq, -1.0f), -1.0f);  // 1 - tanh²
    return Result<Tensor>(B().mul(gradOut, d));
}

// ─── SoftmaxFn ───────────────────────────────────────────────────────────────

ActivationForward SoftmaxFn::forward(const Tensor& x) const {
    Tensor out(x.shape());
    if (x.ndim() == 1) {
        float maxVal = x.flat(0);
        for (int64_t i = 1; i < x.numel(); ++i)
            if (x.flat(i) > maxVal) maxVal = x.flat(i);
        float s = 0.0f;
        for (int64_t i = 0; i < x.numel(); ++i) {
            out.flat(i) = std::exp(x.flat(i) - maxVal);
            s += out.flat(i);
        }
        for (int64_t i = 0; i < x.numel(); ++i) out.flat(i) /= s;
    } else {
        const int64_t N = x.shape()[0], C = x.shape()[1];
        for (int64_t n = 0; n < N; ++n) {
            float maxVal = x.flat(n * C);
            for (int64_t c = 1; c < C; ++c)
                if (x.flat(n*C+c) > maxVal) maxVal = x.flat(n*C+c);
            float s = 0.0f;
            for (int64_t c = 0; c < C; ++c) {
                out.flat(n*C+c) = std::exp(x.flat(n*C+c) - maxVal);
                s += out.flat(n*C+c);
            }
            for (int64_t c = 0; c < C; ++c) out.flat(n*C+c) /= s;
        }
    }
    return { out, out.clone(), /*ctxIsInput=*/false };   // save output
}

Result<Tensor> SoftmaxFn::backward(const Tensor& gradOut,
                                    const ActivationForward& fwd) const {
    // Full Jacobian: dL/dx_i = s_i·(dL/ds_i − Σ_j dL/ds_j·s_j)
    // fwd.ctx = saved softmax output s
    const Tensor& s = fwd.ctx;
    Tensor grad(s.shape());
    if (s.ndim() == 1) {
        float dot = 0.0f;
        for (int64_t c = 0; c < s.numel(); ++c)
            dot += gradOut.flat(c) * s.flat(c);
        for (int64_t c = 0; c < s.numel(); ++c)
            grad.flat(c) = s.flat(c) * (gradOut.flat(c) - dot);
    } else {
        const int64_t N = s.shape()[0], C = s.shape()[1];
        for (int64_t n = 0; n < N; ++n) {
            float dot = 0.0f;
            for (int64_t c = 0; c < C; ++c)
                dot += gradOut.flat(n*C+c) * s.flat(n*C+c);
            for (int64_t c = 0; c < C; ++c)
                grad.flat(n*C+c) = s.flat(n*C+c) * (gradOut.flat(n*C+c) - dot);
        }
    }
    return Result<Tensor>(grad);
}

// ─── GELUFn ──────────────────────────────────────────────────────────────────

ActivationForward GELUFn::forward(const Tensor& x) const {
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i) {
        float xi    = x.flat(i);
        float inner = kSqrt2OverPi * (xi + kGELUCoeff * xi * xi * xi);
        out.flat(i) = 0.5f * xi * (1.0f + std::tanh(inner));
    }
    return { out, x.clone(), /*ctxIsInput=*/true };   // save input
}

Result<Tensor> GELUFn::backward(const Tensor& gradOut,
                                 const ActivationForward& fwd) const {
    // fwd.ctx = saved input x
    Tensor grad(fwd.ctx.shape());
    for (int64_t i = 0; i < fwd.ctx.numel(); ++i) {
        float xi       = fwd.ctx.flat(i);
        float inner    = kSqrt2OverPi * (xi + kGELUCoeff * xi * xi * xi);
        float t        = std::tanh(inner);
        float sech2    = 1.0f - t * t;
        float dinnerDx = kSqrt2OverPi * (1.0f + 3.0f * kGELUCoeff * xi * xi);
        float dGELUdx  = 0.5f * (1.0f + t) + 0.5f * xi * sech2 * dinnerDx;
        grad.flat(i)   = gradOut.flat(i) * dGELUdx;
    }
    return Result<Tensor>(grad);
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
