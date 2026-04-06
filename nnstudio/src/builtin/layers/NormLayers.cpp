/* ============================================================================
 * NormLayers.cpp — Dropout, BatchNorm1d, LayerNorm implementations
 * LGPL v3
 * ============================================================================ */

#include <builtin/layers/NormLayers.h>
#include <core/BackendRegistry.h>

#include <cmath>
#include <cassert>
#include <stdexcept>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace layers {

// ═══════════════════════════════════════════════════════════════════════════════
// Dropout
// ═══════════════════════════════════════════════════════════════════════════════

Dropout::Dropout(float dropRate)
    : dropRate_(dropRate) {}

Result<Shape> Dropout::build(const Shape& inputShape) {
    markBuilt();
    return Result<Shape>(inputShape);
}

Result<Tensor> Dropout::forward(const Tensor& x, EvalTrace* /*trace*/) {
    if (!training_ || dropRate_ <= 0.0f) {
        mask_ = Tensor{};    // clear mask — backward is identity
        return Result<Tensor>(x);
    }
    const float keepProb = 1.0f - dropRate_;
    std::bernoulli_distribution dist(keepProb);
    mask_ = Tensor(x.shape());
    Tensor out(x.shape());
    for (int64_t i = 0; i < x.numel(); ++i) {
        float m      = dist(rng_) ? 1.0f : 0.0f;
        mask_.flat(i) = m;
        out.flat(i)   = x.flat(i) * m / keepProb;
    }
    return Result<Tensor>(out);
}

Result<Tensor> Dropout::backward(const Tensor& gradOut, EvalTrace* /*trace*/) {
    if (!training_ || dropRate_ <= 0.0f || mask_.numel() == 0)
        return Result<Tensor>(gradOut);

    const float keepProb = 1.0f - dropRate_;
    Tensor grad(gradOut.shape());
    for (int64_t i = 0; i < gradOut.numel(); ++i)
        grad.flat(i) = gradOut.flat(i) * mask_.flat(i) / keepProb;
    return Result<Tensor>(grad);
}

// ═══════════════════════════════════════════════════════════════════════════════
// BatchNorm1d
// ═══════════════════════════════════════════════════════════════════════════════

BatchNorm1d::BatchNorm1d(float eps, float momentum)
    : eps_(eps), momentum_(momentum) {
    gamma_.name = "gamma";
    beta_.name  = "beta";
}

Result<Shape> BatchNorm1d::build(const Shape& inputShape) {
    if (built_) return Result<Shape>(inputShape);
    if (inputShape.size() != 2)
        return Result<Shape>(Error{ErrorCode::InvalidArgument,
            "BatchNorm1d expects 2-D input [N, F]"});

    numFeatures_ = inputShape[1];
    gamma_.tensor = Tensor::ones ({numFeatures_});
    beta_.tensor  = Tensor::zeros({numFeatures_});
    gamma_.tensor.setRequiresGrad(true);
    beta_.tensor .setRequiresGrad(true);

    runningMean_ = Tensor::zeros({numFeatures_});
    runningVar_  = Tensor::ones ({numFeatures_});   // init to 1 (unbiased assumption)

    markBuilt();
    return Result<Shape>(inputShape);
}

Result<Tensor> BatchNorm1d::forward(const Tensor& x, EvalTrace* /*trace*/) {
    if (x.ndim() != 2)
        return Result<Tensor>(Error{ErrorCode::InvalidArgument,
            "BatchNorm1d::forward expects [N, F]"});

    const int64_t N = x.shape()[0];
    const int64_t F = x.shape()[1];
    N_ = static_cast<float>(N);

    Tensor mean(Shape{F});         // batch mean [F]
    Tensor var (Shape{F});         // batch variance [F]
    Tensor rstd(Shape{F});         // 1/sqrt(var+eps) [F]
    Tensor xNorm({N, F});          // normalized [N, F]
    Tensor xCentered({N, F});      // x - mean   [N, F]

    // Compute per-feature mean
    for (int64_t f = 0; f < F; ++f) {
        float s = 0.0f;
        for (int64_t n = 0; n < N; ++n) s += x.flat(n * F + f);
        mean.flat(f) = s / N_;
    }

    // Compute per-feature variance (biased: divide by N)
    for (int64_t f = 0; f < F; ++f) {
        float v = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            float d = x.flat(n * F + f) - mean.flat(f);
            xCentered.flat(n * F + f) = d;
            v += d * d;
        }
        var.flat(f) = v / N_;
        rstd.flat(f) = 1.0f / std::sqrt(var.flat(f) + eps_);
    }

    // Normalise and apply gamma/beta
    Tensor out({N, F});
    for (int64_t n = 0; n < N; ++n)
        for (int64_t f = 0; f < F; ++f) {
            xNorm.flat(n * F + f) = xCentered.flat(n * F + f) * rstd.flat(f);
            out.flat(n * F + f)   = gamma_.tensor.flat(f) * xNorm.flat(n * F + f)
                                  + beta_.tensor.flat(f);
        }

    // Save for backward
    xNorm_      = xNorm;
    xCentered_  = xCentered;
    rstd_       = rstd;

    // Update running stats (only in training mode)
    if (training_) {
        for (int64_t f = 0; f < F; ++f) {
            runningMean_.flat(f) = (1.0f - momentum_) * runningMean_.flat(f)
                                 + momentum_ * mean.flat(f);
            runningVar_.flat(f)  = (1.0f - momentum_) * runningVar_.flat(f)
                                 + momentum_ * var.flat(f);
        }
    }

    return Result<Tensor>(out);
}

Result<Tensor> BatchNorm1d::backward(const Tensor& gradOut, EvalTrace* /*trace*/) {
    const int64_t N = static_cast<int64_t>(N_);
    const int64_t F = numFeatures_;

    // dγ[f] = Σ_n gradOut[n,f] * xNorm[n,f]
    // dβ[f] = Σ_n gradOut[n,f]
    Tensor dgamma(Shape{F});
    Tensor dbeta (Shape{F});
    for (int64_t f = 0; f < F; ++f) {
        float sg = 0.0f, sb = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            sg += gradOut.flat(n * F + f) * xNorm_.flat(n * F + f);
            sb += gradOut.flat(n * F + f);
        }
        dgamma.flat(f) = sg;
        dbeta .flat(f) = sb;
    }
    gamma_.tensor.accumulateGrad(dgamma);
    beta_.tensor .accumulateGrad(dbeta);

    // dx_hat[n,f] = gradOut[n,f] * gamma[f]
    Tensor dxHat({N, F});
    for (int64_t n = 0; n < N; ++n)
        for (int64_t f = 0; f < F; ++f)
            dxHat.flat(n * F + f) = gradOut.flat(n * F + f) * gamma_.tensor.flat(f);

    // dvar[f] = Σ_n dx_hat[n,f] * xCentered[n,f] * (-0.5) * rstd[f]^3
    // Note: Σ_n xCentered[n,f] = 0, so dmean simplifies to:
    // dmean[f] = -rstd[f] * Σ_n dx_hat[n,f]
    Tensor dx({N, F});
    for (int64_t f = 0; f < F; ++f) {
        const float r  = rstd_.flat(f);
        const float r3 = r * r * r;

        float dvar  = 0.0f;
        float sumdx = 0.0f;
        for (int64_t n = 0; n < N; ++n) {
            dvar  += dxHat.flat(n * F + f) * xCentered_.flat(n * F + f);
            sumdx += dxHat.flat(n * F + f);
        }
        dvar  *= -0.5f * r3;
        float dmean = -r * sumdx;   // sum(xCentered) == 0, second term drops out

        for (int64_t n = 0; n < N; ++n) {
            dx.flat(n * F + f) = dxHat.flat(n * F + f) * r
                               + dvar  * 2.0f * xCentered_.flat(n * F + f) / N_
                               + dmean / N_;
        }
    }
    return Result<Tensor>(dx);
}

std::vector<Parameter*> BatchNorm1d::parameters() {
    return { &gamma_, &beta_ };
}
std::vector<const Parameter*> BatchNorm1d::parameters() const {
    return { &gamma_, &beta_ };
}

// ═══════════════════════════════════════════════════════════════════════════════
// LayerNorm
// ═══════════════════════════════════════════════════════════════════════════════

LayerNorm::LayerNorm(int64_t normalizedDim, float eps)
    : normalizedDim_(normalizedDim), eps_(eps) {
    gamma_.name = "gamma";
    beta_.name  = "beta";
}

Result<Shape> LayerNorm::build(const Shape& inputShape) {
    if (built_) return Result<Shape>(inputShape);
    if (inputShape.size() < 1)
        return Result<Shape>(Error{ErrorCode::InvalidArgument,
            "LayerNorm expects at least 1-D input"});

    // If user passed -1 (default), infer from last dim of inputShape
    if (normalizedDim_ == -1)
        normalizedDim_ = inputShape.back();

    gamma_.tensor = Tensor::ones ({normalizedDim_});
    beta_.tensor  = Tensor::zeros({normalizedDim_});
    gamma_.tensor.setRequiresGrad(true);
    beta_.tensor .setRequiresGrad(true);

    markBuilt();
    return Result<Shape>(inputShape);
}

Result<Tensor> LayerNorm::forward(const Tensor& x, EvalTrace* /*trace*/) {
    // Support 1-D [D] (single sample) by treating it as [1, D]
    const bool is1D = (x.ndim() == 1);
    const int64_t D = normalizedDim_;
    const int64_t N = is1D ? 1 : x.shape()[0];
    lastN_ = N;

    Tensor xNorm({N, D});
    Tensor rstdVec(Shape{N});
    Tensor out({N, D});

    for (int64_t n = 0; n < N; ++n) {
        const int64_t offset = n * D;

        // Mean over D
        float mean = 0.0f;
        for (int64_t d = 0; d < D; ++d)
            mean += x.flat(is1D ? d : offset + d);
        mean /= static_cast<float>(D);

        // Variance over D
        float var = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float v = x.flat(is1D ? d : offset + d) - mean;
            var += v * v;
        }
        var /= static_cast<float>(D);

        const float rstd = 1.0f / std::sqrt(var + eps_);
        rstdVec.flat(n) = rstd;

        for (int64_t d = 0; d < D; ++d) {
            float xd     = x.flat(is1D ? d : offset + d);
            float xn     = (xd - mean) * rstd;
            xNorm.flat(offset + d) = xn;
            out.flat  (offset + d) = gamma_.tensor.flat(d) * xn + beta_.tensor.flat(d);
        }
    }

    xNorm_ = xNorm;
    rstd_  = rstdVec;

    if (is1D) {
        // Reshape back to 1-D output
        auto r = out.reshape({D});
        if (!r.ok()) return Result<Tensor>(r.error());
        return Result<Tensor>(r.value());
    }
    return Result<Tensor>(out);
}

Result<Tensor> LayerNorm::backward(const Tensor& gradOut, EvalTrace* /*trace*/) {
    const int64_t D = normalizedDim_;
    const int64_t N = lastN_;
    const bool is1D = (gradOut.ndim() == 1);

    // Accumulate dgamma[d] = Σ_n gradOut[n,d] * xNorm[n,d]
    // Accumulate dbeta[d]  = Σ_n gradOut[n,d]
    Tensor dgamma(Shape{D});
    Tensor dbeta (Shape{D});
    for (int64_t d = 0; d < D; ++d) dgamma.flat(d) = dbeta.flat(d) = 0.0f;

    for (int64_t n = 0; n < N; ++n) {
        const int64_t off = n * D;
        for (int64_t d = 0; d < D; ++d) {
            float g = is1D ? gradOut.flat(d) : gradOut.flat(off + d);
            dgamma.flat(d) += g * xNorm_.flat(off + d);
            dbeta .flat(d) += g;
        }
    }
    gamma_.tensor.accumulateGrad(dgamma);
    beta_.tensor .accumulateGrad(dbeta);

    // Efficient backward: for each sample n
    // dx_hat[d] = gradOut[n, d] * gamma[d]
    // k1 = Σ_d (dx_hat[d] * xNorm[n,d]) / D
    // k2 = Σ_d dx_hat[d] / D
    // dx[n, d] = rstd[n] * (dx_hat[d] - k2 - k1 * xNorm[n,d])
    Tensor dx({N, D});
    const float Df = static_cast<float>(D);

    for (int64_t n = 0; n < N; ++n) {
        const int64_t off  = n * D;
        const float   rstd = rstd_.flat(n);

        float k1 = 0.0f, k2 = 0.0f;
        for (int64_t d = 0; d < D; ++d) {
            float g  = is1D ? gradOut.flat(d) : gradOut.flat(off + d);
            float dh = g * gamma_.tensor.flat(d);
            k1 += dh * xNorm_.flat(off + d);
            k2 += dh;
        }
        k1 /= Df;
        k2 /= Df;

        for (int64_t d = 0; d < D; ++d) {
            float g  = is1D ? gradOut.flat(d) : gradOut.flat(off + d);
            float dh = g * gamma_.tensor.flat(d);
            dx.flat(off + d) = rstd * (dh - k2 - k1 * xNorm_.flat(off + d));
        }
    }

    if (is1D) {
        auto r = dx.reshape({D});
        if (!r.ok()) return Result<Tensor>(r.error());
        return Result<Tensor>(r.value());
    }
    return Result<Tensor>(dx);
}

std::vector<Parameter*> LayerNorm::parameters() {
    return { &gamma_, &beta_ };
}
std::vector<const Parameter*> LayerNorm::parameters() const {
    return { &gamma_, &beta_ };
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
