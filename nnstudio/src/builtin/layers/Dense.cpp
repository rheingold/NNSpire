/* ============================================================================
 * Dense.cpp — fully-connected layer implementation
 * LGPL v3
 *
 * WHAT IS "DENSE" / "FULLY CONNECTED"?
 * ─────────────────────────────────────
 * ONE sample:   y [outF]    = W [outF,inF] @ x [inF]     + b [outF]
 * BATCH:        Y [B,outF]  = X [B,inF]   @ W^T [inF,outF] + b [outF]
 *
 * "Fully connected" means every input x_i is connected to every output j
 * with its own independent weight w_{ji}.
 * Total parameters: outF × inF + outF = outF × (inF + 1).
 *
 * @see docs/blueprints.md — §3 Dense walkthrough
 * ============================================================================ */

#include <builtin/layers/Dense.h>
#include <core/BackendRegistry.h>

#include <cmath>
#include <cassert>
#include <random>
#include <optional>
#include <string>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace layers {

static inline IBackend& B() { return backend(); }

// ─── Deterministic seed control (for testing) ──────────────────────────────
namespace {
    std::optional<uint32_t> g_fixedSeed;
}

void Dense::setSeed(uint32_t seed)  noexcept { g_fixedSeed = seed; }
void Dense::resetSeed()             noexcept { g_fixedSeed = std::nullopt; }

// ─── Constructor ─────────────────────────────────────────────────────────────
Dense::Dense(int64_t outFeatures, bool useBias, WeightInit init)
    : outFeatures_(outFeatures), useBias_(useBias), init_(init) {
    weights_.name = "weights";
    bias_.name    = "bias";
}

// ─── Weight initialisation ───────────────────────────────────────────────────
void Dense::initWeights() {
    const int64_t fanIn  = inFeatures_;
    const int64_t fanOut = outFeatures_;

    std::mt19937 rng{g_fixedSeed.has_value() ? *g_fixedSeed
                                              : std::random_device{}()};

    if (init_ == WeightInit::GlorotUniform) {
        float limit = std::sqrt(6.0f / static_cast<float>(fanIn + fanOut));
        std::uniform_real_distribution<float> dist(-limit, limit);
        for (int64_t i = 0; i < weights_.tensor.numel(); ++i)
            weights_.tensor.flat(i) = dist(rng);
    } else if (init_ == WeightInit::HeNormal) {
        float stddev = std::sqrt(2.0f / static_cast<float>(fanIn));
        std::normal_distribution<float> dist(0.0f, stddev);
        for (int64_t i = 0; i < weights_.tensor.numel(); ++i)
            weights_.tensor.flat(i) = dist(rng);
    }

    if (useBias_) {
        for (int64_t i = 0; i < bias_.tensor.numel(); ++i)
            bias_.tensor.flat(i) = 0.0f;
    }
}

// ─── build ───────────────────────────────────────────────────────────────────
Result<Shape> Dense::build(const Shape& inputShape) {
    if (built_) return Result<Shape>({outFeatures_});

    if (inputShape.empty())
        return Result<Shape>(Error{ErrorCode::InvalidArgument,
            "Dense::build: inputShape must not be empty"});

    inFeatures_ = inputShape.back();

    weights_.tensor = (init_ == WeightInit::Zeros) ?
                          Tensor::zeros({outFeatures_, inFeatures_}) :
                      (init_ == WeightInit::Ones)  ?
                          Tensor::ones ({outFeatures_, inFeatures_}) :
                          Tensor::zeros({outFeatures_, inFeatures_});
    weights_.tensor.setRequiresGrad(true);

    if (useBias_) {
        bias_.tensor = Tensor::zeros({outFeatures_});
        bias_.tensor.setRequiresGrad(true);
    }

    initWeights();
    markBuilt();

    Shape outShape = inputShape;
    outShape.back() = outFeatures_;
    return Result<Shape>(outShape);
}

// ─── forward: y = x @ W^T + b ────────────────────────────────────────────────
Result<Tensor> Dense::forward(const Tensor& x) {
    assert(built_ && "Dense::forward called before build()");
    // CpuBackend only supports Float32; fail fast with a typed error rather
    // than silently reinterpreting raw bytes as floats.
    if (x.dtype() != DType::Float32)
        return err(ErrorCode::DTypeMismatch,
                   "Dense::forward: CpuBackend requires Float32 input, got " +
                   std::string(dtypeName(x.dtype())));
    lastInput_ = x;

    Tensor Wt = B().transpose(weights_.tensor, 0, 1);
    Tensor y  = B().matmul(x, Wt);

    if (useBias_) {
        const int64_t batch = x.shape()[0];
        for (int64_t n = 0; n < batch; ++n)
            for (int64_t o = 0; o < outFeatures_; ++o)
                y.flat(n * outFeatures_ + o) += bias_.tensor.flat(o);
    }
    return Result<Tensor>(y);
}

// ─── backward: dW = gradOut^T @ x,  db = sum(gradOut),  dX = gradOut @ W ────
Result<Tensor> Dense::backward(const Tensor& gradOut) {
    assert(built_ && "Dense::backward called before build()");
    assert(lastInput_.numel() > 0 && "Dense::backward called before forward()");

    const int64_t batch = lastInput_.shape()[0];

    // dW = gradOut^T @ x
    Tensor gradOutT = B().transpose(gradOut, 0, 1);
    Tensor dW       = B().matmul(gradOutT, lastInput_);
    weights_.tensor.accumulateGrad(dW);

    // db = sum over batch
    if (useBias_) {
        Tensor db = Tensor::zeros({outFeatures_});
        for (int64_t n = 0; n < batch; ++n)
            for (int64_t o = 0; o < outFeatures_; ++o)
                db.flat(o) += gradOut.flat(n * outFeatures_ + o);
        bias_.tensor.accumulateGrad(db);
    }

    // dX = gradOut @ W
    Tensor dX = B().matmul(gradOut, weights_.tensor);
    return Result<Tensor>(dX);
}

// ─── Parameters ──────────────────────────────────────────────────────────────
std::vector<Parameter*> Dense::parameters() {
    if (useBias_) return {&weights_, &bias_};
    return {&weights_};
}
std::vector<const Parameter*> Dense::parameters() const {
    if (useBias_) return {&weights_, &bias_};
    return {&weights_};
}

// ─── Config ──────────────────────────────────────────────────────────────────
std::unordered_map<std::string,std::string> Dense::config() const {
    return {
        {"out_features", std::to_string(outFeatures_)},
        {"in_features",  std::to_string(inFeatures_)},
        {"use_bias",     useBias_ ? "true" : "false"},
        {"init",         init_ == WeightInit::GlorotUniform ? "glorot_uniform" :
                         init_ == WeightInit::HeNormal      ? "he_normal"      :
                         init_ == WeightInit::Zeros         ? "zeros"          : "ones"},
    };
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
