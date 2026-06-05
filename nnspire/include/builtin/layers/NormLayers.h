/* ============================================================================
 * NormLayers.h — Dropout, BatchNorm1d, LayerNorm  [NNSpire::builtin::layers]
 * LGPL v3
 *
 * All three layers follow the standard ILayer contract (build/forward/backward).
 * BatchNorm1d and LayerNorm have learnable scale (gamma) and shift (beta).
 *
 * Dropout
 * ───────
 * Randomly zeros activations at training time (inverted dropout: kept elements
 * are scaled by 1/(1-dropRate) so the expected value is unchanged).
 * In eval mode, forward() is the identity.
 *
 * BatchNorm1d  (2-D input [N, F])
 * ─────────────────────────────────
 * Normalises over the batch dimension N; learnable gamma[F], beta[F].
 * Tracks running_mean/running_var during training for use in eval mode.
 * Full backward including dW, db, and dX through the normalisation.
 *
 * LayerNorm  (2-D input [N, D], normalise over D)
 * ─────────────────────────────────────────────────
 * Normalises over the feature dimension D; learnable gamma[D], beta[D].
 * Efficient backward via saved x_hat and rstd (reciprocal std-dev).
 *
 * @see docs/blueprints.md §3
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================
*/
#pragma once

#include <core/Layer.h>
#include <string_view>
#include <random>

namespace NNSpire {
namespace builtin {
namespace layers {
using namespace NNSpire::core;

// ─── Dropout ─────────────────────────────────────────────────────────────────
class Dropout : public ILayer {
public:
    /** @param dropRate  Fraction of elements to zero out (0.0 = no dropout). */
    explicit Dropout(float dropRate = 0.5f);

    std::string_view typeName() const noexcept override { return "Dropout"; }

    std::vector<Parameter*>       parameters()       override { return {}; }
    std::vector<const Parameter*> parameters() const override { return {}; }

    Result<Shape>  build   (const Shape& inputShape) override;
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)         override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr)   override;

    /// Toggle training/eval mode.
    void setTraining(bool training) noexcept { training_ = training; }
    bool isTraining()         const noexcept { return training_; }

    std::unordered_map<std::string,std::string> config() const override {
        return {{"drop_rate", std::to_string(dropRate_)}};
    }

    /// Set RNG seed for reproducible tests.
    void setSeed(uint32_t seed) noexcept { rng_.seed(seed); }

private:
    float dropRate_;
    bool  training_{true};
    Tensor mask_;                          ///< saved mask from last forward()
    std::default_random_engine rng_{42u};
};

// ─── BatchNorm1d ─────────────────────────────────────────────────────────────
/**
 * 1-D Batch Normalisation.
 * Input shape: [N, F]  (batch × features).
 * Normalises over the batch dimension N.
 * Learnable parameters: gamma[F] (scale, init=1) and beta[F] (shift, init=0).
 */
class BatchNorm1d : public ILayer {
public:
    /**
     * @param eps       Small constant for numerical stability.
     * @param momentum  Running-stats update rate (0.1 → standard PyTorch default).
     */
    explicit BatchNorm1d(float eps = 1e-5f, float momentum = 0.1f);

    std::string_view typeName() const noexcept override { return "BatchNorm1d"; }

    Result<Shape>  build   (const Shape& inputShape) override;
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)         override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr)   override;

    std::vector<Parameter*>       parameters()       override;
    std::vector<const Parameter*> parameters() const override;

    void setTraining(bool training) noexcept { training_ = training; }
    bool isTraining()         const noexcept { return training_; }

    std::unordered_map<std::string,std::string> config() const override {
        return {{"eps", std::to_string(eps_)},
                {"momentum", std::to_string(momentum_)}};
    }

    // Expose running stats for testing.
    const Tensor& runningMean() const noexcept { return runningMean_; }
    const Tensor& runningVar()  const noexcept { return runningVar_;  }

private:
    float   eps_;
    float   momentum_;
    bool    training_{true};
    int64_t numFeatures_{0};

    Parameter gamma_;          ///< scale  [F] — init to ones
    Parameter beta_;           ///< shift  [F] — init to zeros

    Tensor runningMean_;       ///< [F] running mean for eval
    Tensor runningVar_;        ///< [F] running variance for eval

    // Saved between forward() and backward()
    Tensor xNorm_;             ///< normalised x_hat        [N, F]
    Tensor xCentered_;         ///< x - mean                [N, F]
    Tensor rstd_;              ///< 1/sqrt(var+eps)          [F]
    float  N_{0.0f};           ///< batch size at last forward
};

// ─── LayerNorm ───────────────────────────────────────────────────────────────
/**
 * Layer Normalisation.
 * Input shape: [N, D].
 * Normalises over the last dimension D per sample.
 * Learnable parameters: gamma[D] (scale, init=1) and beta[D] (shift, init=0).
 */
class LayerNorm : public ILayer {
public:
    /**
     * @param normalizedDim  Size of the feature/last dimension to normalise.
     *                       Pass -1 to infer from build() input shape.
     * @param eps            Numerical stability constant.
     */
    explicit LayerNorm(int64_t normalizedDim = -1, float eps = 1e-5f);

    std::string_view typeName() const noexcept override { return "LayerNorm"; }

    Result<Shape>  build   (const Shape& inputShape) override;
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)         override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr)   override;

    std::vector<Parameter*>       parameters()       override;
    std::vector<const Parameter*> parameters() const override;

    std::unordered_map<std::string,std::string> config() const override {
        return {{"normalized_dim", std::to_string(normalizedDim_)},
                {"eps", std::to_string(eps_)}};
    }

private:
    int64_t normalizedDim_;    ///< D — size of normalised dimension
    float   eps_;

    Parameter gamma_;          ///< scale [D] — init to ones
    Parameter beta_;           ///< shift [D] — init to zeros

    // Saved between forward() and backward()
    Tensor xNorm_;             ///< normalised x_hat  [N, D]
    Tensor rstd_;              ///< per-sample 1/σ    [N]    (or [N, 1] conceptually)
    int64_t lastN_{0};         ///< batch size at last forward
};

} // namespace layers
} // namespace builtin
} // namespace NNSpire
