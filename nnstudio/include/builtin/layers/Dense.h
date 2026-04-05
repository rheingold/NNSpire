/* ============================================================================
 * Dense.h — fully-connected (linear) layer  [nnstudio::builtin::layers]
 * LGPL v3
 *
 * Dense is ONE concrete implementation of ILayer — the "no structure assumed"
 * baseline layer.  Every input neuron connects to every output neuron.
 *
 * forward:  y = x @ W^T + b      shape [batch, outFeatures]
 * backward: accumulates dW, db; returns dX for the previous layer
 *
 * Weight init: Glorot/Xavier uniform (default), optional He normal via flag.
 *
 * CALL CHAIN — what actually happens during forward()
 * ────────────────────────────────────────────────────
 *   Dense::forward(x)
 *     B().transpose(weights_.tensor, 0, 1)   → reinterprets strides[], NO data copy
 *     B().matmul(x, Wt)                      → Eigen GEMM on float* pointers
 *     bias loop: y.flat(i) += bias_.flat(o)  → direct indexed access, no Backend
 *
 *   Dense::backward(gradOut)
 *     B().transpose(gradOut, 0, 1)           → gradOut^T, strides only
 *     B().matmul(gradOutT, lastInput_)       → dW = gradOut^T @ x
 *     weights_.tensor.accumulateGrad(dW)     → adds into W.grad_ (lazy alloc)
 *     bias accumulation loop
 *     B().matmul(gradOut, weights_.tensor)   → dX = gradOut @ W  (returned)
 *
 * @see docs/blueprints.md — §3 ILayer + Dense
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#dense-layers
 * ============================================================================ */

#pragma once

#include <core/Layer.h>

#include <optional>
#include <string>

namespace nnstudio {
namespace builtin {
namespace layers {
using namespace nnstudio::core;

// ──────────────────────────────────────────────────────────────────────────────
enum class WeightInit { GlorotUniform, HeNormal, Zeros, Ones };

// ──────────────────────────────────────────────────────────────────────────────
class Dense : public ILayer {
public:
    /**
     * @param outFeatures   Number of output neurons.
     * @param useBias       Whether to add a bias term.
     * @param init          Weight initialisation strategy.
     */
    explicit Dense(int64_t outFeatures,
                   bool useBias = true,
                   WeightInit init = WeightInit::GlorotUniform);

    std::string_view typeName() const noexcept override { return "Dense"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#dense-layers";
    }

    // ── ILayer ───────────────────────────────────────────────────────────────
    Result<Shape>  build   (const Shape& inputShape)  override;
    Result<Tensor> forward (const Tensor& x)          override;
    Result<Tensor> backward(const Tensor& gradOut)    override;

    std::vector<Parameter*>       parameters()       override;
    std::vector<const Parameter*> parameters() const override;

    std::unordered_map<std::string,std::string> config() const override;

    // ── Accessors ─────────────────────────────────────────────────────────────
    int64_t inFeatures()  const noexcept { return inFeatures_; }
    int64_t outFeatures() const noexcept { return outFeatures_; }
    bool    hasBias()     const noexcept { return useBias_; }
    /// Override the random seed for weight initialisation (useful in tests).
    static void setSeed(uint32_t seed) noexcept;
    static void resetSeed() noexcept;  ///< restore std::random_device seeding
private:
    Tensor lastInput_;   ///< saved from forward() for use in backward()
    int64_t    outFeatures_;
    int64_t    inFeatures_{0};
    bool       useBias_;
    WeightInit init_;

    Parameter weights_;
    Parameter bias_;

    void initWeights();
};

} // namespace layers
} // namespace builtin
} // namespace nnstudio
