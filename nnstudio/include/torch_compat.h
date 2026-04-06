#pragma once
/* ============================================================================
 * torch_compat.h — C++ PyTorch-compatible alias shim for NNStudio
 * LGPL v3
 *
 * GOAL
 * ────
 * A translation-unit that uses only the standard PyTorch/LibTorch C++ API
 * can swap this include for <torch/torch.h> and get NNStudio's engine instead,
 * with zero code changes:
 *
 *   // pick ONE of these two lines — everything else is identical:
 *   #include <torch_compat.h>    // ← NNStudio engine
 *   // #include <torch/torch.h>  // ← LibTorch
 *
 *   auto model = torch::nn::Sequential(
 *       torch::nn::Linear(4, 8),
 *       torch::nn::ReLU(),
 *       torch::nn::Linear(8, 1),
 *       torch::nn::Sigmoid()
 *   );
 *   auto optim = torch::optim::Adam(1e-3f);
 *
 * DESIGN
 * ──────
 * • Header-only: zero compiled symbols, zero link-time cost.
 * • Type aliases where constructor signatures match.
 * • Thin wrapper classes where PyTorch and NNStudio constructor order differs.
 * • Sequential implemented as an ILayer that owns and chains component layers.
 * • Non-standard NNStudio extensions are NOT aliased here — they remain in
 *   namespace nnstudio::*.  Code using those will not compile against LibTorch.
 *
 * LIMITATIONS (Phase 1)
 * ─────────────────────
 * • torch::optim constructors do NOT take a params vector — pass params to step().
 * • torch::nn::functional ops use our backend (CpuBackend only until Phase 2).
 * • Mixed-precision / autocast / AMP not mapped.
 * • torch::jit, torch::autograd, torch::data not mapped.
 * • torch::Tensor::grad() / requires_grad() exist but autograd graph is NNStudio's.
 *
 * @see docs/adr/ADR-022-libtorch-backend-vs-scratch-cuda.md
 * @see docs/blueprints.md §9 (cross-framework compat)
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#include <core/Tensor.h>
#include <core/DType.h>
#include <core/Device.h>
#include <core/Layer.h>
#include <core/Result.h>
#include <builtin/layers/Dense.h>
#include <builtin/layers/Conv2D.h>
#include <builtin/layers/Embedding.h>
#include <builtin/layers/MultiHeadAttention.h>
#include <builtin/layers/NormLayers.h>
#include <builtin/layers/Activations.h>
#include <builtin/layers/ActivationFunctors.h>
#include <builtin/layers/ActivationsFnLayer.h>
#include <builtin/losses/Losses.h>
#include <builtin/optimizers/Optimizers.h>

#include <memory>
#include <vector>
#include <utility>
#include <cstdlib>  // std::rand

// ─── torch (top-level) ───────────────────────────────────────────────────────
namespace torch {

using namespace nnstudio::core;

// ── Tensor type & factory functions ──────────────────────────────────────────
using Tensor = nnstudio::core::Tensor;
using Size   = nnstudio::core::Shape;          ///< torch::Size ↔ Shape (vector<int64_t>)

/// dtype constants — torch::kFloat32, torch::kFloat16, torch::kInt8, torch::kInt32
inline constexpr DType kFloat32 = DType::Float32;
inline constexpr DType kFloat16 = DType::Float16;
inline constexpr DType kInt8    = DType::Int8;
inline constexpr DType kInt32   = DType::Int32;
inline constexpr DType kBool    = DType::Bool;

/// device constants
inline constexpr Device kCPU  = Device::CPU;
inline constexpr Device kCUDA = Device::CUDA;

inline Tensor zeros(Shape shape, DType dtype = DType::Float32, Device dev = Device::CPU) {
    return Tensor::zeros(std::move(shape), dtype, dev);
}
inline Tensor ones(Shape shape, DType dtype = DType::Float32, Device dev = Device::CPU) {
    return Tensor::ones(std::move(shape), dtype, dev);
}
inline Tensor full(Shape shape, float value, DType dtype = DType::Float32, Device dev = Device::CPU) {
    return Tensor::full(std::move(shape), value, dtype, dev);
}
/// Uniform random [0, 1) — uses std::rand; seed externally for reproducibility.
inline Tensor rand(Shape shape, DType dtype = DType::Float32, Device dev = Device::CPU) {
    Tensor t(std::move(shape), dtype, dev);
    float* p = t.data();
    for (int64_t i = 0; i < t.numel(); ++i)
        p[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
    return t;
}

// ─── torch::nn ───────────────────────────────────────────────────────────────
namespace nn {

using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::losses;

// ── Module base ──────────────────────────────────────────────────────────────
using Module = nnstudio::core::ILayer;

// ── Layers ───────────────────────────────────────────────────────────────────

/// torch::nn::Linear(in_features, out_features, bias=true)
/// NNStudio Dense defers weight allocation to build(); in_features is stored.
struct Linear : nnstudio::builtin::layers::Dense {
    /// @param in_features  Input size (stored; build() still infers from input shape).
    /// @param out_features Number of output neurons.
    explicit Linear(int64_t in_features, int64_t out_features, bool bias = true)
        : Dense(out_features, bias)
        , in_features(in_features)
    {}
    int64_t in_features;
};

/// torch::nn::Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
struct Conv2d : nnstudio::builtin::layers::Conv2D {
    explicit Conv2d(int64_t in_channels, int64_t out_channels,
                    int64_t kernel_size = 3, int64_t stride = 1,
                    int64_t padding = 0, bool bias = true)
        : Conv2D(out_channels, kernel_size, stride, padding, bias)
        , in_channels(in_channels)
    {}
    int64_t in_channels;
};

/// torch::nn::Embedding(num_embeddings, embedding_dim)
using Embedding = nnstudio::builtin::layers::Embedding;

/// torch::nn::MultiheadAttention(embed_dim, num_heads, dropout, bias, ...)
/// NOTE: PyTorch arg order is (embed_dim, num_heads); NNStudio is (numHeads, embDim).
struct MultiheadAttention : nnstudio::builtin::layers::MultiHeadAttention {
    explicit MultiheadAttention(int64_t embed_dim, int64_t num_heads,
                                float dropout = 0.0f, bool /* bias */ = true)
        : MultiHeadAttention(num_heads, embed_dim, /*causal=*/false, dropout)
    {}
};

/// torch::nn::BatchNorm1d(num_features, eps, momentum, affine)
using BatchNorm1d = nnstudio::builtin::layers::BatchNorm1d;

/// torch::nn::LayerNorm(normalized_shape)
using LayerNorm = nnstudio::builtin::layers::LayerNorm;

/// torch::nn::Dropout(p)
using Dropout = nnstudio::builtin::layers::Dropout;

// ── Activations ──────────────────────────────────────────────────────────────
using ReLU    = nnstudio::builtin::layers::ReLU;
using Sigmoid = nnstudio::builtin::layers::Sigmoid;
using Tanh    = nnstudio::builtin::layers::TanhAct;   ///< TanhAct avoids std::tanh clash
using GELU    = nnstudio::builtin::layers::GELU;
using Softmax = nnstudio::builtin::layers::Softmax;
using LeakyReLU = nnstudio::builtin::layers::LeakyReLU;

// ── Loss functions ────────────────────────────────────────────────────────────
using MSELoss              = nnstudio::builtin::losses::MSE;
using CrossEntropyLoss     = nnstudio::builtin::losses::CrossEntropy;
using BCELoss              = nnstudio::builtin::losses::BCE;
using BCEWithLogitsLoss    = nnstudio::builtin::losses::BCE;  ///< Phase 1: no logit pre-sigmoid
using HuberLoss            = nnstudio::builtin::losses::HuberLoss;

// ── Sequential ───────────────────────────────────────────────────────────────
//
// Thin ILayer that owns a vector of child layers and runs them in order.
// Matches the torch::nn::Sequential(Module1, Module2, ...) constructor.
//
//   auto model = Sequential(Linear(4, 8), ReLU(), Linear(8, 1));
//   model.build({1, 4});              // propagates shapes through all children
//   auto out = model.forward(x);     // calls each child in order
//
class Sequential : public nnstudio::core::ILayer {
public:
    Sequential() = default;

    /// Variadic value constructor — takes any number of ILayer subclass values.
    /// Each argument is move-constructed into a unique_ptr.
    template<typename... Layers>
    explicit Sequential(Layers&&... layers) {
        (pushLayer(std::forward<Layers>(layers)), ...);
    }

    // ── ILayer ──────────────────────────────────────────────────────────────
    std::string_view typeName() const noexcept override { return "Sequential"; }
    std::string_view docRef()   const noexcept override { return ""; }

    std::vector<nnstudio::core::Parameter*> parameters() override {
        std::vector<nnstudio::core::Parameter*> out;
        for (auto& l : layers_) {
            auto lp = l->parameters();
            out.insert(out.end(), lp.begin(), lp.end());
        }
        return out;
    }
    std::vector<const nnstudio::core::Parameter*> parameters() const override {
        std::vector<const nnstudio::core::Parameter*> out;
        for (auto& l : layers_) {
            auto lp = l->parameters();
            out.insert(out.end(), lp.begin(), lp.end());
        }
        return out;
    }

    /// Propagates shape through all children in order.
    nnstudio::core::Result<nnstudio::core::Shape> build(const nnstudio::core::Shape& inputShape) override {
        nnstudio::core::Shape s = inputShape;
        for (auto& l : layers_) {
            auto r = l->build(s);
            if (!r.ok()) return r;
            s = r.value();
        }
        markBuilt();
        return nnstudio::core::Result<nnstudio::core::Shape>(s);
    }

    /// Runs forward through all children in recording order.
    nnstudio::core::Result<nnstudio::core::Tensor> forward(const nnstudio::core::Tensor& x) override {
        nnstudio::core::Tensor t = x;
        for (auto& l : layers_) {
            auto r = l->forward(t);
            if (!r.ok()) return r;
            t = std::move(r).value();
        }
        return nnstudio::core::Result<nnstudio::core::Tensor>(std::move(t));
    }

    /// Runs backward through all children in reverse order.
    nnstudio::core::Result<nnstudio::core::Tensor> backward(const nnstudio::core::Tensor& gradOut) override {
        nnstudio::core::Tensor g = gradOut;
        for (int i = static_cast<int>(layers_.size()) - 1; i >= 0; --i) {
            auto r = layers_[i]->backward(g);
            if (!r.ok()) return r;
            g = std::move(r).value();
        }
        return nnstudio::core::Result<nnstudio::core::Tensor>(std::move(g));
    }

    void setTraining(bool t) noexcept override {
        nnstudio::core::ILayer::setTraining(t);
        for (auto& l : layers_) l->setTraining(t);
    }

    /// Number of children.
    std::size_t size() const noexcept { return layers_.size(); }

private:
    std::vector<std::unique_ptr<nnstudio::core::ILayer>> layers_;

    template<typename L>
    void pushLayer(L&& l) {
        layers_.push_back(std::make_unique<std::decay_t<L>>(std::forward<L>(l)));
    }
};

// ── torch::nn::functional ─────────────────────────────────────────────────────
namespace functional {

using namespace nnstudio::builtin::layers;

/// Stateless activation free functions — match torch.nn.functional.* naming.
inline nnstudio::core::Tensor relu(const nnstudio::core::Tensor& x) {
    return ReLUFn{}.forward(x).output;
}
inline nnstudio::core::Tensor sigmoid(const nnstudio::core::Tensor& x) {
    return SigmoidFn{}.forward(x).output;
}
inline nnstudio::core::Tensor tanh(const nnstudio::core::Tensor& x) {
    return TanhActFn{}.forward(x).output;
}
inline nnstudio::core::Tensor gelu(const nnstudio::core::Tensor& x) {
    return GELUFn{}.forward(x).output;
}
inline nnstudio::core::Tensor softmax(const nnstudio::core::Tensor& x,
                                      int64_t /* dim */ = -1) {
    // Phase 1: SoftmaxFn operates row-wise regardless of dim.
    return SoftmaxFn{}.forward(x).output;
}
inline nnstudio::core::Tensor leaky_relu(const nnstudio::core::Tensor& x,
                                         float negative_slope = 0.01f) {
    return LeakyReLUFn{negative_slope}.forward(x).output;
}
/// Dropout: training=true applies Bernoulli mask; training=false is identity.
inline nnstudio::core::Tensor dropout(const nnstudio::core::Tensor& x,
                                      float p = 0.5f, bool training = true) {
    nnstudio::builtin::layers::Dropout d(p);
    d.setTraining(training);
    d.build(x.shape());
    return d.forward(x).value();
}

} // namespace functional
} // namespace nn

// ─── torch::optim ────────────────────────────────────────────────────────────
namespace optim {

using namespace nnstudio::builtin::optimizers;

// ── Optimizer type aliases ────────────────────────────────────────────────────
//
// NOTE: NNStudio optimizers do NOT take a params vector in the constructor.
// Call step(params) at each training step.
// Constructor signatures match the positional lr/momentum/weight_decay order.
//
using SGD     = nnstudio::builtin::optimizers::SGD;
using Adam    = nnstudio::builtin::optimizers::Adam;
using AdamW   = nnstudio::builtin::optimizers::AdamW;
using RMSprop = nnstudio::builtin::optimizers::RMSProp;   ///< PyTorch spells it RMSprop

} // namespace optim

} // namespace torch
