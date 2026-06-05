/* ============================================================================
 * Functors.h — stateless IActivation functors  [NNSpire::builtin::activations]
 * LGPL v3
 *
 * Each struct is a zero-mutable-state IActivation implementor.  A single
 * instance can be shared safely across multiple threads and graph nodes.
 *
 * The ActivationBase subclasses (ReLU etc.) now delegate their
 * forward()/backward() calls to these functors, eliminating the
 * lastInput_/lastOutput_ mutable fields that made layers non-reentrant.
 *
 * Using with ActivationsFnLayer:
 *   auto layer = ActivationsFnLayer<ReLUFn>{};
 *   auto layer = ActivationsFnLayer<LeakyReLUFn>{LeakyReLUFn{0.2f}};
 *
 * Ctx convention (ActivationForward::ctxIsInput):
 *   true  — ctx holds the input  x  (ReLU, LeakyReLU, GELU)
 *   false — ctx holds the output y  (Sigmoid, TanhAct, Softmax)
 *
 * @see include/core/IActivation.h
 * @see include/builtin/activations/FnLayer.h
 * @see include/builtin/activations/Activations.h  (ILayer wrappers)
 * ============================================================================
*/
#pragma once

#include <core/IActivation.h>

namespace NNSpire {
namespace builtin {
namespace activations {

using namespace NNSpire::core;

// ─── ReLUFn ──────────────────────────────────────────────────────────────────
/// f(x) = max(0, x) — saves input as ctx
struct ReLUFn : IActivation {
    std::string_view  name()     const noexcept override { return "ReLU"; }
    ActivationForward forward (const Tensor& x)                              const override;
    Result<Tensor>    backward(const Tensor& gradOut, const ActivationForward& fwd) const override;
};

// ─── LeakyReLUFn ─────────────────────────────────────────────────────────────
/// f(x) = x if x > 0 else alpha*x — saves input as ctx
struct LeakyReLUFn : IActivation {
    explicit LeakyReLUFn(float alpha = 0.01f) : alpha_(alpha) {}
    std::string_view  name()     const noexcept override { return "LeakyReLU"; }
    ActivationForward forward (const Tensor& x)                              const override;
    Result<Tensor>    backward(const Tensor& gradOut, const ActivationForward& fwd) const override;
    float alpha_;
};

// ─── SigmoidFn ───────────────────────────────────────────────────────────────
/// f(x) = 1/(1+exp(-x)) — saves output as ctx (derivative cheaper via output)
struct SigmoidFn : IActivation {
    std::string_view  name()     const noexcept override { return "Sigmoid"; }
    ActivationForward forward (const Tensor& x)                              const override;
    Result<Tensor>    backward(const Tensor& gradOut, const ActivationForward& fwd) const override;
};

// ─── TanhActFn ───────────────────────────────────────────────────────────────
/// f(x) = tanh(x) — saves output as ctx (1 - tanh² faster via saved output)
struct TanhActFn : IActivation {
    std::string_view  name()     const noexcept override { return "Tanh"; }
    ActivationForward forward (const Tensor& x)                              const override;
    Result<Tensor>    backward(const Tensor& gradOut, const ActivationForward& fwd) const override;
};

// ─── SoftmaxFn ───────────────────────────────────────────────────────────────
/// Numerically stable softmax over last dim — saves output as ctx
struct SoftmaxFn : IActivation {
    std::string_view  name()     const noexcept override { return "Softmax"; }
    ActivationForward forward (const Tensor& x)                              const override;
    Result<Tensor>    backward(const Tensor& gradOut, const ActivationForward& fwd) const override;
};

// ─── GELUFn ──────────────────────────────────────────────────────────────────
/// GELU(x) = 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))) — saves input as ctx
struct GELUFn : IActivation {
    std::string_view  name()     const noexcept override { return "GELU"; }
    ActivationForward forward (const Tensor& x)                              const override;
    Result<Tensor>    backward(const Tensor& gradOut, const ActivationForward& fwd) const override;
};

} // namespace activations
} // namespace builtin
} // namespace NNSpire
