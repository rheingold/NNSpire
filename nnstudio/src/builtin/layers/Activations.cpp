/* ============================================================================
 * Activations.cpp — ILayer wrappers delegating to IActivation functors
 * LGPL v3
 *
 * ADR-020: all math lives in ActivationFunctors.cpp.
 * Each layer stores the ActivationForward context (ctx_) between
 * forward() and backward() calls.
 * ============================================================================ */

#include <builtin/layers/Activations.h>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace layers {

// ─── ReLU ────────────────────────────────────────────────────────────────────

Result<Tensor> ReLU::forward(const Tensor& x) {
    ctx_ = fn_.forward(x);
    return Result<Tensor>(ctx_.output);
}
Result<Tensor> ReLU::backward(const Tensor& gradOut) {
    return fn_.backward(gradOut, ctx_);
}

// ─── LeakyReLU ───────────────────────────────────────────────────────────────

Result<Tensor> LeakyReLU::forward(const Tensor& x) {
    ctx_ = fn_.forward(x);
    return Result<Tensor>(ctx_.output);
}
Result<Tensor> LeakyReLU::backward(const Tensor& gradOut) {
    return fn_.backward(gradOut, ctx_);
}

// ─── Sigmoid ─────────────────────────────────────────────────────────────────

Result<Tensor> Sigmoid::forward(const Tensor& x) {
    ctx_ = fn_.forward(x);
    return Result<Tensor>(ctx_.output);
}
Result<Tensor> Sigmoid::backward(const Tensor& gradOut) {
    return fn_.backward(gradOut, ctx_);
}

// ─── TanhAct ─────────────────────────────────────────────────────────────────

Result<Tensor> TanhAct::forward(const Tensor& x) {
    ctx_ = fn_.forward(x);
    return Result<Tensor>(ctx_.output);
}
Result<Tensor> TanhAct::backward(const Tensor& gradOut) {
    return fn_.backward(gradOut, ctx_);
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

Result<Tensor> Softmax::forward(const Tensor& x) {
    ctx_ = fn_.forward(x);
    return Result<Tensor>(ctx_.output);
}
Result<Tensor> Softmax::backward(const Tensor& gradOut) {
    return fn_.backward(gradOut, ctx_);
}

// ─── GELU ────────────────────────────────────────────────────────────────────

Result<Tensor> GELU::forward(const Tensor& x) {
    ctx_ = fn_.forward(x);
    return Result<Tensor>(ctx_.output);
}
Result<Tensor> GELU::backward(const Tensor& gradOut) {
    return fn_.backward(gradOut, ctx_);
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
