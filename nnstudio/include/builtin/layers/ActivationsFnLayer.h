/* ============================================================================
 * ActivationsFnLayer.h — ILayer adapter for IActivation  [nnstudio::builtin::layers]
 * LGPL v3
 *
 * Bridges the ADR-020 stateless IActivation interface into the ILayer world.
 * ActivationsFnLayer owns or borrows an IActivation and stores the
 * ActivationForward context between forward() and backward() calls, just as
 * ActivationBase did — but the context is mutable state on the *layer*
 * (sequential instance), not on the *activation* (shared functor). This makes
 * the activation itself reentrant while the layer remains the correct owner of
 * per-call state.
 *
 * NAMING
 * ──────
 * "Activations" — belongs to the activations family of layers.
 * "Fn"          — wraps a stateless function object (IActivation implementor).
 * "Layer"       — it IS an ILayer and can be dropped into any Sequential.
 *
 * USAGE
 * ─────
 * Option A — inline concrete type (owning):
 *   auto layer = ActivationsFnLayer<ReLUFn>{};  // ReLUFn implements IActivation
 *
 * Option B — shared functor (non-owning pointer):
 *   auto shared_relu = std::make_shared<ReLUFn>();
 *   auto layer = ActivationsFnLayerPtr{ shared_relu.get() };
 *
 * ActivationsFnLayer has no learnable parameters: parameters() always returns {}.
 * build() passes the input shape through unchanged (like ActivationBase).
 *
 * @see include/core/IActivation.h
 * @see include/builtin/layers/Activations.h  (legacy ActivationBase path)
 * ============================================================================
*/
#pragma once

#include <core/IActivation.h>
#include <core/Layer.h>
#include <string>
#include <string_view>

namespace nnstudio {
namespace builtin {
namespace layers {

using namespace nnstudio::core;

// ─── ActivationsFnLayer<Fn> ──────────────────────────────────────────────────
//
// Fn must satisfy IActivation:
//   ActivationForward forward(const Tensor&) const
//   Result<Tensor>    backward(const Tensor&, const ActivationForward&) const
//   std::string_view  name() const noexcept
//
template<typename Fn>
class ActivationsFnLayer : public ILayer {
public:
    // Owning constructor — creates Fn in-place with forwarded args.
    template<typename... Args>
    explicit ActivationsFnLayer(Args&&... args)
        : fn_(std::forward<Args>(args)...) {}

    // ── ILayer ──────────────────────────────────────────────────────────
    std::string_view typeName() const noexcept override { return fn_.name(); }

    std::vector<Parameter*>       parameters()       override { return {}; }
    std::vector<const Parameter*> parameters() const override { return {}; }

    Result<Shape> build(const Shape& inputShape) override {
        markBuilt();
        return Result<Shape>(inputShape);
    }

    Result<Tensor> forward(const Tensor& x, EvalTrace* /*trace*/ = nullptr) override {
        ctx_ = fn_.forward(x);
        return Result<Tensor>(ctx_.output);
    }

    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* /*trace*/ = nullptr) override {
        return fn_.backward(gradOut, ctx_);
    }

private:
    Fn                fn_;   ///< the stateless activation functor
    ActivationForward ctx_;  ///< context saved by the last forward() call
};

// ─── ActivationsFnLayerPtr (non-owning, for shared functors) ────────────────
//
// Use ActivationsFnLayerPtr when the IActivation lifetime is managed
// externally, e.g. shared across several parallel graph nodes.
//
class ActivationsFnLayerPtr : public ILayer {
public:
    explicit ActivationsFnLayerPtr(IActivation* fn) : fn_(fn) {}

    std::string_view typeName() const noexcept override {
        return fn_ ? fn_->name() : "ActivationsFnLayerPtr(null)";
    }

    std::vector<Parameter*>       parameters()       override { return {}; }
    std::vector<const Parameter*> parameters() const override { return {}; }

    Result<Shape> build(const Shape& inputShape) override {
        markBuilt();
        return Result<Shape>(inputShape);
    }

    Result<Tensor> forward(const Tensor& x, EvalTrace* /*trace*/ = nullptr) override {
        if (!fn_) return Result<Tensor>(
            Error{ErrorCode::InvalidArgument, "ActivationsFnLayerPtr: null IActivation"});
        ctx_ = fn_->forward(x);
        return Result<Tensor>(ctx_.output);
    }

    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* /*trace*/ = nullptr) override {
        if (!fn_) return Result<Tensor>(
            Error{ErrorCode::InvalidArgument, "ActivationsFnLayerPtr: null IActivation"});
        return fn_->backward(gradOut, ctx_);
    }

private:
    IActivation*      fn_;   ///< non-owning pointer
    ActivationForward ctx_;
};

} // namespace layers
} // namespace builtin
} // namespace nnstudio
