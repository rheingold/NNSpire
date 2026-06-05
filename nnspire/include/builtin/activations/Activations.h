/* ============================================================================
 * Activations.h — element-wise activation functions  [NNSpire::builtin::activations]
 * LGPL v3
 *
 * INHERITANCE CHAIN
 * ─────────────────
 * NNSpire::ILayer                                   ← the abstract contract
 *   └─ NNSpire::builtin::layers::ActivationBase     ← factors out two universal no-ops:
 *        │   parameters() → {}               (no learnable weights)
 *        │   build()      → pass shape through unchanged (output shape == input shape)
 *        ├─ ReLU          typeName/forward/backward
 *        ├─ LeakyReLU     typeName/forward/backward  (α hyper-param, not learned)
 *        ├─ Sigmoid       typeName/forward/backward
 *        ├─ TanhAct       typeName/forward/backward
 *        ├─ Softmax       typeName/forward/backward  (NOT elementwise — full Jacobian)
 *        └─ GELU          typeName/forward/backward  (Transformer-era activation)
 *
 * TWO SAVE STRATEGIES
 * ───────────────────
 * forward() must save something for backward() to use:
 *   Save lastInput_  — ReLU, LeakyReLU, GELU
 *   Save lastOutput_ — Sigmoid, TanhAct, Softmax  (derivative via output cheaper)
 *
 * SOFTMAX NOTE
 * ────────────
 * Not elementwise in backward: full Jacobian via dot-product form:
 *   dX_i = s_i · (gradOut_i − dot(gradOut, s))
 *
 * ⚠️  ADR-020 MIGRATION COMPLETE (Phase 1):
 * lastInput_/lastOutput_ have been removed from ILayer and ActivationBase.
 * Each concrete class now owns a functor (e.g. ReLUFn fn_) and stores context
 * as ActivationForward ctx_. Functors are also usable standalone via
 * ActivationsFnLayer<ReLUFn> for reentrant/shared use cases.
 * ============================================================================ */

#pragma once

#include <core/Layer.h>
#include <core/IActivation.h>
#include <core/Tensor.h>
#include <core/Result.h>
#include <builtin/activations/Functors.h>

namespace NNSpire {
namespace builtin {
namespace activations {
using namespace NNSpire::core;

// ─── Shared base that handles the boilerplate ─────────────────────────────────
class ActivationBase : public ILayer {
public:
    // Activations have no learnable parameters
    std::vector<Parameter*>       parameters()       override { return {}; }
    std::vector<const Parameter*> parameters() const override { return {}; }

    // Output shape == input shape
    Result<Shape> build(const Shape& inputShape) override {
        markBuilt();
        return Result<Shape>(inputShape);
    }
protected:
    ActivationForward ctx_;   ///< saved by forward(); consumed by backward()
};

// ─── ReLU: f(x) = max(0, x) ──────────────────────────────────────────────────
class ReLU : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "ReLU"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#relu";
    }
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)       override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;
private:
    ReLUFn fn_;
};

// ─── LeakyReLU: f(x) = x if x>0 else alpha*x ────────────────────────────────────
class LeakyReLU : public ActivationBase {
public:
    explicit LeakyReLU(float alpha = 0.01f) : fn_(alpha) {}
    std::string_view typeName() const noexcept override { return "LeakyReLU"; }

    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)       override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;

    std::unordered_map<std::string,std::string> config() const override {
        return {{"alpha", std::to_string(fn_.alpha_)}};
    }
private:
    LeakyReLUFn fn_;
};

// ─── Sigmoid: f(x) = 1/(1+exp(-x)) ──────────────────────────────────────────
class Sigmoid : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "Sigmoid"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#sigmoid";
    }
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)       override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;
private:
    SigmoidFn fn_;
};

// ─── TanhAct: f(x) = tanh(x) ─────────────────────────────────────────────────────
class TanhAct : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "Tanh"; }
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)       override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;
private:
    TanhActFn fn_;
};

// ─── Softmax: f(x_i) = exp(x_i - max) / sum(exp(x_j - max)) ─────────────────
// Operates on last dimension (dim=-1).
class Softmax : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "Softmax"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#softmax";
    }
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)       override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;
private:
    SoftmaxFn fn_;
};

// ─── GELU: f(x) = 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))) ──────────────────────
// @kb: docs/ai-standards-kb/annexes/A05-LLM-API-Standards.md
class GELU : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "GELU"; }
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)       override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;
private:
    GELUFn fn_;
};

} // namespace activations
} // namespace builtin
} // namespace NNSpire
