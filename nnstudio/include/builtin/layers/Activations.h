/* ============================================================================
 * Activations.h — element-wise activation functions  [nnstudio::builtin::layers]
 * LGPL v3
 *
 * INHERITANCE CHAIN
 * ─────────────────
 * nnstudio::ILayer                                   ← the abstract contract
 *   └─ nnstudio::builtin::layers::ActivationBase     ← factors out two universal no-ops:
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
 * ⚠️  HIGH PRIORITY (ADR-020): ActivationBase stores lastInput_/lastOutput_ making
 * it non-reentrant. Before Phase 2 ABI freeze these must migrate to the functional
 * IActivation pattern (ActivationForward struct + context). See TODO.md.
 *
 * @see docs/blueprints.md — §3.8
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#activation-functions
 * ============================================================================ */

#pragma once

#include <core/Layer.h>
#include <core/Tensor.h>
#include <core/Result.h>

namespace nnstudio {
namespace builtin {
namespace layers {
using namespace nnstudio::core;

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
};

// ─── ReLU: f(x) = max(0, x) ──────────────────────────────────────────────────
class ReLU : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "ReLU"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#relu";
    }

    Result<Tensor> forward (const Tensor& x)        override;
    Result<Tensor> backward(const Tensor& gradOut)  override;
};

// ─── LeakyReLU: f(x) = x if x>0 else alpha*x ────────────────────────────────
class LeakyReLU : public ActivationBase {
public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha_(alpha) {}
    std::string_view typeName() const noexcept override { return "LeakyReLU"; }

    Result<Tensor> forward (const Tensor& x)        override;
    Result<Tensor> backward(const Tensor& gradOut)  override;

    std::unordered_map<std::string,std::string> config() const override {
        return {{"alpha", std::to_string(alpha_)}};
    }

private:
    float alpha_;
};

// ─── Sigmoid: f(x) = 1/(1+exp(-x)) ──────────────────────────────────────────
class Sigmoid : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "Sigmoid"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#sigmoid";
    }

    Result<Tensor> forward (const Tensor& x)        override;
    Result<Tensor> backward(const Tensor& gradOut)  override;
};

// ─── TanhAct: f(x) = tanh(x) ─────────────────────────────────────────────────
class TanhAct : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "Tanh"; }

    Result<Tensor> forward (const Tensor& x)        override;
    Result<Tensor> backward(const Tensor& gradOut)  override;
};

// ─── Softmax: f(x_i) = exp(x_i - max) / sum(exp(x_j - max)) ─────────────────
// Operates on last dimension (dim=-1).
class Softmax : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "Softmax"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#softmax";
    }

    Result<Tensor> forward (const Tensor& x)        override;
    Result<Tensor> backward(const Tensor& gradOut)  override;
};

// ─── GELU: f(x) = 0.5·x·(1+tanh(√(2/π)·(x+0.044715·x³))) ──────────────────
// @kb: docs/ai-standards-kb/annexes/A05-LLM-API-Standards.md
class GELU : public ActivationBase {
public:
    std::string_view typeName() const noexcept override { return "GELU"; }

    Result<Tensor> forward (const Tensor& x)        override;
    Result<Tensor> backward(const Tensor& gradOut)  override;
};

} // namespace layers
} // namespace builtin
} // namespace nnstudio
