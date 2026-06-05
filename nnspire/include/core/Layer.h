/* ============================================================================
 * Layer.h — abstract base class for all trainable modules (ILayer)
 *           + ordered container (Sequential)
 * LGPL v3
 *
 * WHAT IS A LAYER? (vs Backend)
 * ──────────────────────────────
 * Backend  = raw tensor math: matmul, add, exp, sum … pure arithmetic.
 *            It has no concept of neurons, learning, or parameters.
 *
 * Layer    = a meaningful NN computational unit that:
 *              • holds learnable parameters (weights W, bias b)
 *              • uses the Backend to run its specific NN formula in forward()
 *              • computes how wrong each parameter was in backward() (the gradient)
 *              • does NOT update those parameters itself — that is the Optimizer's job
 *
 *   Layer     = the "who is to blame?" accountant.
 *   Optimizer = the "fix it" agent that reads the blame and adjusts weights.
 *
 * SEQUENTIAL — the ordered pipeline
 * ──────────────────────────────────
 * Sequential implements ILayer itself — it is a layer whose "computation" is
 * to thread input through an ordered list of child layers.
 *
 *   • forward()    — passes tensor left-to-right through layers_
 *   • backward()   — passes gradient right-to-left (rbegin → rend)
 *   • build()      — chains shapes: output shape of layer N → input shape of N+1
 *   • parameters() — flattens all child Parameter* into one vector for the Optimizer
 *
 * Because Sequential IS an ILayer, it can be:
 *   • handed to Trainer as ILayer& (same as a bare Dense)
 *   • nested inside another Sequential (sub-network = one layer)
 *   • swapped for a ComputeGraph (DAG version) without changing Trainer
 *
 * HOW LAYERS PLUG INTO THE SYSTEM
 * ────────────────────────────────
 *   User code
 *     └─ Sequential::forward(x)      ← routes to each child in order
 *          └─ Dense::forward(x)       ← concrete ILayer subclass
 *               └─ B().matmul(x, Wt)  ← calls active Backend via free function
 *                    └─ CpuBackend::matmul ← Eigen GEMM on raw float* pointers
 *                         └─ Tensor::data() ← flat std::vector<float> in memory
 *
 * KNOWN CONCRETE ILayer TYPES (current + planned)
 * ─────────────────────────────────────────────────
 *   Dense       — fully-connected; every input → every output (most general)
 *   ReLU        — activation: y = max(0, x), no parameters
 *   Sigmoid     — activation: y = 1/(1+e^-x), no parameters
 *   Conv2D      — spatial filter; local connections only (far fewer weights)
 *   BatchNorm   — normalises activations; 2 params per feature (γ, β)
 *   Dropout     — randomly zeros activations at train time; no params
 *   Embedding   — integer index → float vector (lookup table)
 *
 * Design principles
 *   • No exceptions — forward/backward return Result<Tensor>
 *   • All state lives in layers; ComputeGraph owns the execution order
 *   • Each Layer owns its parameters (weights/bias Tensors)
 *   • Dual-language: Python binding mirrors this API 1-to-1 via pybind11
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#pragma once

#include <core/Tensor.h>
#include <core/Result.h>

#include <string>
#include <string_view>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>

namespace NNSpire::core {

// ─── Parameter descriptor ────────────────────────────────────────────────────
/** A named, learnable Tensor.  Layers expose these so Optimizers can update them. */
struct Parameter {
    std::string name;
    Tensor      tensor;      ///< weight/bias data + gradient
    bool        frozen{false}; ///< if true, optimizer skips this param
};

// ─── Forward declarations (avoids circular includes) ────────────────────────
class ComputeGraph;
struct EvalTrace;

// ─── ILayer — pure abstract module ───────────────────────────────────────────
/**
 * Base class for every feed-forward module.
 *
 * Lifecycle — every layer must support these steps in this order:
 *
 *   Step | Method              | What happens
 *   -----+---------------------+--------------------------------------------------
 *     1  | construct           | Hyper-params stored (e.g. outFeatures_).
 *        |                     | Nothing allocated — inFeatures not yet known.
 *     2  | build(inputShape)   | Now we know inFeatures → allocate W and b.
 *        |                     | Idempotent: repeated calls with same shape = no-op.
 *     3  | forward(x)          | Run the NN formula (e.g. y = x @ W^T + b).
 *        |                     | Save lastInput_ — backward() will need it.
 *        |                     | Return output tensor.
 *     4  | backward(gradOut)   | Receive dLoss/dY from the layer ABOVE.
 *        |                     | Accumulate dW, db into each param's grad_ field.
 *        |                     | Return dX (= dLoss/dX) toward the layer BELOW.
 *     5  | zeroGrad()          | Reset all grad_ fields to 0.
 *        |                     | Trainer calls this before every training step.
 *
 * Important: the layer accumulates gradients ("how wrong were the weights?"),
 * but it does NOT update weight values. That is done by the Optimizer
 * after the full backward pass is complete.
 */
class ILayer {
public:
    virtual ~ILayer() = default;

    // ── Identity ─────────────────────────────────────────────────────────────
    /** Human-readable type name (e.g. "Dense", "ReLU").  Must be stable across versions. */
    virtual std::string_view typeName() const noexcept = 0;

    /** User-assigned display name (default = typeName). */
    const std::string& name() const noexcept { return name_; }
    void setName(std::string n) { name_ = std::move(n); }

    // ── Build ─────────────────────────────────────────────────────────────────
    /**
     * Allocate parameters in response to the first actual input shape.
     * Must be idempotent: repeated calls with the same shape are no-ops.
     * @param inputShape   Shape of ONE sample (without batch dimension).
     */
    virtual Result<Shape> build(const Shape& inputShape) = 0;
    bool isBuilt() const noexcept { return built_; }

    // ── Forward / backward ───────────────────────────────────────────────────
    /**
     * Run the forward pass.
     * @param x  Input tensor, shape [batch, ...inputShape].
     * @return   Output tensor on success.
     */
    virtual Result<Tensor> forward(const Tensor& x, EvalTrace* trace = nullptr) = 0;

    /**
     * Accumulate gradients from a backward pass.
     * @param gradOut  Gradient of the loss w.r.t. this layer's output, same shape as output.
     * @return         Gradient w.r.t. this layer's input (to propagate further back).
     */
    virtual Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) = 0;

    // ── Parameters ───────────────────────────────────────────────────────────
    /** All learnable parameters (including sub-layers if this is a composite). */
    virtual std::vector<Parameter*> parameters() = 0;
    virtual std::vector<const Parameter*> parameters() const = 0;

    /** Count of all learnable float values. */
    int64_t paramCount() const;

    /** Set all gradient tensors to zero. */
    void zeroGrad();

    // ── Training / inference mode ─────────────────────────────────────────────
    bool training() const noexcept { return training_; }
    virtual void setTraining(bool t) noexcept;
    void train()  noexcept { setTraining(true);  }
    void eval()   noexcept { setTraining(false); }

    // ── Trace mode (ComputeGraph overrides; default is no-op) ─────────────────
    /**
     * Enable/disable per-layer EvalTrace recording.
     * ComputeGraph overrides this to activate its trace machinery.
     * All other layers silently ignore this call.
     */
    virtual void setTraceMode(bool /*on*/) noexcept {}

    // ── Serialization helpers ─────────────────────────────────────────────────
    /** Persist layer config (hyper-params only, no weights) as key→value map. */
    virtual std::unordered_map<std::string, std::string> config() const { return {}; }

    /** KB doc reference for the Doxygen @kb alias. */
    virtual std::string_view docRef() const noexcept { return ""; }

protected:
    bool built_{false};
    bool training_{true};
    std::string name_;

    // Call from build() once weights are allocated
    void markBuilt() noexcept { built_ = true; }

    // NOTE (ADR-020): lastInput_/lastOutput_ have been removed from ILayer.
    // Layers that need them declare their own private Tensor members:
    //   Dense         — lastInput_ in Dense.h
    //   ActivationBase — ctx_ (ActivationForward) in Activations.h via functors
};

// ─── LayerPtr convenience alias ───────────────────────────────────────────────
using LayerPtr = std::shared_ptr<ILayer>;

// ─── Sequential container ────────────────────────────────────────────────────
/**
 * Ordered list of layers; forward() threads input through each layer in order.
 * The most common model building block.
 */
class Sequential : public ILayer {
public:
    std::string_view typeName() const noexcept override { return "Sequential"; }

    /** Append a layer to the end of the stack. Returns *this for chaining. */
    Sequential& add(LayerPtr layer);

    /** Convenience: construct-and-add. */
    template<typename L, typename... Args>
    Sequential& add(Args&&... args) {
        return add(std::make_shared<L>(std::forward<Args>(args)...));
    }

    const std::vector<LayerPtr>& layers() const noexcept { return layers_; }
    size_t size() const noexcept { return layers_.size(); }

    Result<Shape> build(const Shape& inputShape) override;
    Result<Tensor> forward(const Tensor& x, EvalTrace* trace = nullptr) override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr) override;

    std::vector<Parameter*>       parameters() override;
    std::vector<const Parameter*> parameters() const override;

    void setTraining(bool t) noexcept override;

private:
    std::vector<LayerPtr> layers_;
};

} // namespace NNSpire::core
