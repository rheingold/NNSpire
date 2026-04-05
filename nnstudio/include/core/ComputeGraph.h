/* ============================================================================
 * ComputeGraph.h — directed-acyclic-graph execution engine + autograd
 * LGPL v3
 *
 * THEORY (Chapter 5: Computational Graphs)
 * ─────────────────────────────────────────
 * A ComputeGraph is the runtime record of HOW a forward pass was computed.
 * After recording, backward() replays the graph in reverse topological order,
 * feeding gradients from later nodes back to earlier ones.
 *
 * WHY DO WE NEED THIS?
 * Sequential's backward() works for *chains* (each layer has exactly one
 * predecessor). ComputeGraph supports arbitrary DAGs:
 *
 *   x₁ ──► Layer A ──┐
 *                     ├──► Layer C ──► loss
 *   x₂ ──► Layer B ──┘
 *
 * In this graph, C receives two inputs (from A and B). Sequential cannot
 * model this; ComputeGraph can.
 *
 * KEY CONCEPTS
 * ─────────────
 *   NodeId    — stable integer handle for a recorded tensor output.
 *               Created by record(); returned to the caller.
 *
 *   OpRecord  — one entry in the forward tape:
 *                 • layer*      — which ILayer produced this output
 *                 • inputId     — NodeId of the input to that layer
 *                 • outputId    — NodeId of the output (= return value of record())
 *
 *   backward  — iterate ops in reverse order:
 *                 each op calls layer->backward(grad_of_output)
 *                 accumulates the returned grad_of_input
 *
 * SIMPLIFICATION FOR PHASE 1
 * ───────────────────────────
 * The Phase 1 graph is still linear-chain compatible.  True multi-input
 * fusion (residual connections, attention skip paths) is planned for Phase 4
 * when ComputeGraph grows a multi-input OpRecord variant.
 * For now each OpRecord has exactly ONE input NodeId.
 *
 * INTEGRATION POINT
 * ──────────────────
 * ComputeGraph implements ILayer so it can be handed to Trainer without changes:
 *
 *   ComputeGraph g;
 *   auto h = g.record(dense, inputNode);     // returns NodeId
 *   auto o = g.record(relu, h);
 *   Trainer trainer(g, bce, adam);           ← identical to Sequential usage
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#compute-graph
 * ============================================================================ */

#pragma once

#include <core/Layer.h>
#include <core/Result.h>
#include <core/Tensor.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace nnstudio::core {

// ─── Node identifier ─────────────────────────────────────────────────────────
using NodeId = uint32_t;
static constexpr NodeId kInvalidNode = UINT32_MAX;

// ─── Single entry in the forward tape ────────────────────────────────────────
struct OpRecord {
    ILayer*  layer;     ///< non-owning; ComputeGraph does NOT own layers
    NodeId   inputId;   ///< NodeId of the tensor fed into this layer
    NodeId   outputId;  ///< NodeId of the tensor produced by this layer
};

// ─── EvalTrace — per-layer snapshot for debugging/inspection ─────────────────
/**
 * When trace mode is enabled (Trainer::setTraceMode(true)), ComputeGraph
 * populates one EvalTrace per recorded op after each forward pass.
 *
 * All tensors are DEEP COPIES (not views) of the live compute tensors so the
 * caller may inspect them after the backward pass without data races.
 */
struct EvalTrace {
    std::string layerName;   ///< ILayer::name() snapshot
    std::string layerType;   ///< ILayer::typeName() snapshot
    Tensor      input;       ///< deep copy of the layer's input
    Tensor      output;      ///< deep copy of the layer's output
    Tensor      gradOutput;  ///< deep copy of the gradient arriving at this layer's output
    Tensor      gradInput;   ///< deep copy of the gradient propagated toward its input
};

// ─── ComputeGraph ─────────────────────────────────────────────────────────────
/**
 * Directed-acyclic-graph execution engine.
 *
 * Usage pattern:
 *   ComputeGraph g;
 *
 *   // 1. Record layers in forward order during model construction:
 *   NodeId h0 = g.input();                 // declare the graph input
 *   NodeId h1 = g.record(&dense1,  h0);    // dense1.forward(x) → h1
 *   NodeId h2 = g.record(&relu,    h1);
 *   NodeId h3 = g.record(&dense2,  h2);
 *   g.setOutput(h3);
 *
 *   // 2. Use as ILayer (hand to Trainer):
 *   Trainer trainer(g, loss, optimizer);
 *   trainer.train(data, epochs, callbacks);
 *
 * Alternatively, record() can be called lazily during the first forward pass.
 */
class ComputeGraph : public ILayer {
public:
    ComputeGraph() = default;

    // ── Graph construction ───────────────────────────────────────────────────

    /** Allocate the special "input" node (NodeId 0 is reserved for it).
     *  Only call once. Returns 0 always. */
    NodeId input();

    /** Record a layer operation: inputId → layer → new NodeId.
     *  Appends the op to the tape. Call in forward order.
     *  The returned NodeId should be passed as inputId to the next layer. */
    NodeId record(ILayer* layer, NodeId inputId);

    /** Mark which NodeId is the graph output (the tensor Trainer will see). */
    void setOutput(NodeId id) noexcept { outputId_ = id; }
    NodeId outputNodeId() const noexcept { return outputId_; }

    // ── Trace mode ───────────────────────────────────────────────────────────

    /** Enable/disable per-op EvalTrace collection. Off by default. */
    void setTraceMode(bool on) noexcept { traceMode_ = on; }
    bool traceMode() const noexcept { return traceMode_; }

    /** Traces collected during the last forward+backward pass.
     *  Empty when traceMode() == false. */
    const std::vector<EvalTrace>& traces() const noexcept { return traces_; }

    // ── ILayer interface ─────────────────────────────────────────────────────

    std::string_view typeName() const noexcept override { return "ComputeGraph"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#compute-graph";
    }

    /**
     * build() propagates the input shape through every layer in recording order,
     * calling each layer's build() with the output shape of its predecessor.
     */
    Result<Shape> build(const Shape& inputShape) override;

    /**
     * forward() executes each recorded op in recording order.
     * The intermediate tensors are stored in nodeValues_ for backward().
     */
    Result<Tensor> forward(const Tensor& x) override;

    /**
     * backward() iterates recorded ops in REVERSE, calling each layer's backward()
     * and propagating the gradient toward the graph input (not used by Trainer,
     * but returned for completeness / nested-graph support).
     */
    Result<Tensor> backward(const Tensor& gradOut) override;

    /** Flattens parameters from all recorded layers (de-duplicated). */
    std::vector<Parameter*>       parameters() override;
    std::vector<const Parameter*> parameters() const override;

    /** Propagates setTraining() to all recorded layers. */
    void setTraining(bool t) noexcept override;

private:
    std::vector<OpRecord> tape_;          ///< forward tape in recording order
    std::unordered_map<NodeId, Tensor> nodeValues_; ///< live tensors per node (forward)
    NodeId  nextId_   {0};
    NodeId  outputId_ {kInvalidNode};
    bool    traceMode_{false};
    std::vector<EvalTrace> traces_;

    NodeId allocNode() noexcept { return nextId_++; }
};

} // namespace nnstudio::core
