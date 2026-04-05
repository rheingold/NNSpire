/* ============================================================================
 * ComputeGraph.cpp — directed-acyclic-graph engine implementation
 * LGPL v3
 * ============================================================================ */

#include <core/ComputeGraph.h>
#include <algorithm>
#include <cassert>

namespace nnstudio::core {

// ─── Graph construction ───────────────────────────────────────────────────────

NodeId ComputeGraph::input() {
    NodeId id = allocNode();          // will be 0 on first call
    nodeValues_[id] = Tensor{};       // placeholder; filled by forward()
    return id;
}

NodeId ComputeGraph::record(ILayer* layer, NodeId inputId) {
    assert(layer != nullptr);
    NodeId outId = allocNode();
    tape_.push_back({layer, inputId, outId});
    if (outputId_ == kInvalidNode) {
        outputId_ = outId;            // auto-track last node as default output
    } else {
        outputId_ = outId;            // keep updating so last record = output
    }
    return outId;
}

// ─── ILayer::build ────────────────────────────────────────────────────────────

Result<Shape> ComputeGraph::build(const Shape& inputShape) {
    if (tape_.empty()) {
        markBuilt();
        return Result<Shape>{inputShape};
    }

    Shape current = inputShape;
    for (auto& op : tape_) {
        auto r = op.layer->build(current);
        if (!r.ok()) return r;
        current = r.value();
    }
    markBuilt();
    return Result<Shape>{current};
}

// ─── ILayer::forward ─────────────────────────────────────────────────────────

Result<Tensor> ComputeGraph::forward(const Tensor& x) {
    if (tape_.empty()) return Result<Tensor>{x};

    if (traceMode_) traces_.clear();

    // The input node id is tape_[0].inputId (== 0 from input() call).
    // Store x into nodeValues_ for that id.
    NodeId inputNodeId = tape_.front().inputId;
    nodeValues_[inputNodeId] = x;

    for (auto& op : tape_) {
        const Tensor& in = nodeValues_[op.inputId];

        // Ensure layer is built
        if (!op.layer->isBuilt()) {
            Shape s  = in.shape();
            // Remove batch dim: build expects single-sample shape
            Shape ss(s.begin() + 1, s.end());
            if (ss.empty()) ss = s;
            auto br = op.layer->build(ss);
            if (!br.ok()) return Result<Tensor>{br.error()};
        }

        auto outR = op.layer->forward(in);
        if (!outR.ok()) return outR;

        nodeValues_[op.outputId] = outR.value();

        if (traceMode_) {
            EvalTrace tr;
            tr.layerName = op.layer->name();
            tr.layerType = std::string{op.layer->typeName()};
            tr.input     = in.clone();
            tr.output    = outR.value().clone();
            // gradOutput / gradInput will be filled in backward()
            traces_.push_back(std::move(tr));
        }
    }

    return Result<Tensor>{nodeValues_[outputId_]};
}

// ─── ILayer::backward ────────────────────────────────────────────────────────

Result<Tensor> ComputeGraph::backward(const Tensor& gradOut) {
    if (tape_.empty()) return Result<Tensor>{gradOut};

    // Seed: gradient of the last node's output
    std::unordered_map<NodeId, Tensor> gradMap;
    gradMap[outputId_] = gradOut;

    // Reverse traversal
    for (int i = static_cast<int>(tape_.size()) - 1; i >= 0; --i) {
        const auto& op        = tape_[static_cast<size_t>(i)];
        const Tensor& gradOfOut = gradMap[op.outputId];

        auto dInR = op.layer->backward(gradOfOut);
        if (!dInR.ok()) return dInR;

        // Accumulate (multi-path graphs would add here; single path = assign)
        auto it = gradMap.find(op.inputId);
        if (it != gradMap.end()) {
            // accumulate: grad += dInR  (for future multi-input support)
            // Phase 1 simplification: just overwrite (single path)
            it->second = dInR.value();
        } else {
            gradMap[op.inputId] = dInR.value();
        }

        if (traceMode_ && i < static_cast<int>(traces_.size())) {
            traces_[static_cast<size_t>(i)].gradOutput = gradOfOut.clone();
            traces_[static_cast<size_t>(i)].gradInput  = dInR.value().clone();
        }
    }

    // Return gradient w.r.t. graph input
    NodeId inputNodeId = tape_.front().inputId;
    auto it = gradMap.find(inputNodeId);
    if (it != gradMap.end()) return Result<Tensor>{it->second};
    return Result<Tensor>(Error{ErrorCode::InvalidArgument, "ComputeGraph::backward: no grad for input node"});
}

// ─── parameters ─────────────────────────────────────────────────────────────

std::vector<Parameter*> ComputeGraph::parameters() {
    std::vector<Parameter*> all;
    for (auto& op : tape_) {
        for (auto* p : op.layer->parameters()) {
            // de-duplicate (same layer might appear twice in theory)
            bool found = false;
            for (auto* q : all) { if (q == p) { found = true; break; } }
            if (!found) all.push_back(p);
        }
    }
    return all;
}

std::vector<const Parameter*> ComputeGraph::parameters() const {
    std::vector<const Parameter*> all;
    for (const auto& op : tape_) {
        for (const auto* p : op.layer->parameters()) {
            bool found = false;
            for (const auto* q : all) { if (q == p) { found = true; break; } }
            if (!found) all.push_back(p);
        }
    }
    return all;
}

// ─── setTraining ────────────────────────────────────────────────────────────

void ComputeGraph::setTraining(bool t) noexcept {
    training_ = t;
    for (auto& op : tape_) op.layer->setTraining(t);
}

} // namespace nnstudio::core
