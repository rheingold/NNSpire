/* ============================================================================
 * Layer.cpp — ILayer and Sequential implementations
 * LGPL v3
 * ============================================================================ */

#include <core/Layer.h>
#include <cassert>
#include <numeric>

namespace nnstudio::core {

// ─── ILayer ──────────────────────────────────────────────────────────────────

int64_t ILayer::paramCount() const {
    int64_t total = 0;
    for (const auto* p : parameters())
        total += p->tensor.numel();
    return total;
}

void ILayer::zeroGrad() {
    for (auto* p : parameters()) {
        if (p->tensor.hasGrad())
            p->tensor.zeroGrad();
    }
}

void ILayer::setTraining(bool t) noexcept {
    training_ = t;
}

// ─── Sequential ──────────────────────────────────────────────────────────────

Sequential& Sequential::add(LayerPtr layer) {
    assert(layer && "Sequential::add: null layer");
    layers_.push_back(std::move(layer));
    return *this;
}

Result<Shape> Sequential::build(const Shape& inputShape) {
    if (layers_.empty())
        return Result<Shape>(Error{ErrorCode::InvalidArgument, "Sequential is empty"});

    Shape current = inputShape;
    for (auto& layer : layers_) {
        auto r = layer->build(current);
        if (!r.ok()) return r;
        current = r.value();
    }
    markBuilt();
    return Result<Shape>(current);
}

Result<Tensor> Sequential::forward(const Tensor& x) {
    Tensor current = x;
    for (auto& layer : layers_) {
        auto r = layer->forward(current);
        if (!r.ok()) return r;
        current = r.value();
    }
    return Result<Tensor>(current);
}

Result<Tensor> Sequential::backward(const Tensor& gradOut) {
    Tensor grad = gradOut;
    // Traverse in reverse
    for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
        auto r = (*it)->backward(grad);
        if (!r.ok()) return r;
        grad = r.value();
    }
    return Result<Tensor>(grad);
}

std::vector<Parameter*> Sequential::parameters() {
    std::vector<Parameter*> all;
    for (auto& layer : layers_) {
        auto ps = layer->parameters();
        all.insert(all.end(), ps.begin(), ps.end());
    }
    return all;
}

std::vector<const Parameter*> Sequential::parameters() const {
    std::vector<const Parameter*> all;
    for (const auto& layer : layers_) {
        auto ps = layer->parameters();
        all.insert(all.end(), ps.begin(), ps.end());
    }
    return all;
}

void Sequential::setTraining(bool t) noexcept {
    training_ = t;
    for (auto& l : layers_) l->setTraining(t);
}

} // namespace nnstudio::core
