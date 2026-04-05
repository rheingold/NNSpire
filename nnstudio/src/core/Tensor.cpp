/* ============================================================================
 * Tensor.cpp — Tensor implementation
 * LGPL v3
 * @kb: ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#include <core/Tensor.h>
#include <core/BackendRegistry.h>

#include <stdexcept>
#include <cstring>
#include <cmath>
#include <sstream>

namespace nnstudio::core {

// ---------------------------------------------------------------------------
// Private constructor (shared-buffer view constructor)
// ---------------------------------------------------------------------------
Tensor::Tensor(Shape shape, Strides strides, DType dtype, Device device,
               std::shared_ptr<float[]> data)
    : shape_(std::move(shape))
    , strides_(std::move(strides))
    , dtype_(dtype)
    , device_(device)
    , numel_(shapeNumel(shape_))
    , data_(std::move(data))
{}

// ---------------------------------------------------------------------------
// Public constructor — allocates owned buffer
// ---------------------------------------------------------------------------
Tensor::Tensor(Shape shape, DType dtype, Device device, bool requires_grad)
    : shape_(std::move(shape))
    , strides_(rowMajorStrides(shape_))
    , dtype_(dtype)
    , device_(device)
    , requires_grad_(requires_grad)
    , numel_(shapeNumel(shape_))
    , data_(std::shared_ptr<float[]>(new float[numel_ > 0 ? numel_ : 1]))
{}

// ---------------------------------------------------------------------------
// Factory methods
// ---------------------------------------------------------------------------
Tensor Tensor::full(Shape shape, float value, DType dtype, Device dev) {
    Tensor t(std::move(shape), dtype, dev, false);
    for (int64_t i = 0; i < t.numel_; ++i) t.data_[i] = value;
    return t;
}

Tensor Tensor::zeros(Shape shape, DType dtype, Device dev) {
    return Tensor::full(std::move(shape), 0.0f, dtype, dev);
}

Tensor Tensor::ones(Shape shape, DType dtype, Device dev) {
    return Tensor::full(std::move(shape), 1.0f, dtype, dev);
}

Tensor Tensor::fromData(const float* data, Shape shape, Device dev) {
    Tensor t(shape, DType::Float32, dev, false);
    std::memcpy(t.data_.get(), data, static_cast<size_t>(t.numel_) * sizeof(float));
    return t;
}

Tensor Tensor::fromData(std::initializer_list<float> data, Shape shape, Device dev) {
    return fromData(data.begin(), std::move(shape), dev);
}

Tensor Tensor::clone() const {
    Tensor t(shape_, dtype_, device_, requires_grad_);
    std::memcpy(t.data_.get(), data_.get(), static_cast<size_t>(numel_) * sizeof(float));
    return t;
}

// ---------------------------------------------------------------------------
// Indexed access
// ---------------------------------------------------------------------------
float Tensor::at(std::initializer_list<int64_t> idx) const {
    assert(static_cast<int>(idx.size()) == ndim());
    int64_t offset = 0;
    auto it = idx.begin();
    for (int i = 0; i < ndim(); ++i, ++it)
        offset += *it * strides_[i];
    return data_[offset];
}

float& Tensor::at(std::initializer_list<int64_t> idx) {
    assert(static_cast<int>(idx.size()) == ndim());
    int64_t offset = 0;
    auto it = idx.begin();
    for (int i = 0; i < ndim(); ++i, ++it)
        offset += *it * strides_[i];
    return data_[offset];
}

// ---------------------------------------------------------------------------
// Gradient
// ---------------------------------------------------------------------------
Tensor& Tensor::grad() {
    assert(grad_ != nullptr && "No gradient — call backward() first or check hasGrad()");
    return *grad_;
}

const Tensor& Tensor::grad() const {
    assert(grad_ != nullptr);
    return *grad_;
}

void Tensor::zeroGrad() {
    if (grad_) {
        for (int64_t i = 0; i < numel_; ++i) grad_->data_[i] = 0.0f;
    }
}

void Tensor::accumulateGrad(const Tensor& g) {
    assert(g.shape_ == shape_);
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(shape_, dtype_, device_, false);
        for (int64_t i = 0; i < numel_; ++i) grad_->data_[i] = 0.0f;
    }
    for (int64_t i = 0; i < numel_; ++i)
        grad_->data_[i] += g.data_[i];
}

// ---------------------------------------------------------------------------
// Reshape / view
// ---------------------------------------------------------------------------
Result<Tensor> Tensor::reshape(Shape newShape) const {
    // Handle -1 (infer)
    int64_t inferIdx = -1;
    int64_t known = 1;
    for (int64_t i = 0; i < static_cast<int64_t>(newShape.size()); ++i) {
        if (newShape[i] == -1) {
            if (inferIdx >= 0)
                return err(ErrorCode::InvalidArgument, "Only one -1 allowed in reshape");
            inferIdx = i;
        } else {
            known *= newShape[i];
        }
    }
    if (inferIdx >= 0) {
        if (numel_ % known != 0)
            return err(ErrorCode::InvalidArgument, "Cannot infer dimension in reshape");
        newShape[inferIdx] = numel_ / known;
    }
    if (shapeNumel(newShape) != numel_)
        return err(ErrorCode::ShapeMismatch, "Reshape numel mismatch");

    if (!isContiguous())
        return err(ErrorCode::InvalidArgument, "reshape requires contiguous tensor; clone first");

    return Tensor(newShape, rowMajorStrides(newShape), dtype_, device_, data_);
}

Result<Tensor> Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0 || dim0 >= ndim() || dim1 < 0 || dim1 >= ndim())
        return err(ErrorCode::InvalidArgument, "transpose dim out of range");
    return backend().transpose(*this, dim0, dim1);
}

// ---------------------------------------------------------------------------
// Arithmetic operators — dispatch to active backend
// ---------------------------------------------------------------------------
Tensor Tensor::operator+(const Tensor& rhs) const { return backend().add(*this, rhs); }
Tensor Tensor::operator-(const Tensor& rhs) const { return backend().sub(*this, rhs); }
Tensor Tensor::operator*(const Tensor& rhs) const { return backend().mul(*this, rhs); }
Tensor Tensor::operator/(const Tensor& rhs) const { return backend().div_(*this, rhs); }
Tensor Tensor::operator-()                  const { return backend().neg(*this); }

Tensor Tensor::operator+(float s) const { return backend().addScalar(*this, s); }
Tensor Tensor::operator-(float s) const { return backend().subScalar(*this, s); }
Tensor Tensor::operator*(float s) const { return backend().mulScalar(*this, s); }
Tensor Tensor::operator/(float s) const { return backend().divScalar(*this, s); }

// ---------------------------------------------------------------------------
// Free functions — dispatch to active backend
// ---------------------------------------------------------------------------
Tensor matmul(const Tensor& a, const Tensor& b)                     { return backend().matmul(a, b); }
Tensor sum   (const Tensor& t, int dim, bool keepdim)               { return backend().sum(t, dim, keepdim); }
Tensor mean  (const Tensor& t, int dim, bool keepdim)               { return backend().mean(t, dim, keepdim); }
Tensor max   (const Tensor& t, int dim, bool keepdim)               { return backend().max(t, dim, keepdim); }
Tensor exp   (const Tensor& t)                                       { return backend().exp(t); }
Tensor log   (const Tensor& t)                                       { return backend().log(t); }
Tensor sqrt  (const Tensor& t)                                       { return backend().sqrt(t); }
Tensor clamp (const Tensor& t, float lo, float hi)                  { return backend().clamp(t, lo, hi); }
Tensor abs   (const Tensor& t)                                       { return backend().abs(t); }
Tensor cat   (const std::vector<Tensor>& tensors, int dim)          { return backend().cat(tensors, dim); }

Tensor broadcastAdd(const Tensor& a, const Tensor& b) {
    // Simple broadcast: if shapes match, use add directly.
    // Otherwise delegate to backend which handles broadcasting.
    return backend().add(a, b);
}

} // namespace nnstudio::core
