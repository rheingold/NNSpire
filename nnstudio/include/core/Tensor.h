#pragma once
/* ============================================================================
 * Tensor.h — primary data container for NNStudio
 * LGPL v3
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 *
 * Design notes:
 *  - Float32 is the primary compute dtype for Phase 1.
 *  - Tensor is a copyable value type; copies share the underlying buffer
 *    (shallow copy, like shared_ptr). Use Tensor::clone() for deep copy.
 *  - All compute operations are dispatched through IBackend (never call
 *    raw BLAS/CUDA here). This keeps backend swapping transparent.
 *  - requires_grad: when true, the ComputeGraph records this tensor's creation.
 *    grad() returns the accumulated gradient after backward().
 * ============================================================================ */

#include <core/DType.h>
#include <core/Device.h>
#include <core/Result.h>

#include <vector>
#include <memory>
#include <numeric>
#include <functional>
#include <initializer_list>
#include <cassert>
#include <cstring>

namespace nnstudio::core {

// ---------------------------------------------------------------------------
// Shape helpers
// ---------------------------------------------------------------------------
using Shape   = std::vector<int64_t>;
using Strides = std::vector<int64_t>;

/// Compute row-major (C-order) strides for a given shape.
inline Strides rowMajorStrides(const Shape& shape) {
    if (shape.empty()) return {};
    Strides s(shape.size());
    s.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        s[i] = s[i + 1] * shape[i + 1];
    return s;
}

/// Total number of elements in a shape.
inline int64_t shapeNumel(const Shape& shape) {
    if (shape.empty()) return 0;
    return std::accumulate(shape.begin(), shape.end(),
                           int64_t{1}, std::multiplies<int64_t>{});
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------
//
// WALKTHROUGH — Chapter 1: The Memory Model
//
// A Tensor is fundamentally simple: a flat heap-allocated float[] buffer
// plus a handful of integers that describe how to interpret it.
//
// The key members in the private section:
//
//   data_    — the actual numbers, owned via shared_ptr<float[]>.
//              shared_ptr means reshape/transpose can return a *view* that
//              shares the same buffer without copying (zero-copy slicing).
//
//   shape_   — e.g. {2, 3} means 2 rows, 3 columns.  A flat array of dims.
//
//   strides_ — the step sizes that translate a logical [d0, d1, d2, ...]
//              index into a flat data_[] offset.  One stride per dimension.
//
//              The formula (always, for any rank):
//                offset = i0*strides_[0] + i1*strides_[1] + i2*strides_[2] + ...
//
//              For a 2-D matrix encoded as one flat vector (the common case),
//              there are exactly two stride numbers — think of them as questions:
//                strides_[0] — "how many elements do I skip in data_[] to move
//                               down by one ROW?"
//                strides_[1] — "how many elements do I skip in data_[] to move
//                               right by one COLUMN?"
//
//              Example — shape {3 rows, 2 cols}, strides {2, 1}:
//                data_ = [ 10, 20, 30, 40, 50, 60 ]
//                           [0] [1] [2] [3] [4] [5]
//
//                  logical grid        data_ index
//                  [row0,col0]=10  →  0*2 + 0*1 = 0  ✓
//                  [row0,col1]=20  →  0*2 + 1*1 = 1  ✓
//                  [row1,col0]=30  →  1*2 + 0*1 = 2  ✓
//                  [row1,col1]=40  →  1*2 + 1*1 = 3  ✓
//
//              The rule for row-major (C-order) strides given shape {d0,d1,...}:
//                strides_[k] = product of all dimensions AFTER k
//                strides_[last] = 1   (last dim is always contiguous)
//
//              Examples:
//                shape {3, 2}       → strides {2, 1}
//                shape {4, 3, 2}    → strides {6, 2, 1}
//                shape {5, 4, 3, 2} → strides {24, 6, 2, 1}
//
//              Strides have nothing to do with computation (W*x+b).
//              They are purely a navigation map: "given a logical index,
//              which float in data_[] is that?"  The result of any
//              computation is always a brand-new Tensor with its own
//              data_[] buffer and its own strides.
//
//              Transpose swaps the stride numbers (and shape) without
//              touching the buffer — see "Transpose is free" block below.
//
//   numel_   — cached product of all shape dims (avoids recomputing).
//              numel_ = shape_[0] * shape_[1] * ... * shape_[N-1]
//              Pure optimisation — for large tensors (e.g. {768, 3072} in a
//              transformer = 2,359,296 elements) recomputing this product on
//              every optimizer step, loss sum, or zeroGrad() call would be
//              costly.  Computed once at construction, read everywhere.
//              Conceptually: "how many floats live in this tensor in total?"
//
//   grad_    — the gradient: slope of the loss surface at the current weight.
//              More precisely: ∂Loss/∂W[i][j] for each element of this tensor.
//              In plain terms: "the slope of the tangent of the curve of the
//              loss-change function, evaluated at the current weight value."
//              Česky: "sklon tečny křivky funkce změny chyby v jejím bodě aktuální
//              hodnoty váhy (nebo biasu)."
//
//              It is null until accumulateGrad() is first called (lazy alloc).
//              Only learnable parameters (W, b) carry a grad_ — intermediate
//              tensors (inputs x, outputs y) do not.
//
//              It does NOT tell you where the weight should end up.
//              It only tells you which direction is downhill on the loss surface
//              from where you currently stand.  A steep positive grad_ means
//              "this weight is too high, loss is rising — step left (decrease)."
//              A near-zero grad_ means "you're near a minimum — tiny step."
//
//              The optimizer (Adam/SGD) subtracts a scaled version each step:
//                W[i][j] -= learning_rate × grad_[i][j]    (SGD, simplest form)
//              Repeat across millions of weights, thousands of times = training.
//
//              Use hasGrad() to check, grad() to access.
//              Intentionally simpler than PyTorch's separate Variable type —
//              gradient lives on the same object as the data it describes.
//
// Computation is NOT on Tensor: Tensor is dumb data storage.
// All math is delegated to IBackend (see BackendRegistry.h / CpuBackend.cpp).
//
// ── IMPORTANT: Tensor is NOT specific to weights ─────────────────────────────
// The same Tensor type is used for EVERYTHING:
//   • weight matrices  W  shape {outFeatures, inFeatures}
//   • bias vectors     b  shape {outFeatures}
//   • layer inputs     x  shape {batch, inFeatures}
//   • layer outputs    y  shape {batch, outFeatures}
//   • gradients       dW  shape {outFeatures, inFeatures}  (same as W)
// It is a generic N-dimensional float array.  Role is determined by context,
// not by the type.
//
// ── How that lives in memory (Dense 2→3 example) ─────────────────────────────
// Input x, shape {1,2} — one sample, two features:
//   data_ = [ 0.6, 0.4 ]   flat, left to right
//             x₁   x₂
//
// Weight W, shape {3,2} — 3 neurons, each with 2 weights:
//   data_ = [ w₀₀, w₀₁,   ← neuron 0
//             w₁₀, w₁₁,   ← neuron 1
//             w₂₀, w₂₁ ]  ← neuron 2
//   strides_ = {2, 1}  — skip 2 floats to reach the next neuron's row,
//                         skip 1 float to reach the next weight in a row.
//
// Output y = x @ W^T + b, shape {1,3}:
//   y[0] = 0.6*w₀₀ + 0.4*w₀₁ + b[0]
//   y[1] = 0.6*w₁₀ + 0.4*w₁₁ + b[1]
//   y[2] = 0.6*w₂₀ + 0.4*w₂₁ + b[2]
// Each of those is one neuron's output — a weighted sum of all inputs plus bias.
//
// ── Batching — why the first dimension exists ─────────────────────────────────
// Processing one sample at a time is wasteful.  Instead we stack B samples
// into a single input tensor, shape {B, inFeatures}:
//   data_ = [ x₁⁽⁰⁾, x₂⁽⁰⁾,   ← sample 0
//             x₁⁽¹⁾, x₂⁽¹⁾,   ← sample 1  ...
//             x₁⁽ᴮ⁾, x₂⁽ᴮ⁾ ]
// The matmul x @ W^T now produces {B, outFeatures} — all B outputs in one
// Eigen (or CUDA) call.  That is the entire purpose of the leading batch dim.
// In the XOR test batch=1 always, so shapes are {1,2}→{1,4}→{1,1}.
//
// ── Transpose is free (strides only) ─────────────────────────────────────────
// Dense::forward needs W^T (shape {inFeatures, outFeatures}).
// Instead of copying data, we just swap strides_:
//   W   shape={3,2}  strides={2,1}
//   W^T shape={2,3}  strides={1,2}   ← same data_, different navigation
// The shared_ptr<float[]> means W and W^T share the same heap allocation.
// Not a single float is moved.  This is why strides_ exists.
// ---------------------------------------------------------------------------
class Tensor {
public:
    // ------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------

    /// Default-construct an empty (null/invalid) Tensor — allows member-variable declarations.
    Tensor() = default;

    /// Construct an uninitialised tensor with the given shape.
    explicit Tensor(Shape shape,
                    DType  dtype         = DType::Float32,
                    Device device        = Device::CPU,
                    bool   requires_grad = false);

    /// Construct and fill with a scalar value.
    static Tensor full(Shape shape, float value,
                       DType dtype = DType::Float32, Device dev = Device::CPU);

    /// Construct filled with zeros.
    static Tensor zeros(Shape shape, DType dtype = DType::Float32, Device dev = Device::CPU);

    /// Construct filled with ones.
    static Tensor ones (Shape shape, DType dtype = DType::Float32, Device dev = Device::CPU);

    /// Construct from raw float data (data is copied).
    static Tensor fromData(const float* data, Shape shape,
                           Device dev = Device::CPU);

    /// Construct from initializer list (data is copied).
    static Tensor fromData(std::initializer_list<float> data, Shape shape,
                           Device dev = Device::CPU);

    /// Deep-copy this tensor (new independent buffer).
    Tensor clone() const;

    // ------------------------------------------------------------------
    // Properties
    // ------------------------------------------------------------------
    const Shape&   shape()        const noexcept { return shape_; }
    const Strides& strides()      const noexcept { return strides_; }
    DType          dtype()        const noexcept { return dtype_; }
    Device         device()       const noexcept { return device_; }
    bool           requiresGrad() const noexcept { return requires_grad_; }
    int64_t        numel()        const noexcept { return numel_; }
    int            ndim()         const noexcept { return static_cast<int>(shape_.size()); }

    bool isContiguous() const noexcept { return strides_ == rowMajorStrides(shape_); }

    void setRequiresGrad(bool v) noexcept { requires_grad_ = v; }

    // ------------------------------------------------------------------
    // Raw data access (CPU float32 only in Phase 1)
    // ------------------------------------------------------------------
    float*       data()       noexcept { return data_.get(); }
    const float* data() const noexcept { return data_.get(); }

    float  at(std::initializer_list<int64_t> idx) const;
    float& at(std::initializer_list<int64_t> idx);

    // Flat (linear) index access — respects strides
    float  flat(int64_t i) const noexcept { return data_[i]; }
    float& flat(int64_t i)       noexcept { return data_[i]; }

    // ------------------------------------------------------------------
    // Gradient
    // ------------------------------------------------------------------
    //
    // Design: no separate autograd graph.  During backward() each layer
    // calls accumulateGrad(dParam) on its weight/bias tensors, then returns
    // dInput so the previous layer can continue the chain.  The optimizer
    // later reads grad() and updates the data buffer in-place, then zeroGrad()
    // clears it ready for the next step.
    //
    bool        hasGrad() const noexcept { return grad_ != nullptr; }
    Tensor&     grad();
    const Tensor& grad() const;
    void         zeroGrad();
    void         accumulateGrad(const Tensor& g);   // grad_ += g (allocates lazily)

    // ------------------------------------------------------------------
    // Reshape / view (returns a new Tensor sharing the same buffer)
    // ------------------------------------------------------------------
    Result<Tensor> reshape(Shape newShape) const;
    Result<Tensor> transpose(int dim0, int dim1) const;  // swaps two dims

    // ------------------------------------------------------------------
    // Comparisons (shape only)
    // ------------------------------------------------------------------
    bool sameShape(const Tensor& other) const noexcept { return shape_ == other.shape_; }
    bool sameDevice(const Tensor& other) const noexcept { return device_ == other.device_; }

    // ------------------------------------------------------------------
    // Operators — dispatch through BackendRegistry (defined in Tensor.cpp)
    // ------------------------------------------------------------------
    Tensor operator+(const Tensor& rhs) const;
    Tensor operator-(const Tensor& rhs) const;
    Tensor operator*(const Tensor& rhs) const;   // element-wise
    Tensor operator/(const Tensor& rhs) const;
    Tensor operator-()                  const;   // negation

    // Scalar broadcast
    Tensor operator+(float s) const;
    Tensor operator-(float s) const;
    Tensor operator*(float s) const;
    Tensor operator/(float s) const;

    // ------------------------------------------------------------------
    // Assigned from friend free functions (see ops/)
    // ------------------------------------------------------------------
    // matmul, sum, mean, etc. are free functions — see TensorOps.h

private:
    Shape                      shape_;
    Strides                    strides_;
    DType                      dtype_   { DType::Float32 };
    Device                     device_  { Device::CPU };
    bool                       requires_grad_ { false };
    int64_t                    numel_         { 0 };
    std::shared_ptr<float[]>   data_;
    std::shared_ptr<Tensor>    grad_;          // allocated lazily

    // Private constructor used by reshape/views (shares buffer)
    Tensor(Shape shape, Strides strides, DType dtype, Device device,
           std::shared_ptr<float[]> data);

};

// ---------------------------------------------------------------------------
// Free-function tensor operations (all dispatch through BackendRegistry)
// @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#matrix-ops
// ---------------------------------------------------------------------------
Tensor matmul(const Tensor& a, const Tensor& b);
Tensor sum   (const Tensor& t, int dim = -1, bool keepdim = false);
Tensor mean  (const Tensor& t, int dim = -1, bool keepdim = false);
Tensor max   (const Tensor& t, int dim = -1, bool keepdim = false);
Tensor exp   (const Tensor& t);
Tensor log   (const Tensor& t);
Tensor sqrt  (const Tensor& t);
Tensor clamp (const Tensor& t, float lo, float hi);
Tensor abs   (const Tensor& t);

/// Concatenate along dim
Tensor cat(const std::vector<Tensor>& tensors, int dim);

/// Broadcast-compatible element-wise addition (shapes must be broadcastable)
Tensor broadcastAdd(const Tensor& a, const Tensor& b);

} // namespace nnstudio::core
