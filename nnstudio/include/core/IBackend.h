#pragma once
/* ============================================================================
 * IBackend.h — abstract compute backend interface
 * LGPL v3
 *
 * All heavy compute is dispatched through this interface; Tensor never calls
 * BLAS / cuBLAS / Eigen directly.  Swapping backends is a single
 * BackendRegistry::setActive() call.
 * ============================================================================ */

#include <core/Tensor.h>
#include <core/Result.h>
#include <string_view>

namespace nnstudio::core {

class IBackend {
public:
    virtual ~IBackend() = default;

    // ------------------------------------------------------------------
    // Identity
    // ------------------------------------------------------------------
    virtual std::string_view name() const noexcept = 0;
    virtual Device           device() const noexcept = 0;

    // ------------------------------------------------------------------
    // Memory
    // ------------------------------------------------------------------
    /// Allocate a raw float buffer of `count` elements on this device.
    virtual std::shared_ptr<float[]> alloc(int64_t count) = 0;

    /// Copy tensor to this backend's device (no-op if already on device).
    virtual Tensor to(const Tensor& t) = 0;

    // ------------------------------------------------------------------
    // BLAS-level ops
    // @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#matrix-ops
    // ------------------------------------------------------------------
    /// Matrix multiplication: C = A @ B
    /// A: [M, K], B: [K, N]  →  C: [M, N]
    /// Also handles batched: A: [B, M, K], B: [B, K, N] → [B, M, N]
    virtual Tensor matmul(const Tensor& a, const Tensor& b) = 0;

    // ------------------------------------------------------------------
    // Element-wise ops
    // ------------------------------------------------------------------
    virtual Tensor add (const Tensor& a, const Tensor& b) = 0;
    virtual Tensor sub (const Tensor& a, const Tensor& b) = 0;
    virtual Tensor mul (const Tensor& a, const Tensor& b) = 0;
    virtual Tensor div_(const Tensor& a, const Tensor& b) = 0; // div_ avoids keyword clash

    virtual Tensor addScalar(const Tensor& a, float s) = 0;
    virtual Tensor subScalar(const Tensor& a, float s) = 0;
    virtual Tensor mulScalar(const Tensor& a, float s) = 0;
    virtual Tensor divScalar(const Tensor& a, float s) = 0;
    virtual Tensor neg      (const Tensor& a)          = 0;

    virtual Tensor exp  (const Tensor& a) = 0;
    virtual Tensor log  (const Tensor& a) = 0;
    virtual Tensor sqrt (const Tensor& a) = 0;
    virtual Tensor clamp(const Tensor& a, float lo, float hi) = 0;
    virtual Tensor abs  (const Tensor& a) = 0;

    // ------------------------------------------------------------------
    // Reduction ops
    // ------------------------------------------------------------------
    /// Sum along dim (-1 = global sum → scalar tensor)
    virtual Tensor sum (const Tensor& a, int dim, bool keepdim) = 0;
    /// Mean along dim (-1 = global mean)
    virtual Tensor mean(const Tensor& a, int dim, bool keepdim) = 0;
    /// Max along dim (-1 = global max)
    virtual Tensor max (const Tensor& a, int dim, bool keepdim) = 0;

    // ------------------------------------------------------------------
    // Shape ops (no compute, but backend may need to adjust memory layout)
    // ------------------------------------------------------------------
    virtual Tensor reshape  (const Tensor& a, const Shape& newShape) = 0;
    virtual Tensor transpose(const Tensor& a, int dim0, int dim1)    = 0;
    virtual Tensor cat      (const std::vector<Tensor>& tensors, int dim) = 0;

    // ------------------------------------------------------------------
    // Synchronisation (for async backends like CUDA)
    // ------------------------------------------------------------------
    virtual void sync() {}
};

} // namespace nnstudio::core
