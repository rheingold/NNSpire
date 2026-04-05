/* ============================================================================
 * CpuBackend.cpp — CPU compute backend using Eigen for BLAS ops
 * LGPL v3
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#matrix-ops
 * ============================================================================ */

#include <builtin/backends/CpuBackend.h>
#include <core/BackendRegistry.h>

#include <Eigen/Dense>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <limits>
#include <cstring>
#include <functional>
#include <stdexcept>

using namespace nnstudio::core;

namespace nnstudio {
namespace builtin {
namespace backends {

// ──────────────────────────────────────────────────────────────────────────────
// Helper: wrap Tensor data as an Eigen Map
// ──────────────────────────────────────────────────────────────────────────────
using EMap  = Eigen::Map<Eigen::MatrixXf>;
using CEMap = Eigen::Map<const Eigen::MatrixXf>;

// ---------------------------------------------------------------------------
// WALKTHROUGH — Chapter 2: Who Does the Math?
//
// Tensor is dumb data storage.  All arithmetic lives here in IBackend.
// The backend is looked up through BackendRegistry::active() — so layers
// never mention CpuBackend directly.  Swapping to a CUDA backend later is
// a single setActive("cuda") call; no layer code changes.
//
// Eigen / Row-major vs Col-major trick:
//   Eigen's Matrix is col-major by default (memory goes down columns first).
//   Our Tensors are row-major (memory goes across rows first, like C arrays).
//
//   Key identity used everywhere:
//     (A_rowmajor @ B_rowmajor) interpreted as col-major = (B^T @ A^T)
//
//   So for C[M,N] = A[M,K] @ B[K,N] we tell Eigen:
//     A is a col-major K×M matrix  → Eigen reads it as A^T
//     B is a col-major N×K matrix  → Eigen reads it as B^T
//     C is a col-major N×M matrix  → Eigen writes it as C^T
//     C^T = B^T @ A^T  ✓ (this is mathematically identical to C = A @ B)
//
//   noalias() tells Eigen the output buffer doesn't overlap either input,
//   skipping a defensive temporary copy — important for performance.
// ---------------------------------------------------------------------------

// ──────────────────────────────────────────────────────────────────────────────
// Registration
// ──────────────────────────────────────────────────────────────────────────────
void CpuBackend::init() {
    BackendRegistry::instance().registerBackend(std::make_shared<CpuBackend>());
}

// ──────────────────────────────────────────────────────────────────────────────
// Memory
// ──────────────────────────────────────────────────────────────────────────────
std::shared_ptr<float[]> CpuBackend::alloc(int64_t count) {
    return std::shared_ptr<float[]>(new float[count]);
}

Tensor CpuBackend::to(const Tensor& t) {
    if (t.device() == Device::CPU) return t;
    return t.clone();
}

// ──────────────────────────────────────────────────────────────────────────────
// BLAS: matmul
// @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#forward-propagation
// ──────────────────────────────────────────────────────────────────────────────
Tensor CpuBackend::matmul(const Tensor& a, const Tensor& b) {
    const bool batched = (a.ndim() == 3 && b.ndim() == 3);
    const bool simple  = (a.ndim() == 2 && b.ndim() == 2);
    assert((simple || batched) && "matmul: only 2-D and 3-D (batched) tensors supported");

    if (simple) {
        const int64_t M = a.shape()[0];
        const int64_t K = a.shape()[1];
        const int64_t N = b.shape()[1];
        assert(K == b.shape()[0] && "matmul inner dim mismatch");

        Tensor c = Tensor::zeros({M, N});

        CEMap At(a.data(), K, M);
        CEMap Bt(b.data(), N, K);
        EMap  Ct(c.data(), N, M);
        Ct.noalias() = Bt * At;
        return c;
    }

    // Batched
    const int64_t Bdim  = a.shape()[0];
    const int64_t M     = a.shape()[1];
    const int64_t K     = a.shape()[2];
    const int64_t N     = b.shape()[2];
    assert(K == b.shape()[1] && "batched matmul inner dim mismatch");
    assert(Bdim == b.shape()[0] && "batched matmul batch size mismatch");

    Tensor c = Tensor::zeros({Bdim, M, N});
    const int64_t sliceA = M * K;
    const int64_t sliceB = K * N;
    const int64_t sliceC = M * N;

    for (int64_t bi = 0; bi < Bdim; ++bi) {
        CEMap At(a.data() + bi * sliceA, K, M);
        CEMap Bt(b.data() + bi * sliceB, N, K);
        EMap  Ct(c.data() + bi * sliceC, N, M);
        Ct.noalias() = Bt * At;
    }
    return c;
}

// ──────────────────────────────────────────────────────────────────────────────
// Element-wise helpers
// ──────────────────────────────────────────────────────────────────────────────
namespace {

Tensor elemWise(const Tensor& a, const Tensor& b,
                std::function<float(float, float)> op) {
    assert(a.sameShape(b) && "element-wise op: shape mismatch");
    Tensor out(a.shape());
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i)
        out.flat(i) = op(a.flat(i), b.flat(i));
    return out;
}

Tensor elemWiseScalar(const Tensor& a, float s,
                      std::function<float(float, float)> op) {
    Tensor out(a.shape());
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i)
        out.flat(i) = op(a.flat(i), s);
    return out;
}

Tensor elemUnary(const Tensor& a, std::function<float(float)> op) {
    Tensor out(a.shape());
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i)
        out.flat(i) = op(a.flat(i));
    return out;
}

} // anonymous namespace

Tensor CpuBackend::add (const Tensor& a, const Tensor& b) { return elemWise(a,b,[](float x,float y){return x+y;}); }
Tensor CpuBackend::sub (const Tensor& a, const Tensor& b) { return elemWise(a,b,[](float x,float y){return x-y;}); }
Tensor CpuBackend::mul (const Tensor& a, const Tensor& b) { return elemWise(a,b,[](float x,float y){return x*y;}); }
Tensor CpuBackend::div_(const Tensor& a, const Tensor& b) { return elemWise(a,b,[](float x,float y){return x/y;}); }

Tensor CpuBackend::addScalar(const Tensor& a, float s) { return elemWiseScalar(a,s,[](float x,float s){return x+s;}); }
Tensor CpuBackend::subScalar(const Tensor& a, float s) { return elemWiseScalar(a,s,[](float x,float s){return x-s;}); }
Tensor CpuBackend::mulScalar(const Tensor& a, float s) { return elemWiseScalar(a,s,[](float x,float s){return x*s;}); }
Tensor CpuBackend::divScalar(const Tensor& a, float s) { return elemWiseScalar(a,s,[](float x,float s){return x/s;}); }
Tensor CpuBackend::neg      (const Tensor& a)          { return elemUnary(a,[](float x){return -x;}); }
Tensor CpuBackend::exp      (const Tensor& a)          { return elemUnary(a,[](float x){return std::exp(x);}); }
Tensor CpuBackend::log      (const Tensor& a)          { return elemUnary(a,[](float x){return std::log(x);}); }
Tensor CpuBackend::sqrt     (const Tensor& a)          { return elemUnary(a,[](float x){return std::sqrt(x);}); }
Tensor CpuBackend::abs      (const Tensor& a)          { return elemUnary(a,[](float x){return std::abs(x);}); }
Tensor CpuBackend::clamp    (const Tensor& a, float lo, float hi) {
    return elemUnary(a, [lo,hi](float x){ return std::max(lo, std::min(hi, x)); });
}

// ──────────────────────────────────────────────────────────────────────────────
// Reductions
// ──────────────────────────────────────────────────────────────────────────────
Tensor CpuBackend::sum(const Tensor& a, int dim, bool keepdim) {
    if (dim == -1) {
        float s = 0.0f;
        for (int64_t i = 0; i < a.numel(); ++i) s += a.flat(i);
        Tensor out = keepdim ? Tensor::full(Shape(a.ndim(), 1), s)
                             : Tensor::full({1}, s);
        return out;
    }
    assert(dim >= 0 && dim < a.ndim());
    Shape outShape = a.shape();
    outShape[dim] = 1;
    Tensor out = Tensor::zeros(outShape);

    const int64_t n = a.numel();
    const int64_t outer = a.strides()[dim] * a.shape()[dim] == 0 ? 1
                        : a.numel() / a.shape()[dim];
    (void)outer;

    for (int64_t i = 0; i < n; ++i) {
        int64_t flat_out = 0;
        int64_t rem = i;
        for (int d = 0; d < a.ndim(); ++d) {
            int64_t coord = rem / a.strides()[d];
            rem          %= a.strides()[d];
            if (d == dim) coord = 0;
            flat_out += coord * out.strides()[d];
        }
        out.flat(flat_out) += a.flat(i);
    }

    if (!keepdim) {
        outShape.erase(outShape.begin() + dim);
        return out.reshape(outShape).value();
    }
    return out;
}

Tensor CpuBackend::mean(const Tensor& a, int dim, bool keepdim) {
    Tensor s = sum(a, dim, keepdim);
    const int64_t n = (dim == -1) ? a.numel() : a.shape()[dim];
    return mulScalar(s, 1.0f / static_cast<float>(n));
}

Tensor CpuBackend::max(const Tensor& a, int dim, bool keepdim) {
    if (dim == -1) {
        float m = a.flat(0);
        for (int64_t i = 1; i < a.numel(); ++i)
            if (a.flat(i) > m) m = a.flat(i);
        return keepdim ? Tensor::full(Shape(a.ndim(), 1), m)
                       : Tensor::full({1}, m);
    }
    Shape outShape = a.shape();
    outShape[dim] = 1;
    Tensor out = Tensor::full(outShape, -std::numeric_limits<float>::infinity());
    const int64_t n = a.numel();
    for (int64_t i = 0; i < n; ++i) {
        int64_t flat_out = 0;
        int64_t rem = i;
        for (int d = 0; d < a.ndim(); ++d) {
            int64_t coord = rem / a.strides()[d];
            rem          %= a.strides()[d];
            if (d == dim) coord = 0;
            flat_out += coord * out.strides()[d];
        }
        if (a.flat(i) > out.flat(flat_out)) out.flat(flat_out) = a.flat(i);
    }
    if (!keepdim) {
        outShape.erase(outShape.begin() + dim);
        return out.reshape(outShape).value();
    }
    return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// Shape ops
// ──────────────────────────────────────────────────────────────────────────────
Tensor CpuBackend::reshape(const Tensor& a, const Shape& newShape) {
    auto r = a.reshape(newShape);
    assert(r.ok());
    return r.value();
}

Tensor CpuBackend::transpose(const Tensor& a, int dim0, int dim1) {
    Shape   newShape   = a.shape();
    Strides newStrides = a.strides();
    std::swap(newShape[dim0],   newShape[dim1]);
    std::swap(newStrides[dim0], newStrides[dim1]);
    Tensor out(newShape);
    const int64_t n = out.numel();
    for (int64_t i = 0; i < n; ++i) {
        int64_t rem = i;
        int64_t srcOff = 0;
        for (int d = 0; d < out.ndim(); ++d) {
            int64_t coord = rem / out.strides()[d];
            rem          %= out.strides()[d];
            srcOff += coord * newStrides[d];
        }
        out.flat(i) = a.data()[srcOff];
    }
    return out;
}

Tensor CpuBackend::cat(const std::vector<Tensor>& tensors, int dim) {
    assert(!tensors.empty());
    int64_t catDim = 0;
    Shape outShape = tensors[0].shape();
    for (size_t ti = 1; ti < tensors.size(); ++ti)
        catDim += tensors[ti].shape()[dim];
    catDim += tensors[0].shape()[dim];
    outShape[dim] = catDim;
    Tensor out(outShape);

    int64_t offset = 0;
    for (const auto& t : tensors) {
        const int64_t sliceSize = t.shape()[dim];
        if (dim == 0) {
            std::memcpy(out.data() + offset, t.data(),
                        static_cast<size_t>(t.numel()) * sizeof(float));
            offset += t.numel();
        } else {
            for (int64_t i = 0; i < t.numel(); ++i) {
                int64_t rem = i;
                int64_t outIdx = 0;
                for (int d = 0; d < t.ndim(); ++d) {
                    int64_t coord = rem / t.strides()[d];
                    rem          %= t.strides()[d];
                    if (d == dim) coord += offset;
                    outIdx += coord * out.strides()[d];
                }
                out.flat(outIdx) = t.flat(i);
            }
            offset += sliceSize;
        }
    }
    return out;
}

} // namespace backends
} // namespace builtin
} // namespace nnstudio
