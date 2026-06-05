/* ============================================================================
 * CudaBackend.h — NVIDIA CUDA/cuBLAS compute backend stub
 *                 [NNSpire::builtin::backends]
 * LGPL v3
 *
 * Phase 1 stub — interface compiles unconditionally; real CUDA operations
 * are ONLY compiled when NN_ENABLE_CUDA=ON (requires CUDAToolkit + cuBLAS).
 *
 * WHY A STUB?
 * ────────────
 * Plugin manifests, unit tests, and feature-flag paths all reference
 * "cuda" as a backend name. By providing this header unconditionally we
 * avoid #ifdefs scattered throughout core. The stub registers itself with
 * the BackendRegistry only when CUDA is actually available at runtime.
 *
 * WHEN IS THIS USED?
 * ───────────────────
 * When NN_ENABLE_CUDA=OFF (the default):
 *   • The header and the stub .cpp compile fine (no CUDA headers needed).
 *   • CudaBackend::init() is a no-op — the backend is not registered.
 *   • Any call to the stub operations triggers a runtime trap.
 *
 * When NN_ENABLE_CUDA=ON:
 *   • CMake links CUDAToolkit (cublas, cudnn).
 *   • CudaBackend.cu / CudaBackend.cpp provide real implementations of every
 *     IBackend virtual method via cuBLAS GEMM + element-wise CUDA kernels.
 *   • CudaBackend::init() registers "cuda" in BackendRegistry.
 *
 * PLANNED (Phase 4)
 * ──────────────────
 * Real implementation in src/builtin/backends/CudaBackend.cu covering:
 *   • matmul via cublasSgemm
 *   • element-wise ops via custom CUDA kernels
 *   • Memory management: cudaMalloc / cudaFree via alloc() override
 *   • Tensor device transfers: cudaMemcpy in to() override
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#backends
 * ============================================================================ */

#pragma once

#include <core/IBackend.h>

namespace NNSpire {
namespace builtin {
namespace backends {

using namespace NNSpire::core;

class CudaBackend : public IBackend {
public:
    std::string_view name()   const noexcept override { return "cuda"; }
    Device           device() const noexcept override { return Device::CUDA; }

    // ── Operations: real impl compiled only with NN_ENABLE_CUDA=ON ───────────
    std::shared_ptr<float[]> alloc(int64_t count)                      override;
    Tensor to        (const Tensor& t)                                  override;
    Tensor matmul    (const Tensor& a, const Tensor& b)                 override;
    Tensor add       (const Tensor& a, const Tensor& b)                 override;
    Tensor sub       (const Tensor& a, const Tensor& b)                 override;
    Tensor mul       (const Tensor& a, const Tensor& b)                 override;
    Tensor div_      (const Tensor& a, const Tensor& b)                 override;
    Tensor addScalar (const Tensor& a, float s)                         override;
    Tensor subScalar (const Tensor& a, float s)                         override;
    Tensor mulScalar (const Tensor& a, float s)                         override;
    Tensor divScalar (const Tensor& a, float s)                         override;
    Tensor neg       (const Tensor& a)                                  override;
    Tensor exp       (const Tensor& a)                                  override;
    Tensor log       (const Tensor& a)                                  override;
    Tensor sqrt      (const Tensor& a)                                  override;
    Tensor clamp     (const Tensor& a, float lo, float hi)              override;
    Tensor abs       (const Tensor& a)                                  override;
    Tensor sum       (const Tensor& a, int dim, bool keepdim)           override;
    Tensor mean      (const Tensor& a, int dim, bool keepdim)           override;
    Tensor max       (const Tensor& a, int dim, bool keepdim)           override;
    Tensor reshape   (const Tensor& a, const Shape& newShape)           override;
    Tensor transpose (const Tensor& a, int dim0, int dim1)              override;
    Tensor cat       (const std::vector<Tensor>& tensors, int dim)      override;

    /**
     * Register this backend with BackendRegistry under the "cuda" key.
     * When NN_ENABLE_CUDA=OFF this is a no-op (stub never registered).
     * Must be called after CpuBackend::init() so CPU remains the fallback.
     */
    static void init();
};

} // namespace backends
} // namespace builtin
} // namespace NNSpire
