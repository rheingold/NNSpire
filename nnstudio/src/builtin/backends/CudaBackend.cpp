/* ============================================================================
 * CudaBackend.cpp — CUDA backend stub (CPU-side glue + not-implemented traps)
 * LGPL v3
 *
 * This file compiles regardless of NN_ENABLE_CUDA.
 * When CUDA is disabled, all virtual methods call ni() (not-implemented trap).
 * When CUDA is enabled, real implementations come from CudaBackend.cu
 * (planned Phase 4) — this file provides only the init() registration and
 * any CUDA-toolkit-independent helpers.
 *
 * Phase 1 status: full stub — only init() is meaningful.
 * ============================================================================ */

#include <builtin/backends/CudaBackend.h>
#include <core/BackendRegistry.h>

namespace nnstudio {
namespace builtin {
namespace backends {

// ─── Registration ─────────────────────────────────────────────────────────────

void CudaBackend::init() {
#ifdef NN_ENABLE_CUDA
    static CudaBackend instance;
    BackendRegistry::instance().registerBackend(
        std::shared_ptr<IBackend>(&instance, [](IBackend*) {}));
    // NOTE: do NOT call setActive("cuda") here; that is the caller's choice.
    // CpuBackend remains the default fallback.
#endif
    // When NN_ENABLE_CUDA is not defined: this is intentionally a no-op.
}

// ─────────────────────────────────────────────────────────────────────────────
// NOT-IMPLEMENTED STUBS
// Every method below traps unless NN_ENABLE_CUDA is defined AND the real
// CUDA implementation has been linked in (CudaBackend.cu, Phase 4).
// ─────────────────────────────────────────────────────────────────────────────

namespace {
[[noreturn]] inline void notImplemented() {
    // No exceptions (-fno-exceptions); abort is the intended behaviour for
    // unsupported operations on the CUDA backend in Phase 1.
    __builtin_trap();
}
} // anonymous namespace

std::shared_ptr<float[]> CudaBackend::alloc(int64_t)                         { notImplemented(); }
Tensor CudaBackend::to       (const Tensor&)                                  { notImplemented(); }
Tensor CudaBackend::matmul   (const Tensor&, const Tensor&)                   { notImplemented(); }
Tensor CudaBackend::add      (const Tensor&, const Tensor&)                   { notImplemented(); }
Tensor CudaBackend::sub      (const Tensor&, const Tensor&)                   { notImplemented(); }
Tensor CudaBackend::mul      (const Tensor&, const Tensor&)                   { notImplemented(); }
Tensor CudaBackend::div_     (const Tensor&, const Tensor&)                   { notImplemented(); }
Tensor CudaBackend::addScalar(const Tensor&, float)                           { notImplemented(); }
Tensor CudaBackend::subScalar(const Tensor&, float)                           { notImplemented(); }
Tensor CudaBackend::mulScalar(const Tensor&, float)                           { notImplemented(); }
Tensor CudaBackend::divScalar(const Tensor&, float)                           { notImplemented(); }
Tensor CudaBackend::neg      (const Tensor&)                                  { notImplemented(); }
Tensor CudaBackend::exp      (const Tensor&)                                  { notImplemented(); }
Tensor CudaBackend::log      (const Tensor&)                                  { notImplemented(); }
Tensor CudaBackend::sqrt     (const Tensor&)                                  { notImplemented(); }
Tensor CudaBackend::clamp    (const Tensor&, float, float)                    { notImplemented(); }
Tensor CudaBackend::abs      (const Tensor&)                                  { notImplemented(); }
Tensor CudaBackend::sum      (const Tensor&, int, bool)                       { notImplemented(); }
Tensor CudaBackend::mean     (const Tensor&, int, bool)                       { notImplemented(); }
Tensor CudaBackend::max      (const Tensor&, int, bool)                       { notImplemented(); }
Tensor CudaBackend::reshape  (const Tensor&, const Shape&)                    { notImplemented(); }
Tensor CudaBackend::transpose(const Tensor&, int, int)                        { notImplemented(); }
Tensor CudaBackend::cat      (const std::vector<Tensor>&, int)                { notImplemented(); }

} // namespace backends
} // namespace builtin
} // namespace nnstudio
