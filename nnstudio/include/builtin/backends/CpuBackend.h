/* ============================================================================ 
 * CpuBackend.h — Eigen-based CPU compute backend  [nnstudio::builtin::backends]
 * LGPL v3
 *
 * CpuBackend is NNStudio's own implementation of IBackend.
 * It lives in nnstudio::builtin::backends:: and is treated identically to any
 * third-party backend plugin — it self-registers via CpuBackend::init().
 *
 * Registration pattern:
 *   CpuBackend::init()  →  BackendRegistry::instance().registerBackend(...)
 *   BackendRegistry::setActive("cpu")  (also done by init() as first/fallback)
 *
 * Then anywhere in the codebase:  B()  (nnstudio::backend() free fn)
 * returns the active IBackend without any layer knowing about CpuBackend.
 *
 * @see docs/blueprints.md — Chapter 2 (IBackend + CpuBackend)
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#matrix-ops
 * ============================================================================ */

#pragma once

#include <core/IBackend.h>

namespace nnstudio {
namespace builtin {
namespace backends {
using namespace nnstudio::core;

class CpuBackend final : public IBackend {
public:
    CpuBackend() = default;

    /** Self-register with BackendRegistry and set as active (call once at startup). */
    static void init();

    std::string_view name()   const noexcept override { return "cpu"; }
    Device           device() const noexcept override { return Device::CPU; }

    std::shared_ptr<float[]> alloc(int64_t count) override;
    Tensor to(const Tensor& t)                    override;

    Tensor matmul   (const Tensor& a, const Tensor& b)          override;
    Tensor add      (const Tensor& a, const Tensor& b)          override;
    Tensor sub      (const Tensor& a, const Tensor& b)          override;
    Tensor mul      (const Tensor& a, const Tensor& b)          override;
    Tensor div_     (const Tensor& a, const Tensor& b)          override;
    Tensor addScalar(const Tensor& a, float s)                  override;
    Tensor subScalar(const Tensor& a, float s)                  override;
    Tensor mulScalar(const Tensor& a, float s)                  override;
    Tensor divScalar(const Tensor& a, float s)                  override;
    Tensor neg      (const Tensor& a)                           override;
    Tensor exp      (const Tensor& a)                           override;
    Tensor log      (const Tensor& a)                           override;
    Tensor sqrt     (const Tensor& a)                           override;
    Tensor abs      (const Tensor& a)                           override;
    Tensor clamp    (const Tensor& a, float lo, float hi)       override;
    Tensor sum      (const Tensor& a, int dim, bool keepdim)    override;
    Tensor mean     (const Tensor& a, int dim, bool keepdim)    override;
    Tensor max      (const Tensor& a, int dim, bool keepdim)    override;
    Tensor reshape  (const Tensor& a, const Shape& newShape)    override;
    Tensor transpose(const Tensor& a, int dim0, int dim1)       override;
    Tensor cat      (const std::vector<Tensor>& tensors, int dim) override;
};

} // namespace backends
} // namespace builtin
} // namespace nnstudio
