/* ============================================================================
 * QuantumBackend.h — quantum-hardware backend stub  [NNSpire::builtin::backends]
 * LGPL v3
 *
 * Phase 1 stub — interface compiles, all operations return NotSupported.
 * Real implementation targeting IBM Quantum / IonQ APIs is Phase 6+.
 *
 * Architecture note:
 *   The quantum backend plugin will NOT be a standard matrix-multiply backend.
 *   Quantum circuits represent tensor contractions natively; the mapping from
 *   feed-forward NN layers to quantum gates is non-trivial and research-grade.
 *   This stub provides the IBackend interface slot so CMake targets, plugin
 *   manifests, and feature-flag calls compile without #ifdefs throughout core.
 * ============================================================================
*/
#pragma once

#include <core/IBackend.h>

namespace NNSpire {
namespace builtin {
namespace backends {
using namespace NNSpire::core;

class QuantumBackend : public IBackend {
public:
    std::string_view name()   const noexcept override { return "quantum"; }
    Device           device() const noexcept override { return Device::Quantum; }

    // ── All operations abort in Phase 1 (stub — not implemented) ─────────
    std::shared_ptr<float[]> alloc(int64_t)                           override { ni(); }
    Tensor to        (const Tensor&)                                   override { ni(); }
    Tensor matmul    (const Tensor&, const Tensor&)                    override { ni(); }
    Tensor add       (const Tensor&, const Tensor&)                    override { ni(); }
    Tensor sub       (const Tensor&, const Tensor&)                    override { ni(); }
    Tensor mul       (const Tensor&, const Tensor&)                    override { ni(); }
    Tensor div_      (const Tensor&, const Tensor&)                    override { ni(); }
    Tensor addScalar (const Tensor&, float)                            override { ni(); }
    Tensor subScalar (const Tensor&, float)                            override { ni(); }
    Tensor mulScalar (const Tensor&, float)                            override { ni(); }
    Tensor divScalar (const Tensor&, float)                            override { ni(); }
    Tensor neg       (const Tensor&)                                   override { ni(); }
    Tensor exp       (const Tensor&)                                   override { ni(); }
    Tensor log       (const Tensor&)                                   override { ni(); }
    Tensor sqrt      (const Tensor&)                                   override { ni(); }
    Tensor clamp     (const Tensor&, float, float)                     override { ni(); }
    Tensor abs       (const Tensor&)                                   override { ni(); }
    Tensor sum       (const Tensor&, int, bool)                        override { ni(); }
    Tensor mean      (const Tensor&, int, bool)                        override { ni(); }
    Tensor max       (const Tensor&, int, bool)                        override { ni(); }
    Tensor reshape   (const Tensor&, const Shape&)                     override { ni(); }
    Tensor transpose (const Tensor&, int, int)                         override { ni(); }
    Tensor cat       (const std::vector<Tensor>&, int)                 override { ni(); }

    /// Register this backend with the BackendRegistry under the "quantum" key.
    static void init();

private:
    [[noreturn]] static void ni() {
        // Quantum backend is not implemented (Phase 6+).
        // Callers should gate on isEnabled(FF_QUANTUM_BACKEND) before dispatch.
        __builtin_trap();
    }
};

} // namespace backends
} // namespace builtin
} // namespace NNSpire
