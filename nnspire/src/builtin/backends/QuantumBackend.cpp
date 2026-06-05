/* ============================================================================
 * QuantumBackend.cpp — quantum backend stub registration
 * LGPL v3
 * ============================================================================ */

#include <builtin/backends/QuantumBackend.h>
#include <core/BackendRegistry.h>

namespace NNSpire {
namespace builtin {
namespace backends {

void QuantumBackend::init() {
    BackendRegistry::instance().registerBackend(
        std::make_shared<QuantumBackend>());
}

} // namespace backends
} // namespace builtin
} // namespace NNSpire
