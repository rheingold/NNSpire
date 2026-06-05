#pragma once
/* ============================================================================
 * BackendRegistry.h — manages available compute backends; one active at a time
 * LGPL v3
 *
 * SINGLETON + B() FREE FUNCTION PATTERN
 * ──────────────────────────────────────
 * BackendRegistry is a singleton (BackendRegistry::instance()).
 * The engine never calls it directly — all dispatch goes through B():
 *
 *   IBackend& B();    // returns the currently active backend
 *
 *   B().matmul(x, w) is the standard idiom throughout Dense, Losses, etc.
 *   setActive("cuda") at startup is all it takes to route everything to GPU.
 *
 * REGISTRATION
 * ────────────
 * Backends self-register by calling BackendRegistry::instance().registerBackend()
 * from a static initialiser or plugin-init function. CpuBackend is always
 * registered first and serves as the fallback if no other backend is set.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#include <core/IBackend.h>
#include <core/Result.h>

#include <unordered_map>
#include <string>
#include <memory>
#include <string_view>

namespace NNSpire::core {

class BackendRegistry {
public:
    static BackendRegistry& instance() noexcept;

    // ------------------------------------------------------------------
    // Registration
    // ------------------------------------------------------------------
    /// Register a backend. Call once at startup (e.g. from backend shared lib init).
    void registerBackend(std::shared_ptr<IBackend> backend);

    // ------------------------------------------------------------------
    // Selection
    // ------------------------------------------------------------------
    /// Set the active backend by name (e.g. "cpu", "cuda").
    Result<void> setActive(std::string_view name);

    /// Get the currently active backend — always non-null after CpuBackend init.
    IBackend& active();

    /// Test whether a named backend is registered.
    bool has(std::string_view name) const noexcept;

    /// List all registered backend names.
    std::vector<std::string> names() const;

private:
    BackendRegistry() = default;
    std::unordered_map<std::string, std::shared_ptr<IBackend>> backends_;
    IBackend* active_ { nullptr };
};

// Shorthand: use the active backend
inline IBackend& backend() { return BackendRegistry::instance().active(); }

} // namespace NNSpire::core
