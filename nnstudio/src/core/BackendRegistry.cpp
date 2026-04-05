/* ============================================================================
 * BackendRegistry.cpp
 * LGPL v3
 * ============================================================================ */

#include <core/BackendRegistry.h>
#include <stdexcept>
#include <cassert>

namespace nnstudio::core {

BackendRegistry& BackendRegistry::instance() noexcept {
    static BackendRegistry inst;
    return inst;
}

void BackendRegistry::registerBackend(std::shared_ptr<IBackend> backend) {
    const std::string name { backend->name() };
    backends_[name] = std::move(backend);
    if (!active_) {
        // First registered backend becomes active automatically.
        active_ = backends_.at(name).get();
    }
}

Result<void> BackendRegistry::setActive(std::string_view name) {
    auto it = backends_.find(std::string{name});
    if (it == backends_.end())
        return err(ErrorCode::BackendError,
                   std::string{"Backend not registered: "} + std::string{name});
    active_ = it->second.get();
    return {};
}

IBackend& BackendRegistry::active() {
    assert(active_ != nullptr &&
           "No backend registered. Call nnstudio::initCpuBackend() at startup.");
    return *active_;
}

bool BackendRegistry::has(std::string_view name) const noexcept {
    return backends_.count(std::string{name}) > 0;
}

std::vector<std::string> BackendRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(backends_.size());
    for (auto& [k, _] : backends_) out.push_back(k);
    return out;
}

} // namespace nnstudio::core
