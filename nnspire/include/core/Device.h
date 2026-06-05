#pragma once
/* ============================================================================
 * Device.h — compute device tag, shared by Tensor and IBackend
 * LGPL v3
 *
 * Every Tensor carries a Device tag that declares where its data lives:
 *   CPU     — main-memory float array; always available; Phase 1 default.
 *   CUDA    — GPU device memory; pointer is a CUDA device address.
 *   Quantum — quantum processing unit (Phase 6 stub; not a float* buffer).
 *
 * Device is a property of the *data*, not the computation.
 * The Backend describes where *operations* run; a mismatch between a
 * Tensor's device tag and the active Backend produces a DeviceMismatch
 * Result<> error before any arithmetic is attempted.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#include <string_view>

namespace NNSpire::core {

enum class Device : uint8_t {
    CPU     = 0,
    CUDA    = 1,
    Quantum = 2,
};

constexpr std::string_view deviceName(Device d) noexcept {
    switch (d) {
        case Device::CPU:     return "cpu";
        case Device::CUDA:    return "cuda";
        case Device::Quantum: return "quantum";
    }
    return "unknown";
}

} // namespace NNSpire::core
