/* ============================================================================
 * PluginLoader.h — dynamic plugin loading with trust verification
 * LGPL v3
 *
 * LOADING SEQUENCE
 * ─────────────────
 *   1. TrustVerifier::verify(path) — check signature against TrustStore
 *   2. If trust < policy minimum → reject (return error, do NOT dlopen)
 *   3. dlopen / LoadLibrary the shared library
 *   4. Resolve "NNSpire_plugin_descriptor" symbol
 *   5. Call descriptor()->api_version check
 *   6. Return LoadedPlugin wrapper (owns dlopen handle RAII)
 *
 * PLATFORM NOTES
 * ───────────────
 * Linux/macOS : dlopen / dlsym / dlclose   (POSIX)
 * Windows     : LoadLibraryW / GetProcAddress / FreeLibrary
 * The .cpp includes the correct platform headers behind #ifdef _WIN32.
 *
 * SANDBOX (Phase 5 — planned)
 * ────────────────────────────
 * Phase 5 will run plugins in a separate process (NNSpire-runner sidecar)
 * communicating over gRPC.  PluginLoader will be extended with a
 * SandboxedLoadPolicy that routes through the sidecar instead of dlopen.
 *
 * @kb: docs/PLUGIN-SDK.md
 * ============================================================================ */

#pragma once

#include <plugin-api/trust/TrustStore.h>
#include <plugin-api/trust/TrustVerifier.h>
#include <plugin-api/NNSpire_plugin.h>
#include <core/Result.h>

#include <memory>
#include <string>
#include <string_view>

namespace NNSpire {
namespace plugins {

using namespace NNSpire::core;

// ─── Minimum trust policy ─────────────────────────────────────────────────────
enum class LoadPolicy {
    RequireEnterprise,   ///< only enterprise-signed plugins allowed
    RequireCommunity,    ///< enterprise OR community-signed plugins allowed
    AllowUntrusted,      ///< any plugin (development/testing only)
};

// ─── LoadedPlugin RAII wrapper ────────────────────────────────────────────────
/**
 * Owns a loaded plugin's shared-library handle.
 * The descriptor() pointer is valid as long as this object is alive.
 * Destructor calls destroy() on the plugin instance and closes the library.
 */
class LoadedPlugin {
public:
    LoadedPlugin() = default;
    ~LoadedPlugin();

    // Move-only (no copy — we own the OS handle)
    LoadedPlugin(LoadedPlugin&&) noexcept;
    LoadedPlugin& operator=(LoadedPlugin&&) noexcept;
    LoadedPlugin(const LoadedPlugin&)            = delete;
    LoadedPlugin& operator=(const LoadedPlugin&) = delete;

    bool valid() const noexcept { return handle_ != nullptr; }

    const NNPluginDescriptor* descriptor() const noexcept { return desc_; }
    const NNSpire::trust::VerifyResult& trustInfo() const noexcept { return trust_; }
    std::string_view path() const noexcept { return path_; }

private:
    friend class PluginLoader;

    void*                            handle_{nullptr};  ///< dlopen handle
    NNPluginHandle                   instance_{nullptr}; ///< plugin->create()
    const NNPluginDescriptor*        desc_{nullptr};
    NNSpire::trust::VerifyResult    trust_;
    std::string                      path_;

    void closeHandle() noexcept;
};

// ─── PluginLoader ─────────────────────────────────────────────────────────────
/**
 * Loads and verifies NNSpire plugins.
 *
 * Usage:
 *   PluginLoader loader(trustStore, LoadPolicy::RequireCommunity);
 *   auto plugR = loader.load("/usr/lib/nn-plugins/bpe-tokenizer.so");
 *   if (!plugR.ok()) { log(plugR.error()); return; }
 *   auto& plugin = plugR.value();
 *   // Use plugin.descriptor()->vtable ...
 */
class PluginLoader {
public:
    explicit PluginLoader(const trust::TrustStore& store,
                          LoadPolicy policy = LoadPolicy::RequireCommunity);

    /**
     * Load and verify a plugin.
     * @param path     Absolute path to the plugin shared library.
     * @return LoadedPlugin on success, error string on failure.
     */
    Result<LoadedPlugin> load(std::string_view path);

    /** Unload a previously loaded plugin (calls destroy() + dlclose). */
    void unload(LoadedPlugin& plugin);

    LoadPolicy              policy() const noexcept { return policy_; }
    const trust::TrustStore& store()  const noexcept { return store_;  }

private:
    const trust::TrustStore& store_;
    trust::TrustVerifier      verifier_;
    LoadPolicy                policy_;

    static bool meetsPolicy(trust::TrustLevel level, LoadPolicy policy) noexcept;
};

} // namespace plugins
} // namespace NNSpire
