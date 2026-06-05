/* ============================================================================
 * PluginLoader.cpp — dlopen-based plugin loading with trust verification
 * LGPL v3
 * Built only when NN_ENABLE_TRUST=ON
 * ============================================================================ */

#include <plugin-api/PluginLoader.h>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace NNSpire {
namespace plugins {

using namespace NNSpire::trust;
using namespace NNSpire::core;

// ─── Platform dlopen wrappers ──────────────────────────────────────────────────
namespace {

void* platformOpen(std::string_view path) {
#ifdef _WIN32
    return static_cast<void*>(LoadLibraryW(
        std::wstring(path.begin(), path.end()).c_str()));
#else
    return dlopen(std::string{path}.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* platformSym(void* handle, const char* sym) {
#ifdef _WIN32
    return reinterpret_cast<void*>(
        GetProcAddress(static_cast<HMODULE>(handle), sym));
#else
    return dlsym(handle, sym);
#endif
}

void platformClose(void* handle) {
    if (!handle) return;
#ifdef _WIN32
    FreeLibrary(static_cast<HMODULE>(handle));
#else
    dlclose(handle);
#endif
}

} // anonymous namespace

// ─── LoadedPlugin ─────────────────────────────────────────────────────────────

void LoadedPlugin::closeHandle() noexcept {
    if (handle_) {
        if (desc_ && instance_) desc_->destroy(instance_);
        platformClose(handle_);
        handle_   = nullptr;
        instance_ = nullptr;
        desc_     = nullptr;
    }
}

LoadedPlugin::~LoadedPlugin() { closeHandle(); }

LoadedPlugin::LoadedPlugin(LoadedPlugin&& o) noexcept
    : handle_(o.handle_), instance_(o.instance_)
    , desc_(o.desc_), trust_(std::move(o.trust_)), path_(std::move(o.path_))
{
    o.handle_   = nullptr;
    o.instance_ = nullptr;
    o.desc_     = nullptr;
}

LoadedPlugin& LoadedPlugin::operator=(LoadedPlugin&& o) noexcept {
    if (this != &o) {
        closeHandle();
        handle_   = o.handle_;   o.handle_   = nullptr;
        instance_ = o.instance_; o.instance_ = nullptr;
        desc_     = o.desc_;     o.desc_     = nullptr;
        trust_    = std::move(o.trust_);
        path_     = std::move(o.path_);
    }
    return *this;
}

// ─── PluginLoader ─────────────────────────────────────────────────────────────

PluginLoader::PluginLoader(const TrustStore& store, LoadPolicy policy)
    : store_(store), verifier_(store), policy_(policy) {}

bool PluginLoader::meetsPolicy(TrustLevel level, LoadPolicy policy) noexcept {
    switch (policy) {
        case LoadPolicy::RequireEnterprise:
            return level >= TrustLevel::Enterprise;
        case LoadPolicy::RequireCommunity:
            return level >= TrustLevel::Community;
        case LoadPolicy::AllowUntrusted:
            return true;
    }
    return false;
}

Result<LoadedPlugin> PluginLoader::load(std::string_view path) {
    // 1. Trust verification
    auto vrR = verifier_.verify(path);
    if (!vrR.ok()) return Result<LoadedPlugin>{"PluginLoader: trust verification error: " + vrR.error()};
    const VerifyResult& vr = vrR.value();

    if (!meetsPolicy(vr.level, policy_)) {
        return Result<LoadedPlugin>{
            "PluginLoader: plugin '" + std::string{path} + "' does not meet trust policy "
            "(level=" + std::to_string(static_cast<int>(vr.level)) + ")"};
    }

    // 2. dlopen
    void* handle = platformOpen(path);
    if (!handle) {
        const char* err =
#ifdef _WIN32
            "(see GetLastError)";
#else
            dlerror();
#endif
        return Result<LoadedPlugin>{"PluginLoader: dlopen failed: " + std::string{err ? err : "unknown"}};
    }

    // 3. Resolve descriptor symbol
    using DescFn = const NNPluginDescriptor*(*)();
    auto fn = reinterpret_cast<DescFn>(platformSym(handle, "NNSpire_plugin_descriptor"));
    if (!fn) {
        platformClose(handle);
        return Result<LoadedPlugin>{"PluginLoader: NNSpire_plugin_descriptor symbol not found"};
    }
    const NNPluginDescriptor* desc = fn();
    if (!desc) { platformClose(handle); return Result<LoadedPlugin>{"PluginLoader: descriptor returned null"}; }

    // 4. API version check
    if (desc->api_version != NNSpire_PLUGIN_API_VERSION) {
        platformClose(handle);
        return Result<LoadedPlugin>{"PluginLoader: API version mismatch (plugin=" +
            std::to_string(desc->api_version) + " host=" +
            std::to_string(NNSpire_PLUGIN_API_VERSION) + ")"};
    }

    // 5. Create plugin instance
    NNPluginHandle instance = desc->create ? desc->create() : nullptr;

    // Build LoadedPlugin
    LoadedPlugin loaded;
    loaded.handle_   = handle;
    loaded.instance_ = instance;
    loaded.desc_     = desc;
    loaded.trust_    = vr;
    loaded.path_     = std::string{path};
    return Result<LoadedPlugin>{std::move(loaded)};
}

void PluginLoader::unload(LoadedPlugin& plugin) {
    plugin.closeHandle();
}

} // namespace plugins
} // namespace NNSpire
