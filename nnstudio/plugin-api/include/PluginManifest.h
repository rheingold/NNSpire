/* ============================================================================
 * PluginManifest.h  —  Plugin manifest parsing and validation
 * LGPL v3
 *
 * Provides:
 *   PluginManifest — parsed view of a plugin.manifest.json
 *   TupManifest    — parsed view of a tup.manifest.json
 *   ManifestLoader — factory: load + validate from file path or JSON text
 *
 * JSON PARSING STRATEGY
 * ─────────────────────
 *   Currently uses a minimal hand-rolled string scanner sufficient to validate
 *   all required fields.  When nlohmann/json is added as a project dependency,
 *   replace with nlohmann::json::parse() for robustness.
 *   See TODO.md §Phase 2 — Plugin SDK.
 *
 * @kb: PLUGIN-SDK.md §4  |  TRUST-ARCHITECTURE.md §TUP
 * ============================================================================ */

#pragma once

#include <core/Result.h>

#include <string>
#include <vector>

namespace nnstudio::plugin_api {

using nnstudio::core::Result;

/* ─── Plugin type (mirrors NNPluginType in nnstudio_plugin.h) ─────────────── */
enum class PluginType : int {
    Layer         =  1,
    Activation    =  2,
    Optimizer     =  3,
    Tokenizer     =  4,
    Backend       =  5,
    InputAdapter  =  6,
    OutputAdapter =  7,
    ContextSource =  8,
    RunnerClient  =  9,
    UiPanel       = 10,
    Unknown       = -1
};

/** Convert a manifest "type" string (e.g. "LAYER") to PluginType. */
PluginType pluginTypeFromString(const std::string& s) noexcept;

/* ─── Implementation entry (cpp or python) ────────────────────────────────── */
struct PluginImplementation {
    std::string language;   ///< "cpp" or "python"
    std::string artifact;   ///< library path (cpp) or module name (python)
    std::string wheel;      ///< Python wheel path (may be empty)
    std::string sha256;     ///< Hex SHA-256 digest
};

/* ─── Parsed plugin.manifest.json ─────────────────────────────────────────── */
struct PluginManifest {
    int         manifestVersion{0};
    std::string id;
    std::string name;
    std::string version;
    PluginType  type{PluginType::Unknown};
    std::string author;
    std::string license;
    std::string minStudioVersion;
    std::string maxStudioVersion;   ///< May be empty (no upper bound)
    std::string description;
    std::vector<std::string> capabilities;
    std::string kbRef;
    std::vector<PluginImplementation> implementations;
    std::string signature;          ///< Path to .p7s signature file
};

/* ─── Revocation entry in a TUP ───────────────────────────────────────────── */
struct TupRevocation {
    std::string serialNumber;
    std::string issuerSubject;
    std::string reason;
    std::string comment;
};

/* ─── Certificate add entry in a TUP ─────────────────────────────────────── */
struct TupCertEntry {
    std::string file;
    std::string sha256;
    std::string role;   ///< "root", "intermediate", or "crl"
    std::string subject;
};

/* ─── Parsed tup.manifest.json ────────────────────────────────────────────── */
struct TupManifest {
    int         manifestVersion{0};
    std::string id;             ///< UUID v4
    std::string version;        ///< Monotonic version
    std::string issuedAt;       ///< ISO 8601 UTC
    std::string issuerSubject;  ///< Signer DN
    std::string description;
    std::vector<TupCertEntry>  addCerts;
    std::vector<TupRevocation> revokeCerts;
    bool selfRevocationGuard{true};
    bool requireUserConfirmation{true};
    std::string signature;
};

/* ─── ManifestLoader ──────────────────────────────────────────────────────── */
/**
 * Loads and validates plugin and TUP manifests.
 *
 * All methods are static. Validation:
 *   - Required fields present and non-empty
 *   - manifestVersion == 1
 *   - Semver pattern for version / minStudioVersion
 *   - At least one implementation entry
 *   - Capabilities list non-empty
 *   - type != "TRUST_UPDATE" in plugin manifests
 */
class ManifestLoader {
public:
    ManifestLoader() = delete;

    /**
     * Parse and validate a plugin.manifest.json from a JSON string.
     * @return Parsed PluginManifest, or an error describing the first violation.
     */
    static Result<PluginManifest> loadPlugin(const std::string& jsonText) noexcept;

    /**
     * Parse and validate a tup.manifest.json from a JSON string.
     * @return Parsed TupManifest, or an error describing the first violation.
     */
    static Result<TupManifest> loadTup(const std::string& jsonText) noexcept;

    /**
     * Read and validate a plugin manifest from a file path.
     */
    static Result<PluginManifest> loadPluginFromFile(const std::string& path) noexcept;

    /**
     * Read and validate a TUP manifest from a file path.
     */
    static Result<TupManifest> loadTupFromFile(const std::string& path) noexcept;
};

} // namespace nnstudio::plugin_api
