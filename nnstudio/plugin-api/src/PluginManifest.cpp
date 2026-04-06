/* ============================================================================
 * PluginManifest.cpp  —  Plugin manifest parsing and validation
 * LGPL v3
 *
 * Minimal JSON field extractor (no external dependency).
 * Handles the small, well-defined plugin/TUP manifest schemas.
 * Replace with nlohmann::json when the project adds it as a dependency.
 * ============================================================================ */

#include <plugin-api/include/PluginManifest.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace nnstudio::plugin_api {

using nnstudio::core::Error;
using nnstudio::core::Result;

/* ─── Tiny JSON scanner helpers ──────────────────────────────────────────── */

namespace {

/** Return true if `text` contains `"key"` somewhere. */
static bool hasKey(const std::string& text, const std::string& key) noexcept {
    std::string needle = "\"" + key + "\"";
    return text.find(needle) != std::string::npos;
}

/**
 * Extract the first string value for `"key": "..."` in text.
 * Returns empty string if not found.
 */
static std::string extractString(const std::string& text,
                                  const std::string& key) noexcept {
    std::string needle = "\"" + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos) return {};

    // skip past key + optional whitespace + colon + optional whitespace
    pos += needle.size();
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
        ++pos;
    if (pos >= text.size() || text[pos] != ':') return {};
    ++pos;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
        ++pos;
    if (pos >= text.size() || text[pos] != '"') return {};
    ++pos;  // skip opening quote

    std::string result;
    while (pos < text.size() && text[pos] != '"') {
        if (text[pos] == '\\' && pos + 1 < text.size()) {
            ++pos;  // skip escape prefix; simplified — just copy escaped char
            result += text[pos];
        } else {
            result += text[pos];
        }
        ++pos;
    }
    return result;
}

/**
 * Extract the first integer value for `"key": <digits>` in text.
 * Returns -1 if not found or not an integer.
 */
static int extractInt(const std::string& text, const std::string& key) noexcept {
    std::string needle = "\"" + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos) return -1;

    pos += needle.size();
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
        ++pos;
    if (pos >= text.size() || text[pos] != ':') return -1;
    ++pos;
    while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\t' || text[pos] == '\n' || text[pos] == '\r'))
        ++pos;
    if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) return -1;

    int val = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos])))
        val = val * 10 + (text[pos++] - '0');
    return val;
}

/**
 * Extract all string items from the first JSON array `"key": [...]` in text.
 */
static std::vector<std::string> extractStringArray(const std::string& text,
                                                    const std::string& key) noexcept {
    std::string needle = "\"" + key + "\"";
    auto pos = text.find(needle);
    if (pos == std::string::npos) return {};

    pos += needle.size();
    // find '[' after key: value separator
    while (pos < text.size() && text[pos] != '[') ++pos;
    if (pos >= text.size()) return {};
    ++pos;  // skip '['

    std::vector<std::string> items;
    while (pos < text.size() && text[pos] != ']') {
        while (pos < text.size() && text[pos] != '"' && text[pos] != ']') ++pos;
        if (pos >= text.size() || text[pos] == ']') break;
        ++pos;  // skip '"'
        std::string item;
        while (pos < text.size() && text[pos] != '"')
            item += text[pos++];
        if (!item.empty()) items.push_back(std::move(item));
        if (pos < text.size()) ++pos;  // skip closing '"'
    }
    return items;
}

/**
 * Check if a string matches the basic semver pattern x.y.z (no pre-release here).
 */
static bool isSemver(const std::string& s) noexcept {
    if (s.empty()) return false;
    int dots = 0;
    for (char c : s) {
        if (c == '.') { ++dots; continue; }
        if (!std::isdigit(static_cast<unsigned char>(c))) return false;
    }
    return dots == 2 && s.front() != '.' && s.back() != '.';
}

/**
 * Check if a string is a valid hex SHA-256 (64 lowercase hex digits).
 */
static bool isSha256(const std::string& s) noexcept {
    if (s.size() != 64) return false;
    for (char c : s)
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    return true;
}

/**
 * Read entire file contents into a string.
 */
static Result<std::string> readFile(const std::string& path) noexcept {
    std::ifstream ifs(path);
    if (!ifs.is_open())
        return Result<std::string>::err(Error{"ManifestLoader: cannot open file: " + path});
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return Result<std::string>::ok(ss.str());
}

/**
 * Extract implementation blocks from json text for "cpp" and "python".
 * Returns a vector of PluginImplementation (0-2 entries).
 */
static std::vector<PluginImplementation> extractImplementations(
    const std::string& text) noexcept
{
    std::vector<PluginImplementation> impls;
    for (const char* lang : {"cpp", "python"}) {
        std::string needle = "\"" + std::string(lang) + "\"";
        auto pos = text.find(needle);
        if (pos == std::string::npos) continue;

        // Find the '{' opening the sub-object
        auto blockStart = text.find('{', pos + needle.size());
        if (blockStart == std::string::npos) continue;

        // Find matching '}'
        int depth = 0;
        auto blockEnd = blockStart;
        while (blockEnd < text.size()) {
            if (text[blockEnd] == '{') ++depth;
            else if (text[blockEnd] == '}') { if (--depth == 0) break; }
            ++blockEnd;
        }

        std::string block = text.substr(blockStart, blockEnd - blockStart + 1);

        PluginImplementation impl;
        impl.language = lang;
        if (std::string(lang) == "cpp") {
            impl.artifact = extractString(block, "library");
        } else {
            impl.artifact = extractString(block, "module");
            impl.wheel    = extractString(block, "wheel");
        }
        impl.sha256   = extractString(block, "sha256");

        if (!impl.artifact.empty() && !impl.sha256.empty())
            impls.push_back(std::move(impl));
    }
    return impls;
}

} // anonymous namespace

/* ─── PluginType helpers ─────────────────────────────────────────────────── */

PluginType pluginTypeFromString(const std::string& s) noexcept {
    if (s == "LAYER")          return PluginType::Layer;
    if (s == "ACTIVATION")     return PluginType::Activation;
    if (s == "OPTIMIZER")      return PluginType::Optimizer;
    if (s == "TOKENIZER")      return PluginType::Tokenizer;
    if (s == "BACKEND")        return PluginType::Backend;
    if (s == "INPUT_ADAPTER")  return PluginType::InputAdapter;
    if (s == "OUTPUT_ADAPTER") return PluginType::OutputAdapter;
    if (s == "CONTEXT_SOURCE") return PluginType::ContextSource;
    if (s == "RUNNER_CLIENT")  return PluginType::RunnerClient;
    if (s == "UI_PANEL")       return PluginType::UiPanel;
    return PluginType::Unknown;
}

/* ─── ManifestLoader::loadPlugin ─────────────────────────────────────────── */

Result<PluginManifest> ManifestLoader::loadPlugin(const std::string& json) noexcept {
    PluginManifest m;

    // manifestVersion
    m.manifestVersion = extractInt(json, "manifestVersion");
    if (m.manifestVersion != 1)
        return Result<PluginManifest>::err(Error{"manifest: 'manifestVersion' must be 1"});

    // type (reject TRUST_UPDATE)
    std::string typeStr = extractString(json, "type");
    if (typeStr == "TRUST_UPDATE")
        return Result<PluginManifest>::err(
            Error{"manifest: type 'TRUST_UPDATE' is reserved for TUP packages; use tup.manifest.json"});
    m.type = pluginTypeFromString(typeStr);
    if (m.type == PluginType::Unknown)
        return Result<PluginManifest>::err(Error{"manifest: unknown plugin type '" + typeStr + "'"});

    // required string fields
    m.id = extractString(json, "id");
    if (m.id.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'id' is required"});

    m.name = extractString(json, "name");
    if (m.name.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'name' is required"});

    m.version = extractString(json, "version");
    if (m.version.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'version' is required"});

    m.author = extractString(json, "author");
    if (m.author.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'author' is required"});

    m.license = extractString(json, "license");
    if (m.license.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'license' is required"});

    m.minStudioVersion = extractString(json, "minStudioVersion");
    if (m.minStudioVersion.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'minStudioVersion' is required"});

    if (!isSemver(m.minStudioVersion))
        return Result<PluginManifest>::err(
            Error{"manifest: 'minStudioVersion' is not a valid x.y.z semver: " + m.minStudioVersion});

    m.signature = extractString(json, "signature");
    if (m.signature.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'signature' is required"});

    // optional string fields
    m.maxStudioVersion = extractString(json, "maxStudioVersion");
    m.description      = extractString(json, "description");
    m.kbRef            = extractString(json, "kbRef");

    // capabilities (required, non-empty array)
    m.capabilities = extractStringArray(json, "capabilities");
    if (m.capabilities.empty())
        return Result<PluginManifest>::err(Error{"manifest: 'capabilities' must have at least one entry"});

    // implementations (at least one)
    m.implementations = extractImplementations(json);
    if (m.implementations.empty())
        return Result<PluginManifest>::err(
            Error{"manifest: 'implementations' must have at least one of 'cpp' or 'python'"});

    // sha256 validation
    for (const auto& impl : m.implementations) {
        if (!isSha256(impl.sha256))
            return Result<PluginManifest>::err(
                Error{"manifest: sha256 for '" + impl.language + "' implementation is not a valid 64-char hex string"});
    }

    return Result<PluginManifest>::ok(std::move(m));
}

/* ─── ManifestLoader::loadTup ────────────────────────────────────────────── */

Result<TupManifest> ManifestLoader::loadTup(const std::string& json) noexcept {
    TupManifest t;

    t.manifestVersion = extractInt(json, "manifestVersion");
    if (t.manifestVersion != 1)
        return Result<TupManifest>::err(Error{"tup: 'manifestVersion' must be 1"});

    // type must be TRUST_UPDATE
    std::string typeStr = extractString(json, "type");
    if (typeStr != "TRUST_UPDATE")
        return Result<TupManifest>::err(Error{"tup: 'type' must be 'TRUST_UPDATE'"});

    t.id = extractString(json, "id");
    if (t.id.empty())
        return Result<TupManifest>::err(Error{"tup: 'id' is required"});

    t.version = extractString(json, "version");
    if (t.version.empty())
        return Result<TupManifest>::err(Error{"tup: 'version' is required"});

    t.issuedAt = extractString(json, "issuedAt");
    if (t.issuedAt.empty())
        return Result<TupManifest>::err(Error{"tup: 'issuedAt' is required"});

    t.issuerSubject = extractString(json, "issuerSubject");
    if (t.issuerSubject.empty())
        return Result<TupManifest>::err(Error{"tup: 'issuerSubject' is required"});

    t.signature = extractString(json, "signature");
    if (t.signature.empty())
        return Result<TupManifest>::err(Error{"tup: 'signature' is required"});

    t.description = extractString(json, "description");

    // addCerts and revokeCerts are validated presence-only here;
    // deep cert parsing happens in TrustUpdateHandler.
    t.selfRevocationGuard    = !hasKey(json, "selfRevocationGuard")  ? true
                               : (json.find("\"selfRevocationGuard\": false") != std::string::npos ? false : true);
    t.requireUserConfirmation= !hasKey(json, "requireUserConfirmation") ? true
                               : (json.find("\"requireUserConfirmation\": false") != std::string::npos ? false : true);

    return Result<TupManifest>::ok(std::move(t));
}

/* ─── File-based loaders ─────────────────────────────────────────────────── */

Result<PluginManifest> ManifestLoader::loadPluginFromFile(const std::string& path) noexcept {
    auto text = readFile(path);
    if (!text.ok()) return Result<PluginManifest>::err(text.error());
    return loadPlugin(text.value());
}

Result<TupManifest> ManifestLoader::loadTupFromFile(const std::string& path) noexcept {
    auto text = readFile(path);
    if (!text.ok()) return Result<TupManifest>::err(text.error());
    return loadTup(text.value());
}

} // namespace nnstudio::plugin_api
