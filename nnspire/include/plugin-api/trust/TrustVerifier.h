/* ============================================================================
 * TrustVerifier.h — verify plugin binary signatures against TrustStore
 * LGPL v3
 *
 * SIGNATURE FORMAT
 * ─────────────────
 * When a plugin is signed by NNSpire-sign, a sidecar file is written:
 *   <plugin>.so  →  <plugin>.so.nnsig
 *
 * The .nnsig format:
 *   [4]  magic     "NNSG"
 *   [4]  version   uint32 le (= 1)
 *   [8]  timestamp int64 le  (signing time, seconds since epoch)
 *   [4]  plugin_id_len uint32 le
 *   [plugin_id_len] UTF-8 plugin reverse-domain id
 *   [4]  semver_len uint32 le
 *   [semver_len] UTF-8 semver string
 *   [4]  cert_len  uint32 le  (DER length of signing certificate)
 *   [cert_len] DER X.509 certificate (issued by a CA in TrustStore)
 *   [4]  sig_len   uint32 le
 *   [sig_len] ECDSA-P256 signature over:
 *                SHA-256(plugin binary) || plugin_id || version || timestamp
 *
 * VERIFICATION STEPS
 * ───────────────────
 *   1. Read .nnsig next to the plugin .so / .dll
 *   2. Parse sidecar, extract certificate + signature
 *   3. Build chain: sidecar cert → CA in TrustStore → root CA (embedded)
 *   4. Verify certificate chain validity (issuer, dates, key usage)
 *   5. Compute SHA-256 of the plugin binary
 *   6. Verify ECDSA signature against the sidecar cert's public key
 *   7. Map the signing CA to a TrustLevel and return
 *
 * FALLBACK POLICY (when no .nnsig exists)
 * ──────────────────────────────────────────
 *   Returns TrustLevel::Untrusted.
 *   Whether Untrusted plugins are loaded is controlled by FeatureFlags
 *   (NN_FLAG_ALLOW_UNTRUSTED_PLUGINS).
 *
 * @kb: docs/TRUST-ARCHITECTURE.md
 * ============================================================================ */

#pragma once

#include <plugin-api/trust/TrustStore.h>
#include <core/Result.h>
#include <string_view>

namespace NNSpire {
namespace trust {

using namespace NNSpire::core;

// ─── Verification result ──────────────────────────────────────────────────────
struct VerifyResult {
    TrustLevel   level;        ///< trust level granted
    std::string  signerDN;     ///< X.509 Subject DN of the signing certificate
    std::string  caDN;         ///< Subject DN of the issuing CA
    int64_t      timestamp;    ///< signing timestamp (seconds since epoch)
    std::string  pluginId;     ///< plugin reverse-domain id from sidecar
    std::string  version;      ///< semver string from sidecar
};

// ─── TrustVerifier ────────────────────────────────────────────────────────────
/**
 * Stateless plugin signature verifier.  Requires OpenSSL (NN_ENABLE_TRUST=ON).
 *
 * Usage:
 *   TrustStore store = TrustStore::createDefault();
 *   store.load("/etc/NNSpire/trust.nnts");
 *
 *   TrustVerifier verifier(store);
 *   auto result = verifier.verify("/usr/lib/nn-plugins/my-plugin.so");
 *   if (!result.ok()) { ... handle error ... }
 *   if (result.value().level < TrustLevel::Community) { ... reject ... }
 */
class TrustVerifier {
public:
    explicit TrustVerifier(const TrustStore& store) : store_(store) {}

    /**
     * Verify the plugin at `pluginPath`.
     * Looks for `pluginPath + ".nnsig"` sidecar.
     * @return VerifyResult on success, error string on failure.
     */
    Result<VerifyResult> verify(std::string_view pluginPath) const;

    /**
     * Quick trust-level check without returning full details.
     * Efficient when you only need pass/fail at a given level.
     */
    Result<bool> atLeast(std::string_view pluginPath, TrustLevel required) const;

private:
    const TrustStore& store_;

    // ── Internal helpers (all require OpenSSL) ────────────────────────────────
    struct SidecarData {
        int64_t              timestamp;
        std::string          pluginId;
        std::string          version;
        std::vector<uint8_t> certDer;
        std::vector<uint8_t> signature;
        std::vector<uint8_t> signedContent;  // the exact bytes that were signed
    };

    static Result<SidecarData>   readSidecar(std::string_view sigPath);
    static Result<std::vector<uint8_t>> hashFile(std::string_view path);
    Result<TrustLevel>            verifyCertChain(const std::vector<uint8_t>& certDer,
                                                  std::string& outSignerDN,
                                                  std::string& outCaDN) const;
    static Result<void>           verifySignature(const std::vector<uint8_t>& certDer,
                                                  const std::vector<uint8_t>& sig,
                                                  const std::vector<uint8_t>& data);
};

} // namespace trust
} // namespace NNSpire
