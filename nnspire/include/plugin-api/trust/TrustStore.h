/* ============================================================================
 * TrustStore.h — persistent CA certificate store for plugin trust verification
 * LGPL v3
 *
 * TRUST MODEL (Phase 2)
 * ──────────────────────
 * NNSpire uses a multi-level trust hierarchy for plugins:
 *
 *   Level 0 — Root CA (embedded in binary, hardware-verified at build time)
 *   Level 1 — Enterprise CA (issued by Root CA; customer deploys to employees)
 *   Level 2 — Community CA (issued by Root CA; open-source plugin authors)
 *   Level 3 — Plugin certificate (issued by either L1 or L2 CA)
 *
 * TrustStore persists the set of trusted CAs (Level 1 + Level 2) on disk in
 * a signed binary format (.nnts). The root CA is NOT stored here — it is
 * embedded in the binary and verified by the NNSpire-sign tool at build time.
 *
 * WHY NOT THE SYSTEM KEYSTORE?
 *   Plugins are NN model components; their trust scope is intentionally
 *   narrower than OS-level code signing. A dedicated store avoids mixing
 *   NN plugin trust with OS PKI and simplifies enterprise policy control.
 *
 * FILE FORMAT (.nnts)
 * ────────────────────
 *   [4]  magic     "NNTS"
 *   [4]  version   uint32 le (= 1)
 *   [8]  timestamp int64 le  (seconds since epoch, store creation time)
 *   [4]  count     uint32 le (number of CA entries)
 *   for each entry:
 *     [4]  type    uint32 le (1=Enterprise, 2=Community)
 *     [4]  der_len uint32 le
 *     [der_len] DER-encoded X.509 certificate
 *   [4]  sig_len   uint32 le
 *   [sig_len] ECDSA-P256 signature over all preceding bytes (root CA signs)
 *
 * API contract (no exceptions throughout):
 *   All methods return Result<void> or Result<T>.
 *   Callers MUST check .ok() before using .value().
 *
 * @kb: docs/TRUST-ARCHITECTURE.md
 * ============================================================================ */

#pragma once

#include <core/Result.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace NNSpire {
namespace trust {

using namespace NNSpire::core;

// ─── Trust levels ─────────────────────────────────────────────────────────────
enum class TrustLevel : uint32_t {
    Untrusted  = 0,  ///< no valid signature found
    Community  = 1,  ///< signed by community CA (open-source authors)
    Enterprise = 2,  ///< signed by enterprise CA (organisation deployment)
    Root       = 3,  ///< signed directly by the embedded root CA (NNSpire core)
};

// ─── CA entry ─────────────────────────────────────────────────────────────────
struct CaEntry {
    TrustLevel         level;    ///< Enterprise or Community
    std::string        subjectDN; ///< human-readable DN (from X.509 Subject)
    std::vector<uint8_t> derCert; ///< DER-encoded X.509 certificate
};

// ─── TrustStore ───────────────────────────────────────────────────────────────
/**
 * Persistent, signed set of trusted CA certificates.
 *
 * Lifecycle:
 *   TrustStore store;
 *   store.load("/etc/NNSpire/trust.nnts");
 *   // ... use verifier ...
 *   store.save("/etc/NNSpire/trust.nnts");
 *
 * All persistent modifications go through TrustUpdateHandler, which verifies
 * the Trust Update Packet before calling addCa() / revokeCa().
 */
class TrustStore {
public:
    TrustStore() = default;

    // ── Persistence ──────────────────────────────────────────────────────────

    /** Load a .nnts store from disk.  Verifies the store's root-CA signature. */
    Result<void> load(std::string_view path);

    /** Persist the current state, signing with the embedded root private key.
     *  NOTE: in production this key resides in an HSM; for development builds
     *  it is a test key stored in third-party-deps/test-root-key.pem. */
    Result<void> save(std::string_view path) const;

    // ── Modification (only called by TrustUpdateHandler after TUP verify) ────

    /** Add a CA certificate to the store. */
    Result<void> addCa(CaEntry entry);

    /** Revoke a CA by its subject DN. Returns error if DN not found. */
    Result<void> revokeCa(std::string_view subjectDN);

    // ── Query ─────────────────────────────────────────────────────────────────

    const std::vector<CaEntry>& entries() const noexcept { return entries_; }
    size_t count() const noexcept { return entries_.size(); }

    /** Find entry by subject DN. Returns nullptr if not found. */
    const CaEntry* findByDN(std::string_view dn) const noexcept;

    /** True when no CAs are loaded (empty store — trust nothing mode). */
    bool empty() const noexcept { return entries_.empty(); }

    // ── Static helpers ────────────────────────────────────────────────────────

    /** Create an empty store with the embedded root CA pre-loaded.
     *  This is the factory used by NNSpire at startup. */
    static TrustStore createDefault();

private:
    std::vector<CaEntry> entries_;
    int64_t              creationTime_{0};
};

} // namespace trust
} // namespace NNSpire
