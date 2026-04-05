/* ============================================================================
 * TrustUpdateHandler.h — process Trust Update Packets (TUPs)
 * LGPL v3
 *
 * WHAT IS A TUP?
 * ───────────────
 * A Trust Update Packet (.tup) is a signed bundle that instructs NNStudio
 * to add or revoke CA certificates in its TrustStore.
 *
 * TUPs are signed by the embedded root CA private key.  The root CA is the
 * only entity that can issue TUPs — enterprise admins and community authors
 * cannot issue them (they can only request them via nnstudio-sign submit).
 *
 * TUP FILE FORMAT (.tup)
 * ──────────────────────
 *   [4]  magic     "NNTU"
 *   [4]  version   uint32 le (= 1)
 *   [8]  timestamp int64 le  (seconds since epoch — prevents replay attacks)
 *   [4]  action_count uint32 le
 *   for each action:
 *     [4]  action_type uint32 le  (1=AddCa, 2=RevokeCa)
 *     if AddCa:
 *       [4]  level uint32 le (1=Enterprise, 2=Community)
 *       [4]  cert_len uint32 le
 *       [cert_len] DER-encoded X.509 certificate
 *     if RevokeCa:
 *       [4]  dn_len uint32 le
 *       [dn_len] UTF-8 subject DN to revoke
 *   [4]  sig_len   uint32 le
 *   [sig_len] ECDSA-P256 signature over all preceding bytes (root CA key)
 *
 * REPLAY PROTECTION
 * ──────────────────
 * TrustStore records the timestamp of the last applied TUP.  Any TUP with
 * a timestamp ≤ lastApplied is rejected.
 *
 * @kb: docs/TRUST-ARCHITECTURE.md
 * ============================================================================ */

#pragma once

#include <plugin-api/trust/TrustStore.h>
#include <core/Result.h>
#include <string_view>

namespace nnstudio {
namespace trust {

using namespace nnstudio::core;

// ─── TUP action ──────────────────────────────────────────────────────────────
enum class TupAction : uint32_t {
    AddCa    = 1,
    RevokeCa = 2,
};

// ─── TrustUpdateHandler ───────────────────────────────────────────────────────
/**
 * Verifies and applies Trust Update Packets to a TrustStore.
 *
 * Usage:
 *   TrustStore store;
 *   store.load("/etc/nnstudio/trust.nnts");
 *
 *   TrustUpdateHandler handler(store);
 *   auto r = handler.applyTup("/tmp/update-2024-01.tup");
 *   if (!r.ok()) { log(r.error()); }
 *   store.save("/etc/nnstudio/trust.nnts");   // persist the update
 */
class TrustUpdateHandler {
public:
    explicit TrustUpdateHandler(TrustStore& store) : store_(store) {}

    /**
     * Read, verify, and apply a TUP file to the bound TrustStore.
     * The TUP must be signed by the embedded root CA key.
     * @return error string if signature invalid, replay attack, or parse error.
     */
    Result<void> applyTup(std::string_view tupPath);

    /**
     * Apply a TUP from an already-read buffer (for network-delivered TUPs).
     */
    Result<void> applyTupFromBuffer(const uint8_t* data, size_t len);

private:
    TrustStore& store_;
    int64_t     lastAppliedTimestamp_{0};

    struct TupPayload {
        int64_t         timestamp;
        struct Action {
            TupAction            type;
            TrustLevel           level;    // for AddCa
            std::vector<uint8_t> certDer;  // for AddCa
            std::string          dn;       // for RevokeCa
        };
        std::vector<Action> actions;
    };

    static Result<TupPayload> parseTup(const uint8_t* data, size_t len,
                                       std::vector<uint8_t>& signedPortionOut,
                                       std::vector<uint8_t>& sigOut);
    Result<void> applyActions(const TupPayload& payload);
};

} // namespace trust
} // namespace nnstudio
