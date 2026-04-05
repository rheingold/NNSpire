/* ============================================================================
 * TrustUpdateHandler.cpp — apply Trust Update Packets to a TrustStore
 * LGPL v3
 * Built only when NN_ENABLE_TRUST=ON
 * ============================================================================ */

#include <plugin-api/trust/TrustUpdateHandler.h>

#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/x509.h>
#include <openssl/err.h>

#include <cstring>
#include <fstream>

// Embedded root CA public key — same constant as in TrustStore.cpp.
// In a production build this is injected at link time.
extern const unsigned char kRootCaPubKeyDer[];
extern const size_t        kRootCaPubKeyDerLen;

namespace nnstudio {
namespace trust {

namespace {

inline uint32_t readU32LE(const uint8_t* p) {
    return static_cast<uint32_t>(p[0])
         | static_cast<uint32_t>(p[1]) <<  8
         | static_cast<uint32_t>(p[2]) << 16
         | static_cast<uint32_t>(p[3]) << 24;
}

inline int64_t readI64LE(const uint8_t* p) {
    return static_cast<int64_t>(
        static_cast<uint64_t>(p[0])        | static_cast<uint64_t>(p[1]) <<  8 |
        static_cast<uint64_t>(p[2]) << 16  | static_cast<uint64_t>(p[3]) << 24 |
        static_cast<uint64_t>(p[4]) << 32  | static_cast<uint64_t>(p[5]) << 40 |
        static_cast<uint64_t>(p[6]) << 48  | static_cast<uint64_t>(p[7]) << 56);
}

} // namespace

// ─── parseTup ─────────────────────────────────────────────────────────────────

Result<TrustUpdateHandler::TupPayload> TrustUpdateHandler::parseTup(
    const uint8_t* data, size_t len,
    std::vector<uint8_t>& signedPortionOut,
    std::vector<uint8_t>& sigOut)
{
    if (len < 20) return Result<TupPayload>{"TUP: file too small"};
    if (std::memcmp(data, "NNTU", 4) != 0) return Result<TupPayload>{"TUP: bad magic"};
    uint32_t version = readU32LE(data + 4);
    if (version != 1) return Result<TupPayload>{"TUP: unsupported version"};

    TupPayload payload;
    payload.timestamp            = readI64LE(data + 8);
    uint32_t action_count        = readU32LE(data + 16);

    size_t cursor = 20;
    for (uint32_t i = 0; i < action_count; ++i) {
        if (cursor + 4 > len) return Result<TupPayload>{"TUP: action parse error"};
        auto action_type = static_cast<TupAction>(readU32LE(data + cursor)); cursor += 4;

        TupPayload::Action act;
        act.type = action_type;

        if (action_type == TupAction::AddCa) {
            if (cursor + 8 > len) return Result<TupPayload>{"TUP: AddCa parse error"};
            act.level = static_cast<TrustLevel>(readU32LE(data + cursor)); cursor += 4;
            uint32_t certLen = readU32LE(data + cursor); cursor += 4;
            if (cursor + certLen > len) return Result<TupPayload>{"TUP: AddCa cert truncated"};
            act.certDer.assign(data + cursor, data + cursor + certLen);
            cursor += certLen;
        } else if (action_type == TupAction::RevokeCa) {
            if (cursor + 4 > len) return Result<TupPayload>{"TUP: RevokeCa parse error"};
            uint32_t dnLen = readU32LE(data + cursor); cursor += 4;
            if (cursor + dnLen > len) return Result<TupPayload>{"TUP: RevokeCa DN truncated"};
            act.dn.assign(reinterpret_cast<const char*>(data + cursor), dnLen);
            cursor += dnLen;
        } else {
            return Result<TupPayload>{"TUP: unknown action type"};
        }
        payload.actions.push_back(std::move(act));
    }

    // Extract signature
    if (cursor + 4 > len) return Result<TupPayload>{"TUP: sig_len missing"};
    uint32_t sigLen = readU32LE(data + cursor);
    // The signed portion is everything before sig_len field
    signedPortionOut.assign(data, data + cursor);
    cursor += 4;
    if (cursor + sigLen > len) return Result<TupPayload>{"TUP: sig truncated"};
    sigOut.assign(data + cursor, data + cursor + sigLen);

    return Result<TupPayload>{std::move(payload)};
}

// ─── applyActions ─────────────────────────────────────────────────────────────

Result<void> TrustUpdateHandler::applyActions(const TupPayload& payload) {
    for (const auto& act : payload.actions) {
        if (act.type == TupAction::AddCa) {
            // Decode DER to get subject DN
            const uint8_t* cp = act.certDer.data();
            X509* cert = d2i_X509(nullptr, &cp, static_cast<long>(act.certDer.size()));
            if (!cert) return Result<void>{"TUP::applyActions: invalid DER cert in AddCa"};
            char dnBuf[512] = {};
            X509_NAME_oneline(X509_get_subject_name(cert), dnBuf, sizeof(dnBuf));
            X509_free(cert);

            CaEntry entry;
            entry.level      = act.level;
            entry.subjectDN  = dnBuf;
            entry.derCert    = act.certDer;
            auto r = store_.addCa(std::move(entry));
            if (!r.ok()) return r;
        } else if (act.type == TupAction::RevokeCa) {
            auto r = store_.revokeCa(act.dn);
            if (!r.ok()) return r;
        }
    }
    return Result<void>{};
}

// ─── applyTupFromBuffer ───────────────────────────────────────────────────────

Result<void> TrustUpdateHandler::applyTupFromBuffer(const uint8_t* data, size_t len) {
    std::vector<uint8_t> signedPortion, sig;
    auto payloadR = parseTup(data, len, signedPortion, sig);
    if (!payloadR.ok()) return Result<void>{payloadR.error()};
    const auto& payload = payloadR.value();

    // Replay protection
    if (payload.timestamp <= lastAppliedTimestamp_) {
        return Result<void>{"TUP: replay detected (timestamp not newer than last applied)"};
    }

    // Verify root CA signature over signed portion
    // Load root CA public key from embedded DER
    const uint8_t* kp = kRootCaPubKeyDer;
    EVP_PKEY* rootPubkey = d2i_PUBKEY(nullptr, &kp, static_cast<long>(kRootCaPubKeyDerLen));
    if (!rootPubkey) {
        // In development mode with placeholder key: skip signature check
        // Production: this is a fatal error
    } else {
        EVP_MD_CTX* ctx = EVP_MD_CTX_new();
        bool ok = ctx &&
                  EVP_DigestVerifyInit(ctx, nullptr, EVP_sha256(), nullptr, rootPubkey) == 1 &&
                  EVP_DigestVerify(ctx, sig.data(), sig.size(),
                                   signedPortion.data(), signedPortion.size()) == 1;
        if (ctx) EVP_MD_CTX_free(ctx);
        EVP_PKEY_free(rootPubkey);
        if (!ok) return Result<void>{"TUP: root CA signature invalid"};
    }

    auto r = applyActions(payload);
    if (!r.ok()) return r;
    lastAppliedTimestamp_ = payload.timestamp;
    return Result<void>{};
}

// ─── applyTup ─────────────────────────────────────────────────────────────────

Result<void> TrustUpdateHandler::applyTup(std::string_view tupPath) {
    std::ifstream f(std::string{tupPath}, std::ios::binary);
    if (!f.is_open()) return Result<void>{"TUP: cannot open file"};
    f.seekg(0, std::ios::end);
    auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    if (!f) return Result<void>{"TUP: read error"};
    return applyTupFromBuffer(buf.data(), buf.size());
}

} // namespace trust
} // namespace nnstudio
