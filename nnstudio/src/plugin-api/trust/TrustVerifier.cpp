/* ============================================================================
 * TrustVerifier.cpp — plugin binary signature verification (OpenSSL)
 * LGPL v3
 * Built only when NN_ENABLE_TRUST=ON
 * ============================================================================ */

#include <plugin-api/trust/TrustVerifier.h>

#include <openssl/bio.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/x509.h>
#include <openssl/x509_vfy.h>
#include <openssl/err.h>

#include <cstring>
#include <ctime>
#include <fstream>

namespace nnstudio {
namespace trust {

// ─── File helpers ─────────────────────────────────────────────────────────────
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

Result<std::vector<uint8_t>> readFileBuf(std::string_view path) {
    std::ifstream f(std::string{path}, std::ios::binary);
    if (!f.is_open()) return Result<std::vector<uint8_t>>{"cannot open file"};
    f.seekg(0, std::ios::end);
    auto sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(sz));
    if (!f) return Result<std::vector<uint8_t>>{"read error"};
    return Result<std::vector<uint8_t>>{std::move(buf)};
}

} // namespace

// ─── readSidecar ──────────────────────────────────────────────────────────────

Result<TrustVerifier::SidecarData> TrustVerifier::readSidecar(std::string_view sigPath) {
    auto bufR = readFileBuf(sigPath);
    if (!bufR.ok()) return Result<SidecarData>{"TrustVerifier: cannot read sidecar: " + bufR.error()};
    const auto& buf = bufR.value();

    if (buf.size() < 20) return Result<SidecarData>{"TrustVerifier: sidecar too small"};
    if (std::memcmp(buf.data(), "NNSG", 4) != 0) return Result<SidecarData>{"TrustVerifier: bad sidecar magic"};
    uint32_t ver = readU32LE(buf.data() + 4);
    if (ver != 1) return Result<SidecarData>{"TrustVerifier: unsupported sidecar version"};

    SidecarData sd;
    sd.timestamp = readI64LE(buf.data() + 8);

    size_t cursor = 16;
    auto readStr = [&](std::string& out) -> bool {
        if (cursor + 4 > buf.size()) return false;
        uint32_t len = readU32LE(buf.data() + cursor); cursor += 4;
        if (cursor + len > buf.size()) return false;
        out.assign(reinterpret_cast<const char*>(buf.data() + cursor), len);
        cursor += len;
        return true;
    };
    auto readVec = [&](std::vector<uint8_t>& out) -> bool {
        if (cursor + 4 > buf.size()) return false;
        uint32_t len = readU32LE(buf.data() + cursor); cursor += 4;
        if (cursor + len > buf.size()) return false;
        out.assign(buf.data() + cursor, buf.data() + cursor + len);
        cursor += len;
        return true;
    };

    if (!readStr(sd.pluginId) || !readStr(sd.version) ||
        !readVec(sd.certDer)  || !readVec(sd.signature)) {
        return Result<SidecarData>{"TrustVerifier: sidecar parse error"};
    }
    // signed content = everything before the final sig_len + sig bytes
    size_t sigTotalLen = 4 + sd.signature.size();
    if (buf.size() < sigTotalLen) return Result<SidecarData>{"TrustVerifier: sidecar truncated"};
    size_t signedLen = buf.size() - sigTotalLen;
    sd.signedContent.assign(buf.data(), buf.data() + signedLen);

    return Result<SidecarData>{std::move(sd)};
}

// ─── hashFile ─────────────────────────────────────────────────────────────────

Result<std::vector<uint8_t>> TrustVerifier::hashFile(std::string_view path) {
    auto bufR = readFileBuf(path);
    if (!bufR.ok()) return Result<std::vector<uint8_t>>{bufR.error()};
    const auto& buf = bufR.value();

    std::vector<uint8_t> digest(SHA256_DIGEST_LENGTH);
    SHA256(buf.data(), buf.size(), digest.data());
    return Result<std::vector<uint8_t>>{std::move(digest)};
}

// ─── verifyCertChain ──────────────────────────────────────────────────────────

Result<TrustLevel> TrustVerifier::verifyCertChain(
    const std::vector<uint8_t>& certDer,
    std::string& outSignerDN,
    std::string& outCaDN) const
{
    // Decode plugin certificate
    const uint8_t* cp = certDer.data();
    X509* cert = d2i_X509(nullptr, &cp, static_cast<long>(certDer.size()));
    if (!cert) return Result<TrustLevel>{"verifyCertChain: invalid plugin cert DER"};

    // Extract Subject DN
    char buf[512] = {};
    X509_NAME_oneline(X509_get_subject_name(cert), buf, sizeof(buf));
    outSignerDN = buf;

    // Extract Issuer DN to match against TrustStore
    char issBuf[512] = {};
    X509_NAME_oneline(X509_get_issuer_name(cert), issBuf, sizeof(issBuf));
    outCaDN = issBuf;

    X509_free(cert);

    // Find issuing CA in TrustStore
    const CaEntry* ca = store_.findByDN(outCaDN);
    if (!ca) return Result<TrustLevel>{TrustLevel::Untrusted};
    return Result<TrustLevel>{ca->level};
}

// ─── verifySignature ──────────────────────────────────────────────────────────

Result<void> TrustVerifier::verifySignature(
    const std::vector<uint8_t>& certDer,
    const std::vector<uint8_t>& sig,
    const std::vector<uint8_t>& data)
{
    const uint8_t* cp = certDer.data();
    X509* cert = d2i_X509(nullptr, &cp, static_cast<long>(certDer.size()));
    if (!cert) return Result<void>{"verifySignature: invalid cert DER"};

    EVP_PKEY* pubkey = X509_get_pubkey(cert);
    X509_free(cert);
    if (!pubkey) return Result<void>{"verifySignature: no public key in cert"};

    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    if (!ctx) { EVP_PKEY_free(pubkey); return Result<void>{"verifySignature: EVP_MD_CTX_new failed"}; }

    int ok = (EVP_DigestVerifyInit(ctx, nullptr, EVP_sha256(), nullptr, pubkey) == 1 &&
              EVP_DigestVerify(ctx, sig.data(), sig.size(), data.data(), data.size()) == 1)
             ? 1 : 0;

    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(pubkey);

    if (!ok) {
        unsigned long e = ERR_get_error();
        char ebuf[256]; ERR_error_string_n(e, ebuf, sizeof(ebuf));
        return Result<void>{std::string{"verifySignature: ECDSA verify failed: "} + ebuf};
    }
    return Result<void>{};
}

// ─── verify ───────────────────────────────────────────────────────────────────

Result<VerifyResult> TrustVerifier::verify(std::string_view pluginPath) const {
    std::string sigPath = std::string{pluginPath} + ".nnsig";

    auto sdR = readSidecar(sigPath);
    if (!sdR.ok()) {
        // No sidecar → untrusted
        return Result<VerifyResult>{VerifyResult{TrustLevel::Untrusted, {}, {}, 0,
                                                 std::string{pluginPath}, ""}};
    }
    const auto& sd = sdR.value();

    // Hash the plugin binary
    auto hashR = hashFile(pluginPath);
    if (!hashR.ok()) return Result<VerifyResult>{hashR.error()};

    // Verify certificate chain
    VerifyResult result;
    result.timestamp  = sd.timestamp;
    result.pluginId   = sd.pluginId;
    result.version    = sd.version;

    auto chainR = verifyCertChain(sd.certDer, result.signerDN, result.caDN);
    if (!chainR.ok()) return Result<VerifyResult>{chainR.error()};
    result.level = chainR.value();

    if (result.level == TrustLevel::Untrusted) return Result<VerifyResult>{std::move(result)};

    // Build signed content: hash || plugin_id || version || timestamp (as 8 bytes LE)
    std::vector<uint8_t> signedData = hashR.value();
    signedData.insert(signedData.end(), sd.pluginId.begin(), sd.pluginId.end());
    signedData.insert(signedData.end(), sd.version.begin(), sd.version.end());
    {
        uint8_t tsbuf[8];
        auto u = static_cast<uint64_t>(sd.timestamp);
        for (int i = 0; i < 8; ++i) tsbuf[i] = static_cast<uint8_t>(u >> (i*8));
        signedData.insert(signedData.end(), tsbuf, tsbuf + 8);
    }

    auto sigR = verifySignature(sd.certDer, sd.signature, signedData);
    if (!sigR.ok()) { result.level = TrustLevel::Untrusted; }

    return Result<VerifyResult>{std::move(result)};
}

// ─── atLeast ─────────────────────────────────────────────────────────────────

Result<bool> TrustVerifier::atLeast(std::string_view pluginPath, TrustLevel required) const {
    auto r = verify(pluginPath);
    if (!r.ok()) return Result<bool>{r.error()};
    return Result<bool>{static_cast<uint32_t>(r.value().level) >= static_cast<uint32_t>(required)};
}

} // namespace trust
} // namespace nnstudio
