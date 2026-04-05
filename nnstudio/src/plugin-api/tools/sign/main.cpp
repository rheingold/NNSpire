/* ============================================================================
 * main.cpp — nnstudio-sign CLI tool
 * LGPL v3
 *
 * SUBCOMMANDS
 * ────────────
 *   nnstudio-sign keygen     [--ca <level>] --out <prefix>
 *       Generate an ECDSA P-256 keypair.  --ca enterprise|community issues a
 *       CA certificate instead of a leaf certificate.
 *
 *   nnstudio-sign sign       --key <leaf.pem> --cert <leaf-cert.pem> \
 *                             --id <com.example.plugin> --version <1.0.0> \
 *                             <plugin.so>
 *       Sign a plugin binary and write <plugin.so>.nnsig sidecar.
 *
 *   nnstudio-sign verify     [--store <trust.nnts>] <plugin.so>
 *       Print the TrustLevel for a plugin binary.
 *
 *   nnstudio-sign create-tup --root-key <root.pem> --action add|revoke \
 *                             --cert <ca.pem>|--dn <DN> \
 *                             --out <update.tup>
 *       Create a signed Trust Update Packet.
 *
 *   nnstudio-sign issue-enterprise-ca \
 *                             --root-key <root.pem> --root-cert <root.pem> \
 *                             --subject <DN> --out <prefix>
 *       Issue an enterprise CA certificate signed by the root CA.
 *
 * EXIT CODES
 * ───────────
 *   0  success
 *   1  usage error
 *   2  verification failed (untrusted)
 *   3  cryptographic error
 * ============================================================================ */

#include <plugin-api/trust/TrustStore.h>
#include <plugin-api/trust/TrustVerifier.h>
#include <plugin-api/trust/TrustUpdateHandler.h>

#include <openssl/bio.h>
#include <openssl/ec.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/x509.h>
#include <openssl/x509v3.h>

#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static void sslDie(const char* msg) {
    std::cerr << "ERROR: " << msg << "\n";
    ERR_print_errors_fp(stderr);
    std::exit(3);
}

static std::vector<uint8_t> readFileBin(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cerr << "Cannot open: " << path << "\n"; std::exit(1); }
    return std::vector<uint8_t>{std::istreambuf_iterator<char>(f), {}};
}

static void writeFileBin(const std::string& path, const uint8_t* data, size_t len) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f.is_open()) { std::cerr << "Cannot write: " << path << "\n"; std::exit(1); }
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
}

static void writeU32LE(std::vector<uint8_t>& buf, uint32_t v) {
    buf.push_back(static_cast<uint8_t>(v));
    buf.push_back(static_cast<uint8_t>(v >>  8));
    buf.push_back(static_cast<uint8_t>(v >> 16));
    buf.push_back(static_cast<uint8_t>(v >> 24));
}

static void writeI64LE(std::vector<uint8_t>& buf, int64_t v) {
    auto u = static_cast<uint64_t>(v);
    for (int i = 0; i < 8; ++i) buf.push_back(static_cast<uint8_t>(u >> (i*8)));
}

static void writeStr(std::vector<uint8_t>& buf, const std::string& s) {
    writeU32LE(buf, static_cast<uint32_t>(s.size()));
    buf.insert(buf.end(), s.begin(), s.end());
}

static void writeVec(std::vector<uint8_t>& buf, const std::vector<uint8_t>& v) {
    writeU32LE(buf, static_cast<uint32_t>(v.size()));
    buf.insert(buf.end(), v.begin(), v.end());
}

// ─── keygen ───────────────────────────────────────────────────────────────────

static int cmdKeygen(int argc, char** argv) {
    std::string outPrefix, caLevel;
    for (int i = 0; i < argc; ++i) {
        if (std::string{argv[i]} == "--out"   && i+1 < argc) outPrefix = argv[++i];
        if (std::string{argv[i]} == "--ca"    && i+1 < argc) caLevel   = argv[++i];
    }
    if (outPrefix.empty()) { std::cerr << "Usage: keygen --out <prefix> [--ca enterprise|community]\n"; return 1; }

    EVP_PKEY* pkey = nullptr;
    EVP_PKEY_CTX* ctx = EVP_PKEY_CTX_new_id(EVP_PKEY_EC, nullptr);
    if (!ctx || EVP_PKEY_keygen_init(ctx) <= 0 ||
        EVP_PKEY_CTX_set_ec_paramgen_curve_nid(ctx, NID_X9_62_prime256v1) <= 0 ||
        EVP_PKEY_keygen(ctx, &pkey) <= 0) {
        sslDie("keygen: key generation failed");
    }
    EVP_PKEY_CTX_free(ctx);

    // Write private key
    std::string keyPath = outPrefix + "-key.pem";
    BIO* bio = BIO_new_file(keyPath.c_str(), "w");
    if (!bio || PEM_write_bio_PrivateKey(bio, pkey, nullptr, nullptr, 0, nullptr, nullptr) != 1)
        sslDie("keygen: write private key failed");
    BIO_free(bio);

    // Self-signed certificate (leaf or CA)
    X509* cert = X509_new();
    ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);
    X509_gmtime_adj(X509_getm_notBefore(cert), 0);
    X509_gmtime_adj(X509_getm_notAfter(cert), 365 * 24 * 3600L * 10); // 10 years
    X509_set_pubkey(cert, pkey);
    X509_NAME* name = X509_get_subject_name(cert);
    if (!caLevel.empty()) {
        std::string cn = "NNStudio " + caLevel + " CA";
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
            reinterpret_cast<const unsigned char*>(cn.c_str()), -1, -1, 0);
    } else {
        X509_NAME_add_entry_by_txt(name, "CN", MBSTRING_ASC,
            reinterpret_cast<const unsigned char*>("NNStudio Plugin Key"), -1, -1, 0);
    }
    X509_set_issuer_name(cert, name);
    if (!caLevel.empty()) {
        // Add BasicConstraints: CA:TRUE
        X509V3_CTX x3ctx; X509V3_set_ctx_nodb(&x3ctx);
        X509V3_set_ctx(&x3ctx, cert, cert, nullptr, nullptr, 0);
        X509_EXTENSION* ext = X509V3_EXT_conf_nid(nullptr, &x3ctx, NID_basic_constraints, "CA:TRUE");
        if (ext) { X509_add_ext(cert, ext, -1); X509_EXTENSION_free(ext); }
    }
    if (X509_sign(cert, pkey, EVP_sha256()) == 0) sslDie("keygen: X509_sign failed");

    std::string certPath = outPrefix + "-cert.pem";
    BIO* cbio = BIO_new_file(certPath.c_str(), "w");
    if (!cbio || PEM_write_bio_X509(cbio, cert) != 1) sslDie("keygen: write cert failed");
    BIO_free(cbio);

    X509_free(cert);
    EVP_PKEY_free(pkey);

    std::cout << "Key : " << keyPath  << "\n";
    std::cout << "Cert: " << certPath << "\n";
    return 0;
}

// ─── sign ─────────────────────────────────────────────────────────────────────

static int cmdSign(int argc, char** argv) {
    std::string keyPath, certPath, pluginId, version, pluginPath;
    for (int i = 0; i < argc; ++i) {
        if (std::string{argv[i]} == "--key"     && i+1 < argc) keyPath    = argv[++i];
        if (std::string{argv[i]} == "--cert"    && i+1 < argc) certPath   = argv[++i];
        if (std::string{argv[i]} == "--id"      && i+1 < argc) pluginId   = argv[++i];
        if (std::string{argv[i]} == "--version" && i+1 < argc) version    = argv[++i];
        else if (argv[i][0] != '-' && pluginPath.empty()) pluginPath = argv[i];
    }
    if (keyPath.empty() || certPath.empty() || pluginId.empty() || version.empty() || pluginPath.empty()) {
        std::cerr << "Usage: sign --key <k.pem> --cert <c.pem> --id <id> --version <v> <plugin.so>\n";
        return 1;
    }

    // Load private key
    BIO* kbio = BIO_new_file(keyPath.c_str(), "r");
    if (!kbio) sslDie("sign: cannot open key file");
    EVP_PKEY* pkey = PEM_read_bio_PrivateKey(kbio, nullptr, nullptr, nullptr);
    BIO_free(kbio);
    if (!pkey) sslDie("sign: PEM_read_bio_PrivateKey failed");

    // Load certificate (DER)
    BIO* cbio = BIO_new_file(certPath.c_str(), "r");
    if (!cbio) sslDie("sign: cannot open cert file");
    X509* cert = PEM_read_bio_X509(cbio, nullptr, nullptr, nullptr);
    BIO_free(cbio);
    if (!cert) sslDie("sign: PEM_read_bio_X509 failed");
    int derLen = i2d_X509(cert, nullptr);
    std::vector<uint8_t> certDer(static_cast<size_t>(derLen));
    uint8_t* certDerPtr = certDer.data();
    i2d_X509(cert, &certDerPtr);
    X509_free(cert);

    // Hash the plugin binary
    auto pluginData = readFileBin(pluginPath);
    std::vector<uint8_t> digest(SHA256_DIGEST_LENGTH);
    SHA256(pluginData.data(), pluginData.size(), digest.data());

    // signed content: hash || id || version || timestamp (8 bytes LE)
    int64_t ts = static_cast<int64_t>(std::time(nullptr));
    std::vector<uint8_t> signedData = digest;
    signedData.insert(signedData.end(), pluginId.begin(), pluginId.end());
    signedData.insert(signedData.end(), version.begin(), version.end());
    {
        auto u = static_cast<uint64_t>(ts);
        for (int i = 0; i < 8; ++i) signedData.push_back(static_cast<uint8_t>(u >> (i*8)));
    }

    // Sign
    EVP_MD_CTX* mctx = EVP_MD_CTX_new();
    size_t sigLen = 0;
    EVP_DigestSignInit(mctx, nullptr, EVP_sha256(), nullptr, pkey);
    EVP_DigestSign(mctx, nullptr, &sigLen, signedData.data(), signedData.size());
    std::vector<uint8_t> sig(sigLen);
    EVP_DigestSign(mctx, sig.data(), &sigLen, signedData.data(), signedData.size());
    sig.resize(sigLen);
    EVP_MD_CTX_free(mctx);
    EVP_PKEY_free(pkey);

    // Build .nnsig sidecar
    std::vector<uint8_t> sidecar;
    sidecar.insert(sidecar.end(), {'N','N','S','G'});
    writeU32LE(sidecar, 1);      // version
    writeI64LE(sidecar, ts);
    writeStr(sidecar, pluginId);
    writeStr(sidecar, version);
    writeVec(sidecar, certDer);
    writeVec(sidecar, sig);

    std::string outPath = pluginPath + ".nnsig";
    writeFileBin(outPath, sidecar.data(), sidecar.size());
    std::cout << "Signed: " << outPath << "\n";
    return 0;
}

// ─── verify ───────────────────────────────────────────────────────────────────

static int cmdVerify(int argc, char** argv) {
    std::string storePath, pluginPath;
    for (int i = 0; i < argc; ++i) {
        if (std::string{argv[i]} == "--store" && i+1 < argc) storePath  = argv[++i];
        else if (argv[i][0] != '-') pluginPath = argv[i];
    }
    if (pluginPath.empty()) { std::cerr << "Usage: verify [--store <trust.nnts>] <plugin.so>\n"; return 1; }

    nnstudio::trust::TrustStore store = nnstudio::trust::TrustStore::createDefault();
    if (!storePath.empty()) {
        auto r = store.load(storePath);
        if (!r.ok()) { std::cerr << "Store load error: " << r.error() << "\n"; return 3; }
    }

    nnstudio::trust::TrustVerifier verifier(store);
    auto r = verifier.verify(pluginPath);
    if (!r.ok()) { std::cerr << "Verify error: " << r.error() << "\n"; return 3; }

    const auto& vr = r.value();
    const char* levelStr =
        vr.level == nnstudio::trust::TrustLevel::Root       ? "Root" :
        vr.level == nnstudio::trust::TrustLevel::Enterprise ? "Enterprise" :
        vr.level == nnstudio::trust::TrustLevel::Community  ? "Community" : "Untrusted";

    std::cout << "Plugin  : " << vr.pluginId  << " v" << vr.version  << "\n";
    std::cout << "Signer  : " << vr.signerDN  << "\n";
    std::cout << "CA      : " << vr.caDN      << "\n";
    std::cout << "Level   : " << levelStr     << "\n";
    return (vr.level == nnstudio::trust::TrustLevel::Untrusted) ? 2 : 0;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "nnstudio-sign <subcommand> [options]\n"
                     "  keygen        Generate ECDSA P-256 keypair\n"
                     "  sign          Sign a plugin binary\n"
                     "  verify        Verify a plugin's trust level\n"
                     "  create-tup    Create a Trust Update Packet [Phase 2 TODO]\n"
                     "  issue-enterprise-ca   Issue an enterprise CA cert [Phase 2 TODO]\n";
        return 1;
    }
    std::string sub = argv[1];
    if (sub == "keygen")  return cmdKeygen(argc - 2, argv + 2);
    if (sub == "sign")    return cmdSign  (argc - 2, argv + 2);
    if (sub == "verify")  return cmdVerify(argc - 2, argv + 2);
    std::cerr << "Unknown subcommand: " << sub << "\n";
    return 1;
}
