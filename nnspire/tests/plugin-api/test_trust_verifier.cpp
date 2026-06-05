/* ============================================================================
 * test_trust_verifier.cpp — TrustVerifier edge-case tests
 * Tests fallback behaviour when no .nnsig sidecar is present.
 * Full cryptographic path requires a real signed plugin (Phase 2 integration).
 * ============================================================================ */

#include <gtest/gtest.h>
#include <plugin-api/trust/TrustVerifier.h>

using namespace NNSpire::trust;

// ─── fixtures ─────────────────────────────────────────────────────────────────

struct TrustVerifierTest : ::testing::Test {
    TrustStore   store;   // empty store — no known CAs
    TrustVerifier verifier{store};
};

// ─── TrustVerifier ────────────────────────────────────────────────────────────

// When no .nnsig sidecar exists next to the plugin, the implementation returns
// TrustLevel::Untrusted (not an error) — the caller decides whether to allow it.
TEST_F(TrustVerifierTest, MissingSidecarReturnsUntrusted) {
    auto r = verifier.verify("no-such-plugin.so");
    ASSERT_TRUE(r.ok()) << "missing sidecar should not be an error — it yields Untrusted";
    EXPECT_EQ(r.value().level,    TrustLevel::Untrusted);
    EXPECT_EQ(r.value().pluginId, "no-such-plugin.so");
    EXPECT_EQ(r.value().version,  "");
    EXPECT_EQ(r.value().signerDN, "");
    EXPECT_EQ(r.value().caDN,     "");
}

// atLeast should succeed but return false — Untrusted does not meet Community.
TEST_F(TrustVerifierTest, AtLeastCommunityReturnsFalseForUnsignedPlugin) {
    auto r = verifier.atLeast("no-such-plugin.so", TrustLevel::Community);
    ASSERT_TRUE(r.ok());
    EXPECT_FALSE(r.value());
}

// atLeast should also return false for Enterprise.
TEST_F(TrustVerifierTest, AtLeastEnterpriseReturnsFalseForUnsignedPlugin) {
    auto r = verifier.atLeast("no-such-plugin.so", TrustLevel::Enterprise);
    ASSERT_TRUE(r.ok());
    EXPECT_FALSE(r.value());
}

// Untrusted meets Untrusted — atLeast(Untrusted) should be true.
TEST_F(TrustVerifierTest, AtLeastUntrustedReturnsTrueForUnsignedPlugin) {
    auto r = verifier.atLeast("no-such-plugin.so", TrustLevel::Untrusted);
    ASSERT_TRUE(r.ok());
    EXPECT_TRUE(r.value());
}

// The TrustLevel ordering: Untrusted < Community < Enterprise < Root.
TEST(TrustLevel, OrderingHolds) {
    EXPECT_LT(static_cast<uint32_t>(TrustLevel::Untrusted),
              static_cast<uint32_t>(TrustLevel::Community));
    EXPECT_LT(static_cast<uint32_t>(TrustLevel::Community),
              static_cast<uint32_t>(TrustLevel::Enterprise));
    EXPECT_LT(static_cast<uint32_t>(TrustLevel::Enterprise),
              static_cast<uint32_t>(TrustLevel::Root));
}
