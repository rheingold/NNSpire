/* ============================================================================
 * test_plugin_loader.cpp — PluginLoader construction and load-path tests
 * Tests policy enforcement and dlopen-failure paths.
 * Full load-success path requires a real built plugin (Phase 2 integration).
 * ============================================================================ */

#include <gtest/gtest.h>
#include <plugin-api/PluginLoader.h>

using namespace NNSpire::plugins;
using namespace NNSpire::trust;

// ─── PluginLoader ─────────────────────────────────────────────────────────────

TEST(PluginLoader, ConstructorAndPolicyAccessor) {
    TrustStore store;
    PluginLoader loader(store, LoadPolicy::RequireCommunity);
    EXPECT_EQ(loader.policy(), LoadPolicy::RequireCommunity);
}

TEST(PluginLoader, StoreAccessorReturnsReference) {
    TrustStore store;
    store.addCa(CaEntry{TrustLevel::Enterprise, "CN=Test", {0x01}});
    PluginLoader loader(store, LoadPolicy::RequireEnterprise);
    EXPECT_EQ(loader.store().count(), 1u);
}

// For a plugin with no .nnsig sidecar:
// verify() returns Untrusted (fallback). RequireEnterprise and RequireCommunity
// reject Untrusted → load() returns a trust-policy error (not a dlopen error).
TEST(PluginLoader, RequireEnterpriseRejectsUnsignedPlugin) {
    TrustStore store;
    PluginLoader loader(store, LoadPolicy::RequireEnterprise);
    auto r = loader.load("no-such-plugin.so");
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().find("trust policy"), std::string::npos)
        << "expected 'trust policy' in error: " << r.error();
}

TEST(PluginLoader, RequireCommunityRejectsUnsignedPlugin) {
    TrustStore store;
    PluginLoader loader(store, LoadPolicy::RequireCommunity);
    auto r = loader.load("no-such-plugin.so");
    ASSERT_FALSE(r.ok());
    EXPECT_NE(r.error().find("trust policy"), std::string::npos)
        << "expected 'trust policy' in error: " << r.error();
}

// AllowUntrusted passes the trust-policy check (Untrusted level is acceptable),
// so the error comes from dlopen failing to open a non-existent file.
TEST(PluginLoader, AllowUntrustedFailsAtDlopenForNonExistentFile) {
    TrustStore store;
    PluginLoader loader(store, LoadPolicy::AllowUntrusted);
    auto r = loader.load("no-such-plugin.so");
    ASSERT_FALSE(r.ok());
    // Should fail at dlopen, not at trust policy
    EXPECT_EQ(r.error().find("trust policy"), std::string::npos)
        << "unexpected 'trust policy' in error: " << r.error();
    EXPECT_NE(r.error().find("dlopen"), std::string::npos)
        << "expected 'dlopen' in error: " << r.error();
}

// ─── LoadPolicy enumeration completeness ─────────────────────────────────────

TEST(LoadPolicy, ValuesAreDistinct) {
    EXPECT_NE(static_cast<int>(LoadPolicy::RequireEnterprise),
              static_cast<int>(LoadPolicy::RequireCommunity));
    EXPECT_NE(static_cast<int>(LoadPolicy::RequireCommunity),
              static_cast<int>(LoadPolicy::AllowUntrusted));
}
