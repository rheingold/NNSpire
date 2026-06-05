/* ============================================================================
 * test_trust_store.cpp — TrustStore in-memory API tests
 * Tests addCa / revokeCa / findByDN / count / empty.
 * No file I/O, no OpenSSL required — pure C++ vector operations.
 * ============================================================================ */

#include <gtest/gtest.h>
#include <plugin-api/trust/TrustStore.h>

using namespace NNSpire::trust;

// ─── helpers ──────────────────────────────────────────────────────────────────

static CaEntry makeEntry(TrustLevel level, const char* dn) {
    // minimal non-empty DER placeholder
    return CaEntry{level, dn, {0x01, 0x02, 0x03}};
}

// ─── TrustStore ───────────────────────────────────────────────────────────────

TEST(TrustStore, InitiallyEmpty) {
    TrustStore store;
    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.count(), 0u);
    EXPECT_EQ(store.entries().size(), 0u);
    EXPECT_EQ(store.findByDN("CN=Test CA"), nullptr);
}

TEST(TrustStore, AddCaIncreasesCount) {
    TrustStore store;
    auto r = store.addCa(makeEntry(TrustLevel::Community, "CN=CommCA"));
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(store.count(), 1u);
    EXPECT_FALSE(store.empty());
}

TEST(TrustStore, AddCaEmptyDerReturnsError) {
    TrustStore store;
    CaEntry bad{TrustLevel::Enterprise, "CN=Bad", {}};
    auto r = store.addCa(std::move(bad));
    EXPECT_FALSE(r.ok());
    EXPECT_FALSE(r.error().empty());
    EXPECT_EQ(store.count(), 0u);
}

TEST(TrustStore, AddCaDuplicateDnIsIdempotent) {
    TrustStore store;
    store.addCa(makeEntry(TrustLevel::Enterprise, "CN=EntCA"));
    auto r = store.addCa(makeEntry(TrustLevel::Enterprise, "CN=EntCA"));
    EXPECT_TRUE(r.ok());          // no error — de-duplicated silently
    EXPECT_EQ(store.count(), 1u); // still exactly one entry
}

TEST(TrustStore, AddMultipleDistinctEntries) {
    TrustStore store;
    store.addCa(makeEntry(TrustLevel::Community,  "CN=CommCA"));
    store.addCa(makeEntry(TrustLevel::Enterprise, "CN=EntCA"));
    store.addCa(makeEntry(TrustLevel::Community,  "CN=CommCA2"));
    EXPECT_EQ(store.count(), 3u);
}

TEST(TrustStore, FindByDNReturnsCorrectEntry) {
    TrustStore store;
    store.addCa(makeEntry(TrustLevel::Community,  "CN=Alpha"));
    store.addCa(makeEntry(TrustLevel::Enterprise, "CN=Beta"));

    const CaEntry* alpha = store.findByDN("CN=Alpha");
    ASSERT_NE(alpha, nullptr);
    EXPECT_EQ(alpha->level,     TrustLevel::Community);
    EXPECT_EQ(alpha->subjectDN, "CN=Alpha");

    const CaEntry* beta = store.findByDN("CN=Beta");
    ASSERT_NE(beta, nullptr);
    EXPECT_EQ(beta->level, TrustLevel::Enterprise);

    EXPECT_EQ(store.findByDN("CN=NotThere"), nullptr);
}

TEST(TrustStore, RevokeCaRemovesEntry) {
    TrustStore store;
    store.addCa(makeEntry(TrustLevel::Community, "CN=ByeBye"));
    ASSERT_EQ(store.count(), 1u);

    auto r = store.revokeCa("CN=ByeBye");
    EXPECT_TRUE(r.ok());
    EXPECT_TRUE(store.empty());
    EXPECT_EQ(store.findByDN("CN=ByeBye"), nullptr);
}

TEST(TrustStore, RevokeCaNotFoundReturnsError) {
    TrustStore store;
    auto r = store.revokeCa("CN=Ghost");
    EXPECT_FALSE(r.ok());
    EXPECT_FALSE(r.error().empty());
}

TEST(TrustStore, RevokeLeavesOtherEntriesIntact) {
    TrustStore store;
    store.addCa(makeEntry(TrustLevel::Community,  "CN=C1"));
    store.addCa(makeEntry(TrustLevel::Community,  "CN=C2"));
    store.addCa(makeEntry(TrustLevel::Enterprise, "CN=E1"));
    ASSERT_EQ(store.count(), 3u);

    store.revokeCa("CN=C2");
    EXPECT_EQ(store.count(), 2u);
    EXPECT_EQ(store.findByDN("CN=C2"), nullptr);
    EXPECT_NE(store.findByDN("CN=C1"), nullptr);
    EXPECT_NE(store.findByDN("CN=E1"), nullptr);
}
