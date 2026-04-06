/* =============================================================================
 * test_bpe_tokenizer.cpp — unit tests for the built-in BPE tokenizer plugin
 *
 * Tests the C ABI exported by BpeTokenizerPlugin.cpp:
 *   - Descriptor fields
 *   - encode() / decode() roundtrip
 *   - vocab_size / token_to_str / str_to_token
 *
 * Compiled with BpeTokenizerPlugin.cpp directly so no dlopen is needed.
 * Phase 2 milestone: validates the plugin C ABI and algorithm correctness.
 * ============================================================================ */

#include <gtest/gtest.h>
#include <cstring>
#include <plugin-api/nnstudio_plugin.h>

// The plugin C ABI — including the real descriptor function
extern "C" const NNPluginDescriptor* nnstudio_plugin_descriptor(void);

// ── Fixture ───────────────────────────────────────────────────────────────────

class BpeTest : public ::testing::Test {
protected:
    void SetUp() override {
        desc_   = nnstudio_plugin_descriptor();
        ASSERT_NE(desc_, nullptr);
        handle_ = desc_->create();
        ASSERT_NE(handle_, nullptr);
        vtable_ = static_cast<const NNTokenizerVTable*>(desc_->vtable);
        ASSERT_NE(vtable_, nullptr);
    }
    void TearDown() override {
        if (desc_ && handle_) desc_->destroy(handle_);
    }

    const NNPluginDescriptor*  desc_{nullptr};
    NNPluginHandle             handle_{nullptr};
    const NNTokenizerVTable*   vtable_{nullptr};
};

// ── Descriptor fields ─────────────────────────────────────────────────────────

TEST_F(BpeTest, Descriptor_ApiVersion) {
    EXPECT_EQ(desc_->api_version, NNSTUDIO_PLUGIN_API_VERSION);
}

TEST_F(BpeTest, Descriptor_Type) {
    EXPECT_EQ(desc_->type, NN_PLUGIN_TOKENIZER);
}

TEST_F(BpeTest, Descriptor_Id) {
    ASSERT_NE(desc_->id, nullptr);
    EXPECT_STREQ(desc_->id, "io.nnstudio.builtin.bpe-tokenizer");
}

TEST_F(BpeTest, Descriptor_License) {
    ASSERT_NE(desc_->license, nullptr);
    EXPECT_STREQ(desc_->license, "MIT");
}

// ── Vocab ─────────────────────────────────────────────────────────────────────

TEST_F(BpeTest, VocabSize_IsPositive) {
    EXPECT_GT(vtable_->vocab_size(handle_), 0u);
}

TEST_F(BpeTest, VocabSize_AtLeast256ByteTokens) {
    EXPECT_GE(vtable_->vocab_size(handle_), 256u);
}

TEST_F(BpeTest, TokenToStr_ByteRange) {
    // Token 0 is the null byte — strlen is 0 by definition in C strings; just check non-null
    ASSERT_NE(vtable_->token_to_str(handle_, 0), nullptr);
    // Tokens 1-255 must map to a single character each
    for (int i = 1; i < 256; ++i) {
        const char* s = vtable_->token_to_str(handle_, i);
        ASSERT_NE(s, nullptr) << "token_to_str returned null for id=" << i;
        EXPECT_EQ(std::strlen(s), 1u) << "byte token id=" << i << " maps to non-single-char";
    }
}

TEST_F(BpeTest, StrToToken_ByteRange) {
    char buf[2] = {0, 0};
    for (int c = 'a'; c <= 'z'; ++c) {
        buf[0] = static_cast<char>(c);
        int32_t id = vtable_->str_to_token(handle_, buf);
        EXPECT_EQ(id, c) << "str_to_token failed for char '" << buf[0] << "'";
    }
}

TEST_F(BpeTest, StrToToken_Unknown_ReturnsNegOne) {
    EXPECT_EQ(vtable_->str_to_token(handle_, "\xFF\xFE\xFD"), -1);
}

// ── Encode ────────────────────────────────────────────────────────────────────

TEST_F(BpeTest, Encode_ReturnsNonEmpty) {
    int32_t* ids = nullptr; int32_t count = 0;
    vtable_->encode(handle_, "hello", &ids, &count);
    EXPECT_GT(count, 0);
    ASSERT_NE(ids, nullptr);
    vtable_->free_result(handle_, ids);
}

TEST_F(BpeTest, Encode_EmptyString_ReturnsZero) {
    int32_t* ids = nullptr; int32_t count = 0;
    vtable_->encode(handle_, "", &ids, &count);
    EXPECT_EQ(count, 0);
    if (ids) vtable_->free_result(handle_, ids);
}

TEST_F(BpeTest, Encode_AllTokenIdsInRange) {
    int32_t* ids = nullptr; int32_t count = 0;
    vtable_->encode(handle_, "the quick brown fox", &ids, &count);
    ASSERT_GT(count, 0);
    uint32_t vs = vtable_->vocab_size(handle_);
    for (int32_t i = 0; i < count; ++i) {
        EXPECT_GE(ids[i], 0) << "negative token id at pos " << i;
        EXPECT_LT(static_cast<uint32_t>(ids[i]), vs)
            << "token id out of vocab at pos " << i;
    }
    vtable_->free_result(handle_, ids);
}

TEST_F(BpeTest, Encode_MergesApplied_TheIsOneMergedToken) {
    // "the" should be merged into a single token (merge rule exists for it)
    int32_t* ids = nullptr; int32_t count = 0;
    vtable_->encode(handle_, "the", &ids, &count);
    ASSERT_GT(count, 0);
    // 'the' as 3 chars would give count=3; check we have fewer (merge happened)
    EXPECT_LE(count, 2) << "Expected BPE merge to reduce 'the' to ≤2 tokens";
    vtable_->free_result(handle_, ids);
}

// ── Decode ────────────────────────────────────────────────────────────────────

TEST_F(BpeTest, Decode_SingleByteToken_Roundtrip) {
    int32_t ids[] = {static_cast<int32_t>('h'),
                     static_cast<int32_t>('i')};
    char* text = vtable_->decode(handle_, ids, 2);
    ASSERT_NE(text, nullptr);
    EXPECT_STREQ(text, "hi");
    vtable_->free_result(handle_, text);
}

TEST_F(BpeTest, Decode_EmptyInput_ReturnsEmptyString) {
    char* text = vtable_->decode(handle_, nullptr, 0);
    // Implementation should return a valid empty string or null gracefully
    if (text) {
        EXPECT_STREQ(text, "");
        vtable_->free_result(handle_, text);
    }
}

TEST_F(BpeTest, Encode_Decode_Roundtrip_ASCII) {
    const char* input = "hello world";
    int32_t* ids = nullptr; int32_t count = 0;
    vtable_->encode(handle_, input, &ids, &count);
    ASSERT_GT(count, 0);

    char* decoded = vtable_->decode(handle_, ids, count);
    ASSERT_NE(decoded, nullptr);
    EXPECT_STREQ(decoded, input);

    vtable_->free_result(handle_, ids);
    vtable_->free_result(handle_, decoded);
}

TEST_F(BpeTest, DocRef_IsNonNull) {
    const char* ref = vtable_->doc_ref(handle_);
    ASSERT_NE(ref, nullptr);
    EXPECT_NE(std::strlen(ref), 0u);
}
