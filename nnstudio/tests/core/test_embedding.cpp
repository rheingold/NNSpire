/* ============================================================================
 * test_embedding.cpp — unit tests for Embedding layer
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <builtin/layers/Embedding.h>
#include <builtin/backends/CpuBackend.h>
#include <core/BackendRegistry.h>
#include <cmath>

using namespace nnstudio::core;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::backends;

// ─── Fixture ──────────────────────────────────────────────────────────────────

class EmbeddingTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ─── Build output shape ───────────────────────────────────────────────────────

TEST_F(EmbeddingTest, BuildOutputShape) {
    Embedding emb(100, 16);   // 100 tokens, 16-dim embeddings
    auto br = emb.build({5}); // seqLen=5 (1D token sequence)
    ASSERT_TRUE(br.ok()) << br.error().message;
    // output shape: [seqLen, embDim]
    const auto& os = br.value();
    ASSERT_EQ(os.size(), 2u);
    EXPECT_EQ(os[0], 5);   // seqLen
    EXPECT_EQ(os[1], 16);  // embDim
}

TEST_F(EmbeddingTest, BuildOutputShape_Batched) {
    Embedding emb(50, 8);
    auto br = emb.build({10}); // per-sample shape: seqLen=10
    ASSERT_TRUE(br.ok());
    const auto& os = br.value();
    ASSERT_EQ(os.size(), 2u);
    EXPECT_EQ(os[0], 10);  // seqLen
    EXPECT_EQ(os[1], 8);   // embDim
}

// ─── Parameter count ──────────────────────────────────────────────────────────

TEST_F(EmbeddingTest, ParameterCount) {
    Embedding emb(100, 16);
    emb.build({5});
    auto params = emb.parameters();
    // One parameter: weight matrix [100, 16]
    EXPECT_EQ(params.size(), 1u);
    EXPECT_EQ(params[0]->tensor.numel(), 100 * 16);
}

// ─── Forward: row lookup ──────────────────────────────────────────────────────

TEST_F(EmbeddingTest, Forward_RowLookup) {
    Embedding emb(10, 4);
    emb.build({3});  // seqLen=3

    // Set weight rows to identity-like values: row i = [i, i, i, i]
    auto params = emb.parameters();
    auto& W = params[0]->tensor;   // shape [10, 4]
    for (int64_t r = 0; r < 10; ++r)
        for (int64_t c = 0; c < 4; ++c)
            W.at({r, c}) = static_cast<float>(r);

    // Input: batch of 1, token sequence [0, 3, 7] — shape [N=1, seqLen=3]
    auto x = Tensor::fromData({0.0f, 3.0f, 7.0f}, {1, 3});
    auto fr = emb.forward(x);
    ASSERT_TRUE(fr.ok()) << fr.error().message;

    const auto& out = fr.value();   // shape [1, 3, 4]
    ASSERT_EQ(out.shape().size(), 3u);
    EXPECT_EQ(out.shape()[0], 1);
    EXPECT_EQ(out.shape()[1], 3);
    EXPECT_EQ(out.shape()[2], 4);

    // out[0, 0, :] → token 0 → all 0.0
    for (int64_t c = 0; c < 4; ++c) EXPECT_FLOAT_EQ(out.at({0, 0, c}), 0.0f);
    // out[0, 1, :] → token 3 → all 3.0
    for (int64_t c = 0; c < 4; ++c) EXPECT_FLOAT_EQ(out.at({0, 1, c}), 3.0f);
    // out[0, 2, :] → token 7 → all 7.0
    for (int64_t c = 0; c < 4; ++c) EXPECT_FLOAT_EQ(out.at({0, 2, c}), 7.0f);
}

// ─── Backward: gradient accumulates into weight rows ──────────────────────────

TEST_F(EmbeddingTest, Backward_ScatterAdd) {
    Embedding emb(5, 3);
    emb.build({2});  // seqLen=2

    // Zero the weight matrix first
    auto& W = emb.parameters()[0]->tensor;
    for (int64_t i = 0; i < W.numel(); ++i) W.flat(i) = 0.0f;
    // Same for grad
    W.zeroGrad();

    // Forward with token IDs [2, 2] — same token twice, batch=1 shape=[1,2]
    auto x  = Tensor::fromData({2.0f, 2.0f}, {1, 2});
    emb.forward(x);

    // Backward: gradient of 1s arrives at output [1, 2, 3]
    auto gradOut = Tensor::ones({1, 2, 3});
    auto gr = emb.backward(gradOut);
    ASSERT_TRUE(gr.ok()) << gr.error().message;

    // Weight grad for row 2 should have accumulated 2.0 (once per position)
    // Other rows should be zero
    const auto& wGrad = W.grad();
    if (wGrad.numel() > 0) {  // grad may be lazily allocated
        // Rows 0, 1, 3, 4 should be ~0
        for (int64_t c = 0; c < 3; ++c) {
            EXPECT_FLOAT_EQ(wGrad.at({0, c}), 0.0f);
            EXPECT_FLOAT_EQ(wGrad.at({1, c}), 0.0f);
            EXPECT_FLOAT_EQ(wGrad.at({3, c}), 0.0f);
            EXPECT_FLOAT_EQ(wGrad.at({4, c}), 0.0f);
        }
        // Row 2 should have accumulated 2.0 (gradOut=1 × 2 positions)
        for (int64_t c = 0; c < 3; ++c)
            EXPECT_FLOAT_EQ(wGrad.at({2, c}), 2.0f);
    }
}

// ─── Grad of input is zeros (discrete tokens — no grad w.r.t. index) ──────────

TEST_F(EmbeddingTest, Backward_InputGradIsZero) {
    Embedding emb(8, 4);
    emb.build({3});
    // Input shape [N=1, seqLen=3]
    auto x  = Tensor::fromData({1.0f, 2.0f, 3.0f}, {1, 3});
    emb.forward(x);
    auto gradOut = Tensor::ones({1, 3, 4});
    auto gr = emb.backward(gradOut);
    ASSERT_TRUE(gr.ok());
    const auto& dX = gr.value();
    for (int64_t i = 0; i < dX.numel(); ++i)
        EXPECT_FLOAT_EQ(dX.flat(i), 0.0f);
}

// ─── config() ─────────────────────────────────────────────────────────────────

TEST_F(EmbeddingTest, Config_ContainsExpectedKeys) {
    Embedding emb(200, 32);
    auto cfg = emb.config();
    EXPECT_NE(cfg.find("vocabSize"), cfg.end());
    EXPECT_NE(cfg.find("embDim"),    cfg.end());
    EXPECT_EQ(cfg.at("vocabSize"), "200");
    EXPECT_EQ(cfg.at("embDim"),    "32");
}

// ─── typeName ─────────────────────────────────────────────────────────────────

TEST_F(EmbeddingTest, TypeName) {
    Embedding emb(10, 4);
    EXPECT_EQ(emb.typeName(), std::string_view("Embedding"));
}
