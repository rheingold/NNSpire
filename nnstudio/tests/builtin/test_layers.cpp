/* ============================================================================
 * test_layers.cpp — unit tests for Dropout, BatchNorm1d, LayerNorm,
 *                   and ActivationsFnLayer reentrance
 * Phase 1 unit tests
 * ============================================================================
*/
#include <gtest/gtest.h>
#include <builtin/layers/NormLayers.h>
#include <builtin/layers/ActivationsFnLayer.h>
#include <builtin/layers/ActivationFunctors.h>
#include <builtin/layers/Dense.h>
#include <builtin/backends/CpuBackend.h>
#include <core/BackendRegistry.h>
#include <core/DType.h>
#include <cmath>
#include <numeric>

using namespace nnstudio::core;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::backends;

class LayerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ─── Dropout ─────────────────────────────────────────────────────────────────

TEST_F(LayerTest, Dropout_TrainingMode_ZerosSomeElements) {
    Dropout drop(0.5f);
    drop.setSeed(12345u);
    drop.build({100});
    auto x = Tensor::ones({100});
    auto res = drop.forward(x);
    ASSERT_TRUE(res.ok());
    int zeros = 0;
    for (int64_t i = 0; i < 100; ++i)
        if (res.value().flat(i) == 0.0f) ++zeros;
    // With p=0.5 and N=100 we expect roughly 50 zeros; allow ±30 for randomness
    EXPECT_GT(zeros,  5);
    EXPECT_LT(zeros, 95);
}

TEST_F(LayerTest, Dropout_EvalMode_IsIdentity) {
    Dropout drop(0.5f);
    drop.build({4});
    drop.setTraining(false);
    auto x   = Tensor::fromData({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto res = drop.forward(x);
    ASSERT_TRUE(res.ok());
    for (int64_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(res.value().flat(i), x.flat(i));
}

TEST_F(LayerTest, Dropout_ZeroRate_IsIdentity) {
    Dropout drop(0.0f);
    drop.build({4});
    auto x   = Tensor::fromData({1.0f, 2.0f, 3.0f, 4.0f}, {4});
    auto res = drop.forward(x);
    ASSERT_TRUE(res.ok());
    for (int64_t i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(res.value().flat(i), x.flat(i));
}

TEST_F(LayerTest, Dropout_Backward_ApplyMask) {
    Dropout drop(0.5f);
    drop.setSeed(99u);
    drop.build({6});
    auto x    = Tensor::ones({6});
    auto fwd  = drop.forward(x);
    ASSERT_TRUE(fwd.ok());
    // Backward with all-ones gradient — result should be same pattern as forward
    auto bwd  = drop.backward(Tensor::ones({6}));
    ASSERT_TRUE(bwd.ok());
    // Each non-zero forward output should correspond to non-zero backward
    for (int64_t i = 0; i < 6; ++i) {
        bool fwdNonZero = (fwd.value().flat(i) != 0.0f);
        bool bwdNonZero = (bwd.value().flat(i) != 0.0f);
        EXPECT_EQ(fwdNonZero, bwdNonZero) << "mask mismatch at i=" << i;
    }
}

// ─── BatchNorm1d ─────────────────────────────────────────────────────────────

TEST_F(LayerTest, BatchNorm1d_HasTwoParameters) {
    BatchNorm1d bn;
    bn.build({3, 4});
    EXPECT_EQ(bn.parameters().size(), 2u);
    EXPECT_EQ(bn.parameters()[0]->name, "gamma");
    EXPECT_EQ(bn.parameters()[1]->name, "beta");
}

TEST_F(LayerTest, BatchNorm1d_GammaOnes_BetaZeros) {
    BatchNorm1d bn;
    bn.build({3, 2});
    const auto& gamma = bn.parameters()[0]->tensor;
    const auto& beta  = bn.parameters()[1]->tensor;
    for (int64_t i = 0; i < 2; ++i) {
        EXPECT_FLOAT_EQ(gamma.flat(i), 1.0f);
        EXPECT_FLOAT_EQ(beta .flat(i), 0.0f);
    }
}

TEST_F(LayerTest, BatchNorm1d_ForwardNormalizesPerFeature) {
    // x = [[1,3],[3,7],[5,5]]
    // col0 mean=3, var=8/3; col1 mean=5, var=8/3
    // With gamma=1, beta=0: output should have ~zero mean, ~unit variance per feature
    BatchNorm1d bn;
    bn.build({3, 2});
    auto x = Tensor::fromData({1.f,3.f, 3.f,7.f, 5.f,5.f}, {3, 2});
    auto res = bn.forward(x);
    ASSERT_TRUE(res.ok());
    const auto& y = res.value();
    // Check mean of each column is near 0
    float sumF0 = 0.f, sumF1 = 0.f;
    for (int64_t n = 0; n < 3; ++n) {
        sumF0 += y.flat(n * 2 + 0);
        sumF1 += y.flat(n * 2 + 1);
    }
    EXPECT_NEAR(sumF0 / 3.f, 0.0f, 1e-5f);
    EXPECT_NEAR(sumF1 / 3.f, 0.0f, 1e-5f);
}

TEST_F(LayerTest, BatchNorm1d_RejectsNonBatchInput) {
    BatchNorm1d bn;
    auto res = bn.build({4});   // 1-D is invalid
    EXPECT_FALSE(res.ok());
}

TEST_F(LayerTest, BatchNorm1d_UpdatesRunningStats) {
    BatchNorm1d bn(1e-5f, 0.5f);   // high momentum for fast update
    bn.build({2, 3});
    auto x = Tensor::fromData({1.f,2.f,3.f, 4.f,5.f,6.f}, {2,3});
    bn.forward(x);
    // Running mean should have moved from 0 towards batch mean
    // batch mean col0=(1+4)/2=2.5
    EXPECT_GT(bn.runningMean().flat(0), 0.0f);
}

// ─── LayerNorm ────────────────────────────────────────────────────────────────

TEST_F(LayerTest, LayerNorm_HasTwoParameters) {
    LayerNorm ln(4);
    ln.build({2, 4});
    EXPECT_EQ(ln.parameters().size(), 2u);
    EXPECT_EQ(ln.parameters()[0]->name, "gamma");
    EXPECT_EQ(ln.parameters()[1]->name, "beta");
}

TEST_F(LayerTest, LayerNorm_GammaOnes_BetaZeros) {
    LayerNorm ln(3);
    ln.build({1, 3});
    const auto& gamma = ln.parameters()[0]->tensor;
    const auto& beta  = ln.parameters()[1]->tensor;
    for (int64_t d = 0; d < 3; ++d) {
        EXPECT_FLOAT_EQ(gamma.flat(d), 1.0f);
        EXPECT_FLOAT_EQ(beta .flat(d), 0.0f);
    }
}

TEST_F(LayerTest, LayerNorm_ForwardZeroMeanPerRow) {
    // x = [[1,3,5],[2,4,6]]
    // Row 0 mean=3, row 1 mean=4 → output rows should have near-zero mean
    LayerNorm ln(-1);  // infer dim
    ln.build({2, 3});
    auto x   = Tensor::fromData({1.f,3.f,5.f, 2.f,4.f,6.f}, {2,3});
    auto res = ln.forward(x);
    ASSERT_TRUE(res.ok());
    const auto& y = res.value();
    float sum0 = y.flat(0) + y.flat(1) + y.flat(2);
    float sum1 = y.flat(3) + y.flat(4) + y.flat(5);
    EXPECT_NEAR(sum0 / 3.0f, 0.0f, 1e-5f);
    EXPECT_NEAR(sum1 / 3.0f, 0.0f, 1e-5f);
}

TEST_F(LayerTest, LayerNorm_ForwardUnitVariancePerRow) {
    LayerNorm ln(-1);
    ln.build({1, 4});
    // x = [1,2,3,4]: mean=2.5, var=1.25
    auto x   = Tensor::fromData({1.f, 2.f, 3.f, 4.f}, {1, 4});
    auto res = ln.forward(x);
    ASSERT_TRUE(res.ok());
    const auto& y = res.value();
    float mean = 0.0f;
    for (int64_t d = 0; d < 4; ++d) mean += y.flat(d);
    mean /= 4.0f;
    float var = 0.0f;
    for (int64_t d = 0; d < 4; ++d) var += (y.flat(d) - mean) * (y.flat(d) - mean);
    var /= 4.0f;
    EXPECT_NEAR(var, 1.0f, 5e-4f);
}

TEST_F(LayerTest, LayerNorm_1DInput) {
    LayerNorm ln(4);
    ln.build({4});
    auto x   = Tensor::fromData({1.f, 2.f, 3.f, 4.f}, {4});
    auto res = ln.forward(x);
    ASSERT_TRUE(res.ok());
    EXPECT_EQ(res.value().ndim(), 1);
    EXPECT_EQ(res.value().numel(), 4);
}

// ─── ActivationsFnLayer reentrance (ADR-020) ──────────────────────────────────

TEST_F(LayerTest, ActivationsFnLayer_TwoInstances_Independent) {
    // Two separate ActivationsFnLayer instances wrap the same type of functor.
    // Each stores its own ctx_, so they are independent and can both run.
    using ReLULayer = ActivationsFnLayer<ReLUFn>;
    ReLULayer layer1, layer2;
    layer1.build({4});
    layer2.build({4});

    auto x1 = Tensor::fromData({-1.f, 2.f, -3.f, 4.f}, {4});
    auto x2 = Tensor::fromData({1.f, -2.f, 3.f, -4.f}, {4});

    auto r1 = layer1.forward(x1);
    auto r2 = layer2.forward(x2);
    ASSERT_TRUE(r1.ok());
    ASSERT_TRUE(r2.ok());

    // Backward on layer1 must use x1's mask, not x2's
    auto g1 = layer1.backward(Tensor::ones({4}));
    ASSERT_TRUE(g1.ok());
    EXPECT_FLOAT_EQ(g1.value().flat(0), 0.0f);  // x1[0]=-1 < 0
    EXPECT_FLOAT_EQ(g1.value().flat(1), 1.0f);  // x1[1]=2  > 0

    // Backward on layer2 must use x2's mask
    auto g2 = layer2.backward(Tensor::ones({4}));
    ASSERT_TRUE(g2.ok());
    EXPECT_FLOAT_EQ(g2.value().flat(0), 1.0f);  // x2[0]=1   > 0
    EXPECT_FLOAT_EQ(g2.value().flat(1), 0.0f);  // x2[1]=-2  < 0
}

// ─── Dense dtype mismatch ──────────────────────────────────────────────────────
// CpuBackend is Float32-only.  Passing an Int32 input must return
// Result::error(DTypeMismatch) rather than silently reinterpreting bits.

TEST_F(LayerTest, Dense_DTypeMismatch_ReturnsError) {
    Dense d(4);
    auto buildR = d.build({2, 3});
    ASSERT_TRUE(buildR.ok());

    // Build the dense layer with Float32 (its normal path) so weights exist,
    // then pass an Int32 tensor to forward() to trigger the dtype guard.
    Tensor bad = Tensor({1, 3}, DType::Int32);
    auto r = d.forward(bad);

    ASSERT_FALSE(r.ok()) << "Expected DTypeMismatch error but got ok()";
    EXPECT_EQ(r.error().code, ErrorCode::DTypeMismatch);
}
