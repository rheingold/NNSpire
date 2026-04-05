/* ============================================================================
 * test_losses.cpp — unit tests for MSE, BCE, CrossEntropy, HuberLoss
 * Phase 1 unit tests
 * ============================================================================
*/
#include <gtest/gtest.h>
#include <core/Tensor.h>
#include <core/BackendRegistry.h>
#include <builtin/backends/CpuBackend.h>
#include <builtin/losses/Losses.h>

#include <cmath>

using namespace nnstudio::core;
using namespace nnstudio::builtin::backends;
using namespace nnstudio::builtin::losses;

// Fixture: ensure CPU backend registered once
class LossTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ─── MSE ──────────────────────────────────────────────────────────────────────

TEST_F(LossTest, MSE_ForwardKnownValues) {
    // pred=[1,2,3], target=[1,1,1]  →  errors=[0,1,2]  →  MSE = (0+1+4)/3 ≈ 1.667
    auto pred   = Tensor::fromData({1.0f, 2.0f, 3.0f}, {3});
    auto target = Tensor::fromData({1.0f, 1.0f, 1.0f}, {3});
    MSE mse;
    auto res = mse.compute(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 5.0f / 3.0f, 1e-5f);
}

TEST_F(LossTest, MSE_ZeroLossWhenIdentical) {
    auto t = Tensor::full({4}, 3.14f);
    MSE mse;
    auto res = mse.compute(t, t);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 0.0f, 1e-6f);
}

TEST_F(LossTest, MSE_GradientKnownValues) {
    // dL/dpred = 2*(pred-target)/N
    // pred=[1,2,3], target=[1,1,1], N=3
    // grad = [0, 2/3, 4/3]
    auto pred   = Tensor::fromData({1.0f, 2.0f, 3.0f}, {3});
    auto target = Tensor::fromData({1.0f, 1.0f, 1.0f}, {3});
    MSE mse;
    auto res = mse.gradient(pred, target);
    ASSERT_TRUE(res.ok());
    const auto& g = res.value();
    EXPECT_NEAR(g.flat(0), 0.0f,   1e-5f);
    EXPECT_NEAR(g.flat(1), 2.0f / 3.0f, 1e-5f);
    EXPECT_NEAR(g.flat(2), 4.0f / 3.0f, 1e-5f);
}

// ─── BCE ──────────────────────────────────────────────────────────────────────

TEST_F(LossTest, BCE_ForwardSingleElement) {
    // pred=0.9, target=1.0  →  L = -log(0.9)
    auto pred   = Tensor::fromData({0.9f}, {1});
    auto target = Tensor::fromData({1.0f}, {1});
    BCE bce;
    auto res = bce.compute(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), -std::log(0.9f), 1e-5f);
}

TEST_F(LossTest, BCE_ForwardBothTerms) {
    // pred=0.6, target=0.0  →  L = -log(1-0.6) = -log(0.4)
    auto pred   = Tensor::fromData({0.6f}, {1});
    auto target = Tensor::fromData({0.0f}, {1});
    BCE bce;
    auto res = bce.compute(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), -std::log(0.4f), 1e-5f);
}

TEST_F(LossTest, BCE_GradientDirection) {
    // target=1, pred=0.8  →  grad = (-1/0.8 + 0) / N = -1.25  (negative = loss goes down as pred increases)
    auto pred   = Tensor::fromData({0.8f}, {1});
    auto target = Tensor::fromData({1.0f}, {1});
    BCE bce;
    auto res = bce.gradient(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), -1.0f / 0.8f, 1e-4f);
}

// ─── CrossEntropy ─────────────────────────────────────────────────────────────

TEST_F(LossTest, CrossEntropy_ForwardOneHot) {
    // pred=[0.7, 0.2, 0.1], target=[1,0,0]  →  L = -log(0.7)
    auto pred   = Tensor::fromData({0.7f, 0.2f, 0.1f}, {1, 3});
    auto target = Tensor::fromData({1.0f, 0.0f, 0.0f}, {1, 3});
    CrossEntropy ce;
    auto res = ce.compute(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), -std::log(0.7f), 1e-5f);
}

TEST_F(LossTest, CrossEntropy_GradientSparsity) {
    // target=[1,0,0]  →  only first element has nonzero gradient
    auto pred   = Tensor::fromData({0.7f, 0.2f, 0.1f}, {1, 3});
    auto target = Tensor::fromData({1.0f, 0.0f, 0.0f}, {1, 3});
    CrossEntropy ce;
    auto res = ce.gradient(pred, target);
    ASSERT_TRUE(res.ok());
    const auto& g = res.value();
    EXPECT_NE(g.flat(0), 0.0f);   // class 0 has gradient
    EXPECT_FLOAT_EQ(g.flat(1), 0.0f);  // class 1 — no contribution
    EXPECT_FLOAT_EQ(g.flat(2), 0.0f);  // class 2 — no contribution
}

// ─── HuberLoss ────────────────────────────────────────────────────────────────

TEST_F(LossTest, Huber_QuadraticRegime) {
    // |error| = 0.5 <= delta=1.0  →  L = 0.5 * 0.5^2 = 0.125
    auto pred   = Tensor::fromData({1.5f}, {1});
    auto target = Tensor::fromData({1.0f}, {1});
    HuberLoss h(1.0f);
    auto res = h.compute(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 0.5f * 0.25f, 1e-6f);  // 0.5 * (0.5)^2
}

TEST_F(LossTest, Huber_LinearRegime) {
    // |error| = 2.0 > delta=1.0  →  L = 1.0*(2.0 - 0.5*1.0) = 1.5
    auto pred   = Tensor::fromData({3.0f}, {1});
    auto target = Tensor::fromData({1.0f}, {1});
    HuberLoss h(1.0f);
    auto res = h.compute(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 1.5f, 1e-6f);
}

TEST_F(LossTest, Huber_GradQuadraticRegime) {
    // gradient in quadratic regime = r/N = 0.5/1 = 0.5
    auto pred   = Tensor::fromData({1.5f}, {1});
    auto target = Tensor::fromData({1.0f}, {1});
    HuberLoss h(1.0f);
    auto res = h.gradient(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 0.5f, 1e-6f);
}

TEST_F(LossTest, Huber_GradLinearRegime) {
    // gradient in linear regime = delta * sign(r) / N = 1.0 * 1.0 / 1 = 1.0
    auto pred   = Tensor::fromData({3.0f}, {1});
    auto target = Tensor::fromData({1.0f}, {1});
    HuberLoss h(1.0f);
    auto res = h.gradient(pred, target);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 1.0f, 1e-6f);
}
