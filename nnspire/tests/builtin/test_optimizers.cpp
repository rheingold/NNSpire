/* ============================================================================
 * test_optimizers.cpp — unit tests for SGD, Adam, AdamW, RMSProp,
 *                       StepDecayScheduler
 * Phase 1 unit tests
 * ============================================================================
*/
#include <gtest/gtest.h>
#include <core/Tensor.h>
#include <core/Layer.h>
#include <core/BackendRegistry.h>
#include <builtin/backends/CpuBackend.h>
#include <builtin/optimizers/Optimizers.h>

#include <cmath>

using namespace NNSpire::core;
using namespace NNSpire::builtin::backends;
using namespace NNSpire::builtin::optimizers;

// Fixture: ensure CPU backend registered once
class OptimizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }

    // Helper: build a single-element Parameter with given value and gradient
    static Parameter makeParam(float value, float grad) {
        Parameter p;
        p.name   = "w";
        p.tensor = Tensor::full({1}, value);
        p.tensor.accumulateGrad(Tensor::full({1}, grad));
        return p;
    }
};

// ─── SGD ──────────────────────────────────────────────────────────────────────

TEST_F(OptimizerTest, SGD_BasicUpdate) {
    // θ ← θ - lr*g  →  1.0 - 0.1*1.0 = 0.9
    auto p = makeParam(1.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    SGD sgd(0.1f);
    sgd.step(params);
    EXPECT_NEAR(p.tensor.flat(0), 0.9f, 1e-6f);
}

TEST_F(OptimizerTest, SGD_NegativeGradient) {
    // θ ← 1.0 - 0.1*(-2.0) = 1.2
    auto p = makeParam(1.0f, -2.0f);
    std::vector<Parameter*> params = {&p};
    SGD sgd(0.1f);
    sgd.step(params);
    EXPECT_NEAR(p.tensor.flat(0), 1.2f, 1e-6f);
}

TEST_F(OptimizerTest, SGD_FrozenParamSkipped) {
    auto p  = makeParam(5.0f, 1.0f);
    p.frozen = true;
    std::vector<Parameter*> params = {&p};
    SGD sgd(0.1f);
    sgd.step(params);
    EXPECT_FLOAT_EQ(p.tensor.flat(0), 5.0f);  // unchanged
}

TEST_F(OptimizerTest, SGD_WeightDecay) {
    // g_effective = g + λ*θ = 1 + 0.1*2 = 1.2   →   θ = 2 - 0.1*1.2 = 1.88
    auto p = makeParam(2.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    SGD sgd(/*lr=*/0.1f, /*momentum=*/0.0f, /*weightDecay=*/0.1f);
    sgd.step(params);
    EXPECT_NEAR(p.tensor.flat(0), 2.0f - 0.1f * (1.0f + 0.1f * 2.0f), 1e-5f);
}

TEST_F(OptimizerTest, SGD_MomentumAccumulates) {
    // Step 1: vel = 0.9*0 + 1 = 1.0  →  θ = 1 - 0.1*1 = 0.9
    // Step 2: grad reset to 1.0 again; vel = 0.9*1 + 1 = 1.9  →  θ = 0.9 - 0.1*1.9 = 0.71
    auto p = makeParam(1.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    SGD sgd(/*lr=*/0.1f, /*momentum=*/0.9f);
    sgd.step(params);
    EXPECT_NEAR(p.tensor.flat(0), 0.9f, 1e-5f);

    // Re-set gradient for step 2
    p.tensor.zeroGrad();
    p.tensor.accumulateGrad(Tensor::full({1}, 1.0f));
    sgd.step(params);
    EXPECT_NEAR(p.tensor.flat(0), 0.9f - 0.1f * 1.9f, 1e-5f);
}

// ─── Adam ─────────────────────────────────────────────────────────────────────

TEST_F(OptimizerTest, Adam_MovesInGradientDirection) {
    // Just verify that after one step θ decreases when gradient is positive
    auto p = makeParam(1.0f, 0.5f);
    std::vector<Parameter*> params = {&p};
    Adam adam(1e-3f);
    adam.step(params);
    EXPECT_LT(p.tensor.flat(0), 1.0f);  // moved in negative direction
}

TEST_F(OptimizerTest, Adam_BiasCorrection_StepOne) {
    // At step 1: bc1 = 1 - 0.9^1 = 0.1,  bc2 = 1 - 0.999^1 ≈ 0.001
    // m=0.1*g, v=0.001*g²,  mHat=m/0.1=g,  vHat=v/0.001=g²
    // update = lr * g / (sqrt(g²) + eps) ≈ lr * g/|g|  (for large g vs eps)
    // With g=1:  update ≈ lr = 1e-3,  θ = 1 - 1e-3 ≈ 0.999
    auto p = makeParam(1.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    Adam adam(1e-3f);
    adam.step(params);
    EXPECT_NEAR(p.tensor.flat(0), 1.0f - 1e-3f, 1e-5f);
}

TEST_F(OptimizerTest, Adam_ZeroGradNoChange) {
    auto p = makeParam(3.0f, 0.0f);
    std::vector<Parameter*> params = {&p};
    Adam adam(1e-3f);
    adam.step(params);
    // With g=0: m=0, v=0, update = 0/(0+eps) = 0  →  no change
    EXPECT_NEAR(p.tensor.flat(0), 3.0f, 1e-6f);
}

// ─── AdamW ────────────────────────────────────────────────────────────────────

TEST_F(OptimizerTest, AdamW_DecoupledDecayApplied) {
    // AdamW applies weight decay as θ *= (1 - lr*λ) BEFORE Adam update
    // With lr=1e-3, λ=0.01: scale = 1 - 1e-3*0.01 = 0.99999
    // Then Adam update ≈ -lr (with g=1 at step 1)
    auto p = makeParam(10.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    AdamW adamw(/*lr=*/1e-3f, /*beta1=*/0.9f, /*beta2=*/0.999f,
                /*eps=*/1e-8f, /*weightDecay=*/0.01f);
    adamw.step(params);

    // After decay: 10 * (1 - 1e-3 * 0.01) = 9.9999
    // After Adam update (≈ -1e-3): ≈ 9.9989
    // Just verify it's less than 10 and within expected range
    EXPECT_LT(p.tensor.flat(0), 10.0f);
    EXPECT_GT(p.tensor.flat(0), 9.99f);
}

TEST_F(OptimizerTest, AdamW_VsAdam_WeightDecayDifference) {
    // AdamW should end up with a smaller weight than Adam (stronger regularization)
    auto p1 = makeParam(5.0f, 0.5f);
    auto p2 = makeParam(5.0f, 0.5f);
    std::vector<Parameter*> params1 = {&p1};
    std::vector<Parameter*> params2 = {&p2};

    Adam  adam(1e-3f,  0.9f, 0.999f, 1e-8f, 0.0f);   // no weight decay
    AdamW adamw(1e-3f, 0.9f, 0.999f, 1e-8f, 0.1f);   // with decoupled decay

    adam.step(params1);
    adamw.step(params2);

    // AdamW applies extra decay → smaller weight
    EXPECT_LT(p2.tensor.flat(0), p1.tensor.flat(0));
}

// ─── RMSProp ──────────────────────────────────────────────────────────────────

TEST_F(OptimizerTest, RMSProp_MovesInGradientDirection) {
    auto p = makeParam(1.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    RMSProp rms(1e-2f);
    rms.step(params);
    EXPECT_LT(p.tensor.flat(0), 1.0f);  // positive gradient → weight decreases
}

TEST_F(OptimizerTest, RMSProp_UpdateMagnitude) {
    // At step 1:  E[g²] = (1-α)*g² = 0.01*1 = 0.01
    //             update = lr * g / (sqrt(E[g²]) + eps) = 0.01 * 1 / (0.1 + 1e-8)
    // ≈ 0.01 / 0.1 = 0.1   →   θ = 1 - 0.1 = 0.9
    auto p = makeParam(1.0f, 1.0f);
    std::vector<Parameter*> params = {&p};
    RMSProp rms(/*lr=*/0.01f, /*alpha=*/0.99f, /*eps=*/1e-8f);
    rms.step(params);
    const float alpha   = 0.99f;
    const float g       = 1.0f;
    const float sq      = (1.0f - alpha) * g * g;  // 0.01
    const float update  = 0.01f * g / (std::sqrt(sq) + 1e-8f);
    EXPECT_NEAR(p.tensor.flat(0), 1.0f - update, 1e-5f);
}

TEST_F(OptimizerTest, RMSProp_WeightDecay) {
    // With weight decay, effective gradient = g + λ*θ
    const float theta = 2.0f, g = 0.5f, wd = 0.1f, lr = 0.01f, alpha = 0.99f;
    auto p = makeParam(theta, g);
    std::vector<Parameter*> params = {&p};
    RMSProp rms(lr, alpha, /*eps=*/1e-8f, wd);
    rms.step(params);
    const float gEff   = g + wd * theta;
    const float sq     = (1.0f - alpha) * gEff * gEff;
    const float update = lr * gEff / (std::sqrt(sq) + 1e-8f);
    EXPECT_NEAR(p.tensor.flat(0), theta - update, 1e-5f);
}

// ─── StepDecayScheduler ───────────────────────────────────────────────────────

TEST_F(OptimizerTest, StepDecay_ReducesLRAtMultiple) {
    SGD sgd(1.0f);
    StepDecayScheduler sched(/*stepSize=*/5, /*gamma=*/0.1f);

    for (uint64_t step = 1; step <= 4; ++step)
        sched.onStep(sgd, step);
    EXPECT_FLOAT_EQ(sgd.lr(), 1.0f);  // unchanged before step 5

    sched.onStep(sgd, 5);
    EXPECT_NEAR(sgd.lr(), 0.1f, 1e-7f);  // reduced at step 5

    for (uint64_t step = 6; step <= 9; ++step)
        sched.onStep(sgd, step);
    EXPECT_NEAR(sgd.lr(), 0.1f, 1e-7f);  // no further reduction

    sched.onStep(sgd, 10);
    EXPECT_NEAR(sgd.lr(), 0.01f, 1e-8f);  // reduced again at step 10
}
