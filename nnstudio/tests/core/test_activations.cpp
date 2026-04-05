/* ============================================================================
 * test_activations.cpp â€” activation forward/backward smoke tests
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <builtin/layers/Activations.h>
#include <builtin/backends/CpuBackend.h>
#include <core/BackendRegistry.h>
#include <cmath>

using namespace nnstudio::core;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::backends;

class ActivationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// â”€â”€ ReLU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TEST_F(ActivationTest, ReLUForward) {
    ReLU relu;
    relu.build({4});
    auto x   = Tensor::fromData({-2.0f, -0.5f, 0.0f, 3.0f}, {4});
    auto out = relu.forward(x);
    ASSERT_TRUE(out.ok());
    EXPECT_FLOAT_EQ(out.value().flat(0), 0.0f);
    EXPECT_FLOAT_EQ(out.value().flat(1), 0.0f);
    EXPECT_FLOAT_EQ(out.value().flat(2), 0.0f);
    EXPECT_FLOAT_EQ(out.value().flat(3), 3.0f);
}

TEST_F(ActivationTest, ReLUBackward) {
    ReLU relu;
    relu.build({4});
    auto x    = Tensor::fromData({-1.0f, 0.5f, 0.0f, 2.0f}, {4});
    relu.forward(x);
    auto grad = Tensor::ones({4});
    auto back = relu.backward(grad);
    ASSERT_TRUE(back.ok());
    EXPECT_FLOAT_EQ(back.value().flat(0), 0.0f);  // -1 < 0
    EXPECT_FLOAT_EQ(back.value().flat(1), 1.0f);  // 0.5 > 0
    EXPECT_FLOAT_EQ(back.value().flat(2), 0.0f);  // 0 â†’ 0
    EXPECT_FLOAT_EQ(back.value().flat(3), 1.0f);  // 2 > 0
}

// â”€â”€ Sigmoid â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TEST_F(ActivationTest, SigmoidForward) {
    Sigmoid sig;
    sig.build({3});
    auto x   = Tensor::fromData({0.0f, 2.0f, -2.0f}, {3});
    auto out = sig.forward(x);
    ASSERT_TRUE(out.ok());
    EXPECT_NEAR(out.value().flat(0), 0.5f,    1e-5f);
    EXPECT_NEAR(out.value().flat(1), 0.8808f, 1e-3f);
    EXPECT_NEAR(out.value().flat(2), 0.1192f, 1e-3f);
}

// â”€â”€ Softmax â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TEST_F(ActivationTest, SoftmaxSumOne) {
    Softmax sm;
    sm.build({1, 4});
    auto x   = Tensor::fromData({1.0f, 2.0f, 3.0f, 4.0f}, {1, 4});
    auto out = sm.forward(x);
    ASSERT_TRUE(out.ok());
    float sum = 0.0f;
    for (int64_t i = 0; i < out.value().numel(); ++i)
        sum += out.value().flat(i);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// â”€â”€ GELU â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TEST_F(ActivationTest, GELUAtZero) {
    GELU gelu;
    gelu.build({1});
    auto x   = Tensor::fromData({0.0f}, {1});
    auto out = gelu.forward(x);
    ASSERT_TRUE(out.ok());
    EXPECT_NEAR(out.value().flat(0), 0.0f, 1e-5f);
}

TEST_F(ActivationTest, GELUPositiveApprox) {
    GELU gelu;
    gelu.build({1});
    // GELU(1) â‰ 0.8413
    auto x   = Tensor::fromData({1.0f}, {1});
    auto out = gelu.forward(x);
    ASSERT_TRUE(out.ok());
    EXPECT_NEAR(out.value().flat(0), 0.8413f, 2e-3f);
}


// --- ActivationsFnLayer + IActivation (ADR-020) ---
#include <core/IActivation.h>
#include <builtin/layers/ActivationsFnLayer.h>
#include <algorithm>

// Minimal stateless ReLU implementing IActivation
struct ReLUFn : nnstudio::core::IActivation {
    std::string_view name() const noexcept override { return "ReLU"; }

    nnstudio::core::ActivationForward
    forward(const nnstudio::core::Tensor& x) const override {
        using namespace nnstudio::core;
        Tensor out(x.shape());
        for (int64_t i = 0; i < x.numel(); ++i)
            out.flat(i) = x.flat(i) > 0.0f ? x.flat(i) : 0.0f;
        return ActivationForward{ out, x.clone(), /*ctxIsInput=*/true };
    }

    nnstudio::core::Result<nnstudio::core::Tensor>
    backward(const nnstudio::core::Tensor& gradOut,
             const nnstudio::core::ActivationForward& fwd) const override {
        using namespace nnstudio::core;
        Tensor g(gradOut.shape());
        for (int64_t i = 0; i < gradOut.numel(); ++i)
            g.flat(i) = fwd.ctx.flat(i) > 0.0f ? gradOut.flat(i) : 0.0f;
        return Result<Tensor>(g);
    }
};

using ReLULayer = nnstudio::builtin::layers::ActivationsFnLayer<ReLUFn>;

TEST_F(ActivationTest, ActivationsFnLayer_ReLU_Forward) {
    ReLULayer layer;
    layer.build({4});
    auto x = Tensor::fromData({-2.0f, -1.0f, 0.0f, 3.0f}, {4});
    auto res = layer.forward(x);
    ASSERT_TRUE(res.ok());
    EXPECT_FLOAT_EQ(res.value().flat(0), 0.0f);
    EXPECT_FLOAT_EQ(res.value().flat(1), 0.0f);
    EXPECT_FLOAT_EQ(res.value().flat(2), 0.0f);
    EXPECT_FLOAT_EQ(res.value().flat(3), 3.0f);
}

TEST_F(ActivationTest, ActivationsFnLayer_ReLU_Backward) {
    ReLULayer layer;
    layer.build({4});
    auto x = Tensor::fromData({-1.0f, 2.0f, 0.0f, 4.0f}, {4});
    layer.forward(x);
    auto grad = Tensor::fromData({1.0f, 1.0f, 1.0f, 1.0f}, {4});
    auto res = layer.backward(grad);
    ASSERT_TRUE(res.ok());
    EXPECT_FLOAT_EQ(res.value().flat(0), 0.0f);  // x<0  -> blocked
    EXPECT_FLOAT_EQ(res.value().flat(1), 1.0f);  // x>0  -> passes
    EXPECT_FLOAT_EQ(res.value().flat(2), 0.0f);  // x==0 -> blocked
    EXPECT_FLOAT_EQ(res.value().flat(3), 1.0f);  // x>0  -> passes
}

TEST_F(ActivationTest, ActivationsFnLayer_NoParams) {
    ReLULayer layer;
    EXPECT_TRUE(layer.parameters().empty());
}