/* ============================================================================
 * test_conv2d.cpp — unit tests for Conv2D layer
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <builtin/layers/Conv2D.h>
#include <builtin/backends/CpuBackend.h>
#include <core/BackendRegistry.h>
#include <cmath>

using namespace nnstudio::core;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::backends;

// ─── Fixture ──────────────────────────────────────────────────────────────────

class Conv2DTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ─── Output shape: valid padding ─────────────────────────────────────────────

TEST_F(Conv2DTest, OutputShape_ValidPadding) {
    Conv2D  conv(8, 3, 1, 0);   // 8 filters, 3x3 kernel, stride 1, no padding
    auto    br = conv.build({1, 8, 8});   // [C_in=1, H=8, W=8]
    ASSERT_TRUE(br.ok()) << br.error().message;
    // outH = outW = (8 + 2*0 - 3) / 1 + 1 = 6
    const auto& os = br.value();
    ASSERT_EQ(os.size(), 3u);
    EXPECT_EQ(os[0], 8);   // C_out
    EXPECT_EQ(os[1], 6);   // outH
    EXPECT_EQ(os[2], 6);   // outW
}

TEST_F(Conv2DTest, OutputShape_WithPadding) {
    Conv2D conv(4, 3, 1, 1);   // same-like padding=1 → outH=outW=input H/W
    conv.build({3, 5, 5});
    // outH = (5 + 2*1 - 3) / 1 + 1 = 5
    Conv2D c2(4, 3, 1, 1);
    auto br = c2.build({3, 5, 5});
    ASSERT_TRUE(br.ok());
    EXPECT_EQ(br.value()[1], 5);
    EXPECT_EQ(br.value()[2], 5);
}

TEST_F(Conv2DTest, OutputShape_Stride2) {
    Conv2D conv(4, 3, 2, 0);   // stride=2
    auto   br = conv.build({1, 7, 7});
    // outH = (7 - 3) / 2 + 1 = 3
    ASSERT_TRUE(br.ok());
    EXPECT_EQ(br.value()[1], 3);
    EXPECT_EQ(br.value()[2], 3);
}

// ─── Parameter count ──────────────────────────────────────────────────────────

TEST_F(Conv2DTest, ParameterCount_WithBias) {
    Conv2D conv(8, 3, 1, 0, true);  // 8 filters, 3x3 kernel, in=1
    conv.build({1, 8, 8});
    auto params = conv.parameters();
    // kernel: [8, 1, 3, 3] = 72 scalars, bias: [8] = 8 scalars
    // Two Parameter objects: kernel_ and bias_
    EXPECT_EQ(params.size(), 2u);
    // total scalars: kernel + bias
    int64_t total = 0;
    for (auto* p : params) total += p->tensor.numel();
    EXPECT_EQ(total, 8 * 1 * 3 * 3 + 8);  // 72 + 8 = 80
}

TEST_F(Conv2DTest, ParameterCount_NoBias) {
    Conv2D conv(4, 5, 1, 0, false);
    conv.build({2, 10, 10});
    auto params = conv.parameters();
    EXPECT_EQ(params.size(), 1u);   // kernel only
    EXPECT_EQ(params[0]->tensor.numel(), 4 * 2 * 5 * 5);  // 200
}

// ─── Forward produces finite outputs ─────────────────────────────────────────

TEST_F(Conv2DTest, Forward_FiniteOutputs) {
    Conv2D conv(4, 3, 1, 0, true);
    conv.build({1, 6, 6});

    // Input: batch of 2, 1 channel, 6x6
    auto x  = Tensor::ones({2, 1, 6, 6});
    auto fr = conv.forward(x);
    ASSERT_TRUE(fr.ok()) << fr.error().message;

    // Expected output shape: [2, 4, 4, 4]
    const auto& out = fr.value();
    ASSERT_EQ(out.shape().size(), 4u);
    EXPECT_EQ(out.shape()[0], 2);
    EXPECT_EQ(out.shape()[1], 4);
    EXPECT_EQ(out.shape()[2], 4);
    EXPECT_EQ(out.shape()[3], 4);

    // All values finite
    for (int64_t i = 0; i < out.numel(); ++i)
        EXPECT_TRUE(std::isfinite(out.flat(i)));
}

// ─── Backward: gradient shapes match ─────────────────────────────────────────

TEST_F(Conv2DTest, Backward_GradInputShape) {
    Conv2D conv(4, 3, 1, 0, true);
    conv.build({1, 5, 5});

    auto x  = Tensor::ones({1, 1, 5, 5});
    auto fr = conv.forward(x);
    ASSERT_TRUE(fr.ok());

    // Feed ones gradient
    auto gradOut = Tensor::ones(fr.value().shape());
    auto gr = conv.backward(gradOut);
    ASSERT_TRUE(gr.ok()) << gr.error().message;

    // dX must match input shape: [1, 1, 5, 5]
    const auto& dX = gr.value();
    ASSERT_EQ(dX.shape().size(), 4u);
    EXPECT_EQ(dX.shape()[0], 1);
    EXPECT_EQ(dX.shape()[1], 1);
    EXPECT_EQ(dX.shape()[2], 5);
    EXPECT_EQ(dX.shape()[3], 5);
}

// ─── Identity kernel produces scaled input sum ───────────────────────────────

TEST_F(Conv2DTest, IdentityKernel_SingleFilter) {
    // 1 filter, 1x1 kernel, no bias → output should equal input
    Conv2D conv(1, 1, 1, 0, false);
    conv.build({1, 4, 4});

    // Zero the kernel then set it to 1.0
    auto params = conv.parameters();
    ASSERT_EQ(params.size(), 1u);
    auto& kTensor = params[0]->tensor;
    ASSERT_EQ(kTensor.numel(), 1);
    kTensor.flat(0) = 1.0f;

    auto x  = Tensor::ones({1, 1, 4, 4});
    for (int64_t i = 0; i < 16; ++i) x.flat(i) = static_cast<float>(i);

    auto fr = conv.forward(x);
    ASSERT_TRUE(fr.ok());
    const auto& out = fr.value();
    EXPECT_EQ(out.numel(), 16);
    for (int64_t i = 0; i < 16; ++i)
        EXPECT_FLOAT_EQ(out.flat(i), static_cast<float>(i));
}

// ─── config() map populated ───────────────────────────────────────────────────

TEST_F(Conv2DTest, Config_ContainsExpectedKeys) {
    Conv2D conv(16, 3, 2, 1, true);
    auto cfg = conv.config();
    EXPECT_NE(cfg.find("outFilters"),  cfg.end());
    EXPECT_NE(cfg.find("kernelSize"),  cfg.end());
    EXPECT_NE(cfg.find("stride"),      cfg.end());
    EXPECT_NE(cfg.find("padding"),     cfg.end());
    EXPECT_EQ(cfg.at("outFilters"), "16");
    EXPECT_EQ(cfg.at("kernelSize"),  "3");
    EXPECT_EQ(cfg.at("stride"),      "2");
    EXPECT_EQ(cfg.at("padding"),     "1");
}
