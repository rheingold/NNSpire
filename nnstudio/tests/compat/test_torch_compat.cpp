/* ============================================================================
 * test_torch_compat.cpp — verify torch_compat.h API surface compiles and runs
 *
 * These tests use ONLY names from the torch:: namespace to confirm that the
 * shim aliases the NNStudio engine correctly.  If this file compiles and all
 * tests pass, a 3-layer MLP written with torch:: names works against the
 * NNStudio backend.
 * ============================================================================ */

#include <gtest/gtest.h>
#include <torch_compat.h>
#include <core/BackendRegistry.h>
#include <builtin/backends/CpuBackend.h>

using namespace nnstudio::builtin::backends;

// Fixture: ensure CPU backend is registered.
class TorchCompatTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!nnstudio::core::BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ─── Tensor factories ─────────────────────────────────────────────────────────

TEST_F(TorchCompatTest, Zeros_Shape) {
    auto t = torch::zeros({2, 3});
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_FLOAT_EQ(t.flat(0), 0.0f);
}

TEST_F(TorchCompatTest, Ones_Shape) {
    auto t = torch::ones({4});
    EXPECT_EQ(t.numel(), 4);
    EXPECT_FLOAT_EQ(t.flat(0), 1.0f);
}

TEST_F(TorchCompatTest, DtypeConstants) {
    auto t = torch::zeros({2}, torch::kFloat32);
    EXPECT_EQ(t.dtype(), nnstudio::core::DType::Float32);
    EXPECT_EQ(t.itemsize(), 4u);
}

// ─── Layer construction ───────────────────────────────────────────────────────

TEST_F(TorchCompatTest, Linear_Constructs) {
    torch::nn::Linear layer(4, 8);
    EXPECT_EQ(layer.in_features, 4);
    EXPECT_EQ(layer.outFeatures(), 8);
}

TEST_F(TorchCompatTest, ReLU_Constructs) {
    torch::nn::ReLU relu;
    EXPECT_EQ(relu.typeName(), std::string_view("ReLU"));
}

TEST_F(TorchCompatTest, Sigmoid_Constructs) {
    torch::nn::Sigmoid sig;
    EXPECT_EQ(sig.typeName(), std::string_view("Sigmoid"));
}

TEST_F(TorchCompatTest, Dropout_Constructs) {
    torch::nn::Dropout dropout(0.5f);
    SUCCEED();
}

TEST_F(TorchCompatTest, Conv2d_Constructs) {
    torch::nn::Conv2d conv(3, 16, 3, 1, 1);   // in=3, out=16, k=3, s=1, p=1
    EXPECT_EQ(conv.outFilters(), 16);
    EXPECT_EQ(conv.kernelSize(), 3);
}

TEST_F(TorchCompatTest, MultiheadAttention_Constructs) {
    // PyTorch order: (embed_dim, num_heads)
    torch::nn::MultiheadAttention mha(64, 8);
    EXPECT_EQ(mha.embDim(),   64);
    EXPECT_EQ(mha.numHeads(), 8);
}

// ─── Sequential — 3-layer MLP forward pass ────────────────────────────────────

TEST_F(TorchCompatTest, Sequential_BuildAndForward) {
    torch::nn::Sequential model(
        torch::nn::Linear(4, 8),
        torch::nn::ReLU(),
        torch::nn::Linear(8, 1),
        torch::nn::Sigmoid()
    );

    // Build with known input shape
    auto buildRes = model.build({1, 4});
    ASSERT_TRUE(buildRes.ok()) << buildRes.error().message;
    EXPECT_EQ(buildRes.value()[0], 1);
    EXPECT_EQ(buildRes.value()[1], 1);

    // Forward pass
    auto x   = torch::zeros({1, 4});
    auto res = model.forward(x);
    ASSERT_TRUE(res.ok()) << res.error().message;
    const auto& out = res.value();
    EXPECT_EQ(out.shape()[0], 1);
    EXPECT_EQ(out.shape()[1], 1);
    // Sigmoid output is always in (0, 1)
    EXPECT_GT(out.flat(0), 0.0f);
    EXPECT_LT(out.flat(0), 1.0f);
}

TEST_F(TorchCompatTest, Sequential_Parameters_Accessible) {
    torch::nn::Sequential model(
        torch::nn::Linear(2, 4),
        torch::nn::ReLU(),
        torch::nn::Linear(4, 1)
    );
    model.build({1, 2});
    auto params = model.parameters();
    // Two Dense layers, each with W + b = 2 params each → 4 total
    EXPECT_EQ(params.size(), 4u);
}

TEST_F(TorchCompatTest, Sequential_SetTraining) {
    torch::nn::Sequential model(
        torch::nn::Dropout(0.5f)
    );
    model.build({1, 4});
    model.setTraining(false);
    auto x   = torch::ones({1, 4});
    auto res = model.forward(x);
    // In eval mode Dropout is identity — output equals input
    ASSERT_TRUE(res.ok());
    for (int64_t i = 0; i < res.value().numel(); ++i)
        EXPECT_FLOAT_EQ(res.value().flat(i), 1.0f);
}

// ─── functional API ───────────────────────────────────────────────────────────

TEST_F(TorchCompatTest, Functional_ReLU) {
    auto x   = nnstudio::core::Tensor::full({3}, -1.0f);
    auto out = torch::nn::functional::relu(x);
    for (int64_t i = 0; i < out.numel(); ++i)
        EXPECT_FLOAT_EQ(out.flat(i), 0.0f);
}

TEST_F(TorchCompatTest, Functional_Sigmoid_Range) {
    auto x   = torch::zeros({4});
    auto out = torch::nn::functional::sigmoid(x);
    for (int64_t i = 0; i < out.numel(); ++i)
        EXPECT_NEAR(out.flat(i), 0.5f, 1e-5f);
}

TEST_F(TorchCompatTest, Functional_Softmax_SumsToOne) {
    auto x   = torch::ones({1, 4});
    auto out = torch::nn::functional::softmax(x);
    float sum = 0.f;
    for (int64_t i = 0; i < out.numel(); ++i) sum += out.flat(i);
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

// ─── Optimizer aliases ────────────────────────────────────────────────────────

TEST_F(TorchCompatTest, Adam_Constructs) {
    torch::optim::Adam adam(1e-3f);
    EXPECT_EQ(adam.name(), std::string_view("Adam"));
}

TEST_F(TorchCompatTest, SGD_Constructs) {
    torch::optim::SGD sgd(0.01f, 0.9f);
    EXPECT_EQ(sgd.name(), std::string_view("SGD"));
}

TEST_F(TorchCompatTest, AdamW_Constructs) {
    torch::optim::AdamW adamw(1e-3f);
    EXPECT_EQ(adamw.name(), std::string_view("AdamW"));
}

TEST_F(TorchCompatTest, RMSprop_Constructs) {
    torch::optim::RMSprop rms(1e-3f);
    EXPECT_EQ(rms.name(), std::string_view("RMSProp"));
}

// ─── Loss aliases ─────────────────────────────────────────────────────────────

TEST_F(TorchCompatTest, MSELoss_Constructs) {
    torch::nn::MSELoss mse;
    EXPECT_EQ(mse.name(), std::string_view("MSE"));
}

TEST_F(TorchCompatTest, CrossEntropyLoss_Constructs) {
    torch::nn::CrossEntropyLoss ce;
    EXPECT_EQ(ce.name(), std::string_view("CrossEntropy"));
}
