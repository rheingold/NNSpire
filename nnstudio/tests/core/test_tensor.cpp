/* ============================================================================
 * test_tensor.cpp — Tensor construction, ops, reshape
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <core/Tensor.h>
#include <core/BackendRegistry.h>
#include <builtin/backends/CpuBackend.h>

using namespace nnstudio::core;
using namespace nnstudio::builtin::backends;

// Fixture: ensure CPU backend registered once
class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

TEST_F(TensorTest, ZerosShape) {
    auto t = Tensor::zeros({2, 3});
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.numel(), 6);
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(t.flat(i), 0.0f);
}

TEST_F(TensorTest, OnesShape) {
    auto t = Tensor::ones({4});
    EXPECT_EQ(t.numel(), 4);
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(t.flat(i), 1.0f);
}

TEST_F(TensorTest, FullValue) {
    auto t = Tensor::full({3, 3}, 7.0f);
    for (int64_t i = 0; i < t.numel(); ++i)
        EXPECT_FLOAT_EQ(t.flat(i), 7.0f);
}

TEST_F(TensorTest, CloneDeep) {
    auto a = Tensor::full({2, 2}, 1.0f);
    auto b = a.clone();
    b.flat(0) = 99.0f;
    EXPECT_FLOAT_EQ(a.flat(0), 1.0f);   // original unchanged
    EXPECT_FLOAT_EQ(b.flat(0), 99.0f);
}

TEST_F(TensorTest, ReshapeValid) {
    auto t = Tensor::zeros({2, 6});
    auto r = t.reshape({3, 4});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().numel(), 12);
    EXPECT_EQ(r.value().shape()[0], 3);
    EXPECT_EQ(r.value().shape()[1], 4);
}

TEST_F(TensorTest, ReshapeInfer) {
    auto t = Tensor::zeros({2, 6});
    auto r = t.reshape({-1, 4});
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().shape()[0], 3);
}

TEST_F(TensorTest, Add) {
    auto a = Tensor::full({3}, 2.0f);
    auto b = Tensor::full({3}, 3.0f);
    auto c = a + b;
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(c.flat(i), 5.0f);
}

TEST_F(TensorTest, Mul) {
    auto a = Tensor::full({4}, 3.0f);
    auto b = Tensor::full({4}, 4.0f);
    auto c = a * b;
    for (int64_t i = 0; i < c.numel(); ++i)
        EXPECT_FLOAT_EQ(c.flat(i), 12.0f);
}

TEST_F(TensorTest, Matmul2D) {
    // [2,3] @ [3,2] = [2,2]
    // A = [[1,2,3],[4,5,6]]
    // B = [[7,8],[9,10],[11,12]]
    // C = [[58,64],[139,154]]
    auto a = Tensor::fromData({1,2,3,4,5,6}, {2,3});
    auto b = Tensor::fromData({7,8,9,10,11,12}, {3,2});
    auto c = backend().matmul(a, b);
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);
    EXPECT_FLOAT_EQ(c.flat(0), 58.0f);
    EXPECT_FLOAT_EQ(c.flat(1), 64.0f);
    EXPECT_FLOAT_EQ(c.flat(2), 139.0f);
    EXPECT_FLOAT_EQ(c.flat(3), 154.0f);
}

// ─── Serialisation ────────────────────────────────────────────────────────────

#include <fstream>
#include <cstdio>

TEST_F(TensorTest, SaveLoadRoundtrip) {
    const char* path = "_test_serde_roundtrip.nns1";
    auto orig = Tensor::fromData({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {2, 3});

    auto saveRes = orig.save(path);
    ASSERT_TRUE(saveRes.ok()) << saveRes.error().message;

    auto loadRes = Tensor::load(path);
    ASSERT_TRUE(loadRes.ok()) << loadRes.error().message;

    const auto& t = loadRes.value();
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.numel(), 6);
    for (int64_t i = 0; i < 6; ++i)
        EXPECT_FLOAT_EQ(t.flat(i), orig.flat(i));

    std::remove(path);
}

TEST_F(TensorTest, SaveLoadScalar) {
    const char* path = "_test_serde_scalar.nns1";
    auto orig = Tensor::full({1}, 3.14159f);
    ASSERT_TRUE(orig.save(path).ok());
    auto res = Tensor::load(path);
    ASSERT_TRUE(res.ok());
    EXPECT_NEAR(res.value().flat(0), 3.14159f, 1e-6f);
    std::remove(path);
}

TEST_F(TensorTest, LoadBadMagicFails) {
    const char* path = "_test_serde_badmagic.tmp";
    { std::ofstream f(path, std::ios::binary); f << "BADD"; }
    auto res = Tensor::load(path);
    EXPECT_TRUE(res.failed());
    std::remove(path);
}

// ── void* buffer refactor tests ──────────────────────────────────────────────

TEST_F(TensorTest, Itemsize_Float32) {
    auto t = Tensor::zeros({3, 4}, DType::Float32);
    EXPECT_EQ(t.itemsize(), 4u);
}

TEST_F(TensorTest, Itemsize_Int8) {
    // Int8 tensor: allocation only — no compute in Phase 1, but buffer is valid.
    Tensor t({5}, DType::Int8);
    EXPECT_EQ(t.itemsize(), 1u);
    EXPECT_EQ(t.numel(), 5);
}

TEST_F(TensorTest, Itemsize_Int32) {
    Tensor t({2, 3}, DType::Int32);
    EXPECT_EQ(t.itemsize(), 4u);
}

TEST_F(TensorTest, RawData_NonNull) {
    auto t = Tensor::ones({2, 2});
    EXPECT_NE(t.rawData(), nullptr);
    // For Float32, rawData() and data() must point to the same address.
    EXPECT_EQ(t.rawData(), static_cast<void*>(t.data()));
}

TEST_F(TensorTest, Dtype_RoundTrip) {
    auto f32 = Tensor::zeros({1}, DType::Float32);
    auto i8  = Tensor({3},      DType::Int8);
    auto i32 = Tensor({3},      DType::Int32);
    EXPECT_EQ(f32.dtype(), DType::Float32);
    EXPECT_EQ(i8.dtype(),  DType::Int8);
    EXPECT_EQ(i32.dtype(), DType::Int32);
    EXPECT_EQ(f32.itemsize(), 4u);
    EXPECT_EQ(i8.itemsize(),  1u);
    EXPECT_EQ(i32.itemsize(), 4u);
}

TEST_F(TensorTest, Item_Float32) {
    auto t = Tensor::full({1}, 3.14f, DType::Float32);
    EXPECT_FLOAT_EQ(t.item<float>(), 3.14f);
}

TEST_F(TensorTest, Item_Int32) {
    // Construct a scalar Int32 tensor and write a value via rawData().
    Tensor t({1}, DType::Int32);
    *static_cast<int32_t*>(t.rawData()) = 42;
    EXPECT_EQ(t.item<int32_t>(), 42);
}
