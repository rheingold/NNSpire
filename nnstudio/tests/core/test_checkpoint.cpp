/* ============================================================================
 * test_checkpoint.cpp — unit tests for Checkpoint::save / Checkpoint::load
 * Phase 1 unit tests  (P1.6)
 * ============================================================================ */

#include <gtest/gtest.h>

#include <core/Checkpoint.h>
#include <core/Tensor.h>
#include <core/Layer.h>
#include <core/Result.h>
#include <core/BackendRegistry.h>
#include <builtin/backends/CpuBackend.h>
#include <builtin/layers/Dense.h>
#include <builtin/optimizers/Optimizers.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

using namespace nnstudio::core;
using namespace nnstudio::builtin::backends;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::optimizers;

// ─── Fixture ──────────────────────────────────────────────────────────────────

class CheckpointTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }

    // Helpers to build and initialise a small Dense layer
    static Dense buildDense(int64_t out) {
        Dense d(out);
        auto br = d.build({4});
        (void)br; // will fail tests implicitly if dims wrong
        return d;
    }

    static const char* kPath() { return "_test_checkpoint.nns"; }

    void TearDown() override {
        std::remove(kPath());
    }
};

// ─── Basic weights round-trip ─────────────────────────────────────────────────

TEST_F(CheckpointTest, SaveLoad_Weights_RoundTrip) {
    // Build a Dense(4→2) layer and fill weights with known values
    Dense d(2);
    ASSERT_TRUE(d.build({4}).ok());

    auto params = d.parameters();
    ASSERT_EQ(params.size(), 2u);  // W and b

    // Write sentinel values into W and b
    for (int64_t i = 0; i < params[0]->tensor.numel(); ++i)
        params[0]->tensor.flat(i) = static_cast<float>(i) * 0.1f + 1.0f;
    for (int64_t i = 0; i < params[1]->tensor.numel(); ++i)
        params[1]->tensor.flat(i) = static_cast<float>(i) * 0.2f - 0.5f;

    // Save
    CheckpointState cs;
    cs.params     = params;
    cs.globalStep = 42;
    cs.epoch      = 7;

    auto sr = Checkpoint::save(kPath(), cs);
    ASSERT_TRUE(sr.ok()) << sr.error().message;

    // Load into a fresh copy of the same model
    Dense d2(2);
    ASSERT_TRUE(d2.build({4}).ok());
    auto params2 = d2.parameters();

    CheckpointState cs2;
    cs2.params = params2;
    auto lr    = Checkpoint::load(kPath(), cs2);
    ASSERT_TRUE(lr.ok()) << lr.error().message;

    // Counters must be restored
    EXPECT_EQ(cs2.globalStep, 42u);
    EXPECT_EQ(cs2.epoch,       7u);

    // Weights must match exactly
    ASSERT_EQ(params2[0]->tensor.numel(), params[0]->tensor.numel());
    ASSERT_EQ(params2[1]->tensor.numel(), params[1]->tensor.numel());

    for (int64_t i = 0; i < params[0]->tensor.numel(); ++i)
        EXPECT_FLOAT_EQ(params2[0]->tensor.flat(i), params[0]->tensor.flat(i));
    for (int64_t i = 0; i < params[1]->tensor.numel(); ++i)
        EXPECT_FLOAT_EQ(params2[1]->tensor.flat(i), params[1]->tensor.flat(i));
}

// ─── Forward still works after loading ───────────────────────────────────────

TEST_F(CheckpointTest, SaveLoad_ForwardAfterLoad_Succeeds) {
    Dense d(2);
    ASSERT_TRUE(d.build({4}).ok());

    CheckpointState cs;
    cs.params = d.parameters();
    cs.globalStep = 1;
    cs.epoch = 0;
    ASSERT_TRUE(Checkpoint::save(kPath(), cs).ok());

    Dense d2(2);
    ASSERT_TRUE(d2.build({4}).ok());
    CheckpointState cs2;
    cs2.params = d2.parameters();
    ASSERT_TRUE(Checkpoint::load(kPath(), cs2).ok());

    // Forward should succeed with proper output shape
    auto x  = Tensor::ones({3, 4});
    auto fr = d2.forward(x);
    ASSERT_TRUE(fr.ok()) << fr.error().message;
    EXPECT_EQ(fr.value().shape()[0], 3);
    EXPECT_EQ(fr.value().shape()[1], 2);
}

// ─── Optimizer state round-trip ───────────────────────────────────────────────

TEST_F(CheckpointTest, SaveLoad_AdamState_RoundTrip) {
    Dense d(2);
    ASSERT_TRUE(d.build({4}).ok());
    auto params = d.parameters();

    Adam adam(0.01f);

    // Run one optimizer step so Adam accumulates moment vectors
    for (auto* p : params)
        p->tensor.accumulateGrad(Tensor::full(p->tensor.shape(), 0.1f));
    adam.step(params);

    uint64_t savedStep = adam.stepCount();
    EXPECT_GT(savedStep, 0u);

    // Save with optimizer
    CheckpointState cs;
    cs.params     = params;
    cs.optimizer  = &adam;
    cs.globalStep = 100;
    cs.epoch      = 5;
    ASSERT_TRUE(Checkpoint::save(kPath(), cs).ok());

    // Load into fresh model + fresh optimizer
    Dense d2(2);
    ASSERT_TRUE(d2.build({4}).ok());
    auto params2 = d2.parameters();
    Adam adam2(0.01f);

    CheckpointState cs2;
    cs2.params    = params2;
    cs2.optimizer = &adam2;
    auto lr = Checkpoint::load(kPath(), cs2);
    ASSERT_TRUE(lr.ok()) << lr.error().message;

    EXPECT_EQ(cs2.globalStep, 100u);
    EXPECT_EQ(cs2.epoch,        5u);
    // Adam step counter must be restored
    EXPECT_EQ(adam2.stepCount(), savedStep);
}

// ─── Counters saved without optimizer ────────────────────────────────────────

TEST_F(CheckpointTest, SaveLoad_NoOptimizer_CountersRestored) {
    Dense d(3);
    ASSERT_TRUE(d.build({2}).ok());

    CheckpointState cs;
    cs.params     = d.parameters();
    cs.optimizer  = nullptr;   // deliberately omit optimizer
    cs.globalStep = 999;
    cs.epoch      = 33;
    ASSERT_TRUE(Checkpoint::save(kPath(), cs).ok());

    Dense d2(3);
    ASSERT_TRUE(d2.build({2}).ok());
    CheckpointState cs2;
    cs2.params    = d2.parameters();
    cs2.optimizer = nullptr;
    ASSERT_TRUE(Checkpoint::load(kPath(), cs2).ok());

    EXPECT_EQ(cs2.globalStep, 999u);
    EXPECT_EQ(cs2.epoch,       33u);
}

// ─── Bad magic returns IoError ────────────────────────────────────────────────

TEST_F(CheckpointTest, Load_BadMagic_ReturnsIoError) {
    // Write a file with wrong magic
    {
        std::ofstream f(kPath(), std::ios::binary);
        f.write("BADD", 4);
    }

    Dense d(1);
    ASSERT_TRUE(d.build({1}).ok());

    CheckpointState cs;
    cs.params = d.parameters();
    auto lr = Checkpoint::load(kPath(), cs);

    ASSERT_TRUE(lr.failed());
    EXPECT_EQ(lr.error().code, ErrorCode::IoError);
}

// ─── Non-existent file returns IoError ───────────────────────────────────────

TEST_F(CheckpointTest, Load_MissingFile_ReturnsIoError) {
    std::remove("_nonexistent_checkpoint.nns");  // ensure it's gone

    Dense d(1);
    ASSERT_TRUE(d.build({1}).ok());

    CheckpointState cs;
    cs.params = d.parameters();
    auto lr = Checkpoint::load("_nonexistent_checkpoint.nns", cs);

    ASSERT_TRUE(lr.failed());
    EXPECT_EQ(lr.error().code, ErrorCode::IoError);
}

// ─── Stream round-trip (Tensor embedded in checkpoint) ───────────────────────

TEST_F(CheckpointTest, TensorStream_SaveLoad_RoundTrip) {
    // This directly exercises the stream overloads added for checkpoint support.
    auto orig = Tensor::fromData({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2});

    std::ostringstream out(std::ios::binary);
    ASSERT_TRUE(orig.save(out).ok());

    std::string blob = out.str();
    std::istringstream in(blob, std::ios::binary);
    auto res = Tensor::load(in);
    ASSERT_TRUE(res.ok()) << res.error().message;

    EXPECT_EQ(res.value().shape(), orig.shape());
    for (int64_t i = 0; i < orig.numel(); ++i)
        EXPECT_FLOAT_EQ(res.value().flat(i), orig.flat(i));
}
