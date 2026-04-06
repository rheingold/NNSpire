/* ============================================================================
 * test_trainer_xor.cpp — Phase 1 Milestone: train a 2-layer MLP to solve XOR
 *
 * XOR truth table:
 *   [0,0] → 0
 *   [0,1] → 1
 *   [1,0] → 1
 *   [1,1] → 0
 *
 * Architecture: Dense(2→4, ReLU) → Dense(4→1, Sigmoid)
 * Loss: BCE
 * Optimizer: Adam(lr=0.05)
 * Expected convergence: < 3000 epochs, final loss < 0.05
 * ============================================================================ */

#include <gtest/gtest.h>
#include <core/Layer.h>
#include <builtin/layers/Dense.h>
#include <builtin/activations/Activations.h>
#include <builtin/losses/Losses.h>
#include <builtin/optimizers/Optimizers.h>
#include <core/Trainer.h>
#include <core/BackendRegistry.h>
#include <builtin/backends/CpuBackend.h>

#include <cmath>

using namespace nnstudio::internal::training;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::activations;
using namespace nnstudio::builtin::losses;
using namespace nnstudio::builtin::optimizers;
using namespace nnstudio::builtin::backends;

class XORTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ---------------------------------------------------------------------------
// WALKTHROUGH — Chapter 5: The XOR Test
//
// XOR is the classic "hello world" for neural networks because it is the
// simplest problem that is NOT linearly separable:
//
//   [0,0] → 0      [0,1] → 1
//   [1,0] → 1      [1,1] → 0
//
// Plot these 4 points on a 2D grid — you cannot draw a single straight line
// that separates the 1s from the 0s.  A single Dense layer (= linear function)
// cannot solve it.  We need at least one hidden layer.
//
// Architecture: Dense(2→4) → ReLU → Dense(4→1) → Sigmoid
//
//   Dense(2→4):  projects the 2D input into a 4D space where XOR becomes
//                linearly separable.  4 neurons give enough degrees of freedom.
//                Glorot init keeps initial activations in a healthy range.
//
//   ReLU:        non-linearity.  Without it, Dense→Dense is still just a
//                linear function (two matrices multiplied = one matrix).
//                ReLU "bends" the decision boundary.
//
//   Dense(4→1):  collapses the 4D representation to a single logit.
//
//   Sigmoid:     squashes the logit to (0, 1) — interpretable as probability.
//
// Loss: BCE (Binary Cross-Entropy) — natural pair for Sigmoid output.
// Optimizer: Adam(lr=0.05) — fast convergence, forgiving of learning rate choice.
//
// The test passes when:
//   - finalLoss < 0.05  (network is confident, not just lucky)
//   - all 4 predictions round to the correct 0 or 1
//
// At ~69ms for 3000 epochs on 4 samples the engine handles ~170k weight
// updates per second on a single CPU thread.
// ---------------------------------------------------------------------------
TEST_F(XORTest, TrainsToConvergence) {
    Dense::setSeed(42);   // deterministic init for reproducibility on all platforms
    // ── Build model ───────────────────────────────────────────────────────────
    //
    // Dense(2→4): each of the 4 hidden neurons learns a half-space in 2D input.
    // Together they can carve out the two XOR-positive regions.
    // Dense(4→1): combines the hidden features into one output logit.
    //
    Sequential model;
    model.add<Dense>(4, true, WeightInit::GlorotUniform)
         .add<ReLU>()
         .add<Dense>(1, true, WeightInit::GlorotUniform)
         .add<Sigmoid>();

    // Build from input shape [2] (features=2, batch dim added at runtime)
    auto buildR = model.build({2});
    ASSERT_TRUE(buildR.ok()) << buildR.error().message;

    // ── XOR dataset (each sample is a separate batch of size 1) ───────────────
    Dataset data = {
        { Tensor::fromData({0.0f, 0.0f}, {1, 2}), Tensor::fromData({0.0f}, {1, 1}) },
        { Tensor::fromData({0.0f, 1.0f}, {1, 2}), Tensor::fromData({1.0f}, {1, 1}) },
        { Tensor::fromData({1.0f, 0.0f}, {1, 2}), Tensor::fromData({1.0f}, {1, 1}) },
        { Tensor::fromData({1.0f, 1.0f}, {1, 2}), Tensor::fromData({0.0f}, {1, 1}) },
    };

    // ── Train ─────────────────────────────────────────────────────────────────
    BCE       bce;
    Adam      adam(0.05f);
    Trainer   trainer(model, bce, adam);

    float finalLoss = 1.0f;
    uint64_t epochsDone = 0;

    TrainCallbacks cb;
    cb.onEpochEnd = [&](const TrainMetrics& m) {
        finalLoss  = m.loss;
        epochsDone = m.epoch + 1;
    };

    auto r = trainer.train(data, 3000, cb);
    ASSERT_TRUE(r.ok()) << r.error().message;

    EXPECT_LT(finalLoss, 0.05f)
        << "XOR not solved after " << epochsDone << " epochs, loss=" << finalLoss;

    // ── Verify predictions ────────────────────────────────────────────────────
    struct Sample { float a, b, expected; };
    std::vector<Sample> samples = {{0,0,0},{0,1,1},{1,0,1},{1,1,0}};

    for (auto& s : samples) {
        auto x    = Tensor::fromData({s.a, s.b}, {1, 2});
        auto predR = trainer.predict(x);
        ASSERT_TRUE(predR.ok());
        float pred = predR.value().flat(0);
        float rounded = pred >= 0.5f ? 1.0f : 0.0f;
        EXPECT_FLOAT_EQ(rounded, s.expected)
            << "XOR(" << s.a << "," << s.b << ")=" << pred
            << " (expected " << s.expected << ")";
    }
}
