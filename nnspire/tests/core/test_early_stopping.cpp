/* ============================================================================
 * test_early_stopping.cpp — unit tests for EarlyStoppingCallback
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <core/EarlyStopping.h>
#include <cmath>
#include <limits>

using namespace NNSpire::internal::training;

// ─── Basic patience ───────────────────────────────────────────────────────────

TEST(EarlyStopping, StopsAfterPatience) {
    EarlyStoppingCallback es(3, 1e-4f);
    auto pred = es.predicate();

    TrainMetrics m{};
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // epoch 1: improvement
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // epoch 2: no change, counter=1
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // epoch 3: counter=2
    m.loss = 1.0f; EXPECT_TRUE (pred(m));  // epoch 4: counter=3 ≥ patience → STOP
}

// ─── Improvement resets counter ──────────────────────────────────────────────

TEST(EarlyStopping, ImprovementResetsCounter) {
    EarlyStoppingCallback es(3, 1e-4f);
    auto pred = es.predicate();

    TrainMetrics m{};
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // first call → initial improvement
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // counter=1
    m.loss = 0.5f; EXPECT_FALSE(pred(m));  // improvement → counter reset to 0
    m.loss = 0.5f; EXPECT_FALSE(pred(m));  // counter=1
    m.loss = 0.5f; EXPECT_FALSE(pred(m));  // counter=2
    m.loss = 0.5f; EXPECT_TRUE (pred(m));  // counter=3 → STOP
}

// ─── minDelta filtering ──────────────────────────────────────────────────────

TEST(EarlyStopping, TinyImprovementIgnoredByMinDelta) {
    EarlyStoppingCallback es(2, 0.1f);   // 0.1 minDelta
    auto pred = es.predicate();

    TrainMetrics m{};
    m.loss = 1.0f;    EXPECT_FALSE(pred(m));  // baseline: bestLoss=1.0
    m.loss = 0.981f;  EXPECT_FALSE(pred(m));  // 1.0 - 0.981 = 0.019 < 0.1 → counter=1
    m.loss = 0.981f;  EXPECT_TRUE (pred(m));  // counter=2 ≥ 2 → STOP
}

TEST(EarlyStopping, BigImprovementAccepted) {
    EarlyStoppingCallback es(2, 0.01f);
    auto pred = es.predicate();

    TrainMetrics m{};
    m.loss = 1.0f;   EXPECT_FALSE(pred(m));  // bestLoss=1.0
    m.loss = 0.5f;   EXPECT_FALSE(pred(m));  // 1.0 - 0.5 = 0.5 >= 0.01 -> improvement, counter=0
    m.loss = 0.48f;  EXPECT_FALSE(pred(m));  // 0.5 - 0.48 = 0.02 >= 0.01 -> improvement, counter=0
    m.loss = 0.48f;  EXPECT_FALSE(pred(m));  // counter=1
    m.loss = 0.48f;  EXPECT_TRUE (pred(m));  // counter=2 -> STOP
}

// ─── reset() ─────────────────────────────────────────────────────────────────

TEST(EarlyStopping, Reset_ClearsState) {
    EarlyStoppingCallback es(2, 0.0f);
    auto pred = es.predicate();

    TrainMetrics m{};
    m.loss = 1.0f; pred(m);  // epoch 1
    m.loss = 1.0f; pred(m);  // counter=1
    // should stop next epoch
    m.loss = 1.0f;
    bool wouldStop = pred(m);
    EXPECT_TRUE(wouldStop);

    // Now reset and verify it doesn't immediately stop
    es.reset();
    // re-create predicate (or just call again — predicate captures by ref)
    pred = es.predicate();
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // epoch 1 after reset
    m.loss = 1.0f; EXPECT_FALSE(pred(m));  // counter=1
    m.loss = 1.0f; EXPECT_TRUE (pred(m));  // counter=2 → STOP again
}

// ─── Accessors match constructor ──────────────────────────────────────────────

TEST(EarlyStopping, Accessors) {
    EarlyStoppingCallback es(7, 5e-3f);
    EXPECT_EQ(es.patience(), 7);
    EXPECT_FLOAT_EQ(es.minDelta(), 5e-3f);
    EXPECT_EQ(es.counter(), 0);
    EXPECT_EQ(es.bestLoss(), std::numeric_limits<float>::infinity());
}
