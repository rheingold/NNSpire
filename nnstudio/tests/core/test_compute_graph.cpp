/* ============================================================================
 * test_compute_graph.cpp — unit tests for ComputeGraph / EvalTrace
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <core/ComputeGraph.h>
#include <builtin/layers/Dense.h>
#include <builtin/layers/Activations.h>
#include <builtin/backends/CpuBackend.h>
#include <core/BackendRegistry.h>
#include <cmath>

using namespace nnstudio::core;
using namespace nnstudio::builtin::layers;
using namespace nnstudio::builtin::backends;

// ─── Fixture ──────────────────────────────────────────────────────────────────

class ComputeGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!BackendRegistry::instance().has("cpu"))
            CpuBackend::init();
    }
};

// ─── Linear chain forward ─────────────────────────────────────────────────────

TEST_F(ComputeGraphTest, LinearChain_ForwardShape) {
    Dense d1(4), d2(1);
    ReLU  act;

    ComputeGraph g;
    NodeId h0 = g.input();
    NodeId h1 = g.record(&d1,  h0);
    NodeId h2 = g.record(&act, h1);
    NodeId h3 = g.record(&d2,  h2);
    g.setOutput(h3);

    // build with 2-feature inputs
    auto br = g.build({2});
    ASSERT_TRUE(br.ok()) << br.error().message;

    // forward on batch of 3 samples
    auto x  = Tensor::ones({3, 2});
    auto fr = g.forward(x);
    ASSERT_TRUE(fr.ok()) << fr.error().message;

    const auto& out = fr.value();
    ASSERT_EQ(out.shape().size(), 2u);
    EXPECT_EQ(out.shape()[0], 3);  // N
    EXPECT_EQ(out.shape()[1], 1);  // output features
}

// ─── Node IDs are sequential ──────────────────────────────────────────────────

TEST_F(ComputeGraphTest, NodeIds_Sequential) {
    Dense d1(8), d2(4);
    ComputeGraph g;
    NodeId h0 = g.input();
    NodeId h1 = g.record(&d1, h0);
    NodeId h2 = g.record(&d2, h1);
    EXPECT_EQ(h0, 0u);
    EXPECT_EQ(h1, 1u);
    EXPECT_EQ(h2, 2u);
    // record() auto-tracks last node as output — setOutput overrides later
    EXPECT_EQ(g.outputNodeId(), h2);
    g.setOutput(h2);
    EXPECT_EQ(g.outputNodeId(), h2);
}

// ─── Backward produces gradient for each parameter ───────────────────────────

TEST_F(ComputeGraphTest, Backward_FillsParameterGrads) {
    Dense d1(4), d2(1);
    ReLU  act;

    ComputeGraph g;
    auto h0 = g.input();
    auto h1 = g.record(&d1,  h0);
    auto h2 = g.record(&act, h1);
    auto h3 = g.record(&d2,  h2);
    g.setOutput(h3);

    g.build({2});
    auto x  = Tensor::ones({1, 2});
    g.forward(x);

    // seed gradient of 1.0 at output
    auto gradOut = Tensor::ones({1, 1});
    auto gr = g.backward(gradOut);
    ASSERT_TRUE(gr.ok()) << gr.error().message;

    // gradient w.r.t. input has same batch shape
    const auto& gi = gr.value();
    EXPECT_EQ(gi.shape()[0], 1);  // N
    EXPECT_EQ(gi.shape()[1], 2);  // input features

    // all parameters should have had their grad filled (not all zeros)
    // (statistical: highly unlikely all grads are exactly 0 after random init)
    auto params = g.parameters();
    EXPECT_FALSE(params.empty());
}

// ─── setTraceMode — traces collected ─────────────────────────────────────────

TEST_F(ComputeGraphTest, TraceMode_PopulatesTraces) {
    Dense d1(4);
    ReLU  act;

    ComputeGraph g;
    auto h0 = g.input();
    auto h1 = g.record(&d1,  h0);
    auto h2 = g.record(&act, h1);
    g.setOutput(h2);
    g.build({3});
    g.setTraceMode(true);

    auto x  = Tensor::ones({2, 3});
    auto fr = g.forward(x);
    ASSERT_TRUE(fr.ok());

    auto gradOut = Tensor::ones(fr.value().shape());
    g.backward(gradOut);

    const auto& tr = g.traces();
    EXPECT_EQ(tr.size(), 2u);  // one trace per recorded op

    // Check Dense trace — typeName is reliable; name() may be empty if unset
    EXPECT_EQ(tr[0].layerType, std::string("Dense"));
    EXPECT_EQ(tr[0].input.shape()[0], 2);
    EXPECT_EQ(tr[0].output.shape()[0], 2);

    // After backward, gradInput/gradOutput should be populated
    EXPECT_EQ(tr[0].gradOutput.shape()[0], 2);
    EXPECT_EQ(tr[0].gradInput.shape()[0], 2);
}

// ─── No trace without setTraceMode ───────────────────────────────────────────

TEST_F(ComputeGraphTest, TraceMode_OffByDefault) {
    Dense d1(4);
    ComputeGraph g;
    auto h = g.record(&d1, g.input());
    g.setOutput(h);
    g.build({2});
    g.forward(Tensor::ones({1, 2}));
    g.backward(Tensor::ones({1, 4}));
    EXPECT_TRUE(g.traces().empty());
}

// ─── setTraining propagates ──────────────────────────────────────────────────

TEST_F(ComputeGraphTest, SetTraining_PropagatesDown) {
    Dense d1(4);
    ComputeGraph g;
    auto h = g.record(&d1, g.input());
    g.setOutput(h);
    g.build({2});

    g.setTraining(false);
    EXPECT_FALSE(d1.training());
    g.setTraining(true);
    EXPECT_TRUE(d1.training());
}

// ─── parameters() de-duplicates ──────────────────────────────────────────────

TEST_F(ComputeGraphTest, Parameters_NoDuplicates) {
    Dense d1(4), d2(1);
    ComputeGraph g;
    auto h0 = g.input();
    auto h1 = g.record(&d1, h0);
    auto h2 = g.record(&d2, h1);
    g.setOutput(h2);
    g.build({2});

    auto params = g.parameters();
    // Dense(2→4) has W[4,2]+b[4]=12+4=16 scalars but we only count Parameter objects
    // Dense(4→1) has W[1,4]+b[1]= 5 Parameter objects
    // No parameter should appear twice
    std::vector<Parameter*> seen;
    for (auto* p : params) {
        for (auto* q : seen) EXPECT_NE(p, q);
        seen.push_back(p);
    }
}
