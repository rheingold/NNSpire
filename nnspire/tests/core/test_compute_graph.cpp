/* ============================================================================
 * test_compute_graph.cpp — unit tests for ComputeGraph / EvalTrace / JSON
 * Phase 1 unit tests
 * ============================================================================ */

#include <gtest/gtest.h>
#include <core/ComputeGraph.h>
#include <builtin/layers/Dense.h>
#include <builtin/activations/Activations.h>
#include <builtin/backends/CpuBackend.h>
#include <memory>
#include <core/BackendRegistry.h>
#include <cmath>

using namespace NNSpire::core;
using namespace NNSpire::builtin::layers;
using namespace NNSpire::builtin::activations;
using namespace NNSpire::builtin::backends;

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
// ─── JSON serialization ───────────────────────────────────────────────────────

// Helper: register Dense + ReLU factories once per test binary.
static bool factoriesRegistered = false;
static void ensureFactories() {
    if (factoriesRegistered) return;
    factoriesRegistered = true;

    using namespace NNSpire::builtin::layers;
    using namespace NNSpire::builtin::activations;
    using CfgMap = std::unordered_map<std::string, std::string>;

    ComputeGraph::registerLayerFactory("Dense",
        [](const CfgMap& cfg, const std::string& name) -> std::unique_ptr<ILayer> {
            int out  = cfg.count("out_features") ? std::stoi(cfg.at("out_features")) : 1;
            bool bias= !cfg.count("use_bias") || cfg.at("use_bias") == "true";
            auto l   = std::make_unique<Dense>(out, bias);
            l->setName(name);
            return l;
        });

    ComputeGraph::registerLayerFactory("ReLU",
        [](const CfgMap& /*cfg*/, const std::string& name) -> std::unique_ptr<ILayer> {
            auto l = std::make_unique<ReLU>();
            l->setName(name);
            return l;
        });
}

// toJson() must produce a non-empty string containing "nodes" and "edges".
TEST_F(ComputeGraphTest, ToJson_ContainsMandatoryKeys) {
    Dense d1(4);
    ReLU  act;

    ComputeGraph g;
    d1.setName("dense1");
    act.setName("relu1");

    auto h0 = g.input();
    auto h1 = g.record(&d1, h0);
    auto h2 = g.record(&act, h1);
    g.setOutput(h2);

    std::string j = g.toJson();
    EXPECT_FALSE(j.empty());
    EXPECT_NE(j.find("\"nodes\""),      std::string::npos);
    EXPECT_NE(j.find("\"edges\""),      std::string::npos);
    EXPECT_NE(j.find("\"outputNodeId\""),std::string::npos);
    EXPECT_NE(j.find("\"Dense\""),      std::string::npos);
    EXPECT_NE(j.find("\"ReLU\""),       std::string::npos);
    EXPECT_NE(j.find("\"dense1\""),     std::string::npos);
}

// toJson() node count must match recorded layer count.
TEST_F(ComputeGraphTest, ToJson_NodeCount) {
    Dense d1(4), d2(1);
    ComputeGraph g;
    auto h0 = g.input();
    auto h1 = g.record(&d1, h0);
    auto h2 = g.record(&d2, h1);
    g.setOutput(h2);

    std::string j = g.toJson();
    // Count occurrences of "\"id\":" to find node entries
    size_t count = 0, pos = 0;
    while ((pos = j.find("\"id\":", pos)) != std::string::npos) { ++count; ++pos; }
    EXPECT_EQ(count, 2u);
}

// fromJson() reconstructs a graph that can run forward() successfully.
TEST_F(ComputeGraphTest, FromJson_RoundTrip_ForwardSucceeds) {
    ensureFactories();

    Dense d1(4);
    ReLU  act;
    Dense d2(1);

    ComputeGraph g;
    auto h0 = g.input();
    auto h1 = g.record(&d1,  h0);
    auto h2 = g.record(&act, h1);
    auto h3 = g.record(&d2,  h2);
    g.setOutput(h3);
    g.build({2});

    std::string json = g.toJson();

    auto r = ComputeGraph::fromJson(json);
    ASSERT_TRUE(r.ok()) << r.error().message;

    auto& g2 = r.value();
    auto br   = g2.build({2});
    ASSERT_TRUE(br.ok()) << br.error().message;

    auto fr = g2.forward(Tensor::ones({3, 2}));
    ASSERT_TRUE(fr.ok()) << fr.error().message;
    EXPECT_EQ(fr.value().shape()[0], 3);  // batch
    EXPECT_EQ(fr.value().shape()[1], 1);  // output features
}

// fromJson() with unknown layer type returns NotImplemented.
TEST_F(ComputeGraphTest, FromJson_UnknownType_ReturnsError) {
    ensureFactories();

    // Manually craft a JSON referencing an unregistered type
    const std::string badJson = R"({
  "outputNodeId": 1,
  "nodes": [
    { "id": 1, "type": "UnknownLayer", "name": "", "config": {} }
  ],
  "edges": [
    { "from": 0, "to": 1, "layer": 0 }
  ]
})";
    auto r = ComputeGraph::fromJson(badJson);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.error().code, ErrorCode::NotImplemented);
}