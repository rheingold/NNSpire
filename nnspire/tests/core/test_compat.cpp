/* ============================================================================
 * test_compat.cpp — unit tests for CompatibilityChecker
 *
 * Coverage:
 *   1.  classify() — known standard_both layers
 *   2.  classify() — torch-only layers (LeakyReLU)
 *   3.  classify() — NNSpire extension (ComputeGraph, unknown)
 *   4.  onnxOp()   — standard layers map to correct ONNX op names
 *   5.  onnxOp()   — extension layers get com.NNSpire.* prefix
 *   6.  check()    — empty graph → empty report, no extensions
 *   7.  check()    — all-standard graph → no extensions, overallLevel = BothStandard
 *   8.  check()    — graph with one extension → hasExtensions(), summary non-empty
 *   9.  CompatReport::overallLevel() — mixed graph → NNSpireExtension
 *  10.  CompatReport::extensionNodes() — returns only extension nodes
 * ============================================================================ */

#include <gtest/gtest.h>

#include <builtin/layers/Dense.h>
#include <builtin/activations/Activations.h>
#include <builtin/activations/FnLayer.h>
#include <core/CompatibilityChecker.h>
#include <core/ComputeGraph.h>

using namespace NNSpire::core;
using namespace NNSpire::builtin::layers;
using namespace NNSpire::builtin::activations;

// ─── 1. classify — BothStandard layers ───────────────────────────────────────

TEST(CompatibilityCheckerTest, Classify_Dense_BothStandard)
{
    auto c = CompatibilityChecker::classify("Dense");
    EXPECT_EQ(c, CompatClass::BothStandard);
    EXPECT_TRUE(isTorchStandard(c));
    EXPECT_TRUE(isKerasStandard(c));
    EXPECT_FALSE(isExtension(c));
}

TEST(CompatibilityCheckerTest, Classify_ReLU_BothStandard)
{
    auto c = CompatibilityChecker::classify("ReLU");
    EXPECT_EQ(c, CompatClass::BothStandard);
}

TEST(CompatibilityCheckerTest, Classify_Softmax_BothStandard)
{
    auto c = CompatibilityChecker::classify("Softmax");
    EXPECT_EQ(c, CompatClass::BothStandard);
}

// ─── 2. classify — TorchStandard only ────────────────────────────────────────

TEST(CompatibilityCheckerTest, Classify_LeakyReLU_TorchStandard)
{
    auto c = CompatibilityChecker::classify("LeakyReLU");
    EXPECT_EQ(c, CompatClass::TorchStandard);
    EXPECT_TRUE(isTorchStandard(c));
    EXPECT_FALSE(isKerasStandard(c));
    EXPECT_FALSE(isExtension(c));
}

// ─── 3. classify — NNSpire extension ────────────────────────────────────────

TEST(CompatibilityCheckerTest, Classify_ComputeGraph_Extension)
{
    auto c = CompatibilityChecker::classify("ComputeGraph");
    EXPECT_EQ(c, CompatClass::NNSpireExtension);
    EXPECT_TRUE(isExtension(c));
}

TEST(CompatibilityCheckerTest, Classify_Unknown_Extension)
{
    auto c = CompatibilityChecker::classify("SomeUnknownLayer");
    EXPECT_EQ(c, CompatClass::NNSpireExtension);
    EXPECT_TRUE(isExtension(c));
}

// ─── 4. onnxOp — standard layers ─────────────────────────────────────────────

TEST(CompatibilityCheckerTest, OnnxOp_Dense_Gemm)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("Dense"), "Gemm");
}

TEST(CompatibilityCheckerTest, OnnxOp_Conv2D_Conv)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("Conv2D"), "Conv");
}

TEST(CompatibilityCheckerTest, OnnxOp_Embedding_Gather)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("Embedding"), "Gather");
}

TEST(CompatibilityCheckerTest, OnnxOp_BatchNorm_BatchNormalization)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("BatchNorm1d"), "BatchNormalization");
}

TEST(CompatibilityCheckerTest, OnnxOp_LayerNorm_LayerNormalization)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("LayerNorm"), "LayerNormalization");
}

TEST(CompatibilityCheckerTest, OnnxOp_ReLU_Relu)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("ReLU"), "Relu");
}

TEST(CompatibilityCheckerTest, OnnxOp_GELU_Gelu)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("GELU"), "Gelu");
}

TEST(CompatibilityCheckerTest, OnnxOp_LeakyReLU_LeakyRelu)
{
    EXPECT_EQ(CompatibilityChecker::onnxOp("LeakyReLU"), "LeakyRelu");
}

// ─── 5. onnxOp — extension layers ────────────────────────────────────────────

TEST(CompatibilityCheckerTest, OnnxOp_UnknownLayer_CustomDomain)
{
    const std::string op = CompatibilityChecker::onnxOp("FancyLayer");
    EXPECT_EQ(op, "com.NNSpire.FancyLayer");
}

// ─── 6. check — empty graph ───────────────────────────────────────────────────

TEST(CompatibilityCheckerTest, Check_EmptyGraph_NoExtensions)
{
    ComputeGraph g;
    CompatReport r = CompatibilityChecker::check(g);
    EXPECT_TRUE(r.nodes.empty());
    EXPECT_FALSE(r.hasExtensions());
    EXPECT_EQ(r.overallLevel(), CompatClass::BothStandard);
    EXPECT_TRUE(r.summary().empty());
}

// ─── 7. check — all-standard graph ───────────────────────────────────────────

TEST(CompatibilityCheckerTest, Check_StandardGraph_NoExtensions)
{
    // Dense(2,4) → ReLU
    Dense   dense(4);
    ReLU    relu;

    dense.setName("d1");
    relu.setName("r1");

    ComputeGraph g;
    NodeId in  = g.input();
    NodeId h1  = g.record(&dense, in);
    NodeId out = g.record(&relu,  h1);
    g.setOutput(out);

    CompatReport r = CompatibilityChecker::check(g);
    EXPECT_EQ(r.nodes.size(), 2u);
    EXPECT_FALSE(r.hasExtensions());
    EXPECT_EQ(r.overallLevel(), CompatClass::BothStandard);
    EXPECT_TRUE(r.summary().empty());

    // Check per-node classification
    EXPECT_EQ(r.nodes[0].typeName, "Dense");
    EXPECT_EQ(r.nodes[0].compat,   CompatClass::BothStandard);
    EXPECT_EQ(r.nodes[0].onnxOp,   "Gemm");

    EXPECT_EQ(r.nodes[1].typeName, "ReLU");
    EXPECT_EQ(r.nodes[1].compat,   CompatClass::BothStandard);
    EXPECT_EQ(r.nodes[1].onnxOp,   "Relu");
}

// ─── 8. check — graph with extension ─────────────────────────────────────────

namespace {

/// Minimal ILayer that reports itself as an unknown NNSpire-extension layer.
class FancyLayer final : public ILayer {
public:
    std::string_view typeName() const noexcept override { return "FancyLayer"; }

    Result<Shape>  build(const Shape& s) override                  { return s; }
    Result<Tensor> forward(const Tensor& x, EvalTrace*) override   { return x; }
    Result<Tensor> backward(const Tensor& g, EvalTrace*) override  { return g; }
    std::vector<Parameter*>       parameters() override            { return {}; }
    std::vector<const Parameter*> parameters() const override      { return {}; }
};

} // anonymous namespace

TEST(CompatibilityCheckerTest, Check_GraphWithExtension_HasExtensions)
{
    Dense      dense(4);
    FancyLayer fancy;
    fancy.setName("my_fancy");

    ComputeGraph g;
    NodeId in  = g.input();
    NodeId h1  = g.record(&dense, in);
    NodeId out = g.record(&fancy, h1);
    g.setOutput(out);

    CompatReport r = CompatibilityChecker::check(g);
    EXPECT_EQ(r.nodes.size(), 2u);
    EXPECT_TRUE(r.hasExtensions());
    EXPECT_EQ(r.overallLevel(), CompatClass::NNSpireExtension);
    EXPECT_FALSE(r.summary().empty());
}

// ─── 9. overallLevel — mixed (torch + extension) ─────────────────────────────

TEST(CompatibilityCheckerTest, OverallLevel_MixedGraph_Extension)
{
    Dense      dense(4);
    LeakyReLU  lrelu;      // TorchStandard only
    FancyLayer fancy;

    ComputeGraph g;
    NodeId in  = g.input();
    NodeId h1  = g.record(&dense, in);
    NodeId h2  = g.record(&lrelu, h1);
    NodeId out = g.record(&fancy, h2);
    g.setOutput(out);

    CompatReport r = CompatibilityChecker::check(g);
    EXPECT_EQ(r.overallLevel(), CompatClass::NNSpireExtension);
}

// ─── 10. extensionNodes ───────────────────────────────────────────────────────

TEST(CompatibilityCheckerTest, ExtensionNodes_OnlyExtensionsReturned)
{
    Dense      dense(4);
    FancyLayer fancy;

    ComputeGraph g;
    NodeId in  = g.input();
    NodeId h1  = g.record(&dense, in);
    NodeId out = g.record(&fancy, h1);
    g.setOutput(out);

    CompatReport r = CompatibilityChecker::check(g);
    auto exts = r.extensionNodes();
    ASSERT_EQ(exts.size(), 1u);
    EXPECT_EQ(exts[0].typeName, "FancyLayer");
    EXPECT_EQ(exts[0].onnxOp,   "com.NNSpire.FancyLayer");
}
