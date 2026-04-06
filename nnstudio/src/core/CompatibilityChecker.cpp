/* ============================================================================
 * CompatibilityChecker.cpp — static analysis pass over ComputeGraph nodes
 * LGPL v3
 * ============================================================================ */

#include <core/CompatibilityChecker.h>
#include <core/ComputeGraph.h>
#include <core/Layer.h>

#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nnstudio::core {

// ─── Classification table ────────────────────────────────────────────────────
//
// Keys   : ILayer::typeName() strings (from builtin layers + Sequential + graph types)
// Values : {CompatClass, onnxOpName}
//
// ONNX op names refer to opset 17 (the stable long-term release as of 2023).
// MultiHeadAttention: ONNX added "Attention" in contrib ops, no stable opset op;
//   we use "Attention" here (Microsoft contrib domain is the defacto standard).
// Sequential / ComputeGraph: containers, not individual ops → empty onnxOp.

namespace {

struct Entry {
    CompatClass  compat;
    const char*  onnxOp;
};

// NOLINTNEXTLINE(*-avoid-non-const-global-variables)
static const std::unordered_map<std::string_view, Entry> kTable {
    // ── Standard in PyTorch AND Keras ────────────────────────────────────────
    { "Dense",              { CompatClass::BothStandard, "Gemm"                } },
    { "Conv2D",             { CompatClass::BothStandard, "Conv"                } },
    { "Embedding",          { CompatClass::BothStandard, "Gather"              } },
    { "MultiHeadAttention", { CompatClass::BothStandard, "Attention"           } },
    { "Dropout",            { CompatClass::BothStandard, "Dropout"             } },
    { "BatchNorm1d",        { CompatClass::BothStandard, "BatchNormalization"  } },
    { "LayerNorm",          { CompatClass::BothStandard, "LayerNormalization"  } },
    { "ReLU",               { CompatClass::BothStandard, "Relu"                } },
    { "Sigmoid",            { CompatClass::BothStandard, "Sigmoid"             } },
    { "Tanh",               { CompatClass::BothStandard, "Tanh"                } },
    { "Softmax",            { CompatClass::BothStandard, "Softmax"             } },
    { "GELU",               { CompatClass::BothStandard, "Gelu"                } },
    { "Sequential",         { CompatClass::BothStandard, ""                    } },

    // ── Standard in PyTorch only ──────────────────────────────────────────────
    //   (Keras 3.x added get_custom_objects but LeakyReLU is not a core Keras class)
    { "LeakyReLU",          { CompatClass::TorchStandard, "LeakyRelu"          } },

    // ── NNStudio-internal containers ──────────────────────────────────────────
    //   ComputeGraph is our DAG engine; it has no ONNX equivalent.
    //   ActivationsFnLayer is a thin ILayer wrapper — the underlying activation
    //   (ReLU, GELU, …) is already Standard; the wrapper itself is an artefact.
    //   We treat the wrapper class itself as extension so users see what's hidden.
    { "ComputeGraph",       { CompatClass::NNStudioExtension, "" } },
    { "ActivationsFnLayer", { CompatClass::NNStudioExtension, "" } },
};

} // anonymous namespace

// ─── CompatibilityChecker ────────────────────────────────────────────────────

CompatClass CompatibilityChecker::classify(std::string_view typeName) noexcept
{
    auto it = kTable.find(typeName);
    if (it == kTable.end()) {
        return CompatClass::NNStudioExtension;
    }
    return it->second.compat;
}

std::string CompatibilityChecker::onnxOp(std::string_view typeName)
{
    auto it = kTable.find(typeName);
    if (it == kTable.end()) {
        // Unknown / extension — use custom-op domain
        return std::string("com.nnstudio.") + std::string(typeName);
    }
    const char* op = it->second.onnxOp;
    if (op == nullptr || op[0] == '\0') {
        // Known type but no single ONNX op (e.g. Sequential, ComputeGraph)
        return "";
    }
    return std::string(op);
}

CompatReport CompatibilityChecker::check(const ComputeGraph& graph)
{
    CompatReport report;
    report.nodes.reserve(graph.ops().size());

    for (const OpRecord& op : graph.ops()) {
        if (op.layer == nullptr) { continue; }

        NodeCompatInfo info;
        info.typeName   = std::string(op.layer->typeName());
        info.layerName  = std::string(op.layer->name());
        info.compat     = classify(info.typeName);
        info.onnxOp     = onnxOp(info.typeName);

        report.nodes.push_back(std::move(info));
    }

    return report;
}

// ─── CompatReport ────────────────────────────────────────────────────────────

bool CompatReport::hasExtensions() const noexcept
{
    for (const NodeCompatInfo& n : nodes) {
        if (isExtension(n.compat)) { return true; }
    }
    return false;
}

CompatClass CompatReport::overallLevel() const noexcept
{
    if (nodes.empty()) {
        return CompatClass::BothStandard;  // vacuously true
    }

    // Accumulate the intersection of flags: a layer must be standard in a
    // framework for the WHOLE graph to be considered standard in that framework.
    uint8_t common = static_cast<uint8_t>(CompatClass::BothStandard);  // 0x03

    for (const NodeCompatInfo& n : nodes) {
        common &= static_cast<uint8_t>(n.compat);
        if (common == 0x00) { break; }
    }

    return static_cast<CompatClass>(common);
}

std::vector<NodeCompatInfo> CompatReport::extensionNodes() const
{
    std::vector<NodeCompatInfo> out;
    for (const NodeCompatInfo& n : nodes) {
        if (isExtension(n.compat)) { out.push_back(n); }
    }
    return out;
}

std::string CompatReport::summary() const
{
    auto exts = extensionNodes();
    if (exts.empty()) { return {}; }

    std::ostringstream oss;
    oss << "WARNING: " << exts.size()
        << " NNStudio-extension op(s) found:\n";

    for (std::size_t i = 0; i < exts.size(); ++i) {
        const NodeCompatInfo& n = exts[i];
        oss << "  [" << i << "]"
            << "  type=" << n.typeName
            << "  name=" << n.layerName;
        if (!n.onnxOp.empty()) {
            oss << "  onnx=" << n.onnxOp;
        }
        oss << '\n';
    }

    return oss.str();
}

} // namespace nnstudio::core
