/* ============================================================================
 * CompatibilityChecker.h — static analysis pass over ComputeGraph nodes
 *
 * PURPOSE
 * ───────
 * After a model is constructed (ComputeGraph recorded), pass it to
 * CompatibilityChecker::check() to learn:
 *   • which layers are standard PyTorch ops
 *   • which layers are standard Keras ops
 *   • which are NNSpire-only extensions that will need adapters on export
 *
 * USAGE
 * ─────
 *   ComputeGraph g;
 *   // … record layers …
 *   auto report = CompatibilityChecker::check(g);
 *
 *   if (report.hasExtensions()) {
 *       std::cout << report.summary();   // lists non-standard ops
 *   }
 *
 * ONNX ALIGNMENT
 * ──────────────
 * check() also populates NodeCompatInfo::onnxOp for every layer that maps
 * to a stable ONNX operator (opset 17).  Layers that have NO standard ONNX
 * equivalent are marked with the custom-op domain prefix "com.NNSpire.*".
 * ============================================================================ */

#pragma once

#include <core/ComputeGraph.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace NNSpire::core {

// ─── Compatibility classification ────────────────────────────────────────────

/**
 * Each layer type is zero or more of: PyTorch-standard, Keras-standard,
 * or exclusively an NNSpire extension.
 *
 * Represented as a bitmask so a layer can be "both" standard.
 */
enum class CompatClass : uint8_t {
    NNSpireExtension = 0x00,   ///< Not found in PyTorch or Keras standard surface
    TorchStandard     = 0x01,   ///< Standard PyTorch/LibTorch op
    KerasStandard     = 0x02,   ///< Standard Keras (tf.keras) op
    BothStandard      = 0x03,   ///< Standard in BOTH PyTorch and Keras
};

/// Returns true iff the layer is a standard PyTorch op.
constexpr bool isTorchStandard (CompatClass c) noexcept { return (static_cast<uint8_t>(c) & 0x01) != 0; }
/// Returns true iff the layer is a standard Keras op.
constexpr bool isKerasStandard (CompatClass c) noexcept { return (static_cast<uint8_t>(c) & 0x02) != 0; }
/// Returns true iff the layer is NOT in any standard framework.
constexpr bool isExtension     (CompatClass c) noexcept { return c == CompatClass::NNSpireExtension; }

// ─── Per-node result ──────────────────────────────────────────────────────────

struct NodeCompatInfo {
    std::string typeName;   ///< ILayer::typeName() value
    std::string layerName;  ///< ILayer::name() value (user-assigned display name)
    CompatClass compat;     ///< classification bitmask
    std::string onnxOp;     ///< ONNX operator name (opset 17); "com.NNSpire.*" if extension
};

// ─── Report ───────────────────────────────────────────────────────────────────

struct CompatReport {
    std::vector<NodeCompatInfo> nodes;  ///< one entry per OpRecord in the graph

    /** True if any node is an NNSpire extension (not in torch or keras). */
    bool hasExtensions() const noexcept;

    /**
     * Overall compatibility level.
     *
     * Returns:
     *   BothStandard      — every layer is in both PyTorch and Keras
     *   TorchStandard     — every layer is at least PyTorch-standard
     *   KerasStandard     — every layer is at least Keras-standard
     *   NNSpireExtension — one or more layers are extension-only
     */
    CompatClass overallLevel() const noexcept;

    /**
     * Human-readable summary string.
     * Empty string if no extensions are present.
     *
     * Example:
     *   "WARNING: 2 NNSpire-extension op(s) found:\n"
     *   "  [0] type=FancyLayer   name=fancy  onnx=com.NNSpire.FancyLayer\n"
     */
    std::string summary() const;

    /** List only the extension nodes. */
    std::vector<NodeCompatInfo> extensionNodes() const;
};

// ─── CompatibilityChecker ─────────────────────────────────────────────────────

/**
 * Static analysis pass over a recorded ComputeGraph.
 *
 * All methods are static; the class has no state.
 */
class CompatibilityChecker {
public:
    /**
     * Run the compatibility pass over every recorded op in the graph.
     *
     * The graph does NOT need to have been forward-executed; it only needs to
     * have been populated via record() calls.
     *
     * O(n) in the number of recorded ops.
     */
    static CompatReport check(const ComputeGraph& graph);

    /**
     * Classify a single layer type name.
     *
     * @param typeName  The value returned by ILayer::typeName().
     * @returns         The CompatClass for that type.
     */
    static CompatClass  classify(std::string_view typeName) noexcept;

    /**
     * Return the ONNX operator name (opset 17) for the given layer type.
     *
     * For standard layers this is the canonical ONNX op string (e.g. "Gemm",
     * "BatchNormalization", "Relu").  For NNSpire extensions the string is
     * prefixed with the custom-op domain: "com.NNSpire.<typeName>".
     */
    static std::string onnxOp(std::string_view typeName);
};

} // namespace NNSpire::core
