#pragma once
/* ============================================================================
 * FeatureFlags.h
 * NNSpire Feature Flag Registry — LGPL v3
 *
 * All Studio features are registered here with a tier level.
 * STANDING COMMITMENT (see LICENSING.md §2.1):
 *   The categories FREE_LEARNING, FREE_VISUALIZATION, FREE_PIPELINE,
 *   FREE_RUNNERS, FREE_KB_HELP, and FREE_EXPORT are permanently FREE.
 *   Any change from FREE to PRO must be a documented, deliberate decision
 *   with a rationale comment and recorded in ai.md.
 * ============================================================================ */

#include <string_view>

namespace NNSpire::core {

enum class Tier {
    FREE,        ///< Always enabled; no license check
    PRO,         ///< Requires Pro license (not currently in use)
    ENTERPRISE   ///< Requires Enterprise license (not currently in use)
};

struct FeatureFlag {
    std::string_view id;
    Tier             tier;
    bool             enabled;    ///< Runtime override; currently always true for FREE
    std::string_view description;
};

/// Returns true if the feature should be accessible to the user.
/// Currently: all FREE features return true unconditionally.
/// Future: PRO/ENTERPRISE will check a license token here.
inline constexpr bool isEnabled(const FeatureFlag& f) noexcept {
    return f.tier == Tier::FREE && f.enabled;
    // When Pro tier is introduced, extend this to check a license token.
}

/* ────────────────────────────────────────────────────────────────────────────
 * Feature declarations
 * All items below MUST remain FREE per LICENSING.md §2.1
 * ──────────────────────────────────────────────────────────────────────────── */

// Learning & educational features
inline constexpr FeatureFlag FF_WIZARD_MODE =
    { "wizard_mode",      Tier::FREE, true, "Guided walkthrough / wizard mode" };
inline constexpr FeatureFlag FF_LEARNING_KB =
    { "learning_kb",      Tier::FREE, true, "In-app KB help and context popups" };
inline constexpr FeatureFlag FF_NEURON_VIEWER =
    { "neuron_viewer",    Tier::FREE, true, "Neuron-level visualization for small networks" };

// Visualization
inline constexpr FeatureFlag FF_WEIGHT_VIEWER =
    { "weight_viewer",    Tier::FREE, true, "Weight heatmaps and activation histograms" };
inline constexpr FeatureFlag FF_LOSS_CHART =
    { "loss_chart",       Tier::FREE, true, "Live training loss / metric charts" };
inline constexpr FeatureFlag FF_TOPOLOGY_DIAGRAM =
    { "topology_diagram", Tier::FREE, true, "Auto-generated model topology diagram" };
inline constexpr FeatureFlag FF_GRADIENT_VIEW =
    { "gradient_view",    Tier::FREE, true, "Gradient magnitude per layer during training" };

// Pipeline
inline constexpr FeatureFlag FF_PIPELINE =
    { "pipeline",         Tier::FREE, true, "Full multi-stage inference pipeline" };
inline constexpr FeatureFlag FF_INPUT_TEXT =
    { "input_text",       Tier::FREE, true, "Text input adapter" };
inline constexpr FeatureFlag FF_INPUT_IMAGE =
    { "input_image",      Tier::FREE, true, "Image input adapter" };
inline constexpr FeatureFlag FF_INPUT_AUDIO =
    { "input_audio",      Tier::FREE, true, "Audio input adapter" };
inline constexpr FeatureFlag FF_INPUT_STRUCTURED =
    { "input_structured", Tier::FREE, true, "Structured data (CSV/Arrow) input adapter" };
inline constexpr FeatureFlag FF_INPUT_MCP =
    { "input_mcp",        Tier::FREE, true, "MCP message input adapter" };
inline constexpr FeatureFlag FF_CONTEXT_RAG =
    { "context_rag",      Tier::FREE, true, "RAG context retrieval stage" };

// Runner connectors
inline constexpr FeatureFlag FF_RUNNER_TRITON =
    { "runner_triton",    Tier::FREE, true, "Triton Inference Server connector" };
inline constexpr FeatureFlag FF_RUNNER_TFS =
    { "runner_tfs",       Tier::FREE, true, "TensorFlow Serving connector" };
inline constexpr FeatureFlag FF_RUNNER_KSERVE =
    { "runner_kserve",    Tier::FREE, true, "KServe V2 connector" };
inline constexpr FeatureFlag FF_RUNNER_ORT =
    { "runner_ort",       Tier::FREE, true, "ONNX Runtime embedded runner" };
inline constexpr FeatureFlag FF_RUNNER_OPENAI =
    { "runner_openai",    Tier::FREE, true, "OpenAI-compatible REST connector" };

// Export formats
inline constexpr FeatureFlag FF_EXPORT_ONNX =
    { "export_onnx",      Tier::FREE, true, "Export to ONNX (.onnx)" };
inline constexpr FeatureFlag FF_EXPORT_NNS =
    { "export_nns",       Tier::FREE, true, "Save as NNSpire project (.nns)" };
inline constexpr FeatureFlag FF_EXPORT_NNSR =
    { "export_nnsr",      Tier::FREE, true, "Export runner bundle (.nnsr)" };
inline constexpr FeatureFlag FF_EXPORT_TORCHSCRIPT =
    { "export_torchscript", Tier::FREE, true, "Export to TorchScript (.pt)" };
inline constexpr FeatureFlag FF_CODE_EXPORT_PYTHON =
    { "code_export_python", Tier::FREE, true, "Generate Python source code from model" };
inline constexpr FeatureFlag FF_CODE_EXPORT_CPP =
    { "code_export_cpp",    Tier::FREE, true, "Generate C++ source code from model" };

// Plugin system (the mechanism is FREE; commercial cert issuance is the revenue model)
inline constexpr FeatureFlag FF_PLUGIN_SYSTEM =
    { "plugin_system",    Tier::FREE, true, "Plugin loading and trust verification" };

} // namespace NNSpire::core
