/* ============================================================================
 * {{PLUGIN_NAME}} — NNStudio Layer Plugin implementation
 *
 * Replace every {{PLACEHOLDER}} with your actual values before distribution.
 * ============================================================================ */

#include "my_layer.h"

#include <nnstudio_plugin.h>

/* ── Helper: access typed handle ──────────────────────────────────────────── */
static inline {{PLUGIN_CLASS_NAME}}* self(NNPluginHandle h) noexcept {
    return static_cast<{{PLUGIN_CLASS_NAME}}*>(h);
}

/* ── Instance lifecycle ────────────────────────────────────────────────────── */

static NNPluginHandle create() noexcept {
    return new (std::nothrow) {{PLUGIN_CLASS_NAME}}{};
}

static void destroy(NNPluginHandle h) noexcept {
    delete self(h);
}

/* ── VTable implementations ────────────────────────────────────────────────── */

static void vtable_forward(NNPluginHandle h,
                            const NNTensorView* in,
                            NNTensorView* out) noexcept {
    self(h)->forward(in, out);
}

static void vtable_backward(NNPluginHandle h,
                             const NNTensorView* grad_out,
                             NNTensorView* grad_in) noexcept {
    self(h)->backward(grad_out, grad_in);
}

static void vtable_parameters(NNPluginHandle h,
                               NNTensorView** params,
                               int32_t* count) noexcept {
    self(h)->parameters(params, count);
}

static const char* vtable_doc_ref(NNPluginHandle h) noexcept {
    return self(h)->docRef();
}

/* ── Static vtable and descriptor ─────────────────────────────────────────── */

static const NNLayerVTable LAYER_VTABLE = {
    /* forward     */ vtable_forward,
    /* backward    */ vtable_backward,
    /* parameters  */ vtable_parameters,
    /* doc_ref     */ vtable_doc_ref,
};

static const NNPluginDescriptor DESCRIPTOR = {
    /* api_version */ NNSTUDIO_PLUGIN_API_VERSION,
    /* type        */ NN_PLUGIN_LAYER,
    /* id          */ "{{PLUGIN_REVERSE_DOMAIN_ID}}",
    /* name        */ "{{PLUGIN_NAME}}",
    /* version     */ "{{PLUGIN_VERSION}}",
    /* author      */ "{{PLUGIN_AUTHOR}}",
    /* license     */ "{{PLUGIN_LICENSE}}",
    /* create      */ create,
    /* destroy     */ destroy,
    /* vtable      */ const_cast<NNLayerVTable*>(&LAYER_VTABLE),
};

/* ── Mandatory entry point ─────────────────────────────────────────────────── */

extern "C" const NNPluginDescriptor* nnstudio_plugin_descriptor() noexcept {
    return &DESCRIPTOR;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Implement your layer methods below.
 * ══════════════════════════════════════════════════════════════════════════════ */

{{PLUGIN_CLASS_NAME}}::{{PLUGIN_CLASS_NAME}}() {
    // TODO: initialise weights, allocate buffers
}

{{PLUGIN_CLASS_NAME}}::~{{PLUGIN_CLASS_NAME}}() {
    // TODO: free any heap allocations
}

void {{PLUGIN_CLASS_NAME}}::forward(const NNTensorView* in, NNTensorView* out) {
    (void)in; (void)out;
    // TODO: compute forward pass
    //
    // Example:
    //   for (int i = 0; i < numElements(in); ++i)
    //       out->data[i] = myActivation(in->data[i]);
    //
    // Save any intermediate values needed in backward().
}

void {{PLUGIN_CLASS_NAME}}::backward(const NNTensorView* grad_out,
                                     NNTensorView* grad_in) {
    (void)grad_out; (void)grad_in;
    // TODO: compute backward pass
    //
    // Compute grad_in  = dLoss/dInput  (chain rule through forward())
    // Accumulate param gradients internally so the optimizer can read them.
}

void {{PLUGIN_CLASS_NAME}}::parameters(NNTensorView** params, int32_t* count) {
    (void)params;
    *count = 0;
    // TODO: fill params[] with views of your trainable weight tensors.
    // If stateless, leave count = 0 and set vtable.parameters = nullptr.
}

const char* {{PLUGIN_CLASS_NAME}}::docRef() const noexcept {
    return "ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md";
}
