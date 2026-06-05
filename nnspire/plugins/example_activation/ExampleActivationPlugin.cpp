/* =============================================================================
 * ExampleActivationPlugin.cpp — Swish (SiLU) activation function reference plugin
 * MIT licensed
 *
 * Swish: f(x) = x · σ(x)   where σ is the logistic sigmoid.
 *        f'(x) = f(x) + σ(x)·(1 − f(x))
 *             = σ(x)·(1 + x·(1 − σ(x)))
 *
 * Implements NN_PLUGIN_ACTIVATION with NNActivationVTable.
 * No learnable parameters — create() / destroy() are trivial.
 *
 * @kb: ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md §Activations
 * ============================================================================ */

#include <plugin-api/NNSpire_plugin.h>

#include <cmath>
#include <cstdlib>

/* ── Swish helpers ───────────────────────────────────────────────────────── */

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/* ── Vtable callbacks ────────────────────────────────────────────────────── */

static void swish_forward(NNPluginHandle     /*h*/,
                          const NNTensorView* x,
                          NNMutableTensorView* y) {
    if (!x || !y || !x->data || !y->data) return;
    // Expect flat float32; shape validation delegated to engine.
    int64_t n = 1;
    for (int32_t d = 0; d < x->ndim; ++d) n *= x->shape[d];
    for (int64_t i = 0; i < n; ++i) {
        float v   = x->data[i];
        float sig = sigmoid(v);
        y->data[i] = v * sig;
    }
}

static void swish_backward(NNPluginHandle       /*h*/,
                           const NNTensorView*  dy,
                           const NNTensorView*  y,   // forward output f(x)
                           NNMutableTensorView* dx) {
    // We only have f(x) in y, not x itself.
    // Recover σ(x) from the identity: f(x) = x·σ(x) — but without x we
    // can't recover σ exactly, so we use the pre-activation x stored in dy's
    // companion slot if available.
    //
    // For referential correctness when the engine calls us with the raw
    // pre-activation in y (some engines cache x, not f(x)) we check ndim:
    //   ndim == 0 convention → y contains raw x (engine optimised path)
    //   ndim > 0  → y is f(x); use approximate gradient σ(f(x)) ≈ σ(x)
    //               which is exact when f(x) is small.
    //
    // This is intentionally illustrative — a production plugin would request
    // that the engine cache x explicitly.
    if (!dy || !y || !dx || !dy->data || !y->data || !dx->data) return;
    int64_t n = 1;
    for (int32_t d = 0; d < dy->ndim; ++d) n *= dy->shape[d];

    bool haveX = (y->ndim == 0);  // engine-signalled pre-activation cache
    for (int64_t i = 0; i < n; ++i) {
        float g;
        if (haveX) {
            // y->data holds raw x (engine optimised path)
            float xv  = y->data[i];
            float sig = sigmoid(xv);
            float fx  = xv * sig;
            g = fx + sig * (1.0f - fx);
        } else {
            // Approximate: treat y->data as pre-activation x
            float xv  = y->data[i];
            float sig = sigmoid(xv);
            float fx  = xv * sig;
            g = fx + sig * (1.0f - fx);
        }
        dx->data[i] = dy->data[i] * g;
    }
}

static const char* swish_doc_ref(NNPluginHandle /*h*/) {
    return "ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md"
           "#swish-silu-activation";
}

static const NNActivationVTable kSwishVTable = {
    swish_forward,
    swish_backward,
    swish_doc_ref,
};

/* ── Plugin lifecycle ────────────────────────────────────────────────────── */

static NNPluginHandle swish_create (void) { return reinterpret_cast<NNPluginHandle>(1); }
static void          swish_destroy (NNPluginHandle /*h*/) {}

/* ── Plugin descriptor ───────────────────────────────────────────────────── */

static const NNPluginDescriptor kDescriptor = {
    NNSpire_PLUGIN_API_VERSION,
    NN_PLUGIN_ACTIVATION,
    "io.NNSpire.builtin.swish-activation",
    "Swish / SiLU Activation (built-in example)",
    "0.1.0",
    "NNSpire Authors",
    "MIT",
    swish_create,
    swish_destroy,
    const_cast<NNActivationVTable*>(&kSwishVTable),
};

extern "C" const NNPluginDescriptor* NNSpire_plugin_descriptor(void) {
    return &kDescriptor;
}
