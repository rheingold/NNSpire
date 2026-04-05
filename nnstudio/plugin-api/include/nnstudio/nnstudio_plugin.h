/* ============================================================================
 * nnstudio_plugin.h
 * NNStudio Plugin C ABI — PUBLIC API, LGPL v3
 *
 * This header defines the stable C-linkage interface between NNStudio and all
 * plugins (C++, Python via pybind11 bridge, or any other compiled language).
 * NO C++ in this header — pure C types and function pointers only.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */
#ifndef NNSTUDIO_PLUGIN_H
#define NNSTUDIO_PLUGIN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ──────────────────────────────────────────────────────────────── */
#define NNSTUDIO_PLUGIN_API_VERSION 1

/* ── Plugin types ─────────────────────────────────────────────────────────── */
typedef enum NNPluginType {
    NN_PLUGIN_LAYER          = 1,
    NN_PLUGIN_ACTIVATION     = 2,
    NN_PLUGIN_OPTIMIZER      = 3,
    NN_PLUGIN_TOKENIZER      = 4,
    NN_PLUGIN_BACKEND        = 5,
    NN_PLUGIN_INPUT_ADAPTER  = 6,
    NN_PLUGIN_OUTPUT_ADAPTER = 7,
    NN_PLUGIN_CONTEXT_SOURCE = 8,
    NN_PLUGIN_RUNNER_CLIENT  = 9,
    NN_PLUGIN_UI_PANEL       = 10,
    NN_PLUGIN_TRUST_UPDATE   = 99    /* handled by TrustUpdateHandler, not PluginLoader */
} NNPluginType;

/* ── Tensor view (read-only window into engine-owned memory) ──────────────── */
typedef struct NNTensorView {
    const float*   data;      /* raw float32 data pointer (device: CPU only in ABI) */
    const int64_t* shape;     /* dimension sizes */
    int32_t        ndim;      /* number of dimensions */
    int32_t        dtype;     /* 0=float32, 1=float16, 2=int8, 3=int32, 4=bool */
} NNTensorView;

/* Mutable tensor view for outputs written by plugins */
typedef struct NNMutableTensorView {
    float*         data;
    const int64_t* shape;
    int32_t        ndim;
    int32_t        dtype;
} NNMutableTensorView;

/* ── Opaque plugin instance handle ───────────────────────────────────────── */
typedef void* NNPluginHandle;

/* ── Plugin descriptor (one per shared library) ──────────────────────────── */
typedef struct NNPluginDescriptor {
    uint32_t       api_version; /* MUST equal NNSTUDIO_PLUGIN_API_VERSION */
    NNPluginType   type;
    const char*    id;          /* Reverse-domain string: com.example.my-plugin */
    const char*    name;        /* Human-readable display name */
    const char*    version;     /* Semver string: "1.0.0" */
    const char*    author;
    const char*    license;     /* SPDX identifier: "MIT", "Apache-2.0", etc. */

    NNPluginHandle (*create) (void);
    void           (*destroy)(NNPluginHandle);

    void*          vtable;      /* Cast per type; see vtable structs below */
} NNPluginDescriptor;

/* Every plugin shared library MUST export this symbol with this exact name */
const NNPluginDescriptor* nnstudio_plugin_descriptor(void);


/* ════════════════════════════════════════════════════════════════════════════
 * Type-specific vtable structs
 * ════════════════════════════════════════════════════════════════════════════ */

/* ── Layer vtable ─────────────────────────────────────────────────────────── */
typedef struct NNLayerVTable {
    /** Forward pass: compute output from input. */
    void (*forward)(
        NNPluginHandle          handle,
        const NNTensorView*     input,
        NNMutableTensorView*    output);

    /** Backward pass: compute grad_input from grad_output.
     *  Also accumulate gradients into trainable parameters if any. */
    void (*backward)(
        NNPluginHandle          handle,
        const NNTensorView*     grad_output,
        NNMutableTensorView*    grad_input);

    /** Return pointers to all trainable parameter tensors (for optimizer).
     *  May be NULL if the layer has no trainable parameters. */
    void (*parameters)(
        NNPluginHandle          handle,
        NNMutableTensorView**   params_out,     /* engine allocates, plugin fills */
        int32_t*                count_out);

    /** Return KB path#anchor for mathematical documentation. */
    const char* (*doc_ref)(NNPluginHandle handle);
} NNLayerVTable;


/* ── Activation vtable ────────────────────────────────────────────────────── */
typedef struct NNActivationVTable {
    void (*forward) (NNPluginHandle, const NNTensorView* x, NNMutableTensorView* y);
    void (*backward)(NNPluginHandle, const NNTensorView* dy,
                     const NNTensorView* y, NNMutableTensorView* dx);
    const char* (*doc_ref)(NNPluginHandle);
} NNActivationVTable;


/* ── Optimizer vtable ─────────────────────────────────────────────────────── */
typedef struct NNOptimizerVTable {
    /** Zero all parameter gradients. */
    void (*zero_grad)(NNPluginHandle);

    /** Perform one parameter update step.
     *  @param params      array of parameter tensor views
     *  @param grads       corresponding gradient tensor views
     *  @param count       number of parameter/gradient pairs
     *  @param step_count  global step index (for schedulers, Adam bias correction) */
    void (*step)(
        NNPluginHandle          handle,
        NNMutableTensorView**   params,
        const NNTensorView**    grads,
        int32_t                 count,
        uint64_t                step_count);

    const char* (*doc_ref)(NNPluginHandle);
} NNOptimizerVTable;


/* ── Tokenizer vtable ─────────────────────────────────────────────────────── */
typedef struct NNTokenizerVTable {
    /** Encode text to token IDs. Engine calls free_result(ids) when done. */
    void    (*encode)   (NNPluginHandle, const char* text,
                         int32_t** ids_out, int32_t* count_out);

    /** Decode token IDs to text. Engine calls free_result(text) when done. */
    char*   (*decode)   (NNPluginHandle, const int32_t* ids, int32_t count);

    /** Free a buffer previously returned by encode or decode. */
    void    (*free_result)(NNPluginHandle, void* ptr);

    uint32_t    (*vocab_size)   (NNPluginHandle);
    const char* (*token_to_str) (NNPluginHandle, int32_t id);
    int32_t     (*str_to_token) (NNPluginHandle, const char* token); /* -1 if unknown */

    const char* (*doc_ref)(NNPluginHandle);
} NNTokenizerVTable;


/* ── UI Panel vtable ──────────────────────────────────────────────────────── */
typedef struct NNUIPanelVTable {
    /** URL of the QML component to load (qrc:// or file://).
     *  The QML component communicates with the plugin via the controller interface. */
    const char* (*qml_source_url)(NNPluginHandle);
    const char* (*panel_title)   (NNPluginHandle);
    const char* (*panel_icon_url)(NNPluginHandle);  /* optional; may return NULL */
} NNUIPanelVTable;


/* ── Runner client vtable ─────────────────────────────────────────────────── */
typedef struct NNRunnerClientVTable {
    int32_t (*connect) (NNPluginHandle, const char* url);   /* 0=ok, <0=error */
    int32_t (*load_model)(NNPluginHandle, const char* name, int32_t version);
    void    (*infer)   (NNPluginHandle,
                        const char* model_name,
                        const NNTensorView* input,
                        NNMutableTensorView* output);
    int32_t (*health)  (NNPluginHandle);   /* 0=ok, 1=degraded, <0=unavailable */
    void    (*disconnect)(NNPluginHandle);
    const char* (*doc_ref)(NNPluginHandle);
} NNRunnerClientVTable;


/* ── Context source vtable ────────────────────────────────────────────────── */
typedef struct NNContextSourceVTable {
    int32_t (*connect)  (NNPluginHandle, const char* config_json);
    /** Retrieve top_k chunks relevant to query_embedding.
     *  chunks_json_out: JSON array of retrieved text chunks (caller calls free_result). */
    void    (*retrieve) (NNPluginHandle,
                         const NNTensorView* query_embedding,
                         int32_t top_k,
                         char** chunks_json_out);
    void    (*free_result)(NNPluginHandle, void* ptr);
    void    (*disconnect)(NNPluginHandle);
    const char* (*doc_ref)(NNPluginHandle);
} NNContextSourceVTable;


#ifdef __cplusplus
}
#endif
#endif /* NNSTUDIO_PLUGIN_H */
