/* ============================================================================
 * NNSpire_plugin.h  —  NNSpire Plugin ABI  —  LGPL v3
 *
 * PUBLIC API — C linkage only (no C++ in this header).
 *
 * Every NNSpire plugin shared library / Python module must expose the symbol
 *   const NNPluginDescriptor* NNSpire_plugin_descriptor(void);
 * which returns a pointer to a statically-allocated NNPluginDescriptor.
 *
 * VERSIONING
 * ──────────
 *   api_version in NNPluginDescriptor must equal NNSpire_PLUGIN_API_VERSION.
 *   A mismatch causes PluginLoader to hard-reject the plugin before any other
 *   code in the shared library executes.
 *
 * THREAD SAFETY
 * ─────────────
 *   All vtable entry-points are called from the Engine's worker thread.
 *   Plugins may spawn internal threads but must synchronise externally.
 *
 * MEMORY OWNERSHIP
 * ────────────────
 *   Unless a vtable documents otherwise:
 *     • Data pointed to by const char* return values is owned by the plugin.
 *     • Callers must call free_result() (if present) to release heap allocations
 *       that cross the ABI boundary.
 *     • NNTensorView is a non-owning view; the engine manages tensor lifetimes.
 *
 * @kb: PLUGIN-SDK.md §3
 * ============================================================================ */

#ifndef NNSpire_PLUGIN_H
#define NNSpire_PLUGIN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── ABI version ─────────────────────────────────────────────────────────── */
#define NNSpire_PLUGIN_API_VERSION  1

/* ─── Plugin types ────────────────────────────────────────────────────────── */
typedef enum {
    NN_PLUGIN_LAYER          =  1,  /**< Custom layer: forward() + backward()        */
    NN_PLUGIN_ACTIVATION     =  2,  /**< Custom activation function + derivative      */
    NN_PLUGIN_OPTIMIZER      =  3,  /**< Custom parameter-update optimizer            */
    NN_PLUGIN_TOKENIZER      =  4,  /**< Tokenizer: encode/decode + vocab management  */
    NN_PLUGIN_BACKEND        =  5,  /**< Compute backend (hardware, FPGA, etc.)       */
    NN_PLUGIN_INPUT_ADAPTER  =  6,  /**< New input modality adapter                  */
    NN_PLUGIN_OUTPUT_ADAPTER =  7,  /**< New output modality adapter                 */
    NN_PLUGIN_CONTEXT_SOURCE =  8,  /**< Vector DB or knowledge source connector      */
    NN_PLUGIN_RUNNER_CLIENT  =  9,  /**< Inference-runner connector                   */
    NN_PLUGIN_UI_PANEL       = 10,  /**< QML component enriching the Studio UI        */
    NN_PLUGIN_TRUST_UPDATE   = 99   /**< Trust Update Package (engine-internal only)  */
} NNPluginType;

/* ─── Tensor view (non-owning) ────────────────────────────────────────────── */
/**
 * A non-owning, read-only view of a dense float32 tensor.
 * The engine always passes valid (data, shape, ndim) triples.
 * Plugins must NOT free or retain these pointers beyond the vtable call.
 */
typedef struct {
    const float*   data;   /**< Row-major flat buffer, float32               */
    const int64_t* shape;  /**< Dimension sizes, length = ndim               */
    int32_t        ndim;   /**< Number of dimensions (0 = scalar)            */
} NNTensorView;

/* ─── Plugin handle (opaque per-instance state) ───────────────────────────── */
typedef void* NNPluginHandle;

/* ─── Descriptor ──────────────────────────────────────────────────────────── */
/**
 * Statically returned by NNSpire_plugin_descriptor().
 * All const char* fields must remain alive for the lifetime of the process.
 */
typedef struct {
    uint32_t       api_version; /**< Must equal NNSpire_PLUGIN_API_VERSION  */
    NNPluginType   type;
    const char*    id;          /**< Reverse-domain: com.example.my-layer    */
    const char*    name;        /**< Human-readable display name             */
    const char*    version;     /**< semver string e.g. "1.0.0"              */
    const char*    author;
    const char*    license;     /**< SPDX identifier e.g. "MIT" or "LGPL-3.0" */
    NNPluginHandle (*create) (void);           /**< Allocate instance state  */
    void           (*destroy)(NNPluginHandle); /**< Free instance state      */
    /**
     * Type-specific vtable pointer.
     * PluginLoader casts this to the appropriate NNXxxVTable* after
     * validating `type`.  Must not be NULL for loadable plugin types.
     */
    void*          vtable;
} NNPluginDescriptor;

/* ─── Entry point symbol (every plugin must export this) ─────────────────── */
/**
 * NNSpire_plugin_descriptor — mandatory export
 *
 * Returns a pointer to a process-lifetime static NNPluginDescriptor.
 * The engine calls this immediately after dlopen()/LoadLibrary(); no other
 * plugin symbols are referenced before this returns successfully.
 */
const NNPluginDescriptor* NNSpire_plugin_descriptor(void);

/* ========================================================================== */
/*  Type-specific vtable structs                                               */
/* ========================================================================== */

/* ─── NN_PLUGIN_LAYER ─────────────────────────────────────────────────────── */
/**
 * A trainable layer with forward and backward passes.
 *
 * forward()    — compute output tensor from input tensor.
 * backward()   — given upstream gradient (grad_out), produce gradient wrt input (grad_in)
 *                and accumulate gradients into internal parameter gradients.
 * parameters() — return views of all trainable weight tensors (for the optimizer).
 * doc_ref()    — return KB path#anchor documenting this layer type (may be NULL).
 */
typedef struct {
    void        (*forward)   (NNPluginHandle, const NNTensorView* in,      NNTensorView* out);
    void        (*backward)  (NNPluginHandle, const NNTensorView* grad_out, NNTensorView* grad_in);
    void        (*parameters)(NNPluginHandle, NNTensorView** params, int32_t* count);
    const char* (*doc_ref)   (NNPluginHandle); /**< KB path, may be NULL    */
} NNLayerVTable;

/* ─── NN_PLUGIN_ACTIVATION ────────────────────────────────────────────────── */
/**
 * Element-wise activation function.
 *
 * apply()       — apply activation to every element of in → out.
 * derivative()  — compute activation derivative at every element of in → out.
 *                 Used by the engine's autograd for backpropagation.
 */
typedef struct {
    void (*apply)     (NNPluginHandle, const NNTensorView* in, NNTensorView* out);
    void (*derivative)(NNPluginHandle, const NNTensorView* in, NNTensorView* out);
} NNActivationVTable;

/* ─── NN_PLUGIN_OPTIMIZER ─────────────────────────────────────────────────── */
/**
 * Parameter update rule.
 *
 * step()     — update params[i] in-place using grads[i] (count entries).
 * zero_grad() — zero all gradient buffers in grads (count entries).
 * set_lr()   — runtime learning-rate adjustment (called by lr-scheduler).
 */
typedef struct {
    void (*step)     (NNPluginHandle, NNTensorView** params, NNTensorView** grads,
                      int32_t count);
    void (*zero_grad)(NNPluginHandle, NNTensorView** grads, int32_t count);
    void (*set_lr)   (NNPluginHandle, float lr);
    float (*get_lr)  (NNPluginHandle);
} NNOptimizerVTable;

/* ─── NN_PLUGIN_TOKENIZER ─────────────────────────────────────────────────── */
/**
 * Text ↔ token-id tokenizer.
 *
 * encode()       — tokenize `text` into an int32 id array.
 *                  *ids and *count are written; call free_result(h, *ids) when done.
 * decode()       — convert id array back to text; returned string is heap-allocated;
 *                  call free_result(h, returned_ptr) when done.
 * free_result()  — release any pointer returned by encode() or decode().
 * vocab_size()   — total vocabulary cardinality.
 * token_to_str() — reverse-lookup: id → display string (permanent storage, do not free).
 */
typedef struct {
    void        (*encode)      (NNPluginHandle, const char* text,
                                int32_t** ids, int32_t* count);
    char*       (*decode)      (NNPluginHandle, const int32_t* ids, int32_t count);
    void        (*free_result) (NNPluginHandle, void* ptr);
    uint32_t    (*vocab_size)  (NNPluginHandle);
    const char* (*token_to_str)(NNPluginHandle, int32_t id);
} NNTokenizerVTable;

/* ─── NN_PLUGIN_BACKEND ───────────────────────────────────────────────────── */
/**
 * Hardware or software compute backend.
 *
 * device_name()  — human-readable device string (e.g. "NVIDIA A100").
 * is_available() — returns 1 if the hardware is detected and ready; 0 otherwise.
 * matmul()       — C = A × B  (both 2-D, row-major).
 * add()          — element-wise addition; in-place (out may alias a or b).
 * relu()         — element-wise ReLU in-place.
 */
typedef struct {
    const char* (*device_name) (NNPluginHandle);
    int32_t     (*is_available)(NNPluginHandle);
    void        (*matmul)      (NNPluginHandle, const NNTensorView* a,
                                const NNTensorView* b, NNTensorView* out);
    void        (*add)         (NNPluginHandle, const NNTensorView* a,
                                const NNTensorView* b, NNTensorView* out);
    void        (*relu)        (NNPluginHandle, const NNTensorView* in,
                                NNTensorView* out);
} NNBackendVTable;

/* ─── NN_PLUGIN_INPUT_ADAPTER ─────────────────────────────────────────────── */
/**
 * Converts an external data source into a tensor the engine can forward().
 *
 * modality_name() — e.g. "image/png", "audio/wav", "mesh/obj".
 * load()          — read from `path` and produce an NNTensorView in *out.
 *                   Engine calls free_result(h, out->data) when done.
 * free_result()   — release data allocated by load().
 */
typedef struct {
    const char* (*modality_name)(NNPluginHandle);
    void        (*load)         (NNPluginHandle, const char* path, NNTensorView* out);
    void        (*free_result)  (NNPluginHandle, void* ptr);
} NNInputAdapterVTable;

/* ─── NN_PLUGIN_OUTPUT_ADAPTER ────────────────────────────────────────────── */
/**
 * Converts an engine output tensor into an external artefact.
 *
 * modality_name() — e.g. "image/png", "text/plain".
 * save()          — write the tensor to `path` in the plugin's format.
 */
typedef struct {
    const char* (*modality_name)(NNPluginHandle);
    void        (*save)         (NNPluginHandle, const NNTensorView* tensor,
                                 const char* path);
} NNOutputAdapterVTable;

/* ─── NN_PLUGIN_CONTEXT_SOURCE ────────────────────────────────────────────── */
/**
 * Vector database or knowledge-source connector (RAG).
 *
 * source_name()  — display name for the data source.
 * query()        — given a dense embedding vector `query_vec`, write at most
 *                  `max_results` nearest-neighbour results into *results/*count.
 *                  Each NNTensorView in *results is a retrieved embedding.
 *                  Call free_results() when done.
 * free_results() — release the array allocated by query().
 * get_document() — return the original text document for result at `idx`
 *                  (permanent storage within the plugin; do not free).
 */
typedef struct {
    const char* (*source_name) (NNPluginHandle);
    void        (*query)       (NNPluginHandle, const NNTensorView* query_vec,
                                int32_t max_results,
                                NNTensorView** results, int32_t* count);
    void        (*free_results)(NNPluginHandle, NNTensorView** results, int32_t count);
    const char* (*get_document)(NNPluginHandle, int32_t idx);
} NNContextSourceVTable;

/* ─── NN_PLUGIN_RUNNER_CLIENT ─────────────────────────────────────────────── */
/**
 * Inference-runner connector (e.g. connects to NNSpire-runner sidecar,
 * Triton, TensorFlow Serving, etc.).
 *
 * runner_name()  — display name.
 * connect()      — connect to `endpoint` (URL/path); returns 0 on success.
 * disconnect()   — release connection.
 * infer()        — send `in` tensor and receive `out` tensor synchronously.
 */
typedef struct {
    const char* (*runner_name)(NNPluginHandle);
    int32_t     (*connect)    (NNPluginHandle, const char* endpoint);
    void        (*disconnect) (NNPluginHandle);
    void        (*infer)      (NNPluginHandle, const NNTensorView* in,
                               NNTensorView* out);
} NNRunnerClientVTable;

/* ─── NN_PLUGIN_UI_PANEL ──────────────────────────────────────────────────── */
/**
 * QML component enriching the Studio UI.
 *
 * qml_source_url() — qrc:// or file:// URL to the root QML document.
 * panel_title()    — display title for the dockable panel.
 */
typedef struct {
    const char* (*qml_source_url)(NNPluginHandle);
    const char* (*panel_title)   (NNPluginHandle);
} NNUIPanelVTable;

/* ─── NN_PLUGIN_TRUST_UPDATE ──────────────────────────────────────────────── */
/**
 * Trust Update Package — handled entirely by the engine's TrustUpdateHandler.
 * Third-party plugins must NOT set type = NN_PLUGIN_TRUST_UPDATE; the engine
 * will reject such descriptors before the create() factory is called.
 * vtable for TUP is engine-internal and not exposed here.
 */

#ifdef __cplusplus
}
#endif

#endif /* NNSpire_PLUGIN_H */
