# NNSpire — Plugin SDK Whitepaper

**Version**: 0.1 (Phase 0 — design)  
**Date**: 2026-03-31  
**Status**: DRAFT — pre-implementation baseline

---

## 1. Design principles

1. **Pure C ABI at the boundary** — the public plugin interface (`NNSpire_plugin.h`) uses only C linkage (`extern "C"`), plain structs, and function pointers. No C++ in the ABI. This means plugins can be written in any language that can produce a shared library with C-exported symbols: C, C++, Rust, Zig, D — and via ctypes/cffi bridges, Python.

2. **Dual-language first class** — every plugin *type* has a C++ scaffold template AND a Python scaffold template. Both produce functionally identical plugins. Python plugins are loaded via the pybind11 bridge.

3. **Trust-gated loading** — no plugin code executes before `TrustVerifier` has validated the plugin's signature and manifest. See `TRUST-ARCHITECTURE.md`.

4. **Declarative capabilities** — plugins declare all their capabilities in `plugin.manifest.json` before loading. The Studio can display available plugins, their types, and their parameters without loading any code.

5. **UI extensibility** — plugins may ship a QML component that is dynamically loaded into the Studio's panel system. The QML component communicates with the plugin via the controller interface; no direct engine access from QML plugin code.

---

## 2. Plugin types

| Type constant | Description |
|---|---|
| `NN_PLUGIN_LAYER` | Custom layer type with forward() + backward() |
| `NN_PLUGIN_ACTIVATION` | Custom activation function + derivative |
| `NN_PLUGIN_OPTIMIZER` | Custom optimizer with parameter update step |
| `NN_PLUGIN_TOKENIZER` | Tokenizer: encode() + decode() + vocab management |
| `NN_PLUGIN_BACKEND` | Compute backend (custom hardware, FPGA, etc.) |
| `NN_PLUGIN_INPUT_ADAPTER` | New input modality adapter |
| `NN_PLUGIN_OUTPUT_ADAPTER` | New output modality adapter |
| `NN_PLUGIN_CONTEXT_SOURCE` | New vector DB or knowledge source connector |
| `NN_PLUGIN_RUNNER_CLIENT` | New inference runner connector |
| `NN_PLUGIN_UI_PANEL` | QML component enriching the Studio UI |
| `NN_PLUGIN_TRUST_UPDATE` | Trust Update Package (special — see TRUST-ARCHITECTURE.md) |

---

## 3. C ABI header (`NNSpire_plugin.h`)

```c
/* NNSpire_plugin.h  —  LGPL v3  —  PUBLIC API, C linkage only */
#ifndef NNSpire_PLUGIN_H
#define NNSpire_PLUGIN_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NNSpire_PLUGIN_API_VERSION 1

typedef enum {
    NN_PLUGIN_LAYER         = 1,
    NN_PLUGIN_ACTIVATION    = 2,
    NN_PLUGIN_OPTIMIZER     = 3,
    NN_PLUGIN_TOKENIZER     = 4,
    NN_PLUGIN_BACKEND       = 5,
    NN_PLUGIN_INPUT_ADAPTER = 6,
    NN_PLUGIN_OUTPUT_ADAPTER= 7,
    NN_PLUGIN_CONTEXT_SOURCE= 8,
    NN_PLUGIN_RUNNER_CLIENT = 9,
    NN_PLUGIN_UI_PANEL      = 10,
    NN_PLUGIN_TRUST_UPDATE  = 99
} NNPluginType;

typedef struct {
    /* Raw float tensors passed as (data_ptr, shape_ptr, ndim) triples.
       Backend dispatch is handled by the engine; plugins never call cuBLAS etc. directly. */
    const float* data;
    const int64_t* shape;
    int32_t ndim;
} NNTensorView;

/* Returned by create(); freed by destroy() */
typedef void* NNPluginHandle;

typedef struct {
    uint32_t        api_version;    /* Must equal NNSpire_PLUGIN_API_VERSION */
    NNPluginType    type;
    const char*     id;             /* Reverse-domain: com.example.my-layer */
    const char*     name;
    const char*     version;        /* semver string */
    const char*     author;
    const char*     license;
    NNPluginHandle  (*create)  (void);
    void            (*destroy) (NNPluginHandle);
    /* Type-specific vtable pointer follows; cast by PluginLoader per type */
    void*           vtable;
} NNPluginDescriptor;

/* Every plugin shared library must export this symbol */
const NNPluginDescriptor* NNSpire_plugin_descriptor(void);

/* ── Layer vtable ──────────────────────────────────────────────────────────── */
typedef struct {
    void (*forward) (NNPluginHandle, const NNTensorView* in,  NNTensorView* out);
    void (*backward)(NNPluginHandle, const NNTensorView* grad_out, NNTensorView* grad_in);
    void (*parameters)(NNPluginHandle, NNTensorView** params, int32_t* count);
    const char* (*doc_ref)(NNPluginHandle);   /* KB path#anchor */
} NNLayerVTable;

/* ── Tokenizer vtable ─────────────────────────────────────────────────────── */
typedef struct {
    void     (*encode)(NNPluginHandle, const char* text, int32_t** ids, int32_t* count);
    char*    (*decode)(NNPluginHandle, const int32_t* ids, int32_t count);
    void     (*free_result)(NNPluginHandle, void* ptr);
    uint32_t (*vocab_size)(NNPluginHandle);
    const char* (*token_to_str)(NNPluginHandle, int32_t id);
} NNTokenizerVTable;

/* ── UI Panel vtable ──────────────────────────────────────────────────────── */
typedef struct {
    const char* (*qml_source_url)(NNPluginHandle);  /* qrc:// or file:// URL */
    const char* (*panel_title)   (NNPluginHandle);
} NNUIPanelVTable;

#ifdef __cplusplus
}
#endif
#endif /* NNSpire_PLUGIN_H */
```

Additional vtable structs for `NN_PLUGIN_ACTIVATION`, `NN_PLUGIN_OPTIMIZER`, `NN_PLUGIN_BACKEND`, `NN_PLUGIN_INPUT_ADAPTER`, `NN_PLUGIN_OUTPUT_ADAPTER`, `NN_PLUGIN_CONTEXT_SOURCE`, and `NN_PLUGIN_RUNNER_CLIENT` follow the same pattern. Full definitions in `NNSpire/plugin-api/NNSpire_plugin.h`.

---

## 4. Plugin manifest (`plugin.manifest.json`)

Every plugin package (C++ or Python) must include a `plugin.manifest.json`:

```json
{
  "manifestVersion": 1,
  "id": "com.example.my-custom-layer",
  "name": "My Custom Layer",
  "version": "1.0.0",
  "type": "LAYER",
  "author": "Example Corp",
  "authorUrl": "https://example.com",
  "license": "MIT",
  "licenseUrl": "https://opensource.org/licenses/MIT",
  "minStudioVersion": "1.0.0",
  "capabilities": ["LAYER_FORWARD", "LAYER_BACKWARD"],
  "kbRef": "ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#custom-layers",
  "implementations": {
    "cpp": {
      "library": "my-custom-layer.dll",
      "sha256": "<hex>"
    },
    "python": {
      "module": "my_custom_layer",
      "wheel": "my_custom_layer-1.0.0-py3-none-any.whl",
      "sha256": "<hex>"
    }
  },
  "signature": "my-custom-layer.p7s"
}
```

The detached PKCS#7 signature (`signature` field) covers the UTF-8 canonical JSON of this manifest plus all binary artefacts listed in `implementations`. Generated by `NNSpire-sign sign`.

---

## 5. Plugin lifecycle

```
Discovery
  └─ Scan plugin directories (system, user, project-local)
  └─ Parse plugin.manifest.json (no code loaded yet)

Trust verification
  └─ TrustVerifier.verify(manifest, signature)
  └─ If UNVERIFIED → user prompted (modal, every session)
  └─ If REVOKED → hard block

Loading
  └─ C++ path: QPluginLoader.load() → cast vtable → register with PluginRegistry
  └─ Python path: embedded Python importer → pybind11 wrapper → same PluginRegistry

Active
  └─ Accessible to engine and UI
  └─ UI panel (if NN_PLUGIN_UI_PANEL) loaded into dock system

Unload
  └─ Plugin.destroy() called
  └─ QPluginLoader.unload() / Python module reference dropped
  └─ Hot-reload: load new version without restarting Studio (plugin must support it)
```

Plugin directories scanned (in order, higher index = higher specificity):
1. `<install>/plugins/builtin/` — signed built-in plugins shipped with Studio
2. `<OS system plugin dir>/NNSpire/` — system-wide installed plugins
3. `<user home>/.NNSpire/plugins/` — user-installed plugins
4. `<project dir>/plugins/` — project-local plugins (useful for enterprise deployments)

---

## 6. Writing a plugin — C++ scaffold

The scaffold is generated by `NNSpire-sign keygen --type layer --id com.example.my-layer --out my-layer/`.

### Directory structure generated

```
my-layer/
├── CMakeLists.txt
├── plugin.manifest.json  (template, sha256 left blank until signing)
├── include/
│   └── my_layer.h
├── src/
│   └── my_layer.cpp
└── README.md
```

### `my_layer.h`

```cpp
#pragma once
#include <NNSpire/plugin-api/NNSpire_plugin.h>

class MyLayer {
public:
    void forward (const NNTensorView* in, NNTensorView* out);
    void backward(const NNTensorView* grad_out, NNTensorView* grad_in);
    // ...
};
```

### `my_layer.cpp` (key symbols)

```cpp
#include "my_layer.h"

// Per-instance storage
static NNPluginHandle create()  { return new MyLayer{}; }
static void destroy(NNPluginHandle h) { delete static_cast<MyLayer*>(h); }

static NNLayerVTable VTABLE = {
    /* forward   */ [](NNPluginHandle h, const NNTensorView* in, NNTensorView* out) {
        static_cast<MyLayer*>(h)->forward(in, out);
    },
    /* backward  */ [](NNPluginHandle h, const NNTensorView* gout, NNTensorView* gin) {
        static_cast<MyLayer*>(h)->backward(gout, gin);
    },
    /* parameters*/ nullptr,  // implement if layer has trainable weights
    /* doc_ref   */ [](NNPluginHandle) -> const char* {
        return "ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md";
    }
};

static const NNPluginDescriptor DESCRIPTOR = {
    NNSpire_PLUGIN_API_VERSION, NN_PLUGIN_LAYER,
    "com.example.my-layer", "My Layer", "1.0.0", "Example Corp", "MIT",
    create, destroy, &VTABLE
};

extern "C" const NNPluginDescriptor* NNSpire_plugin_descriptor() {
    return &DESCRIPTOR;
}
```

---

## 7. Writing a plugin — Python scaffold

Generated by `NNSpire-sign keygen --type layer --id com.example.my-layer --lang python --out my-layer-py/`.

### Directory structure

```
my-layer-py/
├── pyproject.toml
├── plugin.manifest.json
├── my_layer/
│   ├── __init__.py
│   └── layer.py
└── README.md
```

### `layer.py`

```python
# my_layer/layer.py
from NNSpire import Tensor, LayerPlugin, plugin_descriptor

class MyLayer(LayerPlugin):
    """Custom layer.
    @kb: ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
    """
    plugin_id      = "com.example.my-layer"
    plugin_version = "1.0.0"
    plugin_type    = "LAYER"

    def forward(self, x: Tensor) -> Tensor:
        # implement forward pass
        raise NotImplementedError

    def backward(self, grad_output: Tensor) -> Tensor:
        # implement backward pass
        raise NotImplementedError

    def parameters(self) -> list[Tensor]:
        return []
```

### `__init__.py`

```python
from .layer import MyLayer
__all__ = ["MyLayer"]
```

The `LayerPlugin` base class in the pybind11 bridge registers `MyLayer` with the C++ `PluginRegistry` automatically on import.

---

## 8. `NNSpire-sign` CLI

```
NNSpire-sign <subcommand> [options]

Subcommands:
  keygen        Generate plugin key pair + CSR
                --type <layer|tokenizer|...>
                --id <reverse-domain-id>
                --lang <cpp|python|both>
                --out <directory>

  sign          Sign a built plugin
                --manifest plugin.manifest.json
                --cert plugin_signing.crt
                --key  plugin_signing.key
                --out  plugin.manifest.json  (updates sha256 + writes .p7s)

  verify        Verify plugin signature offline
                --manifest plugin.manifest.json
                --trust-dir <path to truststore roots/>

  submit        Submit CSR to NNSpire Plugin Registry
                --csr plugin.csr
                --registry https://registry.NNSpire.dev
                --type community|commercial

  create-tup    Build and sign a Trust Update Package
                --add-cert   new_root.crt
                --revoke-sn  <serial-number>
                --signer-cert intermediate.crt
                --signer-key  intermediate.key
                --out trust-update.NNSpire-trust

  issue-enterprise-ca   (Admin / project owner only)
                --csr enterprise.csr
                --org "Company Name"
                --validity-days 1095
                --out enterprise_intermediate.crt
```

---

## 9. Dual-language compliance verification

Before a plugin is accepted by the Registry (community or commercial), the submission tool verifies:

- `implementations.cpp` and `implementations.python` both present in manifest
- Both sha256 hashes valid
- Both build artefacts loadable in the Studio sandbox environment
- Behavioural equivalence test: a set of reference input tensors produces tensors within float32 epsilon between C++ and Python implementations (automated by `NNSpire-sign verify --behavioral-check`)
