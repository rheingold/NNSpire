# NNSpire

A multiplatform (Qt 6) neural-network design, training, deployment, and learning workbench.
Built "inside out" — from the mathematics of a single neuron outward to full LLM pipelines, hardware runners, and eventually quantum computation backends.

---

## Document map

| Document | Purpose |
|---|---|
| [README.md](README.md) | This file — project overview and signpost |
| [ai.md](ai.md) | AI-assistant knowledge base: conventions, standing decisions, key paths, build commands |
| [TODO.md](TODO.md) | Full hierarchical project checklist — the living task tracker |
| [docs/blueprints.md](docs/blueprints.md) | Engine walkthrough: every layer, loss, optimizer, trainer — with maths and numerical traces |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System whitepaper: component model, data flow, threading, backend abstraction |
| [docs/PIPELINE.md](docs/PIPELINE.md) | Full inference/training chain: human inputs → tokenization → context DB → NN graph → outputs |
| [docs/PLUGIN-SDK.md](docs/PLUGIN-SDK.md) | Plugin architecture: C ABI contract, manifest format, lifecycle, extensibility points |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | Runner strategy: export formats, runner connectors, `.nns`/`.nnsr` bundle spec |
| [docs/TRUST-ARCHITECTURE.md](docs/TRUST-ARCHITECTURE.md) | PKI trust chain: certificate hierarchy, Trust Update Packages, enterprise CA delegation |
| [docs/HISTORY.md](docs/HISTORY.md) | Historical origins of layer types — biology vs. engineering, era-by-era |
| [LICENSING.md](LICENSING.md) | License split, plugin charter, community and commercial plugin policy |
| [docs/adr/](docs/adr/README.md) | Architecture Decision Records — one ADR per significant design choice |

---

## Source tree (after Phase 1+)

```
Studio/                         ← project root (this folder)
├── NNSpire/                   ← all source code
│   ├── core/                   ← C++17 engine library (NNSpire-core, LGPL v3)
│   ├── backends/               ← CPU / CUDA / Quantum backend plugins
│   ├── plugin-api/             ← C ABI headers + TrustVerifier + PluginLoader
│   ├── plugins/                ← built-in reference plugins (BPE tokenizer, FAISS, etc.)
│   ├── pipeline/               ← input adapters, tokenization, context, execution, output
│   ├── deployment/             ← export wizards, runner connectors, bundle builder
│   ├── python-bridge/          ← pybind11 bridge exposing core to Python
│   ├── app/                    ← Qt 6 QML application (GPL v3)
│   └── tests/                  ← GoogleTest unit + integration tests
├── docs/                       ← all project documentation (Doxygen config, ADRs, design docs, history, KB)
│   ├── adr/                    ← Architecture Decision Records
│   ├── ai-standards-kb/        ← AI/ML standards reference KB (read-only reference)
│   └── examples/               ← annotated C++ call-chain examples
└── <root .md files>            ← only .md files and .gitignore live at root
```

---

## Quick start (Phase 0 — before any source exists)

```bash
# Clone
git clone <repo-url> NNSpire && cd NNSpire

# Read the plan
cat TODO.md                  # full task checklist
cat docs/ARCHITECTURE.md     # system design
cat docs/PIPELINE.md         # pipeline design
cat docs/blueprints.md       # engine walkthrough

# When source exists (Phase 1+):
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
./build/NNSpire/app/NNSpire
```

---

## Key principles (non-negotiable)

1. **Dual-language everywhere**: every plugin, export, script, and runner client ships both Python and C++ implementations.
2. **Inside out**: engine mathematics first, UI second, pipeline third, deployment fourth.
3. **Portable by default**: xcopy-deployable on Windows, `.app` on macOS, AppImage on Linux. No mandatory installer.
4. **Free and open core**: learning features, visualizations, full pipeline, and all standard runners are permanently `FREE`. See [LICENSING.md](LICENSING.md).
5. **PKI trust chain**: plugin signing is mandatory for silent loading. See [TRUST-ARCHITECTURE.md](TRUST-ARCHITECTURE.md).
6. **Standards-first**: use existing standards (ONNX, gRPC V2, OpenAPI, Model Cards) wherever they reach; use the `.nns`/`.nnsr` custom format only for the gaps. See [DEPLOYMENT.md](DEPLOYMENT.md).
