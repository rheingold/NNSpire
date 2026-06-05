# NNSpire — Pipeline Whitepaper

**Version**: 0.1 (Phase 0 — design)  
**Date**: 2026-03-31  
**Status**: DRAFT — pre-implementation baseline

> Standards references: KB A03 (RAG), A05 (LLM APIs), A06 (Tokenization), A01 (Embeddings),
> A02 (Vector DBs), 03 (OpenAPI/MCP), 08 (gRPC)

---

## 1. Overview

The NNSpire pipeline describes the full data flow from raw human input through all processing stages to final output. Each stage is an independently configurable component that can be swapped, extended by plugins, and inspected in the Studio UI. The same pipeline description is executable both from the UI and from exported Python/C++ code.

```
Human Input
    │
    ▼
┌──────────────────────────────────┐
│  Input Adapters                  │  text / voice / image / audio /
│  (modality detection + norm.)    │  structured data / MCP message
└─────────────────┬────────────────┘
                  │  raw tensor(s)
                  ▼
┌──────────────────────────────────┐
│  Tokenization Chain              │  BPE / WordPiece / SentencePiece /
│  (plugin-based, composable)      │  image patch / audio frame / custom
└─────────────────┬────────────────┘
                  │  token-ID tensors + attention masks
                  ▼
┌──────────────────────────────────┐
│  Context Stage  (optional)       │  vector DB lookup → retrieved chunks
│  (RAG: retrieve + rerank)        │  prepended to token sequence
└─────────────────┬────────────────┘
                  │  enriched input tensors
                  ▼
┌──────────────────────────────────┐
│  Execution Stage                 │  one or more ComputeGraphs chained;
│  (NN graph chain)                │  output of graph N → input of graph N+1
└─────────────────┬────────────────┘
                  │  output tensors
                  ▼
┌──────────────────────────────────┐
│  Output Adapters                 │  text / image / audio / JSON /
│  (modality-specific decode)      │  structured data
└─────────────────┬────────────────┘
                  │
                  ▼
              Final Output
```

---

## 2. Input adapters

### 2.1 Text adapter

Accepts UTF-8 string. Passes directly to the tokenization chain.

```cpp
// C++
class TextInputAdapter : public IInputAdapter {
public:
    Tensor<int32_t> adapt(const std::string& text) override;
    // returns raw character or byte sequence tensor before tokenization
};
```

```python
# Python
class TextInputAdapter(InputAdapter):
    def adapt(self, text: str) -> Tensor: ...
```

### 2.2 Voice / Audio adapter

Accepts WAV, FLAC, MP3 (via system codec or libsndfile). Produces:
- **Waveform tensor**: shape `[samples]`, float32, sample rate normalized to model's expected rate
- **Spectrogram tensor** (optional preprocessing): shape `[freq_bins, time_frames]`

Audio tokenization (converting to discrete tokens for audio models) is handled by a `TokenizerPlugin` of type `AUDIO_TOKENIZER`. See KB A06 §Audio tokenization.

### 2.3 Image adapter

Accepts any Qt-loadable image format + PNG/JPEG/TIFF/WebP. Produces tensor `[C, H, W]` float32, normalized to `[0, 1]` or `[-1, 1]` depending on downstream model declaration in the `.nns` project file.

Image tokenization (patch splitting for Vision Transformers) is handled by an `IMAGE_TOKENIZER` plugin. See KB A06 §Image tokenization.

### 2.4 Structured data adapter

Accepts CSV (header row inferred as feature names) and Apache Arrow IPC format. Produces tensor `[batch, features]`. Categorical columns are label-encoded unless a plugin overrides this.

### 2.5 MCP message adapter

Accepts a JSON message conforming to the Model Context Protocol schema (KB doc 03 §MCP). Deserialises to the appropriate tensor based on the `content_type` field of the MCP message. Enables NNSpire pipelines to be triggered as MCP tool endpoints.

---

## 3. Tokenization chain

### 3.1 Architecture

```
TokenizationChain
    ├── TokenizerPlugin[0]   (e.g. text normalizer / BPE)
    ├── TokenizerPlugin[1]   (e.g. image patcher)
    └── TokenizerPlugin[2]   (e.g. audio framer)
```

Each plugin in the chain receives the output of the previous. The chain produces a **unified token sequence** combining all modalities, using special separator tokens declared in the model's vocabulary. This is how multimodal models (text + image, text + audio) are assembled.

### 3.2 TokenizationChain outputs

| Tensor | Shape | Dtype | Description |
|---|---|---|---|
| `input_ids` | `[batch, seq_len]` | int32 | Token IDs |
| `attention_mask` | `[batch, seq_len]` | int32 | 1 = real token, 0 = padding |
| `position_ids` | `[batch, seq_len]` | int32 | Absolute position (optional) |
| `token_type_ids` | `[batch, seq_len]` | int32 | Segment IDs for multi-segment models |

### 3.3 Built-in tokenizer plugins

| Plugin | Algorithm | Reference |
|---|---|---|
| `BpeTokenizer` | Byte-pair encoding | KB A06 §BPE |
| `WordPieceTokenizer` | WordPiece | KB A06 §WordPiece |
| `SentencePieceTokenizer` | Unigram / BPE | KB A06 §SentencePiece |
| `ImagePatchTokenizer` | 16×16 / 32×32 patch split (ViT style) | KB A06 §Image |
| `AudioFrameTokenizer` | Mel spectrogram + VQ-VAE discrete codes | KB A06 §Audio |

### 3.4 Tokenization cache

Repeated encoding of the same vocabulary lookups is cached with an LRU cache keyed on input hash. Cache is per-session; persisted optionally across sessions via `<app_data>/tokenize_cache/`.

---

## 4. Context stage (RAG)

This stage is **optional** and bypassed if no context source is configured.

### 4.1 RAG pipeline

```
query tensor
    │
    ▼  (query embedding via an Embedding layer or separate model)
query embedding vector
    │
    ▼
VectorDB.search(query, top_k)
    │
    ▼
Retrieved chunks (text/structured data)
    │
    ▼  (re-ranker plugin, optional)
Reranked chunks
    │
    ▼  (tokenize chunks, prepend to input_ids sequence)
enriched input_ids
```

See KB A03 §RAG pipeline for detailed chunking strategies, reranking, and agentic RAG patterns.

### 4.2 Supported vector backends (connect-on-demand, no bundled servers)

| Backend | Protocol | KB reference |
|---|---|---|
| FAISS (embedded) | In-process C++ / Python | KB A02 §FAISS |
| Qdrant | HTTP REST | KB A02 §Qdrant |
| Chroma | HTTP REST | KB A02 §Chroma |
| Weaviate | HTTP/gRPC | KB A02 §Weaviate |
| Milvus | gRPC | KB A02 §Milvus |
| Pinecone | HTTP REST | KB A02 §Pinecone |

FAISS is the only database that can run embedded (no separate process); all others require a running server that the Studio connects to.

### 4.3 Reranker interface

A `RerankerPlugin` slot in the context stage. A cross-encoder reranker (see KB A03 §Reranking) takes `(query, [candidates])` and returns a scored, sorted list. The default (no reranker) returns top-K results by raw cosine similarity.

---

## 5. Execution stage

### 5.1 Single graph execution

```cpp
Tensor<float> output = graph.forward(enriched_input);
```

The `ComputeGraph` runs through all layers in topological order, dispatching each operation to the active `IBackend`.

### 5.2 Chained graph execution

Multiple `ComputeGraph` instances can be connected:

```cpp
PipelineChain chain;
chain.add(embedder_graph);   // text → embedding
chain.add(encoder_graph);    // embedding → latent
chain.add(decoder_graph);    // latent → output tokens
auto output = chain.forward(input_ids);
```

The `.nns` project file records the wiring. The Pipeline Studio Panel renders this as a coarse node graph.

### 5.3 Streaming execution

For autoregressive generation (token-by-token LLM output), the execution stage supports a streaming callback:

```cpp
graph.forwardStreaming(input, [](const Tensor<float>& token_logits) {
    // called once per generated token
    // UI controller queues this to main thread via signal
});
```

```python
for token_logits in graph.forward_streaming(input_ids):
    decoded_token = tokenizer.decode(token_logits.argmax())
    print(decoded_token, end='', flush=True)
```

---

## 6. Output adapters

### 6.1 Text decode

Calls the inverse of the tokenization chain: `TokenizerPlugin.decode(token_ids)`. For autoregressive generation, decoding is applied per-token in streaming mode.

Sampling strategies (greedy, top-K, top-P nucleus, temperature) are applied to logits in the output adapter before calling `argmax`/`sample`. These strategies are configurable in the Pipeline Panel.

### 6.2 Image synthesis (stub — Phase 4+)

Accepts tensor `[C, H, W]` and converts to a Qt-displayable `QImage`. Diffusion model latent → pixel decode is a `POST_PROCESS` plugin slot.

### 6.3 Audio synthesis (stub — Phase 4+)

Accepts waveform tensor `[samples]` and writes to WAV/FLAC via libsndfile. Vocoder models (Griffin-Lim, HiFi-GAN) are `POST_PROCESS` plugins.

### 6.4 Structured JSON output

Enforced-schema JSON output: the adapter verifies output token sequence against a JSON schema (for tool-call / structured generation scenarios, following KB A05 §Function calling patterns).

---

## 7. Code export

The entire pipeline configuration is exportable as **both** Python and C++ source code from the Studio UI.

### Python export (example skeleton)

```python
import NNSpire as nn

# Build pipeline
input_adapter = nn.TextInputAdapter()
tokenizer = nn.BpeTokenizer.load("vocab.json")
context = nn.FaissContextStage("index.faiss", top_k=5)
graph = nn.ComputeGraph.load("model.nns")
output_adapter = nn.TextDecodeAdapter(tokenizer, strategy="top_p", p=0.9)

# Run
raw = "What is backpropagation?"
tokens = tokenizer.encode(input_adapter.adapt(raw))
enriched = context.retrieve(tokens)
logits = graph.forward(enriched)
print(output_adapter.decode(logits))
```

### C++ export (equivalent)

```cpp
#include <NNSpire/pipeline/pipeline.h>
using namespace NNSpire;

int main() {
    auto input   = TextInputAdapter{};
    auto tok     = BpeTokenizer::load("vocab.json");
    auto context = FaissContextStage{"index.faiss", /*top_k=*/5};
    auto graph   = ComputeGraph::load("model.nns");
    auto output  = TextDecodeAdapter{tok, SamplingStrategy::TopP{0.9f}};

    std::string query = "What is backpropagation?";
    auto tokens   = tok.encode(input.adapt(query));
    auto enriched = context.retrieve(tokens);
    auto logits   = graph.forward(enriched);
    std::cout << output.decode(logits) << std::endl;
}
```

The exported C++ code uses `NNSpire-core` as a library and includes a `CMakeLists.txt`.

---

## 8. MCP / A2A integration

NNSpire pipelines can be exposed as **MCP tool endpoints** (KB doc 03 §MCP):

- The Studio generates an OpenAPI 3.1 descriptor for a configured pipeline
- A lightweight embedded HTTP server (single-pipeline mode) serves the endpoint
- MCP messages with `content_type: text | image | audio | structured` are routed to the appropriate input adapter
- Responses are streamed back in MCP-compatible SSE format

This allows NNSpire pipelines to participate in multi-agent systems without any additional server infrastructure.
