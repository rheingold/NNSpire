# NNSpire — Blueprints Addendum: Model Weight File Formats

> **Companion to:** [blueprints.md](blueprints.md)  
> **Reading order:** Read [Chapter 1 (Tensor)](blueprints.md#chapter-1--the-memory-model-tensor) and [§1.x (DType)](blueprints.md#1x--foundation-types-device-dtype-resultt) of blueprints.md first. This addendum explains how the logical Tensor + DType + layer-parameter model maps onto the on-disk representations used by the industry.

---

## Motivation — Why File Formats Matter to NNSpire

Every trained model is a collection of `Tensor` instances that represent learned parameters — the `W_` and `b_` fields in every `Dense`, `Conv2D`, `Embedding`, and `BatchNorm` layer (see [§3.1](blueprints.md#31-lifecycle--the-contract-every-layer-must-honour)).  
Persisting and loading them requires a serialisation contract. That contract is a file format.

The critical engineering choice is: **what information must be stored alongside the raw float bytes to reconstruct a working model?**

The answer maps directly back to NNSpire's core concepts:

| Logical concept in NNSpire | What must be stored on disk |
|---|---|
| `Tensor::shape_` — the navigation map | Dimension sizes for every parameter tensor |
| `Tensor::strides_` — the traversal rule | Usually implicit (row-major assumed) or explicit if non-standard |
| `DType` — element type (`Float32`, `Float16`, `Int8`, …) | dtype tag per tensor — determines how many bytes each element occupies |
| `Device` — where data lives | Usually CPU on disk; target device is chosen at load time |
| Layer name / parameter name | A string key so the loader knows which `W_` or `b_` each tensor belongs to |
| Architecture (layer types, connectivity) | May or may not be stored — formats differ radically here |
| Quantisation metadata | Scale factors and zero-points used when `DType` is `Int8` or `Int4` |
| Tokeniser / model config | Separate JSON or embedded metadata block |

The formats below each make different trade-offs across those dimensions.

---

## Format Gallery

### 1 — SafeTensors (Hugging Face, 2022)

**Design philosophy:** Minimal, safe, zero-copy. Store *only* the tensor data and their metadata. Architecture is the caller's responsibility.

**File layout:**

```
safetensors file
│
├── [8 bytes]  header_size  (little-endian uint64)
│                           → tells reader how many bytes the JSON header occupies
│
├── [header_size bytes]  JSON header
│   │
│   ├── "__metadata__"  : { "format": "pt", "custom_key": "value", … }
│   │                     ↑ optional free-form metadata dict
│   │
│   └── per-tensor entries (one per parameter):
│       "model.layer.weight" : {
│           "dtype"   : "F32"           ← maps to NNSpire DType::Float32
│           "shape"   : [768, 3072]     ← maps to NNSpire Tensor::shape_
│           "data_offsets": [0, 9437184]  ← byte range in the DATA section below
│       }
│       "model.layer.bias" : {
│           "dtype"   : "F32"
│           "shape"   : [3072]
│           "data_offsets": [9437184, 9449472]
│       }
│       …
│
└── [variable bytes]  DATA section
    │   flat concatenation of all tensor buffers
    │   each buffer is little-endian, row-major
    │
    ├── bytes 0 … 9437183      → "model.layer.weight" raw floats
    ├── bytes 9437184 … 9449471 → "model.layer.bias" raw floats
    └── …
```

**Supported `dtype` strings → NNSpire `DType` mapping:**

| SafeTensors dtype | NNSpire DType | Bytes/elem | Typical use |
|---|---|---|---|
| `"F64"` | — (not in Phase 1) | 8 | Double-precision research |
| `"F32"` | `Float32` | 4 | Full-precision training (NNSpire Phase 1 native) |
| `"BF16"` | — (Phase 2+) | 2 | Mixed-precision training (Google TPU standard) |
| `"F16"` | `Float16` | 2 | Inference compression; GPU-friendly |
| `"I32"` | `Int32` | 4 | Token IDs (Embedding layer input indices) |
| `"I16"` | — | 2 | Intermediate quantised |
| `"I8"` | `Int8` | 1 | Post-training quantisation (PTQ) |
| `"U8"` | — | 1 | Rare; pixel data |
| `"BOOL"` | `Bool` | 1 | Attention masks |

**What SafeTensors does NOT store:**

- Layer architecture (which layer type produced each tensor)
- Activation functions (ReLU, GELU, Sigmoid — see [§3.8](blueprints.md#38-activation-functions--iactivation-functors-and-activationbase))
- Optimizer state (`Adam`'s `m_` and `v_` moment tensors)
- Quantisation scale factors (a plain `F32` file is always dequantised)

**Relation to NNSpire Tensor memory model:**

When NNSpire loads a SafeTensors file, each JSON entry maps to exactly one `Tensor` instance:
- `shape` → `Tensor::shape_` and `Tensor::numel_` (product of dims)
- `data_offsets` → byte range read into the `float*` buffer at `Tensor::data_`
- `dtype` → `Tensor::dtype_` (determines `dtypeBytes()` stride calculation — see [§1.x](blueprints.md#1x--foundation-types-device-dtype-resultt))
- `strides_` — always reconstructed as row-major: `strides_[k] = product of all dims after k`

**Security property:** The header is validated before any memory access. There is no pickle/arbitrary-code path — this is the primary reason Hugging Face migrated away from PyTorch's `.pt` format.

---

### 2 — GGUF (llama.cpp / ggml, 2023)

**Design philosophy:** Self-contained. Everything needed to run a model — architecture config, tokeniser, quantised weights — in a single file. Optimised for CPU inference with 4-bit quantised blocks.

**File layout:**

```
GGUF file
│
├── [4 bytes]  magic  "GGUF"  (0x46554747)
├── [4 bytes]  version         (current: 3)
├── [8 bytes]  tensor_count    (uint64)
├── [8 bytes]  metadata_kv_count  (uint64)
│
├── METADATA SECTION  (key-value pairs, variable count)
│   ├── "general.architecture"   = "llama"  | "mistral" | "phi" | "falcon" | …
│   ├── "general.name"           = "Meta-Llama-3-8B"
│   ├── "general.file_type"      = 2   ← GGML_TYPE_Q4_0 (quantisation kind enum)
│   │
│   ├── Model hyperparameters (architecture config — NOT stored in SafeTensors):
│   │   ├── "llama.context_length"         = 8192   ← max sequence length
│   │   ├── "llama.embedding_length"       = 4096   ← model dimension d_model
│   │   ├── "llama.block_count"            = 32     ← number of Transformer blocks
│   │   ├── "llama.feed_forward_length"    = 14336  ← FFN hidden dim
│   │   ├── "llama.attention.head_count"   = 32     ← number of attention heads
│   │   ├── "llama.attention.head_count_kv"= 8      ← GQA key/value heads
│   │   ├── "llama.rope.freq_base"         = 500000 ← RoPE θ base frequency
│   │   └── "llama.attention.layer_norm_rms_epsilon" = 1e-5
│   │
│   ├── Tokeniser (embedded — NOT in SafeTensors):
│   │   ├── "tokenizer.ggml.model"         = "llama"   | "gpt2" | "bert"
│   │   ├── "tokenizer.ggml.tokens"        = ["<unk>","<s>","</s>", …]   ← vocab list
│   │   ├── "tokenizer.ggml.scores"        = [0.0, 0.0, …]  ← BPE merge scores
│   │   ├── "tokenizer.ggml.token_type"    = [2, 3, 3, 1, 1, …]  ← NORMAL/BOS/EOS/…
│   │   └── "tokenizer.ggml.bos_token_id" = 1
│   │
│   └── Chat template (optional):
│       └── "tokenizer.chat_template" = "{% for message in messages %}…"
│
├── TENSOR INFO SECTION  (one entry per tensor)
│   ├── entry 0:
│   │   ├── name_length  : uint64
│   │   ├── name         : "token_embd.weight"
│   │   ├── n_dims       : uint32   (= 2)
│   │   ├── dims         : [32000, 4096]   ← vocab_size × d_model
│   │   ├── type         : GGML_TYPE_Q4_K  ← quantisation type enum
│   │   └── offset       : byte offset into DATA section
│   │
│   ├── entry 1:
│   │   ├── name         : "blk.0.attn_q.weight"
│   │   ├── dims         : [4096, 4096]
│   │   ├── type         : GGML_TYPE_Q4_K
│   │   └── offset       : …
│   │
│   └── … (32 blocks × ~9 tensors each + embeddings + output = ~300 tensors total for Llama-3-8B)
│
├── [PADDING]  align to 32-byte boundary
│
└── DATA SECTION  (raw tensor buffers, quantised blocks)
    ├── "token_embd.weight"  ← Q4_K quantised blocks
    │   Each block: 256 elements packed as 4-bit integers + 2× F16 scale factors
    │   Block layout: [scale_f16][min_f16][32 × uint8 packing 2 nibbles each]
    │
    ├── "blk.0.attn_q.weight"  ← Q4_K
    ├── "blk.0.attn_k.weight"  ← Q4_K
    ├── "blk.0.attn_v.weight"  ← Q4_K
    ├── "blk.0.attn_output.weight"  ← Q4_K
    ├── "blk.0.ffn_gate.weight"     ← Q4_K   (SwiGLU gate — replaces GELU in LLaMA)
    ├── "blk.0.ffn_up.weight"       ← Q4_K
    ├── "blk.0.ffn_down.weight"     ← Q4_K
    ├── "blk.0.attn_norm.weight"    ← F32    (RMSNorm scale γ — kept in full precision)
    ├── "blk.0.ffn_norm.weight"     ← F32    (RMSNorm scale γ)
    └── … (blocks 1–31, then "output_norm.weight" and "output.weight")
```

**GGML quantisation types — the weight type zoo:**

GGUF's most distinctive feature is its rich quantisation type system. Each type packs 4-bit or 5-bit integer weights with per-block floating-point scales, dramatically reducing model size while retaining most accuracy.

| Type | Bits/weight | Block size | Scale precision | Notes |
|---|---|---|---|---|
| `F32` | 32 | 1 | — | Full precision; used for norms, biases |
| `F16` | 16 | 1 | — | Half precision; GPU-native |
| `BF16` | 16 | 1 | — | Brain float; exponent-rich, less mantissa |
| `Q8_0` | 8 | 32 | F32 scale | Symmetric int8; minimal quality loss |
| `Q4_0` | 4 | 32 | F32 scale | Older 4-bit; 4× smaller than F32 |
| `Q4_1` | 4 | 32 | F32 scale + F32 min | Asymmetric 4-bit; slightly better accuracy |
| `Q5_0` | 5 | 32 | F32 scale | 5-bit symmetric |
| `Q5_1` | 5 | 32 | F32 scale + F32 min | 5-bit asymmetric |
| `Q2_K` | ~2.5 | 256 | F16 super-scale | "K-quant" family; super-block with nested scales |
| `Q3_K` | ~3.5 | 256 | F16 super-scale | K-quant; good quality/size trade-off |
| `Q4_K` | ~4.5 | 256 | 2× F16 | Default for most LLMs; recommended |
| `Q5_K` | ~5.5 | 256 | 2× F16 | Near-lossless at 55% the F32 size |
| `Q6_K` | ~6.5 | 256 | F32 | Very close to F16 quality |
| `IQ2_XXS`..`IQ4_XS` | 2–4 | 256 | Learned importance | "i-quant" imatrix-calibrated; state-of-art small |

**Relation to NNSpire's mathematical model:**

The weight matrix `W_` of a `Dense` layer (see [§3.3](blueprints.md#33-what-is-dense--from-one-neuron-to-a-matrix)) is stored as one GGUF tensor entry. When the model is Q4_K quantised:

```
Dense layer W_ — logical shape [outF, inF] — stored as Q4_K blocks:

Each 256-weight super-block on disk:
┌────────────────────────────────────────────────────────────────────────┐
│  scale_d   [F16]  — super-block scale (divides all sub-block scales)   │
│  sub_scales[8×F16]— one scale per 32-element sub-block                 │
│  packed    [128 bytes]— 256 × 4 bits = 128 bytes of quantised values   │
└────────────────────────────────────────────────────────────────────────┘

Dequantisation (to recover the F32 for compute):
  w_f32[i] = (nibble[i] - 8) * sub_scale[i/32] * scale_d
             ↑ signed centred   ↑ per-32 scale    ↑ super-block scale
```

The `Dense::forward()` equation `Y = X @ W^T + b` (see [§3.4](blueprints.md#34-forward-pass-y--x--wt--b)) remains unchanged in mathematics. Quantisation only affects what `W_` physically stores — at inference, weights are dequantised (or kept quantised with a quantised GEMM kernel) before the matmul. The activation functions (ReLU, GELU, SwiGLU — [§3.8](blueprints.md#38-activation-functions--iactivation-functors-and-activationbase)) operate on dequantised activations; intermediate activation tensors are not quantised in standard GGUF inference.

**Activation stored in GGUF:** none — activations (ReLU, GELU, SwiGLU) are architectural metadata in the `general.architecture` key, not tensors. The consumer selects the correct activation by reading `general.architecture`.

---

### 2.1 — GGUF Deep Dive: Transformer Attention Weights and What They Map To

The tensor names in a GGUF Transformer model (`attn_q.weight`, `attn_k.weight`, `attn_v.weight`, …) can look opaque at first. This section explains exactly what each tensor *is* mathematically, how each one maps to a `Dense` layer (or not), and where the non-parametric layer specifications live.

#### Multi-Head Attention — the mathematical foundation

A single Transformer attention block receives an input matrix $X$ of shape $[T, d]$ where $T$ is the sequence length and $d$ is the model dimension (`llama.embedding_length` in GGUF metadata). It produces an output of the same shape.

The block applies four **projection matrices** — all of which are standard `Dense` layers without bias in most modern LLMs:

$$Q = X \, W_Q^T, \quad K = X \, W_K^T, \quad V = X \, W_V^T$$
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
$$\text{Output} = \text{Attention}(\cdot) \, W_O^T$$

Each equation is exactly the Dense forward pass `Y = X @ W^T` from [§3.4](blueprints.md#34-forward-pass-y--x--wt--b) — with bias omitted (a design choice, not a constraint of the layer type).

#### Tensor-name → Dense layer → shape → math

```
GGUF tensor name        Dense layer role     Shape [outF, inF]        Math
─────────────────────────────────────────────────────────────────────────────────────────────
blk.N.attn_q.weight     Query projection     [n_heads*d_head, d_model]   W_Q   Q = X @ W_Q^T
blk.N.attn_k.weight     Key   projection     [n_kv_heads*d_head, d_model] W_K  K = X @ W_K^T
blk.N.attn_v.weight     Value projection     [n_kv_heads*d_head, d_model] W_V  V = X @ W_V^T
blk.N.attn_output.weight Output projection   [d_model, n_heads*d_head]   W_O   O = Attn @ W_O^T
blk.N.ffn_gate.weight   FFN SwiGLU gate      [d_ffn, d_model]            W_gate  G = X @ W_gate^T
blk.N.ffn_up.weight     FFN SwiGLU up-proj   [d_ffn, d_model]            W_up    U = X @ W_up^T
blk.N.ffn_down.weight   FFN down-projection  [d_model, d_ffn]            W_down  Y = (G⊙σ(G)) @ W_down^T
blk.N.attn_norm.weight  RMSNorm scale γ      [d_model]                   scalar per-feature scale
blk.N.ffn_norm.weight   RMSNorm scale γ      [d_model]                   scalar per-feature scale
token_embd.weight       Embedding lookup     [vocab_size, d_model]       row-select (not matmul)
output_norm.weight      Final RMSNorm γ      [d_model]                   scalar per-feature scale
output.weight           LM head (unembedding)[vocab_size, d_model]       W_lm   logits = H @ W_lm^T
```

**Concrete shape example — Llama-3-8B** (`d_model=4096`, `n_heads=32`, `n_kv_heads=8`, `d_head=128`, `d_ffn=14336`):

```
blk.0.attn_q.weight       [4096, 4096]   = [32×128, 4096]   Q-projection
blk.0.attn_k.weight       [1024, 4096]   = [ 8×128, 4096]   K-projection (GQA — fewer KV heads)
blk.0.attn_v.weight       [1024, 4096]   = [ 8×128, 4096]   V-projection (GQA)
blk.0.attn_output.weight  [4096, 4096]   = [4096,  32×128]  O-projection
blk.0.ffn_gate.weight     [14336, 4096]                      SwiGLU gate
blk.0.ffn_up.weight       [14336, 4096]                      SwiGLU up
blk.0.ffn_down.weight     [4096, 14336]                      FFN down
blk.0.attn_norm.weight    [4096]                             RMSNorm γ (not a Dense layer)
blk.0.ffn_norm.weight     [4096]                             RMSNorm γ (not a Dense layer)
```

#### Why Q and K shapes differ from V in GQA (Grouped Query Attention)

Standard Multi-Head Attention uses `n_heads` heads for Q, K, and V equally — all three projections have the same output dimension.  
**GQA** (Ainslie et al. 2023, used in Llama-2/3, Mistral) allows K and V to use fewer heads (`n_kv_heads < n_heads`), with each KV head shared across a group of Q heads:

```
Standard MHA:   n_heads = n_kv_heads = 32
  W_Q  [32×128, d] = [4096, 4096]
  W_K  [32×128, d] = [4096, 4096]
  W_V  [32×128, d] = [4096, 4096]

GQA (Llama-3-8B, n_kv_heads = 8, group_size = 4):
  W_Q  [32×128, d] = [4096, 4096]   ← 32 query heads, unchanged
  W_K  [ 8×128, d] = [1024, 4096]   ← only 8 key heads; each serves 4 query heads
  W_V  [ 8×128, d] = [1024, 4096]   ← only 8 value heads
```

In NNSpire terms, each of `W_Q`, `W_K`, `W_V`, `W_O` is exactly one `Dense` layer's `W_` tensor — with `requires_grad=true` during training, and shape `[outF, inF]` matching the table above. The `Dense::forward()` call for the Q-projection is:

$$Y_{[T,\, n\_heads \cdot d\_head]} = X_{[T,\, d\_model]} \;\mathbin{@}\; W_Q^T_{[d\_model,\, n\_heads \cdot d\_head]}$$

which is byte-for-byte the same formula as any other `Dense` — only the interpretation of the dimensions changes.

---

#### Why Q, K, and V are three separate Dense layers — their distinct mathematical roles

The most common confusion: *if Q, K, and V are all just `Y = X @ W^T`, why are there three of them? Why not one?*

The answer is that the **formula** is the same but the **role** of each output is completely different. Think of it like a library:

| Projection | Metaphor | Mathematical role |
|---|---|---|
| **Q** (Query) | *"What am I looking for?"* | Compared against every key to produce similarity scores |
| **K** (Key) | *"What do I advertise?"* | Matched against every query — determines who attends to whom |
| **V** (Value) | *"What do I actually carry?"* | The content that gets blended according to the attention weights |
| **O** (Output) | *"Consolidate what I found"* | Projects the blended values back to model dimension |

The three matrices `W_Q`, `W_K`, `W_V` project the same input `X` into **three different learned subspaces**. They are initialised differently, trained differently, and after training they encode completely different geometric transformations. The only thing they share is the functional form `Y = X @ W^T`.

**Tiny worked example** — simplified attention with $d_{model}=4$, $T=3$ tokens, $d_{head}=2$, 1 head:

Input $X$ — three tokens, four features each:
```
X = [  0.9  0.2  0.1  0.4  ]   <- token 0  (e.g. "cat")
    [  0.1  0.8  0.3  0.6  ]   <- token 1  (e.g. "sat")
    [  0.5  0.5  0.9  0.1  ]   <- token 2  (e.g. "mat")
    shape [3, 4]   (T=3, d_model=4)
```

Weight matrices — each independently learned, shape [2, 4] (d_head=2, d_model=4):
```
W_Q = [ 1.0  0.0  0.0  1.0 ]   <- "look for features 0 and 3"
      [ 0.0  1.0  1.0  0.0 ]   <- "look for features 1 and 2"

W_K = [ 1.0  0.0  1.0  0.0 ]   <- "advertise features 0 and 2"
      [ 0.0  1.0  0.0  1.0 ]   <- "advertise features 1 and 3"

W_V = [ 0.5  0.5  0.0  0.0 ]   <- "carry mix of features 0+1"
      [ 0.0  0.0  0.5  0.5 ]   <- "carry mix of features 2+3"
```

Step 1 — compute Q, K, V (each is Dense::forward = X @ W^T):
```
Q = X @ W_Q^T                              K = X @ W_K^T
token 0: [0.9*1+0.4*1, 0.2*1+0.1*1]         token 0: [0.9*1+0.1*1, 0.2*1+0.4*1]
       = [1.3,  0.3]                                 = [1.0,  0.6]
token 1: [0.1+0.6, 0.8+0.3] = [0.7, 1.1]    token 1: [0.1+0.3, 0.8+0.6] = [0.4, 1.4]
token 2: [0.5+0.1, 0.5+0.9] = [0.6, 1.4]    token 2: [0.5+0.9, 0.5+0.1] = [1.4, 0.6]
  Q shape [3,2]                                K shape [3,2]

V = X @ W_V^T
token 0: [0.9*0.5+0.2*0.5, 0.1*0.5+0.4*0.5] = [0.55, 0.25]
token 1: [0.1*0.5+0.8*0.5, 0.3*0.5+0.6*0.5] = [0.45, 0.45]
token 2: [0.5*0.5+0.5*0.5, 0.9*0.5+0.1*0.5] = [0.50, 0.50]
  V shape [3,2]
```

Step 2 — attention scores S = Q @ K^T / sqrt(d_head):
```
Raw scores Q @ K^T  (each entry = how much token i "matches" token j):

  S_raw[0,0] = Q[0]·K[0] = 1.3*1.0 + 0.3*0.6 = 1.48   <- token 0 vs token 0 (self)
  S_raw[0,1] = Q[0]·K[1] = 1.3*0.4 + 0.3*1.4 = 0.94   <- token 0 vs token 1
  S_raw[0,2] = Q[0]·K[2] = 1.3*1.4 + 0.3*0.6 = 2.00   <- token 0 vs token 2 (highest!)
  S_raw[1,0] = Q[1]·K[0] = 0.7*1.0 + 1.1*0.6 = 1.36
  S_raw[1,1] = Q[1]·K[1] = 0.7*0.4 + 1.1*1.4 = 1.82   <- token 1 attends to itself most
  S_raw[1,2] = Q[1]·K[2] = 0.7*1.4 + 1.1*0.6 = 1.64
  S_raw[2,0] = Q[2]·K[0] = 0.6*1.0 + 1.4*0.6 = 1.44
  S_raw[2,1] = Q[2]·K[1] = 0.6*0.4 + 1.4*1.4 = 2.20   <- token 2 attends to token 1 most
  S_raw[2,2] = Q[2]·K[2] = 0.6*1.4 + 1.4*0.6 = 1.68

S = S_raw / sqrt(2) = S_raw / 1.414:
  [ 1.047  0.665  1.414 ]
  [ 0.962  1.287  1.160 ]
  [ 1.018  1.556  1.188 ]
```

Step 3 — softmax row by row (each row sums to 1.0):
```
A = softmax(S, dim=-1):
  row 0: exp([1.047, 0.665, 1.414]) = [2.849, 1.945, 4.112]  sum=8.906
         -> [0.320, 0.218, 0.462]   <- token 0 attends most to token 2 ("mat")
  row 1: exp([0.962, 1.287, 1.160]) = [2.617, 3.621, 3.190]  sum=9.428
         -> [0.278, 0.384, 0.338]   <- token 1 attends most to itself ("sat")
  row 2: exp([1.018, 1.556, 1.188]) = [2.767, 4.737, 3.281]  sum=10.785
         -> [0.257, 0.439, 0.304]   <- token 2 attends most to token 1 ("sat")
```

Step 4 — context = A @ V (weighted blend of V vectors):
```
context[0] = 0.320*V[0] + 0.218*V[1] + 0.462*V[2]
           = 0.320*[0.55,0.25] + 0.218*[0.45,0.45] + 0.462*[0.50,0.50]
           = [0.176,0.080] + [0.098,0.098] + [0.231,0.231]
           = [0.505, 0.409]  <- token 0 ("cat") new representation

context[1] = 0.278*[0.55,0.25] + 0.384*[0.45,0.45] + 0.338*[0.50,0.50]
           = [0.153,0.070] + [0.173,0.173] + [0.169,0.169]
           = [0.495, 0.412]  <- token 1 ("sat") new representation

context[2] = 0.257*[0.55,0.25] + 0.439*[0.45,0.45] + 0.304*[0.50,0.50]
           = [0.141,0.064] + [0.198,0.198] + [0.152,0.152]
           = [0.491, 0.414]  <- token 2 ("mat") new representation
```

**The key insight from this example:**

Token 0 ("cat") started as raw features `[0.9, 0.2, 0.1, 0.4]` and ended up as a blended 2-D vector `[0.505, 0.409]` — a weighted mix of all three tokens' V vectors, where the blend weights were determined by how well its Q matched each token's K. It *looked for* features 0 and 3 (via W_Q), *found* that token 2 matched its query best (score 1.414 was highest), and *received* the most content (0.462) from token 2's V vector.

**If you used the same matrix for Q, K, and V:** every token's Q would be identical to its own K (same projection), so `S[i,i]` would always dominate (each token matches itself best), attention weights would approach an identity matrix, and `context ≈ V = X @ W^T` — the context would just be a single linear projection of the input with no cross-token interaction. That is exactly a Dense layer — nothing is gained by the attention mechanism.

Three separate trained matrices are what give attention its power to route information across positions.

#### The attention score computation — where Softmax lives

The inner attention computation between the projections is **not** a Dense layer. It is a sequence of backend operations:

```
Step 1 — Scale dot-product:
  scores [n_heads, T, T] = Q [n_heads, T, d_head]  @  K^T [n_heads, d_head, T]
                           ─────────────────────────────────────────────────────
                           IBackend::matmul (batched, one GEMM per head)

Step 2 — Apply causal mask (lower-triangular):
  scores[i,j] = −∞  for  j > i
               ─────────────────
               IBackend::add with a mask tensor of −∞ values

Step 3 — Softmax over key dimension:
  weights [n_heads, T, T] = softmax(scores / √d_head, dim=-1)
                           ────────────────────────────────────
                           SoftmaxFn::forward() — same functor as standalone Softmax activation
                           (see §3.8, blueprints.md#38)

Step 4 — Weighted sum of values:
  context [n_heads, T, d_head] = weights @ V [n_heads, T, d_head]
                                ─────────────────────────────────────
                                IBackend::matmul (batched)

Step 5 — Concatenate heads, project:
  output [T, d_model] = reshape(context) @ W_O^T
                       ───────────────────────────
                       Dense forward (W_O above)
```

**No tensors are stored for steps 1–4.** They are pure computation, driven by the architecture tag `"llama"` (or `"mistral"`, `"phi"`, etc.) in the GGUF metadata. The softmax here is the same `SoftmaxFn` functor as in NNSpire's activation suite — reused as a computational primitive inside the attention algorithm, not as a standalone layer.

#### FFN layers — SwiGLU is two Denses fused through a gate

Modern LLMs replace the classic `Dense → GELU → Dense` FFN with a **gated variant** (Shazeer 2020). GGUF stores this as three separate weight tensors:

```
SwiGLU FFN forward pass (Llama-3 style):
  gate  = X  @ W_gate^T          ← Dense, shape [T, d_ffn]
  up    = X  @ W_up^T            ← Dense, shape [T, d_ffn]
  fused = gate ⊙ σ(gate)         ← SiLU elementwise gate applied to gate projection
                                    (σ = sigmoid; SiLU(x) = x·σ(x))
          ─────────────────────────
          IBackend::mul (Hadamard — see §D.3, blueprints.md#d3)
          + SigmoidFn::forward()
  ffn_out = fused ⊙ up           ← Hadamard product of gated signal with up-projection
  output  = ffn_out @ W_down^T   ← Dense, shape [T, d_model]
```

The distinction from classic GELU FFN:

| Architecture | Stored tensors | Activation | Formula |
|---|---|---|---|
| Classic (BERT, GPT-2) | `ffn_fc1.weight`, `ffn_fc2.weight` | GELU node | `output = GELU(X @ W1^T) @ W2^T` |
| SwiGLU (Llama, Mistral, Phi-3) | `ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight` | SiLU (implicit) | `output = (gate ⊙ SiLU(gate) ⊙ up) @ W_down^T` |

The activation function (SiLU, GELU) is **never stored as a tensor**. It is inferred from `general.architecture`.

#### RMSNorm weight — the one parametric "normalisation" tensor

`attn_norm.weight` and `ffn_norm.weight` are shape `[d_model]` vectors. They are the learned scale $\gamma$ of **Root Mean Square Layer Normalisation** (Zhang & Sennrich 2019):

$$\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^2 + \varepsilon}} \cdot \gamma_i$$

This is **not a Dense layer**. It has no weight matrix, no bias, and performs no cross-neuron mixing. It is a per-feature rescaling after a scalar normalisation. In NNSpire's layer taxonomy (see [§3.7](blueprints.md#37-dense-is-one-of-many-ilayer-types)) it is most similar to `BatchNorm` — two learned scalars ($\gamma$ and $\beta$) per feature — except RMSNorm drops the $\beta$ bias entirely and replaces mean-centring with RMS normalisation.

```
Stored tensors per block:
  attn_norm.weight  [d_model]   ← γ vector, F32, learned
                                ← NO β bias (RMSNorm design choice)
                                ← NO running mean / variance (unlike BatchNorm)
                                ← The RMS is computed fresh each forward call

Not stored (implicit in architecture):
  ε (epsilon)   ← read from metadata key "llama.attention.layer_norm_rms_epsilon"
  normalisation formula ← determined by general.architecture = "llama"
```

#### Complete parametric vs non-parametric map for one Transformer block

```
One Transformer block (e.g. blk.0.*)
│
├── PRE-ATTENTION NORM
│   ├── attn_norm.weight  [d_model]        ← PARAMETRIC — RMSNorm γ, learned F32 vector
│   └── normalisation op                   ← NON-PARAMETRIC — RMS formula, no stored tensor
│
├── SELF-ATTENTION
│   ├── attn_q.weight     [n_h*d_h, d]     ← PARAMETRIC — Dense W_Q (no bias in LLaMA)
│   ├── attn_k.weight     [n_kv*d_h, d]    ← PARAMETRIC — Dense W_K
│   ├── attn_v.weight     [n_kv*d_h, d]    ← PARAMETRIC — Dense W_V
│   ├── attn_output.weight [d, n_h*d_h]    ← PARAMETRIC — Dense W_O
│   │
│   ├── QK^T / √d_k  (score)               ← NON-PARAMETRIC — IBackend::matmul + divScalar
│   ├── causal mask                         ← NON-PARAMETRIC — IBackend::add with −∞ mask
│   ├── softmax(scores)                     ← NON-PARAMETRIC — SoftmaxFn (no params)
│   └── weights @ V                         ← NON-PARAMETRIC — IBackend::matmul
│
├── RESIDUAL ADD (x = x + attn_output)      ← NON-PARAMETRIC — IBackend::add
│
├── PRE-FFN NORM
│   ├── ffn_norm.weight   [d_model]         ← PARAMETRIC — RMSNorm γ
│   └── normalisation op                    ← NON-PARAMETRIC
│
├── FEED-FORWARD NETWORK (SwiGLU)
│   ├── ffn_gate.weight   [d_ffn, d]        ← PARAMETRIC — Dense W_gate
│   ├── ffn_up.weight     [d_ffn, d]        ← PARAMETRIC — Dense W_up
│   ├── ffn_down.weight   [d, d_ffn]        ← PARAMETRIC — Dense W_down
│   │
│   ├── SiLU(gate) = gate ⊙ σ(gate)        ← NON-PARAMETRIC — SigmoidFn + IBackend::mul
│   ├── fused = SiLU(gate) ⊙ up            ← NON-PARAMETRIC — IBackend::mul (Hadamard)
│   └── output = fused @ W_down^T          ← (W_down is parametric above)
│
└── RESIDUAL ADD                            ← NON-PARAMETRIC — IBackend::add

Legend:
  PARAMETRIC     → stored as a tensor in the file; has a name entry in GGUF / SafeTensors
  NON-PARAMETRIC → computed at runtime from architecture metadata; zero bytes in the file
```

**Where are non-parametric layer specs stored?**

| Format | Location of non-parametric layer specs |
|---|---|
| **GGUF** | `general.architecture` metadata key (e.g. `"llama"`) → consumer hardcodes the block formula for that architecture |
| **ONNX** | Explicit `NodeProto` entries in `graph.node[]` — every `Softmax`, `Relu`, `Div`, `Add` is a named graph node with operator type |
| **SafeTensors** | Nowhere — caller must supply architecture separately (e.g. a `config.json` sidecar) |
| **PyTorch `.pt`** | Implicit in the saved `nn.Module` class hierarchy (pickle-reconstructed) |
| **Keras `.h5`** | `model_config` attribute on the root HDF5 group — JSON string of the full `keras.Model` config |
| **NNSpire `.nns`/`.nnsr`** | `ARCHITECTURE` JSON block (see §7 above) — explicit layer type list, self-contained |

---

### 3 — ONNX (Open Neural Network Exchange, 2017–present)

**Design philosophy:** Interoperability. A computation graph (operators + weights) that any conforming runtime can execute.

**File layout (Protocol Buffers binary):**

```
ONNX ModelProto
│
├── ir_version          : int64  (ONNX IR version, e.g. 9)
├── opset_imports       : [{ domain: "", version: 19 }]  ← which op-set version
├── producer_name       : "torch" | "tf2onnx" | "NNSpire"
├── model_version       : int64
├── doc_string          : string
│
├── graph  : GraphProto
│   │
│   ├── name : "main_graph"
│   │
│   ├── node  (one NodeProto per operator — the computation graph)
│   │   ├── node[0]:
│   │   │   ├── op_type  : "MatMul"
│   │   │   ├── inputs   : ["input", "layer0.weight"]
│   │   │   ├── outputs  : ["matmul_out0"]
│   │   │   └── attribute: {}
│   │   ├── node[1]:
│   │   │   ├── op_type  : "Add"
│   │   │   ├── inputs   : ["matmul_out0", "layer0.bias"]
│   │   │   └── outputs  : ["linear_out0"]
│   │   ├── node[2]:
│   │   │   ├── op_type  : "Relu"        ← activation encoded as an operator node!
│   │   │   ├── inputs   : ["linear_out0"]
│   │   │   └── outputs  : ["relu_out0"]
│   │   ├── node[3]:
│   │   │   ├── op_type  : "MatMul"
│   │   │   ├── inputs   : ["relu_out0", "layer1.weight"]
│   │   │   └── outputs  : ["matmul_out1"]
│   │   └── … (complete op DAG, matches NNSpire's ComputeGraph topology)
│   │
│   ├── input  : [{ name: "input", type: FLOAT, shape: ["batch", 2] }]
│   ├── output : [{ name: "output", type: FLOAT, shape: ["batch", 1] }]
│   │
│   └── initializer  (the weight tensors — one TensorProto per parameter)
│       ├── TensorProto:
│       │   ├── name      : "layer0.weight"
│       │   ├── dims      : [4, 2]         ← outF=4, inF=2
│       │   ├── data_type : 1              ← FLOAT = float32 (NNSpire DType::Float32)
│       │   └── raw_data  : [raw bytes]    ← flat row-major float32 array
│       ├── TensorProto:
│       │   ├── name      : "layer0.bias"
│       │   ├── dims      : [4]
│       │   ├── data_type : 1
│       │   └── raw_data  : [raw bytes]
│       └── … (one entry per W_ and b_ across all layers)
│
└── (optional external data files for large models — weights stored separately as .bin)
```

**ONNX data types → NNSpire DType:**

| ONNX `data_type` int | ONNX name | NNSpire DType | Notes |
|---|---|---|---|
| 1 | `FLOAT` | `Float32` | Phase 1 native |
| 2 | `UINT8` | — | Quantised output |
| 3 | `INT8` | `Int8` | PTQ weights |
| 5 | `INT16` | — | Rare |
| 6 | `INT32` | `Int32` | Token indices |
| 7 | `INT64` | — | Shape/index metadata |
| 10 | `FLOAT16` | `Float16` | Inference compression |
| 16 | `BFLOAT16` | — | Mixed-precision |

**Key distinction from SafeTensors and GGUF:**

ONNX stores the **computation graph**, not just the weights. Every `Relu`, `Sigmoid`, `Softmax`, `Gelu` node in the graph corresponds directly to NNSpire's `IActivation` implementations (see [§3.8](blueprints.md#38-activation-functions--iactivation-functors-and-activationbase)). The XOR model's four-layer graph (Dense→ReLU→Dense→Sigmoid) becomes exactly four `node` entries in the ONNX `GraphProto`.

This is NNSpire's primary export target (see [ADR-008](adr/ADR-008-onnx-primary-export.md)).

---

### 4 — PyTorch `.pt` / `.pth` (pickle-based, 2016–present)

**Design philosophy:** Maximal flexibility. Python `pickle` serialisation of arbitrary Python objects — including `torch.nn.Module` state dicts, optimizer states, training metadata.

**File layout (ZIP archive of pickle streams):**

```
model.pt  (ZIP archive)
│
├── archive/
│   ├── data.pkl           ← Python pickle stream (entry point)
│   │   Deserialises to:
│   │   {
│   │     "epoch": 42,
│   │     "model_state_dict": {
│   │       "layer0.weight": <Tensor reference → storage/0>,
│   │       "layer0.bias":   <Tensor reference → storage/1>,
│   │       "layer1.weight": <Tensor reference → storage/2>,
│   │       "layer1.bias":   <Tensor reference → storage/3>,
│   │     },
│   │     "optimizer_state_dict": {
│   │       "state": {
│   │         0: { "step": 1000,
│   │              "exp_avg":    <Tensor reference → storage/4>,   ← Adam m_
│   │              "exp_avg_sq": <Tensor reference → storage/5> }, ← Adam v_
│   │         …
│   │       },
│   │       "param_groups": [{ "lr": 1e-4, "betas": [0.9, 0.999], … }]
│   │     },
│   │     "loss": 0.023
│   │   }
│   │
│   └── data/
│       ├── 0    ← raw float32 bytes for "layer0.weight" [4, 2]
│       ├── 1    ← raw float32 bytes for "layer0.bias"   [4]
│       ├── 2    ← raw float32 bytes for "layer1.weight" [1, 4]
│       ├── 3    ← raw float32 bytes for "layer1.bias"   [1]
│       ├── 4    ← raw float32 bytes for Adam m_
│       └── 5    ← raw float32 bytes for Adam v_
```

**Why `.pt` files are dangerous (and why NNSpire does not use them):**

The `data.pkl` pickle stream can encode arbitrary Python objects and class instantiations. Loading an untrusted `.pt` file can execute arbitrary code on the host — a well-known supply-chain attack vector. SafeTensors was created specifically to replace this format.

**Optimizer state — what it maps to in NNSpire:**

NNSpire's `Adam` optimizer (see [Chapter 6](blueprints.md#chapter-6--fixing-the-weights-optimizer)) maintains moment tensors `m_` (first moment, exponential moving average of gradients) and `v_` (second moment, EMA of squared gradients) for each parameter:

```
Adam update rule (for each parameter W_):
  m_ ← β₁·m_ + (1−β₁)·∇W          ← momentum: exponential average of gradient history
  v_ ← β₂·v_ + (1−β₂)·(∇W)²       ← velocity: exponential average of squared gradients
  m̂  = m_ / (1−β₁ᵗ)               ← bias-corrected first moment
  v̂  = v_ / (1−β₂ᵗ)               ← bias-corrected second moment
  W_ ← W_ − lr · m̂ / (√v̂ + ε)    ← parameter update step
```

In a `.pt` checkpoint, `exp_avg` = `m_` and `exp_avg_sq` = `v_`. Both are full `Tensor` instances identical in shape to `W_`. A full training checkpoint therefore stores **three copies** of every parameter tensor: `W_`, `m_`, `v_`.

---

### 5 — HDF5 / Keras `.h5` (2015–present)

**Design philosophy:** Hierarchical, self-describing scientific data. Originally from HPC; adopted by Keras as the default training checkpoint format.

**File layout (HDF5 group hierarchy):**

```
model.h5  (HDF5 file)
│
├── /  (root group)
│   ├── model_weights/                       ← HDF5 group
│   │   ├── dense/                           ← layer name group
│   │   │   ├── dense/
│   │   │   │   ├── kernel:0   ← HDF5 dataset — shape [2, 4], dtype float32
│   │   │   │   │              ← "kernel" = NNSpire W_  (note: Keras uses [inF, outF], not [outF, inF])
│   │   │   │   └── bias:0     ← HDF5 dataset — shape [4], dtype float32
│   │   │   └── weight_names: ["dense/kernel:0", "dense/bias:0"]  ← attribute
│   │   ├── dense_1/
│   │   │   ├── dense_1/
│   │   │   │   ├── kernel:0   ← shape [4, 1]
│   │   │   │   └── bias:0     ← shape [1]
│   │   │   └── weight_names: ["dense_1/kernel:0", "dense_1/bias:0"]
│   │   └── layer_names: ["dense", "dense_1"]  ← attribute listing all layers
│   │
│   └── optimizer_weights/                   ← optimizer state (if saved)
│       └── Adam/
│           ├── iter:0                       ← step counter (scalar int64)
│           ├── dense/dense/kernel:0_m       ← Adam m_ for layer0 W_
│           ├── dense/dense/kernel:0_v       ← Adam v_ for layer0 W_
│           └── …
```

**Keras weight shape convention — the transpose trap:**

Keras stores `kernel` in shape `[inF, outF]`, the transpose of NNSpire's `W_` shape `[outF, inF]`. This is because Keras computes `y = x @ kernel + bias` (no explicit transpose), while NNSpire computes `Y = X @ W^T + b` (see [§3.4](blueprints.md#34-forward-pass-y--x--wt--b)). When loading Keras weights into NNSpire, the kernel tensor must be transposed:

```
Keras kernel [inF, outF]   →   NNSpire W_ [outF, inF]
      transpose (free in NNSpire — just swap strides_[0] and strides_[1])
```

This is a perfect example of why NNSpire's stride-swap transpose (see [§1: "Transpose is free"](blueprints.md#transpose-is-free--strides-without-copying)) has real practical value.

---

### 6 — TensorFlow SavedModel (2019–present)

**Design philosophy:** Production deployment. A directory containing the computation graph (as a TF function trace) plus weights as a VariableList — fully self-contained for serving.

**Directory layout:**

```
saved_model/
│
├── saved_model.pb             ← Protocol Buffers: MetaGraphDef + SignatureDefs
│   ├── MetaGraphDef
│   │   ├── graph_def          ← TF computation graph (NodeDef list)
│   │   │   ├── "MatMul"  node — inputs: [activations, kernel]
│   │   │   ├── "BiasAdd" node — inputs: [matmul_out, bias]
│   │   │   ├── "Relu"    node
│   │   │   └── "Sigmoid" node
│   │   └── saver_def
│   └── SignatureDef           ← input/output tensor names for serving
│       ├── "serving_default"
│       │   ├── inputs:  { "input_1": { dtype: DT_FLOAT, shape: [None, 2] } }
│       │   └── outputs: { "output":  { dtype: DT_FLOAT, shape: [None, 1] } }
│
└── variables/
    ├── variables.index        ← SSTable index: maps variable name → byte offset
    └── variables.data-00000-of-00001  ← raw float bytes (sharded for large models)
        ├── "dense/kernel"     ← shape [2, 4], dtype float32
        ├── "dense/bias"       ← shape [4],    dtype float32
        ├── "dense_1/kernel"   ← shape [4, 1], dtype float32
        └── "dense_1/bias"     ← shape [1],    dtype float32
```

---

### 7 — NNSpire Native: `.nns` / `.nnsr` (NNSpire, see ADR-009)

**Design philosophy:** NNSpire's own formats. `.nns` is the full training artifact (weights + architecture + optimizer state). `.nnsr` is the read-only runtime slice — minimal, signed, plugin-loadable.

**`.nns` layout (planned):**

```
model.nns
│
├── HEADER
│   ├── magic       : "NNS\0"
│   ├── version     : uint32
│   ├── flags       : uint64  (has_optimizer_state | has_grad | …)
│   └── schema_hash : SHA256 of the architecture JSON below
│
├── ARCHITECTURE  (JSON, similar to ONNX GraphProto but NNSpire-native)
│   ├── layers: [
│   │   { "type": "Dense",   "in": 2, "out": 4 },
│   │   { "type": "ReLU"                        },  ← IActivation identity (see §3.8)
│   │   { "type": "Dense",   "in": 4, "out": 1 },
│   │   { "type": "Sigmoid"                     }
│   │ ]
│   └── backend_hint: "cpu"
│
├── TENSORS  (one block per parameter — same role as SafeTensors DATA section)
│   ├── tensor_block:
│   │   ├── name     : "dense0.W"    ← NNSpire parameter naming convention
│   │   ├── dtype    : DType::Float32   ← NNSpire DType enum value
│   │   ├── shape    : [4, 2]        ← [outF, inF] (NNSpire row convention)
│   │   └── data     : [raw bytes]
│   ├── tensor_block: "dense0.b"   shape [4]
│   ├── tensor_block: "dense1.W"   shape [1, 4]
│   └── tensor_block: "dense1.b"   shape [1]
│
└── OPTIMIZER_STATE  (if flags has has_optimizer_state)
    ├── optimizer_type : "Adam"
    ├── step_count     : uint64
    ├── hyperparams    : { "lr": 1e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8 }
    └── moment_tensors : (same block format as TENSORS above, for m_ and v_)
        ├── "dense0.W.m"   shape [4, 2]
        ├── "dense0.W.v"   shape [4, 2]
        └── …
```

**`.nnsr` (runtime, read-only, signed):**

```
model.nnsr
│
├── HEADER        (same as .nns)
├── ARCHITECTURE  (same as .nns — read to reconstruct the ILayer graph)
├── TENSORS       (same as .nns — no optimizer state)
└── TRUST_BLOCK   (see TRUST-ARCHITECTURE.md and ADR-007)
    ├── publisher_cert   : PEM X.509 certificate chain
    ├── signature        : Ed25519 signature over SHA256(HEADER + ARCHITECTURE + TENSORS)
    └── policy_uri       : "https://nnspire.example.com/trust/policy.json"
```

The Plugin SDK loads `.nnsr` files after verifying the `TRUST_BLOCK` against the local trust store (see [§10.7](blueprints.md#107-plugin-sdk--trust-architecture)).

---

## Cross-Format Comparison

| Feature | SafeTensors | GGUF | ONNX | PyTorch `.pt` | Keras `.h5` | `.nns` | `.nnsr` |
|---|---|---|---|---|---|---|---|
| Architecture stored | ✗ | ✓ (config JSON) | ✓ (full graph) | Partial | Partial | ✓ | ✓ |
| Activation functions | ✗ | Implicit (arch key) | ✓ (operator nodes) | ✗ | ✗ | ✓ | ✓ |
| Optimizer state | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |
| Tokeniser embedded | ✗ | ✓ | ✗ | ✗ | ✗ | Optional | Optional |
| Quantisation (INT4/8) | ✗ (always F32+) | ✓ (Q4_K etc.) | ✓ (QLinear ops) | ✓ (via quantize_dynamic) | Partial | Planned | Planned |
| Safe to load untrusted | ✓ | ✓ | ✓ | ✗ (pickle RCE) | ✓ | ✓ | ✓ + signed |
| Zero-copy mmap | ✓ | ✓ | Partial | ✗ | ✗ | ✓ | ✓ |
| Human-readable header | ✓ (JSON) | ✗ (binary KV) | ✗ (protobuf) | ✗ (pickle) | ✗ (HDF5) | ✓ (JSON) | ✓ (JSON) |
| NNSpire Phase 1 export | — | — | **Primary (ADR-008)** | — | Via bridge | Native | Native |

---

## How Weight Types Map to NNSpire's Mathematical Model

The mathematical operations described in blueprints.md remain identical regardless of which dtype stores the weights. What changes is precision and memory footprint.

### Dense layer — `Y = X @ W^T + b` — weight type impact

```
W_ stored as Float32 (training, NNSpire Phase 1 native):
  Each element: 32 bits = sign(1) + exponent(8) + mantissa(23)
  Dynamic range: ≈ ±3.4×10³⁸
  Representation: exact IEEE 754 round-trip

W_ stored as Float16 (inference compression):
  Each element: 16 bits = sign(1) + exponent(5) + mantissa(10)
  Dynamic range: ≈ ±65504  ← overflow risk for large weight values
  Precision: ~3 decimal digits vs ~7 for F32

W_ stored as BFloat16 (mixed-precision training, Google TPU):
  Each element: 16 bits = sign(1) + exponent(8) + mantissa(7)
  Dynamic range: same as F32 (8-bit exponent) ← key advantage over F16
  Precision: ~2 decimal digits (fewer mantissa bits than F16)
  Rounding: truncation of F32 mantissa → trivially convert F32↔BF16

W_ stored as Int8 (post-training quantisation):
  Each element: 8 bits = signed integer [-128, 127]
  Dequantise formula: w_f32 = (w_int8 - zero_point) * scale
  scale and zero_point: F32 scalars stored once per tensor or per channel

W_ stored as Q4_K (GGUF 4-bit block quantisation):
  Each 256-weight super-block:
    nibble[i] ∈ [0, 15]   → 4 bits
    w_f32[i] = (nibble[i] - 8) * sub_scale[i/32] * super_scale
  Memory: 256 weights × 4 bits / 8 = 128 bytes + 18 bytes overhead ≈ 0.57 bytes/weight
  vs Float32: 256 × 4 = 1024 bytes → ~5.7× compression
```

### Activation function tensors — always full precision at runtime

Regardless of how `W_` is stored, the intermediate activation tensors flowing between layers (the `Y` tensors that `Dense::forward()` returns and `ReLU::forward()` transforms — see [§3.8](blueprints.md#38-activation-functions--iactivation-functors-and-activationbase)) are typically kept in `Float32` or `Float16` during inference. Quantising activations mid-computation is a separate technique (activation quantisation / QAT) not covered by the weight formats above.

The activation functions themselves are never stored as weights because they have no learnable parameters (except for `LeakyReLU`'s `alpha_` hyper-parameter, which is stored as a scalar attribute):

| Activation | NNSpire class | Stored in file as | Notes |
|---|---|---|---|
| `ReLU` | `ReLUFn` / `ReLU` | Operator node (ONNX) or layer type tag | No parameters; just a type identifier |
| `GELU` | `GELUFn` / `GELU` | Operator node or type tag | Two variants: `tanh` approx vs exact `erf` |
| `Sigmoid` | `SigmoidFn` / `Sigmoid` | Operator node or type tag | No parameters |
| `Tanh` | `TanhActFn` / `TanhAct` | Operator node or type tag | No parameters |
| `Softmax` | `SoftmaxFn` / `Softmax` | Operator node with `axis` attribute | `axis` = which dimension to normalise |
| `LeakyReLU` | `LeakyReLUFn` | Operator node with `alpha` attribute | `alpha` stored as a float attribute |
| `SwiGLU` | Planned (LLaMA) | Two operator nodes: `Sigmoid` + `Mul` | Composed activation; no dedicated node in GGUF |
| `RMSNorm` | Planned (LLaMA) | Two tensors: scale `γ` [d_model] + no bias | `γ` is a learned F32 vector — IS a weight tensor |

### Embedding layer — the lookup-table weight

The `Embedding` layer (see [§3.7](blueprints.md#37-dense-is-one-of-many-ilayer-types)) is a special case: its parameter is a dense float matrix of shape `[vocab_size, d_model]` — the largest single tensor in most LLMs (e.g., `[32000, 4096]` for Llama-3-8B = 128 million float32 values = 512 MB before quantisation).

```
GGUF: "token_embd.weight"  dims=[32000, 4096]  type=Q4_K
                           └─────────────────────────────── same as a Dense W_, but:
                              • input is an integer token ID, not a float vector
                              • forward = table lookup (one row copy), not matmul
                              • backward = gradient scatter-add to the selected row
```

After quantisation to Q4_K this drops to ~74 MB — the dominant storage saving in 4-bit LLMs.

---

## Reading Guide — Format by Use Case

| Goal | Recommended format | Rationale |
|---|---|---|
| Load a pretrained transformer (HuggingFace) | SafeTensors | Safe, zero-copy, ecosystem standard |
| Run a local LLM on CPU, low RAM | GGUF Q4_K | 4-bit quantisation; llama.cpp optimised kernels |
| Export NNSpire model for deployment | ONNX | ADR-008; universally accepted by ONNX Runtime, TensorRT, TF Lite |
| Checkpoint a training run | `.nns` | Full state: weights + optimizer moments + architecture |
| Distribute a plugin model | `.nnsr` | Signed, trust-verified, minimal (no optimizer state) |
| Interop with Keras / TF | `.h5` / SavedModel | NNSpire Python bridge reads both (see §10.9) |

---

> **See also:**
> - [ADR-008 — ONNX as primary export](adr/ADR-008-onnx-primary-export.md)
> - [ADR-009 — NNS/NNSR custom formats](adr/ADR-009-nns-nnsr-custom-formats.md)
> - [ADR-007 — PKI trust chain](adr/ADR-007-pki-trust-chain.md)
> - [§1.x — DType in NNSpire](blueprints.md#1x--foundation-types-device-dtype-resultt)
> - [§3.4 — Dense forward pass](blueprints.md#34-forward-pass-y--x--wt--b)
> - [§3.8 — Activation functions](blueprints.md#38-activation-functions--iactivation-functors-and-activationbase)
> - [TRUST-ARCHITECTURE.md](TRUST-ARCHITECTURE.md)

---

## Annex — What the Hell is Quantisation?

> **Short answer:** quantisation replaces 32-bit floats with small integers (4-bit, 8-bit), stores a small number of floating-point *scale factors* alongside them, and reconstructs approximate float values on the fly during inference. The Dense forward pass `Y = X @ W^T` runs on those approximate values. The result is slightly wrong — but only slightly, and the model is 4–8× smaller.

### A.1 — The problem quantisation solves

A Llama-3-8B model in full `Float32` precision:
```
  ~8 billion parameters x 4 bytes/param = 32 GB
```
That does not fit in the RAM of any consumer GPU. In `Float16` it is 16 GB — still tight. In Q4_K (~4.5 bits/param) it is ~4.5 GB — fits on a 6 GB GPU or even CPU RAM.

The price is **representational error**: instead of storing the exact float `0.31415926...`, you store a 4-bit integer `5` and a shared scale factor `0.06`, giving `5 x 0.06 = 0.30` — an error of `0.014`. For a weight matrix with thousands of such weights contributing to a dot product, the errors largely cancel each other out and the final prediction barely degrades.

### A.2 — Int8 symmetric quantisation — the simplest case

This is the conceptual core. All other quantisation schemes are refinements of this idea.

**Quantise a weight vector to Int8:**

Original weights (Float32):
```
w = [ 0.80,  -0.42,  0.15,  -0.93,  0.61,  0.07,  -0.31,  0.50 ]
```

Step 1 — find the absolute maximum:
```
max_abs = max(|w|) = 0.93
```

Step 2 — compute the scale factor (Int8 range is [-128, 127]; we use 127 for symmetry):
```
scale = max_abs / 127  =  0.93 / 127  = 0.007323
```

Step 3 — quantise: divide by scale, round to nearest integer:
```
w_int8 = round(w / scale)
       = round([ 0.80,  -0.42,  0.15, -0.93,  0.61,  0.07, -0.31,  0.50 ] / 0.007323)
       = round([ 109.2, -57.4,  20.5, -127.0, 83.3,   9.6, -42.3,  68.3 ])
       =       [ 109,   -57,    21,   -127,    83,    10,   -42,    68   ]  (8 x int8)
```

**What is stored on disk:**
```
  8 x int8 bytes  =  8 bytes   (the quantised weights)
  1 x float32     =  4 bytes   (the scale factor)
  ────────────────────────────
  Total           = 12 bytes
  vs original     = 32 bytes   (8 x float32)
  Savings: 62.5%
```

Step 4 — dequantise at inference time (before the matmul):
```
w_approx = w_int8 x scale
         = [ 109, -57,  21, -127,  83,  10, -42,  68 ] x 0.007323
         = [ 0.7982, -0.4174, 0.1538, -0.9301, 0.6078, 0.0732, -0.3076, 0.4980 ]

Original  [ 0.800, -0.420, 0.150, -0.930, 0.610, 0.070, -0.310, 0.500 ]
Recovered [ 0.798, -0.417, 0.154, -0.930, 0.608, 0.073, -0.308, 0.498 ]
Error     [ 0.002, -0.003, 0.004,  0.000, 0.002, 0.003, -0.002, 0.002 ]
            ^                                                             ^
            max absolute error ~= 0.004 ~= 0.5 x scale   (always true by construction)
```

### A.3 — Impact on the Dense forward pass: error propagation

Consider a tiny Dense layer: input `x` (Float32, exact), weight row `w` (quantised then dequantised to `w_approx`).

The forward pass computes one output neuron as:

$$y = \sum_{i=1}^{n} w_i \cdot x_i$$

With quantisation error $\delta_i = w_{approx,i} - w_i$, the actual computed value is:

$$\hat{y} = \sum_{i=1}^{n} (w_i + \delta_i) \cdot x_i = y + \underbrace{\sum_{i=1}^{n} \delta_i \cdot x_i}_{\text{quantisation noise}}$$

**Tiny matrix example** — 2×3 Dense, one sample (continuing the weights above):

```
x (input, Float32, exact):  [ 1.0,  0.5, -0.3 ]

W (original Float32, two neurons using the first 3 weights each):
  neuron 0: [  0.80, -0.42,  0.15 ]   y0 = 1.0*0.80 + 0.5*(-0.42) + (-0.3)*0.15
                                          = 0.800 - 0.210 - 0.045 = 0.545
  neuron 1: [ -0.93,  0.61,  0.07 ]   y1 = 1.0*(-0.93) + 0.5*0.61 + (-0.3)*0.07
                                          = -0.930 + 0.305 - 0.021 = -0.646

W_approx (after Int8 quantise -> dequantise, same scale = 0.007323):
  neuron 0: [  0.798, -0.417,  0.154 ]   y0_hat = 0.798 - 0.209 - 0.046 = 0.543
  neuron 1: [ -0.930,  0.608,  0.073 ]   y1_hat = -0.930 + 0.304 - 0.022 = -0.648

Errors:
  neuron 0:  0.543 - 0.545 = -0.002   (< 0.5% of output magnitude)
  neuron 1: -0.648 - (-0.646) = -0.002
```

For a layer with `inF = 4096` (typical in LLMs), the 4096 errors $\delta_i \cdot x_i$ are approximately independent and random in sign — they accumulate as a random walk, so the total noise grows as $O(\sqrt{n} \cdot \delta_{max} \cdot \bar{x})$ rather than $O(n)$. For Int8 with $\delta_{max} \approx 0.004$, $n = 4096$, and typical activation magnitude $\bar{x} \approx 0.1$:

$$\text{expected output noise} \approx \sqrt{4096} \times 0.004 \times 0.1 = 64 \times 0.0004 \approx 0.026$$

That is small relative to typical layer output magnitudes, which is why Int8 quantisation is nearly lossless in practice.

### A.4 — Why one scale per tensor is not enough: the channel problem

The single-scale approach breaks down when different rows of `W` have very different value ranges. If row 0 lives in `[-0.01, 0.01]` and row 1 lives in `[-5.0, 5.0]`, a single global scale tuned for row 1 will quantise row 0 to nearly all zeros:

```
W = [  0.010  -0.008   0.012   0.005 ]  <- row 0: tiny values
    [ -4.800   3.200  -2.100   1.500 ]  <- row 1: large values

Global scale = 4.8 / 127 = 0.0378

Row 0 quantised: round([0.010, -0.008, 0.012, 0.005] / 0.0378)
               = round([0.26, -0.21, 0.32, 0.13])
               = [  0,   0,   0,   0 ]  <- ENTIRE ROW ZEROED OUT
Row 1 quantised: round([-4.8, 3.2, -2.1, 1.5] / 0.0378)
               = [-127, 85, -56, 40]    <- fine

Per-channel fix: one scale per row
  scale_row0 = 0.012 / 127 = 0.0000945
  Row 0 quantised: round([0.010, -0.008, 0.012, 0.005] / 0.0000945)
                 = [106, -85, 127, 53]  <- preserved with full 8-bit resolution
```

Per-channel quantisation stores one extra `float32` scale per output neuron (per row of `W`). For a `[4096, 4096]` Dense layer that is 4096 extra floats = 16 KB overhead versus 64 MB of weight data — negligible.

### A.5 — GGUF block quantisation (Q4_K) — the file-level anatomy

GGUF goes further than per-channel Int8. It uses **super-blocks** of 256 weights, each subdivided into 8 sub-blocks of 32 weights, with two levels of scales:

```
One Q4_K super-block (256 weights stored on disk):

  Byte offset   Content
  ───────────   ──────────────────────────────────────────────────────────────────
  0–1           super_scale  [float16]  divides all sub-block scales
  2–3           super_min    [float16]  asymmetric offset at super-block level
  4–9           sub_scales   [8 x 6-bit packed into 6 bytes]  one per 32-weight group
  10–15         sub_mins     [8 x 6-bit packed into 6 bytes]  one per 32-weight group
  16–143        nibbles      [128 bytes]  256 x 4-bit weights, two per byte

  Total: 144 bytes for 256 weights = 0.5625 bytes/weight = 4.5 bits/weight
  vs Float32:  256 x 4 = 1024 bytes          ->  7.1x compression

Dequantise weight i (in sub-block b = floor(i/32)):
  effective_scale = sub_scales[b] * super_scale
  effective_min   = sub_mins[b]   * super_min
  w_approx[i]     = effective_scale * nibble[i] - effective_min
                    nibble[i] in [0, 15]   (unsigned 4-bit, asymmetric)
```

**Tiny Q4_K worked example** — 8 weights (one sub-block fragment):

```
Original F32 weights:
  w = [ 0.42,  -0.31,  0.78,  -0.55,  0.10,  -0.88,  0.62,  -0.20 ]

Sub-block range:
  min = -0.88,  max = 0.78
  effective_scale = (max - min) / 15 = 1.66 / 15 = 0.11067
  effective_min   = min = -0.88

Quantise (nibble = round((w - min) / scale), clamp to [0, 15]):
  w[0] = 0.42:   (0.42 - (-0.88)) / 0.11067 = 1.30 / 0.11067 = 11.7  ->  nibble = 12
  w[1] =-0.31:   (-0.31 + 0.88)  / 0.11067 = 0.57 / 0.11067 =  5.1  ->  nibble =  5
  w[2] = 0.78:   (0.78 + 0.88)  / 0.11067 = 1.66 / 0.11067 = 15.0  ->  nibble = 15
  w[3] =-0.55:   (0.33)  / 0.11067 = 2.98  ->  nibble =  3
  w[4] = 0.10:   (0.98)  / 0.11067 = 8.85  ->  nibble =  9
  w[5] =-0.88:   (0.00)  / 0.11067 = 0.00  ->  nibble =  0
  w[6] = 0.62:   (1.50)  / 0.11067 = 13.55 ->  nibble = 14
  w[7] =-0.20:   (0.68)  / 0.11067 = 6.15  ->  nibble =  6

Stored nibbles: [ 12,  5, 15,  3,  9,  0, 14,  6 ]

Packed bytes (low nibble = even index, high nibble = odd index):
  byte 0 = (nibble[1] << 4) | nibble[0] = (5 << 4) | 12 = 0x5C
  byte 1 = (nibble[3] << 4) | nibble[2] = (3 << 4) | 15 = 0x3F
  byte 2 = (nibble[5] << 4) | nibble[4] = (0 << 4) |  9 = 0x09
  byte 3 = (nibble[7] << 4) | nibble[6] = (6 << 4) | 14 = 0x6E

Dequantise back:
  w_approx[0] = 12 * 0.11067 - 0.88 =  1.328 - 0.88 =  0.448  (orig: 0.42,  err: +0.028)
  w_approx[1] =  5 * 0.11067 - 0.88 =  0.553 - 0.88 = -0.327  (orig:-0.31,  err: -0.017)
  w_approx[2] = 15 * 0.11067 - 0.88 =  1.660 - 0.88 =  0.780  (orig: 0.78,  err: +0.000)
  w_approx[3] =  3 * 0.11067 - 0.88 =  0.332 - 0.88 = -0.548  (orig:-0.55,  err: +0.002)
  w_approx[4] =  9 * 0.11067 - 0.88 =  0.996 - 0.88 =  0.116  (orig: 0.10,  err: +0.016)
  w_approx[5] =  0 * 0.11067 - 0.88 =  0.000 - 0.88 = -0.880  (orig:-0.88,  err:  0.000)
  w_approx[6] = 14 * 0.11067 - 0.88 =  1.549 - 0.88 =  0.669  (orig: 0.62,  err: +0.049)
  w_approx[7] =  6 * 0.11067 - 0.88 =  0.664 - 0.88 = -0.216  (orig:-0.20,  err: -0.016)

Max error: 0.049  vs  theoretical max = 0.5 * scale = 0.5 * 0.11067 = 0.055  [ok, within bound]
```

In a real GGUF file those 4 bytes are part of the 128-byte nibble array of a full 256-weight super-block, alongside the 16 bytes of scale/min metadata — 144 bytes total per super-block.

### A.6 — What quantisation does and does NOT affect

```
                    Transformer forward pass
                    ─────────────────────────────────────────────────────────────

QUANTISED (stored as int4/int8; dequantised before compute):
  attn_q.weight      W_Q   Q = X @ W_Q^T
  attn_k.weight      W_K   K = X @ W_K^T
  attn_v.weight      W_V   V = X @ W_V^T
  attn_output.weight W_O   O = context @ W_O^T
  ffn_gate.weight    W_gate gate = X @ W_gate^T
  ffn_up.weight      W_up  up   = X @ W_up^T
  ffn_down.weight    W_down out  = act(gate,up) @ W_down^T
  token_embd.weight  Embedding lookup table (largest single tensor in most LLMs)
  output.weight      LM head  logits = H @ W_lm^T

NOT QUANTISED (kept Float32 even in Q4_K files):
  attn_norm.weight   RMSNorm gamma [d_model]  <- too few values; precision matters
  ffn_norm.weight    RMSNorm gamma [d_model]
  output_norm.weight Final RMSNorm gamma

NOTHING TO QUANTISE (no stored weights at all):
  Softmax, ReLU, GELU/SiLU    <- no parameters; computation only
  Causal mask                  <- derived at runtime from sequence length
  Residual adds                <- IBackend::add, no weights
```

**Why are norm weights kept in Float32?**
RMSNorm has only `d_model = 4096` parameters — 16 KB at Float32. Negligible. But those 4096 scalars gate the *magnitude* of every activation in the residual stream. A 4-bit scale error there propagates as a systematic bias across the entire model rather than averaging out. Float32 here costs almost nothing and avoids a whole class of degradation.

### A.7 — Quantisation quality ladder

Typical perplexity increase on a 7–8B model vs Float16 baseline (lower = better):

```
  Format      Bits/param  Size (8B)  Perplexity increase  Notes
  ────────────────────────────────────────────────────────────────────────────────────
  F32         32.0        32 GB      +0.00 (baseline)     Training precision
  BF16        16.0        16 GB      ~+0.00               Near-lossless; same exponent range as F32
  F16         16.0        16 GB      ~+0.01               Near-lossless
  Q8_0         8.0         8 GB      ~+0.02               Effectively lossless for most tasks
  Q6_K         6.5         6.5 GB    ~+0.05               Excellent quality
  Q5_K         5.5         5.5 GB    ~+0.10               Very good; preferred over Q4 if RAM allows
  Q4_K         4.5         4.5 GB    ~+0.20               Recommended default for consumer hardware
  Q3_K         3.5         3.5 GB    ~+0.50               Noticeable but acceptable degradation
  Q2_K         2.5         2.5 GB    ~+2.0                Poor; last resort when RAM is critical
  IQ4_XS       4.3         4.3 GB    ~+0.15               Importance-calibrated; often beats Q4_K
  IQ2_XXS      2.1         2.1 GB    ~+1.0                Best available 2-bit option
```

**Importance-calibrated quantisation (IQ-family):** not all weights are equally important. An activation-aware calibration pass (imatrix) measures which weight channels most affect model output. High-importance channels get more bits; low-importance ones get fewer. At the same average bit-rate this produces lower perplexity than uniform quantisation — at the cost of a one-time calibration step on a representative dataset.

---

> **See also:**
> - [ADR-008 — ONNX as primary export](adr/ADR-008-onnx-primary-export.md)
> - [ADR-009 — NNS/NNSR custom formats](adr/ADR-009-nns-nnsr-custom-formats.md)
> - [ADR-007 — PKI trust chain](adr/ADR-007-pki-trust-chain.md)
> - [§1.x — DType in NNSpire](blueprints.md#1x--foundation-types-device-dtype-resultt)
> - [§3.4 — Dense forward pass](blueprints.md#34-forward-pass-y--x--wt--b)
> - [§3.8 — Activation functions](blueprints.md#38-activation-functions--iactivation-functors-and-activationbase)
> - [TRUST-ARCHITECTURE.md](TRUST-ARCHITECTURE.md)
