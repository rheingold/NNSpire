/* ============================================================================
 * MultiHeadAttention.h — scaled dot-product multi-head attention
 *                         [nnstudio::builtin::layers]
 * LGPL v3
 *
 * WHAT IS MULTI-HEAD ATTENTION?
 * ──────────────────────────────
 * Multi-Head Attention (MHA) allows a model to jointly attend to information
 * from different representation sub-spaces at different positions.
 *
 * ALGORITHM (scaled dot-product, Vaswani et al. 2017 "Attention is All You Need")
 * ─────────────────────────────────────────────────────────────────────
 *   Given input x ∈ ℝ^{N × L × d_model}:
 *
 *   1. Linear projections:
 *      Q = x Wq,  K = x Wk,  V = x Wv     each → [N, L, d_model]
 *
 *   2. Split each into h heads (d_k = d_v = d_model / h):
 *      Q_h = Q[:, :, h*d_k : (h+1)*d_k]   → [N, L, d_k]
 *      (reshape to [N*h, L, d_k] for batched ops)
 *
 *   3. Scaled dot-product attention per head:
 *      Scores = Q_h K_h^T / sqrt(d_k)      → [N*h, L, L]
 *      Attn   = softmax(Scores, dim=-1)     → [N*h, L, L]
 *      Head_h = Attn V_h                   → [N*h, L, d_k]
 *
 *   4. Concatenate heads → [N, L, d_model]
 *
 *   5. Output projection:
 *      Output = concat Wo                  → [N, L, d_model]
 *
 * SHAPES
 * ──────
 *   Input  : [N, L, d_model]
 *   Wq/Wk/Wv: [d_model, d_model]
 *   Wo     : [d_model, d_model]
 *   Output : [N, L, d_model]
 *
 * CAUSAL MASKING
 * ──────────────
 * When causal=true, the attention score matrix is masked to -∞ above the
 * diagonal so position i cannot attend to j > i (autoregressive models).
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#attention
 * ============================================================================ */

#pragma once

#include <core/Layer.h>

namespace nnstudio {
namespace builtin {
namespace layers {

using namespace nnstudio::core;

class MultiHeadAttention : public ILayer {
public:
    /**
     * @param numHeads   Number of attention heads h.
     * @param embDim     Model dimension d_model (must be divisible by numHeads).
     * @param causal     If true, apply causal (lower-triangular) mask.
     * @param dropout    Attention dropout probability (applied during training).
     *                   Phase 1: accepted but not yet applied (planned for Phase 4).
     */
    explicit MultiHeadAttention(int64_t numHeads,
                                int64_t embDim,
                                bool    causal  = false,
                                float   dropout = 0.0f);

    std::string_view typeName() const noexcept override { return "MultiHeadAttention"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#attention";
    }

    // ── ILayer ───────────────────────────────────────────────────────────────
    Result<Shape>  build   (const Shape& inputShape) override;
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)         override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr)   override;

    std::vector<Parameter*>       parameters()       override;
    std::vector<const Parameter*> parameters() const override;

    std::unordered_map<std::string,std::string> config() const override;

    int64_t numHeads() const noexcept { return numHeads_; }
    int64_t embDim()   const noexcept { return embDim_;   }
    bool    causal()   const noexcept { return causal_;   }

private:
    int64_t numHeads_;
    int64_t embDim_;
    int64_t dK_;        ///< embDim_ / numHeads_
    bool    causal_;
    float   dropout_;

    Parameter Wq_, Wk_, Wv_, Wo_;   ///< each [embDim, embDim]
    Parameter bq_, bk_, bv_, bo_;   ///< each [embDim]

    // Saved tensors for backward
    Tensor lastInput_;
    Tensor lastQ_, lastK_, lastV_;      // after projections: [N, L, embDim]
    Tensor lastAttn_;                   // [N*h, L, L] attention weights
    Tensor lastContext_;                // [N*h, L, dK] after attn @ V

    int64_t lastN_{0}, lastL_{0};

    // ── Helper: 3-D "batched dense" projection ─────────────────────────────
    // x: [N, L, in]   W: [in, out]   b: [out]
    // returns [N, L, out]
    static Tensor project(const Tensor& x,
                          const Tensor& W,
                          const Tensor& b);

    // ── Helper: softmax over last dimension ────────────────────────────────
    // t: [B, M, N] → returns softmax over N, in-place result tensor
    static Tensor softmaxLastDim(const Tensor& t);

    // ── Helper: reshape 3→4 and 4→3 for head splitting ─────────────────────
    // [N, L, embDim] → [N*h, L, dK]
    static Tensor splitHeads(const Tensor& t, int64_t numHeads, int64_t dK);
    // [N*h, L, dK]  → [N, L, embDim]
    static Tensor mergeHeads(const Tensor& t, int64_t N, int64_t L, int64_t embDim);
};

} // namespace layers
} // namespace builtin
} // namespace nnstudio
