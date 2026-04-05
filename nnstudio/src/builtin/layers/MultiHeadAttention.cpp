/* ============================================================================
 * MultiHeadAttention.cpp — scaled dot-product multi-head attention
 * LGPL v3
 *
 * Implementation follows Vaswani et al. (2017) "Attention is All You Need".
 * Phase 1: CPU reference implementation; CUDA-accelerated flash attention
 * is planned for Phase 4 (CUDA backend).
 * ============================================================================ */

#include <builtin/layers/MultiHeadAttention.h>
#include <core/BackendRegistry.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>

namespace nnstudio {
namespace builtin {
namespace layers {

using namespace nnstudio::core;

// ─── convenience ─────────────────────────────────────────────────────────────
static inline IBackend& B() { return nnstudio::core::backend(); }

// ─── Constructor ─────────────────────────────────────────────────────────────

MultiHeadAttention::MultiHeadAttention(int64_t numHeads,
                                       int64_t embDim,
                                       bool    causal,
                                       float   dropout)
    : numHeads_(numHeads)
    , embDim_(embDim)
    , dK_(embDim / numHeads)
    , causal_(causal)
    , dropout_(dropout)
{
    Wq_.name = "Wq";  Wk_.name = "Wk";  Wv_.name = "Wv";  Wo_.name = "Wo";
    bq_.name = "bq";  bk_.name = "bk";  bv_.name = "bv";  bo_.name = "bo";
}

// ─── Static helpers ───────────────────────────────────────────────────────────

// project: [N, L, in] → [N, L, out]  via W[in,out] + b[out]
Tensor MultiHeadAttention::project(const Tensor& x, const Tensor& W, const Tensor& b) {
    int64_t N   = x.shape()[0];
    int64_t L   = x.shape()[1];
    int64_t in  = x.shape()[2];
    int64_t out = W.shape()[1];

    // Reshape to [N*L, in]
    auto xfR = x.reshape({N * L, in});
    if (!xfR.ok()) { assert(false && "MHA::project reshape failed"); return Tensor{}; }
    const Tensor& xf = xfR.value();

    // matmul [N*L, in] @ [in, out] = [N*L, out]
    Tensor yf = B().matmul(xf, W);

    // Add bias (broadcast over rows)
    float* yd = yf.data();
    const float* bd = b.data();
    for (int64_t i = 0; i < N * L; ++i)
        for (int64_t o = 0; o < out; ++o)
            yd[i * out + o] += bd[o];

    // Reshape back to [N, L, out]
    auto outR = yf.reshape({N, L, out});
    if (!outR.ok()) { assert(false && "MHA::project reshape-back failed"); return Tensor{}; }
    return outR.value();
}

// softmax over last dim: t [B, M, N] → each [M, N] row softmax'd
Tensor MultiHeadAttention::softmaxLastDim(const Tensor& t) {
    assert(t.ndim() == 3);
    int64_t B = t.shape()[0];
    int64_t M = t.shape()[1];
    int64_t N = t.shape()[2];
    Tensor out = Tensor::zeros({B, M, N});
    const float* td = t.data();
    float* od = out.data();

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t m = 0; m < M; ++m) {
            const float* row = td + (b * M + m) * N;
            float* orow = od + (b * M + m) * N;
            // Numerically stable: subtract max
            float maxv = row[0];
            for (int64_t n = 1; n < N; ++n) if (row[n] > maxv) maxv = row[n];
            float sumexp = 0.0f;
            for (int64_t n = 0; n < N; ++n) { orow[n] = std::exp(row[n] - maxv); sumexp += orow[n]; }
            float inv = 1.0f / (sumexp + 1e-9f);
            for (int64_t n = 0; n < N; ++n) orow[n] *= inv;
        }
    }
    return out;
}

// splitHeads: [N, L, embDim] → [N*h, L, dK]
Tensor MultiHeadAttention::splitHeads(const Tensor& t, int64_t h, int64_t dK) {
    int64_t N = t.shape()[0];
    int64_t L = t.shape()[1];
    // Currently stored as [N, L, h, dK] in flat memory when h=numHeads, dK=embDim/h
    // We need to permute to [N, h, L, dK] then reshape to [N*h, L, dK]
    // Since memory is [N, L, h*dK] = [N, L, embDim]:
    //   element [n, l, hh, dk] = flat index n*(L*h*dK) + l*(h*dK) + hh*dK + dk
    // We want result [N*h, L, dK]:
    //   element [n*h+hh, l, dk] = flat (n*h+hh)*(L*dK) + l*dK + dk
    Tensor out = Tensor::zeros({N * h, L, dK});
    const float* src = t.data();
    float* dst = out.data();
    for (int64_t n = 0; n < N; ++n)
        for (int64_t l = 0; l < L; ++l)
            for (int64_t hh = 0; hh < h; ++hh)
                for (int64_t dk = 0; dk < dK; ++dk)
                    dst[(n*h+hh)*(L*dK) + l*dK + dk] =
                        src[n*(L*h*dK) + l*(h*dK) + hh*dK + dk];
    return out;
}

// mergeHeads: [N*h, L, dK] → [N, L, embDim=h*dK]
Tensor MultiHeadAttention::mergeHeads(const Tensor& t, int64_t N, int64_t L, int64_t embDim) {
    int64_t Nh = t.shape()[0];
    int64_t h  = Nh / N;
    int64_t dK = embDim / h;
    Tensor out = Tensor::zeros({N, L, embDim});
    const float* src = t.data();
    float* dst = out.data();
    for (int64_t n = 0; n < N; ++n)
        for (int64_t hh = 0; hh < h; ++hh)
            for (int64_t l = 0; l < L; ++l)
                for (int64_t dk = 0; dk < dK; ++dk)
                    dst[n*(L*embDim) + l*embDim + hh*dK + dk] =
                        src[(n*h+hh)*(L*dK) + l*dK + dk];
    return out;
}

// ─── build ────────────────────────────────────────────────────────────────────

Result<Shape> MultiHeadAttention::build(const Shape& inputShape) {
    // inputShape: [L, embDim] (per-sample; batch is prepended at runtime)
    if (inputShape.size() != 2) {
        return Result<Shape>(Error{ErrorCode::InvalidArgument, "MultiHeadAttention::build: inputShape must be [L, embDim]"});
    }
    if (embDim_ % numHeads_ != 0) {
        return Result<Shape>(Error{ErrorCode::InvalidArgument, "MultiHeadAttention::build: embDim must be divisible by numHeads"});
    }
    if (isBuilt()) return Result<Shape>{{inputShape[0], embDim_}};

    // Xavier uniform init
    float limit = std::sqrt(6.0f / static_cast<float>(embDim_ + embDim_));
    auto initW = [&](Parameter& p, const std::string& n) {
        p.name = n;
        p.tensor = Tensor{{embDim_, embDim_}};
        uint32_t state = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(&p));
        state ^= 0x9e3779b9u;
        float* d = p.tensor.data();
        for (int64_t i = 0, sz = p.tensor.numel(); i < sz; ++i) {
            state = state * 1664525u + 1013904223u;
            float u = static_cast<float>(state >> 16) / 65535.0f;
            d[i] = -limit + 2.0f * limit * u;
        }
    };
    auto initB = [&](Parameter& p, const std::string& n) {
        p.name = n;
        p.tensor = Tensor::zeros({embDim_});
    };

    initW(Wq_, "Wq");  initW(Wk_, "Wk");  initW(Wv_, "Wv");  initW(Wo_, "Wo");
    initB(bq_, "bq");  initB(bk_, "bk");  initB(bv_, "bv");  initB(bo_, "bo");

    markBuilt();
    return Result<Shape>{{inputShape[0], embDim_}};
}

// ─── forward ──────────────────────────────────────────────────────────────────

Result<Tensor> MultiHeadAttention::forward(const Tensor& x) {
    // x: [N, L, embDim]
    if (x.ndim() != 3) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "MultiHeadAttention::forward: input must be rank 3 [N,L,embDim]"});
    if (!isBuilt())    return Result<Tensor>(Error{ErrorCode::InvalidArgument, "MultiHeadAttention::forward: call build() first"});

    int64_t N = x.shape()[0];
    int64_t L = x.shape()[1];
    float   scale = 1.0f / std::sqrt(static_cast<float>(dK_));

    lastInput_ = x.clone();
    lastN_ = N;  lastL_ = L;

    // 1. Linear projections: [N, L, embDim]
    lastQ_ = project(x, Wq_.tensor, bq_.tensor);
    lastK_ = project(x, Wk_.tensor, bk_.tensor);
    lastV_ = project(x, Wv_.tensor, bv_.tensor);

    // 2. Split heads: [N*h, L, dK]
    Tensor Q_h = splitHeads(lastQ_, numHeads_, dK_);
    Tensor K_h = splitHeads(lastK_, numHeads_, dK_);
    Tensor V_h = splitHeads(lastV_, numHeads_, dK_);

    // 3. scores = Q_h @ K_h^T / sqrt(dK)   → [N*h, L, L]
    //    K_h^T: transpose dims 1 and 2 of [N*h, L, dK] → [N*h, dK, L]
    Tensor K_h_T = B().transpose(K_h, 1, 2);
    Tensor scores = B().matmul(Q_h, K_h_T);  // [N*h, L, L]
    // scale
    {
        float* sd = scores.data();
        for (int64_t i = 0; i < scores.numel(); ++i) sd[i] *= scale;
    }

    // Causal mask: set upper triangle above diagonal to -inf
    if (causal_) {
        int64_t Nh = N * numHeads_;
        float* sd = scores.data();
        for (int64_t b = 0; b < Nh; ++b)
            for (int64_t i = 0; i < L; ++i)
                for (int64_t j = i + 1; j < L; ++j)
                    sd[b*(L*L) + i*L + j] = -1e9f;
    }

    // 4. attn = softmax(scores)  → [N*h, L, L]
    lastAttn_ = softmaxLastDim(scores);

    // 5. context = attn @ V_h   → [N*h, L, dK]
    lastContext_ = B().matmul(lastAttn_, V_h);  // [N*h, L, dK]

    // 6. Merge heads: [N, L, embDim]
    Tensor merged = mergeHeads(lastContext_, N, L, embDim_);

    // 7. Output projection: [N, L, embDim]
    Tensor out = project(merged, Wo_.tensor, bo_.tensor);
    return Result<Tensor>{std::move(out)};
}

// ─── backward ─────────────────────────────────────────────────────────────────

Result<Tensor> MultiHeadAttention::backward(const Tensor& gradOut) {
    // gradOut: [N, L, embDim]
    if (gradOut.ndim() != 3) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "MultiHeadAttention::backward: gradOut must be rank 3"});

    int64_t N   = lastN_;
    int64_t L   = lastL_;
    int64_t h   = numHeads_;
    float   scale = 1.0f / std::sqrt(static_cast<float>(dK_));

    // ── Step 1: backward through output projection ────────────────────────────
    // out = merged @ Wo + bo   where merged=[N,L,embDim], Wo=[embDim,embDim]
    // dmerged = dOut @ Wo^T
    // dWo = merged.reshape(N*L, embDim)^T @ dOut.reshape(N*L, embDim)
    Tensor merged = mergeHeads(lastContext_, N, L, embDim_);

    // Reshape for 2D matmul
    auto dOutFR = gradOut.reshape({N * L, embDim_});
    auto mergedFR = merged.reshape({N * L, embDim_});
    if (!dOutFR.ok() || !mergedFR.ok())
        return Result<Tensor>(Error{ErrorCode::InvalidArgument, "MultiHeadAttention::backward: reshape failed (step1)"});
    const Tensor& dOutF   = dOutFR.value();
    const Tensor& mergedF = mergedFR.value();

    // dWo = mergedF^T @ dOutF   [embDim, N*L] @ [N*L, embDim] = [embDim, embDim]
    Tensor WoT = B().transpose(Wo_.tensor, 0, 1);  // [embDim, embDim] → [embDim, embDim]
    Tensor mergedFT = B().transpose(mergedF, 0, 1); // [N*L, embDim] → [embDim, N*L]
    Tensor dWo = B().matmul(mergedFT, dOutF);       // [embDim, embDim]
    Wo_.tensor.accumulateGrad(dWo);
    // dbo = sum over N*L
    Tensor dbo = Tensor::zeros({embDim_});
    { float* dbd = dbo.data(); const float* df = dOutF.data();
      for (int64_t i = 0; i < N*L; ++i) for (int64_t o = 0; o < embDim_; ++o) dbd[o] += df[i*embDim_+o]; }
    bo_.tensor.accumulateGrad(dbo);

    // dmergedF = dOutF @ Wo^T  [N*L, embDim] @ [embDim, embDim] = [N*L, embDim]
    Tensor dMergedF = B().matmul(dOutF, WoT);
    auto dMergedR = dMergedF.reshape({N, L, embDim_});
    if (!dMergedR.ok()) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "MHA::backward: reshape dMerged"});
    Tensor dMerged = dMergedR.value();

    // ── Step 2: backward through mergeHeads → dContext
    // mergeHeads maps [N*h, L, dK] → [N, L, embDim]
    // The inverse: split dMerged back into [N*h, L, dK]
    Tensor dContext = splitHeads(dMerged, h, dK_);

    // ── Step 3: backward through context = attn @ V_h
    // dAttn = dContext @ V_h^T    [N*h, L, dK] @ [N*h, dK, L] = [N*h, L, L]
    Tensor V_h  = splitHeads(lastV_, h, dK_);
    Tensor V_hT = B().transpose(V_h, 1, 2);       // [N*h, dK, L]
    Tensor dAttn = B().matmul(dContext, V_hT);     // [N*h, L, L]

    // dV_h = attn^T @ dContext   [N*h, L, L]^T @ [N*h, L, dK] = [N*h, L, dK]
    Tensor attnT = B().transpose(lastAttn_, 1, 2); // [N*h, L, L]
    Tensor dV_h  = B().matmul(attnT, dContext);    // [N*h, L, dK]

    // ── Step 4: backward through softmax
    // dScores[b,i,j] = attn[b,i,j] * (dAttn[b,i,j] - sum_k(dAttn[b,i,k]*attn[b,i,k]))
    int64_t Nh  = N * h;
    Tensor dScores = Tensor::zeros({Nh, L, L});
    { const float* atd = lastAttn_.data();
      const float* dad = dAttn.data();
      float*       dsd = dScores.data();
      for (int64_t b = 0; b < Nh; ++b) {
        for (int64_t i = 0; i < L; ++i) {
            float dot = 0.0f;
            for (int64_t j = 0; j < L; ++j) dot += dad[b*(L*L)+i*L+j] * atd[b*(L*L)+i*L+j];
            for (int64_t j = 0; j < L; ++j)
                dsd[b*(L*L)+i*L+j] = atd[b*(L*L)+i*L+j] * (dad[b*(L*L)+i*L+j] - dot);
        }
      }
    }

    // ── Step 5: backward through scores = Q_h @ K_h^T * scale
    // dQ_h = dScores @ K_h * scale          [N*h, L, L] @ [N*h, L, dK] = [N*h, L, dK]
    // dK_h = dScores^T @ Q_h * scale        [N*h, L, L]^T @ [N*h, L, dK] = [N*h, L, dK]
    Tensor Q_h = splitHeads(lastQ_, h, dK_);
    Tensor K_h = splitHeads(lastK_, h, dK_);

    Tensor dQ_h = B().matmul(dScores, K_h);            // [N*h, L, dK]
    Tensor dScoresT = B().transpose(dScores, 1, 2);
    Tensor dK_h = B().matmul(dScoresT, Q_h);           // [N*h, L, dK]
    // scale
    { float* dp = dQ_h.data(); for (int64_t i = 0; i < dQ_h.numel(); ++i) dp[i] *= scale; }
    { float* dp = dK_h.data(); for (int64_t i = 0; i < dK_h.numel(); ++i) dp[i] *= scale; }

    // ── Step 6: merge head gradients → [N, L, embDim]
    Tensor dQ = mergeHeads(dQ_h, N, L, embDim_);
    Tensor dK = mergeHeads(dK_h, N, L, embDim_);
    Tensor dV = mergeHeads(dV_h, N, L, embDim_);

    // ── Step 7: backward through Q/K/V projections → accumulate dWq, dWk, dWv, dx

    auto backProj = [&](const Tensor& dy, const Tensor& W, const Tensor& lastX,
                         Parameter& Wp, Parameter& bp) -> Tensor {
        // dy: [N, L, embDim], lastX: [N, L, embDim], W: [embDim, embDim]
        auto dyFR  = dy.reshape({N * L, embDim_});
        auto xFR   = lastX.reshape({N * L, embDim_});
        if (!dyFR.ok() || !xFR.ok()) { assert(false); return Tensor{}; }
        const Tensor& dyF = dyFR.value();
        const Tensor& xF  = xFR.value();
        // dW = xF^T @ dyF    [embDim, N*L] @ [N*L, embDim] = [embDim, embDim]
        Tensor xFT = B().transpose(xF, 0, 1);
        Tensor dW  = B().matmul(xFT, dyF);
        Wp.tensor.accumulateGrad(dW);
        // db
        Tensor db = Tensor::zeros({embDim_});
        { float* dbd = db.data(); const float* df = dyF.data();
          for (int64_t i = 0; i < N*L; ++i) for (int64_t o = 0; o < embDim_; ++o) dbd[o] += df[i*embDim_+o]; }
        bp.tensor.accumulateGrad(db);
        // dx_proj = dyF @ W^T   [N*L, embDim] @ [embDim, embDim] = [N*L, embDim]
        Tensor WT  = B().transpose(W, 0, 1);
        Tensor dxF = B().matmul(dyF, WT);
        auto dxR = dxF.reshape({N, L, embDim_});
        if (!dxR.ok()) { assert(false); return Tensor{}; }
        return dxR.value();
    };

    Tensor dxFromQ = backProj(dQ, Wq_.tensor, lastInput_, Wq_, bq_);
    Tensor dxFromK = backProj(dK, Wk_.tensor, lastInput_, Wk_, bk_);
    Tensor dxFromV = backProj(dV, Wv_.tensor, lastInput_, Wv_, bv_);

    // dx = dxFromQ + dxFromK + dxFromV
    Tensor dx = Tensor::zeros({N, L, embDim_});
    { float* d = dx.data();
      const float* q = dxFromQ.data();
      const float* k = dxFromK.data();
      const float* v = dxFromV.data();
      for (int64_t i = 0; i < dx.numel(); ++i) d[i] = q[i] + k[i] + v[i]; }

    return Result<Tensor>{std::move(dx)};
}

// ─── parameters ───────────────────────────────────────────────────────────────

std::vector<Parameter*> MultiHeadAttention::parameters() {
    return {&Wq_, &Wk_, &Wv_, &Wo_, &bq_, &bk_, &bv_, &bo_};
}
std::vector<const Parameter*> MultiHeadAttention::parameters() const {
    return {&Wq_, &Wk_, &Wv_, &Wo_, &bq_, &bk_, &bv_, &bo_};
}

// ─── config ──────────────────────────────────────────────────────────────────

std::unordered_map<std::string,std::string> MultiHeadAttention::config() const {
    return {
        {"numHeads", std::to_string(numHeads_)},
        {"embDim",   std::to_string(embDim_)},
        {"dK",       std::to_string(dK_)},
        {"causal",   causal_ ? "true" : "false"},
        {"dropout",  std::to_string(dropout_)},
    };
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
