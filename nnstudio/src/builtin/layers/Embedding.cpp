/* ============================================================================
 * Embedding.cpp — token embedding lookup implementation
 * LGPL v3
 * ============================================================================ */

#include <builtin/layers/Embedding.h>
#include <cmath>
#include <cstring>
#include <string>

namespace nnstudio {
namespace builtin {
namespace layers {

// ─── Constructor ──────────────────────────────────────────────────────────────

Embedding::Embedding(int64_t vocabSize, int64_t embDim)
    : vocabSize_(vocabSize), embDim_(embDim)
{
    weight_.name = "weight";
}

// ─── build ────────────────────────────────────────────────────────────────────

Result<Shape> Embedding::build(const Shape& inputShape) {
    // inputShape: [seqLen]  (per-sample; batch dim will be prepended by Trainer)
    // Or [N, seqLen] at runtime — we accept either and just need the embedding dim.
    if (isBuilt()) {
        auto seqLen = inputShape.back();
        return Result<Shape>{{seqLen, embDim_}};
    }

    // Xavier-like init: uniform [-limit, limit]
    float limit = std::sqrt(1.0f / static_cast<float>(vocabSize_));
    weight_.tensor = Tensor{{vocabSize_, embDim_}};
    {
        int64_t n = weight_.tensor.numel();
        float* d  = weight_.tensor.data();
        uint32_t state = 0xdeadbeef;
        for (int64_t i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            float u = static_cast<float>(state >> 16) / 65535.0f;
            d[i] = -limit + 2.0f * limit * u;
        }
    }

    markBuilt();
    // Return output shape for a single-sample input of shape [seqLen]:
    auto seqLen = inputShape.empty() ? int64_t{1} : inputShape.back();
    return Result<Shape>{{seqLen, embDim_}};
}

// ─── forward ──────────────────────────────────────────────────────────────────

Result<Tensor> Embedding::forward(const Tensor& x) {
    // x: [N, seqLen] — float values used as integer token IDs
    if (x.ndim() != 2) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Embedding::forward: input must be rank 2 [N, seqLen]"});
    if (!isBuilt())    return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Embedding::forward: call build() first"});

    int64_t N      = x.shape()[0];
    int64_t seqLen = x.shape()[1];

    lastIndices_ = x.clone();
    lastN_       = N;
    lastSeqLen_  = seqLen;

    // Output: [N, seqLen, embDim]
    Tensor out = Tensor::zeros({N, seqLen, embDim_});
    const float* xd  = x.data();
    const float* wd  = weight_.tensor.data();
    float*       od  = out.data();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t t = 0; t < seqLen; ++t) {
            int64_t idx = static_cast<int64_t>(xd[n * seqLen + t]);
            if (idx < 0 || idx >= vocabSize_) {
                return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Embedding::forward: token id out of range"});
            }
            const float* row = wd + idx * embDim_;
            float*       dst = od + (n * seqLen + t) * embDim_;
            for (int64_t d = 0; d < embDim_; ++d) {
                dst[d] = row[d];
            }
        }
    }

    return Result<Tensor>{std::move(out)};
}

// ─── backward ─────────────────────────────────────────────────────────────────

Result<Tensor> Embedding::backward(const Tensor& gradOut) {
    // gradOut: [N, seqLen, embDim]
    if (gradOut.ndim() != 3) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Embedding::backward: gradOut must be rank 3"});

    int64_t N      = lastN_;
    int64_t seqLen = lastSeqLen_;

    Tensor dW = Tensor::zeros({vocabSize_, embDim_});
    float* dwd = dW.data();
    const float* xd = lastIndices_.data();
    const float* gd = gradOut.data();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t t = 0; t < seqLen; ++t) {
            int64_t idx = static_cast<int64_t>(xd[n * seqLen + t]);
            if (idx < 0 || idx >= vocabSize_) continue;
            const float* gsrc = gd + (n * seqLen + t) * embDim_;
            float*       dst  = dwd + idx * embDim_;
            for (int64_t d = 0; d < embDim_; ++d) {
                dst[d] += gsrc[d];
            }
        }
    }

    weight_.tensor.accumulateGrad(dW);

    // Embedding has no gradient w.r.t. the integer inputs (they are discrete).
    // Return a zero tensor of appropriate shape as dX.
    return Result<Tensor>{Tensor::zeros({N, seqLen})};
}

// ─── parameters ───────────────────────────────────────────────────────────────

std::vector<Parameter*>       Embedding::parameters()       { return {&weight_}; }
std::vector<const Parameter*> Embedding::parameters() const { return {&weight_}; }

// ─── config ──────────────────────────────────────────────────────────────────

std::unordered_map<std::string,std::string> Embedding::config() const {
    return {
        {"vocabSize", std::to_string(vocabSize_)},
        {"embDim",    std::to_string(embDim_)},
    };
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
