/* ============================================================================
 * Embedding.h — token embedding lookup table  [nnstudio::builtin::layers]
 * LGPL v3
 *
 * WHAT IS AN EMBEDDING LAYER?
 * ────────────────────────────
 * An Embedding maps discrete integer token IDs to dense float vectors.
 * Conceptually it is just a matrix W with shape [vocabSize, embDim]
 * where row i is the embedding of token i.
 *
 * Forward:  for each token id t in input,  output = W[t, :]
 * Backward: dW[t, :] += gradOut[:] for every position that selected token t.
 *           (Multiple positions selecting the same token accumulate gradients.)
 *
 * Shapes:
 *   Input  : [N, seqLen]  — filled with float values representing token IDs
 *                           (cast to int64 internally; fractional parts ignored)
 *   Output : [N, seqLen, embDim]
 *   Weights: [vocabSize, embDim]
 *
 * Note on integer inputs:
 *   The engine's primary dtype is float32 (Phase 1).  Integer token IDs are
 *   stored as float32 and cast to int64_t during forward/backward.
 *   A dedicated int32/int64 DType path is planned for Phase 4.
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#embedding
 * ============================================================================ */

#pragma once

#include <core/Layer.h>

namespace nnstudio {
namespace builtin {
namespace layers {

using namespace nnstudio::core;

class Embedding : public ILayer {
public:
    /**
     * @param vocabSize  Number of distinct token IDs (vocabulary size).
     * @param embDim     Dimension of each embedding vector.
     */
    explicit Embedding(int64_t vocabSize, int64_t embDim);

    std::string_view typeName() const noexcept override { return "Embedding"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#embedding";
    }

    // ── ILayer ───────────────────────────────────────────────────────────────
    Result<Shape>  build   (const Shape& inputShape) override;
    Result<Tensor> forward (const Tensor& x)         override;
    Result<Tensor> backward(const Tensor& gradOut)   override;

    std::vector<Parameter*>       parameters()       override;
    std::vector<const Parameter*> parameters() const override;

    std::unordered_map<std::string,std::string> config() const override;

    int64_t vocabSize() const noexcept { return vocabSize_; }
    int64_t embDim()    const noexcept { return embDim_;    }

private:
    int64_t   vocabSize_;
    int64_t   embDim_;
    Parameter weight_;    ///< shape [vocabSize, embDim]

    // saved for backward
    Tensor    lastIndices_;   ///< shape [N, seqLen] — float-typed integer IDs
    int64_t   lastN_{0};
    int64_t   lastSeqLen_{0};
};

} // namespace layers
} // namespace builtin
} // namespace nnstudio
