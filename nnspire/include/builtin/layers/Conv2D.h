/* ============================================================================
 * Conv2D.h — 2-D spatial convolution layer  [NNSpire::builtin::layers]
 * LGPL v3
 *
 * WHAT IS A CONV2D LAYER?
 * ────────────────────────
 * Conv2D slides a small "filter" (kernel) over an image (or feature map),
 * computing a dot product at each position.  This gives the network:
 *   • Local connectivity — each output pixel sees only a small receptive field
 *   • Weight sharing     — the same kernel is applied everywhere ⇒ far fewer params
 *   • Translation equivariance — features are detected wherever they appear
 *
 * Shapes (NCHW — batch-first, channels before spatial dims):
 *   Input  : [N, C_in,  H,    W   ]
 *   Kernel : [C_out, C_in, kH, kW ]   (outFilters, inChannels, kernelH, kernelW)
 *   Bias   : [C_out]
 *   Output : [N, C_out, outH, outW]
 *
 *   outH = (H + 2*padding - kernelH) / stride + 1
 *   outW = (W + 2*padding - kernelW) / stride + 1
 *
 * Implementation strategy (Phase 1): direct nested-loop convolution.
 * No im2col is used here — correctness is prioritised over peak throughput.
 * Backend GEMM acceleration (im2col + GEMM) is planned for Phase 4.
 *
 * Memory layout follows NCHW throughout (no format conversion).
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#conv2d
 * ============================================================================ */

#pragma once

#include <core/Layer.h>

namespace NNSpire {
namespace builtin {
namespace layers {

using namespace NNSpire::core;

// ─── Conv2DPadding ────────────────────────────────────────────────────────────
enum class Conv2DPadding { Valid, Same };

// ─── Conv2D ───────────────────────────────────────────────────────────────────
class Conv2D : public ILayer {
public:
    /**
     * @param outFilters  Number of output feature maps (= number of kernels).
     * @param kernelSize  Square kernel dimension (kernelSize × kernelSize).
     * @param stride      Step size for the sliding window.
     * @param padding     0 = Valid (no padding), or explicit pad value.
     * @param useBias     Whether to add a per-filter bias.
     */
    explicit Conv2D(int64_t outFilters,
                    int64_t kernelSize = 3,
                    int64_t stride     = 1,
                    int64_t padding    = 0,
                    bool    useBias    = true);

    std::string_view typeName() const noexcept override { return "Conv2D"; }
    std::string_view docRef()   const noexcept override {
        return "docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md#conv2d";
    }

    // ── ILayer ───────────────────────────────────────────────────────────────
    Result<Shape>  build   (const Shape& inputShape) override;
    Result<Tensor> forward (const Tensor& x, EvalTrace* trace = nullptr)         override;
    Result<Tensor> backward(const Tensor& gradOut, EvalTrace* trace = nullptr)   override;

    std::vector<Parameter*>       parameters()       override;
    std::vector<const Parameter*> parameters() const override;

    std::unordered_map<std::string,std::string> config() const override;

    // ── Accessors ─────────────────────────────────────────────────────────────
    int64_t outFilters()  const noexcept { return outFilters_; }
    int64_t kernelSize()  const noexcept { return kernelSize_; }
    int64_t stride()      const noexcept { return stride_;     }
    int64_t padding()     const noexcept { return padding_;    }
    bool    hasBias()     const noexcept { return useBias_;    }

private:
    int64_t     outFilters_;
    int64_t     kernelSize_;
    int64_t     stride_;
    int64_t     padding_;
    bool        useBias_;

    int64_t     inChannels_{0};
    int64_t     inH_{0}, inW_{0};

    Parameter   kernel_;   ///< shape [outFilters, inChannels, kernelH, kernelW]
    Parameter   bias_;     ///< shape [outFilters]

    Tensor      lastInput_;  ///< saved for backward
};

} // namespace layers
} // namespace builtin
} // namespace NNSpire
