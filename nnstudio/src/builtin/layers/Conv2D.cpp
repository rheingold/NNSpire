/* ============================================================================
 * Conv2D.cpp — 2-D spatial convolution implementation
 * LGPL v3
 *
 * Phase 1: direct nested-loop convolution (correctness > throughput).
 * Backend GEMM acceleration via im2col is planned for Phase 4.
 * ============================================================================ */

#include <builtin/layers/Conv2D.h>
#include <cmath>
#include <cstring>
#include <string>

namespace nnstudio {
namespace builtin {
namespace layers {

// ─── Constructor ──────────────────────────────────────────────────────────────

Conv2D::Conv2D(int64_t outFilters,
               int64_t kernelSize,
               int64_t stride,
               int64_t padding,
               bool    useBias)
    : outFilters_(outFilters)
    , kernelSize_(kernelSize)
    , stride_(stride)
    , padding_(padding)
    , useBias_(useBias)
{
    kernel_.name = "kernel";
    if (useBias_) bias_.name = "bias";
}

// ─── build ────────────────────────────────────────────────────────────────────

Result<Shape> Conv2D::build(const Shape& inputShape) {
    // inputShape is the per-sample shape: [C_in, H, W]
    if (inputShape.size() != 3) {
        return Result<Shape>(Error{ErrorCode::InvalidArgument, "Conv2D::build: inputShape must be rank 3 [C,H,W]"});
    }
    if (isBuilt()) {
        // Idempotent check: same shape → no-op
        if (inputShape[0] == inChannels_ &&
            inputShape[1] == inH_        &&
            inputShape[2] == inW_) {
            int64_t outH = (inH_ + 2*padding_ - kernelSize_) / stride_ + 1;
            int64_t outW = (inW_ + 2*padding_ - kernelSize_) / stride_ + 1;
            return Result<Shape>{{outFilters_, outH, outW}};
        }
    }

    inChannels_ = inputShape[0];
    inH_        = inputShape[1];
    inW_        = inputShape[2];

    int64_t outH = (inH_ + 2*padding_ - kernelSize_) / stride_ + 1;
    int64_t outW = (inW_ + 2*padding_ - kernelSize_) / stride_ + 1;

    if (outH <= 0 || outW <= 0) {
        return Result<Shape>(Error{ErrorCode::InvalidArgument, "Conv2D::build: kernel+stride incompatible with input spatial dims"});
    }

    // Glorot uniform init for kernel
    Shape kShape = {outFilters_, inChannels_, kernelSize_, kernelSize_};
    kernel_.tensor = Tensor{kShape};
    {
        float fan_in  = static_cast<float>(inChannels_ * kernelSize_ * kernelSize_);
        float fan_out = static_cast<float>(outFilters_ * kernelSize_ * kernelSize_);
        float limit   = std::sqrt(6.0f / (fan_in + fan_out));
        int64_t n     = kernel_.tensor.numel();
        float* d      = kernel_.tensor.data();
        // Simple LCG for determinism in tests (avoids <random> thread-safety issues)
        uint32_t state = 0x9e3779b9u;
        for (int64_t i = 0; i < n; ++i) {
            state = state * 1664525u + 1013904223u;
            float u = static_cast<float>(state >> 16) / 65535.0f;   // [0,1]
            d[i] = -limit + 2.0f * limit * u;
        }
    }

    if (useBias_) {
        bias_.tensor = Tensor::zeros({outFilters_});
    }

    markBuilt();
    return Result<Shape>{{outFilters_, outH, outW}};
}

// ─── forward ──────────────────────────────────────────────────────────────────

Result<Tensor> Conv2D::forward(const Tensor& x) {
    // x: [N, C_in, H, W]
    if (x.ndim() != 4) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Conv2D::forward: input must be rank 4 [N,C,H,W]"});
    if (!isBuilt())    return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Conv2D::forward: call build() first"});

    int64_t N     = x.shape()[0];
    int64_t C_in  = x.shape()[1];
    int64_t H     = x.shape()[2];
    int64_t W     = x.shape()[3];
    int64_t F     = outFilters_;
    int64_t kH    = kernelSize_;
    int64_t kW    = kernelSize_;
    int64_t outH  = (H + 2*padding_ - kH) / stride_ + 1;
    int64_t outW  = (W + 2*padding_ - kW) / stride_ + 1;

    lastInput_ = x.clone();

    Tensor out = Tensor::zeros({N, F, outH, outW});

    const float* xd  = x.data();
    const float* kd  = kernel_.tensor.data();
    float*       od  = out.data();

    // Strides for input (NCHW, contiguous)
    int64_t xsN = C_in * H * W;
    int64_t xsC = H * W;
    int64_t xsH = W;
    // Strides for kernel [F, C_in, kH, kW]
    int64_t ksF = C_in * kH * kW;
    int64_t ksC = kH * kW;
    int64_t ksH = kW;
    // Strides for output [N, F, outH, outW]
    int64_t osN = F * outH * outW;
    int64_t osF = outH * outW;
    int64_t osH = outW;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t f = 0; f < F; ++f) {
            // Add bias
            float bval = useBias_ ? bias_.tensor.flat(f) : 0.0f;
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float acc = bval;
                    for (int64_t c = 0; c < C_in; ++c) {
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t ih = oh * stride_ + kh - padding_;
                                int64_t iw = ow * stride_ + kw - padding_;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    float xv = xd[n*xsN + c*xsC + ih*xsH + iw];
                                    float kv = kd[f*ksF + c*ksC + kh*ksH + kw];
                                    acc += xv * kv;
                                }
                            }
                        }
                    }
                    od[n*osN + f*osF + oh*osH + ow] = acc;
                }
            }
        }
    }

    return Result<Tensor>{std::move(out)};
}

// ─── backward ─────────────────────────────────────────────────────────────────

Result<Tensor> Conv2D::backward(const Tensor& gradOut) {
    // gradOut: [N, F, outH, outW]
    if (gradOut.ndim() != 4) return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Conv2D::backward: gradOut must be rank 4"});
    if (!isBuilt())           return Result<Tensor>(Error{ErrorCode::InvalidArgument, "Conv2D::backward: call build() first"});

    int64_t N     = gradOut.shape()[0];
    int64_t F     = gradOut.shape()[1];
    int64_t outH  = gradOut.shape()[2];
    int64_t outW  = gradOut.shape()[3];
    int64_t C_in  = inChannels_;
    int64_t H     = inH_;
    int64_t W     = inW_;
    int64_t kH    = kernelSize_;
    int64_t kW    = kernelSize_;

    const float* grad_d = gradOut.data();
    const float* xd     = lastInput_.data();
    const float* kd     = kernel_.tensor.data();

    // Allocate gradient tensors
    Tensor dK = Tensor::zeros(kernel_.tensor.shape());
    Tensor dX = Tensor::zeros({N, C_in, H, W});

    float* dkd = dK.data();
    float* dxd = dX.data();

    // Strides
    int64_t xsN = C_in * H * W;  int64_t xsC = H * W;  int64_t xsH = W;
    int64_t ksF = C_in * kH * kW; int64_t ksC = kH * kW; int64_t ksH = kW;
    int64_t gsN = F * outH * outW; int64_t gsF = outH * outW; int64_t gsH = outW;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t f = 0; f < F; ++f) {
            for (int64_t oh = 0; oh < outH; ++oh) {
                for (int64_t ow = 0; ow < outW; ++ow) {
                    float g = grad_d[n*gsN + f*gsF + oh*gsH + ow];
                    for (int64_t c = 0; c < C_in; ++c) {
                        for (int64_t kh = 0; kh < kH; ++kh) {
                            for (int64_t kw = 0; kw < kW; ++kw) {
                                int64_t ih = oh * stride_ + kh - padding_;
                                int64_t iw = ow * stride_ + kw - padding_;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    // dK += grad * x
                                    dkd[f*ksF + c*ksC + kh*ksH + kw]
                                        += g * xd[n*xsN + c*xsC + ih*xsH + iw];
                                    // dX += grad * kernel
                                    dxd[n*xsN + c*xsC + ih*xsH + iw]
                                        += g * kd[f*ksF + c*ksC + kh*ksH + kw];
                                }
                            }
                        }
                    }
                    // dBias
                    if (useBias_) bias_.tensor.accumulateGrad(
                        Tensor::full({F}, 0.0f));  // handled below
                }
            }
        }
    }

    // Accumulate kernel gradient
    kernel_.tensor.accumulateGrad(dK);

    // Accumulate bias gradient: dBias[f] = sum_n_oh_ow gradOut[n,f,oh,ow]
    if (useBias_) {
        Tensor dB = Tensor::zeros({F});
        float* dbd = dB.data();
        for (int64_t n = 0; n < N; ++n)
            for (int64_t f = 0; f < F; ++f)
                for (int64_t oh = 0; oh < outH; ++oh)
                    for (int64_t ow = 0; ow < outW; ++ow)
                        dbd[f] += grad_d[n*gsN + f*gsF + oh*gsH + ow];
        bias_.tensor.accumulateGrad(dB);
    }

    return Result<Tensor>{std::move(dX)};
}

// ─── parameters ───────────────────────────────────────────────────────────────

std::vector<Parameter*> Conv2D::parameters() {
    if (useBias_) return {&kernel_, &bias_};
    return {&kernel_};
}
std::vector<const Parameter*> Conv2D::parameters() const {
    if (useBias_) return {&kernel_, &bias_};
    return {&kernel_};
}

// ─── config ──────────────────────────────────────────────────────────────────

std::unordered_map<std::string,std::string> Conv2D::config() const {
    return {
        {"outFilters",  std::to_string(outFilters_)},
        {"kernelSize",  std::to_string(kernelSize_)},
        {"stride",      std::to_string(stride_)},
        {"padding",     std::to_string(padding_)},
        {"useBias",     useBias_ ? "true" : "false"},
    };
}

} // namespace layers
} // namespace builtin
} // namespace nnstudio
