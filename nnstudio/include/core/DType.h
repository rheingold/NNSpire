#pragma once
/* ============================================================================
 * DType.h — data type enum for Tensor elements
 * LGPL v3
 *
 * Each Tensor stores one homogeneous dtype for all its elements.
 * Why these five?
 *   Float32 — primary training dtype; maximum precision; full hardware support.
 *   Float16 — inference-time compression; halves memory; needs FP16 GPU.
 *   Int8    — quantized inference; 4× smaller than Float32; some accuracy loss.
 *   Int32   — index tensors (token IDs, class labels) and reduction accumulators.
 *   Bool    — attention masks, padding masks; 1 byte per element in this impl.
 *
 * dtypeBytes(d) — byte width for memory layout / stride calculations.
 * dtypeName(d)  — stable lowercase string for serialisation (ONNX, .nns format).
 *
 * Phase 1 note: the engine computes in Float32 only.  Float16 / Int8 paths
 * are reserved for the CUDA backend (Phase 2+) and quantization export (Phase 4+).
 *
 * @kb: docs/ai-standards-kb/standards/01-Neural-Networks-Fundamentals.md
 * ============================================================================ */

#include <cstdint>
#include <string_view>

namespace nnstudio::core {

enum class DType : uint8_t {
    Float32 = 0,
    Float16 = 1,
    Int8    = 2,
    Int32   = 3,
    Bool    = 4,
};

constexpr std::string_view dtypeName(DType d) noexcept {
    switch (d) {
        case DType::Float32: return "float32";
        case DType::Float16: return "float16";
        case DType::Int8:    return "int8";
        case DType::Int32:   return "int32";
        case DType::Bool:    return "bool";
    }
    return "unknown";
}

constexpr size_t dtypeBytes(DType d) noexcept {
    switch (d) {
        case DType::Float32: return 4;
        case DType::Float16: return 2;
        case DType::Int8:    return 1;
        case DType::Int32:   return 4;
        case DType::Bool:    return 1;
    }
    return 0;
}

} // namespace nnstudio::core
