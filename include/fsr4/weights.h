// FSR 4 All — Weight Loader
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Loads FSR4 model weights from extracted binary files + manifest.json.
// Handles NHWC→NCHW conversion and ConvTranspose2d weight repacking.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <string>

namespace fsr4 {

// A single convolution layer's GPU-resident weights.
struct ConvLayer {
    int8_t* weight  = nullptr;   // INT8 weight on GPU [C_out, C_in*kH*kW]
    __half* bias    = nullptr;   // FP16 bias on GPU [C_out], or nullptr
    __half* w_scale = nullptr;   // FP16 [C_out] GPU array (per-tensor value broadcast)
    float quant_scale = 0.0f;   // Original per-tensor scale (host, for debugging)
    int C_in, C_out, kH, kW;
    int weight_bytes = 0;
    int bias_bytes   = 0;
};

// ConvTranspose2d layer (k=2 s=2) with repacked weights.
struct ConvTransposeLayer {
    int8_t* weight  = nullptr;   // INT8 weight on GPU [4*C_out, C_in] (repacked)
    __half* bias    = nullptr;   // FP16 bias on GPU [C_out], or nullptr
    __half* w_scale = nullptr;   // FP16 [C_out] GPU array (per-tensor value broadcast)
    float quant_scale = 0.0f;   // Original per-tensor scale (host, for debugging)
    int C_in, C_out;
    int weight_bytes = 0;
    int bias_bytes   = 0;
};

// FP16 convolution layer (used only for encoder1).
struct FP16ConvLayer {
    __half* weight = nullptr;   // FP16 weight on GPU [C_out, C_in, kH, kW]
    __half* bias   = nullptr;   // FP16 bias on GPU [C_out]
    int C_in, C_out, kH, kW;
    int weight_bytes = 0;
    int bias_bytes   = 0;
};

// ConvNextBlock: conv_3x3 → pw_expand → GELU → pw_contract + skip
struct ConvNextBlock {
    ConvLayer conv;       // 3×3, C→C
    ConvLayer pw_expand;  // 1×1, C→2C
    ConvLayer pw_contract;// 1×1, 2C→C
};

// FasterNetBlock: partial_conv_3x3(first 16ch) → cat → pw_expand → GELU → pw_contract + skip
struct FasterNetBlock {
    ConvLayer spatial;    // 3×3, 16→C_spatial_out (partial: only first 16 input channels)
    ConvLayer pw_expand;  // 1×1, C→2C
    ConvLayer pw_contract;// 1×1, 2C→C
};

// Complete FSR4 model weights.
struct FSR4Weights {
    // --- Encoder 1: FP16 strided conv (7ch→16ch, k=2 s=2) ---
    FP16ConvLayer enc1_downscale;  // Conv2d(7→16, k=2, s=2)

    // --- Encoder 2: 16ch, ConvNextBlocks + downscale ---
    ConvNextBlock enc2_res[2];
    ConvLayer     enc2_downscale;  // Conv2d(16→32, k=2, s=2)

    // --- Encoder 3: 32ch, FasterNetBlocks + downscale ---
    FasterNetBlock enc3_res[2];
    ConvLayer      enc3_downscale; // Conv2d(32→64, k=2, s=2)

    // --- Bottleneck: 64ch, FasterNetBlocks + upscale ---
    FasterNetBlock     bn_res[3];
    ConvTransposeLayer bn_upscale; // ConvTranspose2d(64→32, k=2, s=2)

    // --- Decoder 3: 32ch, FasterNetBlocks + upscale ---
    // Note: ResidualBlock indices are 1,2 (not 0,1) in manifest
    FasterNetBlock     dec3_res[2];
    ConvTransposeLayer dec3_upscale; // ConvTranspose2d(32→16, k=2, s=2)

    // --- Decoder 2: 16ch, ConvNextBlocks + upscale ---
    ConvNextBlock      dec2_res[2];
    ConvTransposeLayer dec2_upscale; // ConvTranspose2d(16→8, k=2, s=2)

    // Load from extracted weight directory (containing manifest.json + .bin files).
    // Returns true on success.
    bool load(const std::string& weight_dir);

    // Free all GPU memory.
    void free();

    // Print summary (layer counts, total bytes).
    void print_summary() const;

    // Total GPU memory used (bytes).
    size_t total_gpu_bytes() const;
};

} // namespace fsr4
