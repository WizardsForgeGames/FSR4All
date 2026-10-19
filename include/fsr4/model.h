// FSR 4 All — Model Forward Pass
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// FSR4 U-Net INT8 inference. Channel progression: 7->16->32->64->32->16->8.
// All intermediate activations are FP16. Weight quantization is INT8 per-tensor.
#pragma once

#include <fsr4/weights.h>
#include <fsr4/profile.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fsr4 {

// Pre-allocated GPU buffers for the forward pass.
struct FSR4Buffers {
    // Intermediate activation buffers (FP16).
    // NHWC path uses 3-buffer rotation with skip aliasing:
    //   buf_a/buf_b/buf_c rotate through encoder/decoder stages
    //   skip2 aliases buf_b (enc2_mega output), skip3 aliases buf_a (enc3_mega output)
    //   bn_scratch + buf_c used for bottleneck ping-pong
    // NCHW path uses buf_a/buf_b/buf_c + separate skip buffers.
    __half* buf_a = nullptr;   // Primary working buffer / skip3 alias (NHWC)
    __half* buf_b = nullptr;   // Secondary working buffer / skip2 alias (NHWC)
    __half* buf_c = nullptr;   // Tertiary working buffer (NHWC)
    __half* skip2 = nullptr;   // Encoder2 skip connection (aliases buf_b in NHWC)
    __half* skip3 = nullptr;   // Encoder3 skip connection (aliases buf_a in NHWC)
    __half* bn_scratch = nullptr; // Bottleneck ping-pong buffer (NHWC only)

    // INT8 conv2d workspace (reused across all layers).
    void* workspace = nullptr;

    size_t buf_a_bytes = 0;
    size_t buf_b_bytes = 0;
    size_t buf_c_bytes = 0;
    size_t skip2_bytes = 0;
    size_t skip3_bytes = 0;
    size_t bn_scratch_bytes = 0;
    size_t workspace_bytes = 0;

    // Allocate all buffers for a given input resolution.
    // H_in, W_in = input resolution (e.g., 1080, 1920 for 1080p).
    bool allocate(int H_in, int W_in, bool nhwc_only = false);
    void free();
    size_t total_bytes() const;
};

// Complete FSR4 model: weights + forward pass.
struct FSR4Model {
    FSR4Weights weights;
    FSR4Buffers buffers;

    // Load weights and allocate buffers for the given input resolution.
    // nhwc_only: if true, skip NCHW workspace allocation (saves 400-900+ MB).
    //            Only forward_nhwc() will work; forward() requires nhwc_only=false.
    bool init(const std::string& weight_dir, int H_in, int W_in,
              bool nhwc_only = false);

    // Run the full U-Net forward pass (NCHW layout throughout).
    // Input:  FP16 [1, 7, H_in, W_in] on GPU (NCHW)
    // Output: FP16 [1, 8, H_in, W_in] on GPU (NCHW)
    // If profile is non-null, records per-layer CUDA event timings.
    void forward(const __half* input, __half* output,
                 int H_in, int W_in, cudaStream_t stream = nullptr,
                 FSR4Profile* profile = nullptr);

    // NHWC-optimized forward pass. Same computation, but keeps all activations
    // in NHWC [H*W, C] layout between layers, eliminating transpose overhead.
    // Uses CUDA Graph for launch overhead elimination when not profiling.
    // Input:  FP16 [H_in*W_in, 7] on GPU (NHWC)
    // Output: FP16 [H_in*W_in, 8] on GPU (NHWC)
    void forward_nhwc(const __half* input, __half* output,
                      int H_in, int W_in, cudaStream_t stream = nullptr,
                      FSR4Profile* profile = nullptr);

    void free();

private:
    // CUDA Graph state for NHWC forward pass (captured on first non-profiled call)
    cudaStream_t nhwc_capture_stream_ = nullptr;
    cudaGraphExec_t nhwc_graph_exec_ = nullptr;
    const __half* nhwc_graph_input_ = nullptr;
    __half* nhwc_graph_output_ = nullptr;
    int nhwc_graph_H_ = 0, nhwc_graph_W_ = 0;

    // Kernel dispatch for NHWC forward (called by both profiled and graph paths)
    void forward_nhwc_kernels(const __half* input, __half* output,
                               int H_in, int W_in, cudaStream_t stream,
                               FSR4Profile* prof);
};

// =========================================================================
// Helper kernels (exposed for testing)
// =========================================================================

// FP16 element-wise ReLU: out[i] = max(in[i], 0)
void fp16_relu(const __half* input, __half* output, int count,
               cudaStream_t stream = nullptr);

// FP16 element-wise add: out[i] = a[i] + b[i]
void fp16_add(const __half* a, const __half* b, __half* output, int count,
              cudaStream_t stream = nullptr);

// FP16 strided conv2d k=2 s=2 (for encoder1 FP16 weights).
// Input:  [1, C_in, H, W]
// Weight: [C_out, C_in, 2, 2]
// Bias:   [C_out]
// Output: [1, C_out, H/2, W/2]
void fp16_conv2d_k2s2(
    const __half* input, const __half* weight, const __half* bias,
    __half* output,
    int C_in, int H, int W, int C_out,
    cudaStream_t stream = nullptr);

// Copy first N channels from src to dst (for FasterNet channel split).
// src: [1, C_total, H, W], dst: [1, C_first, H, W]
void channel_slice(const __half* src, __half* dst,
                   int C_total, int C_first, int H, int W,
                   cudaStream_t stream = nullptr);

// Concatenate two tensors along channel dimension.
// a: [1, C_a, H, W], b: [1, C_b, H, W] -> out: [1, C_a+C_b, H, W]
void channel_concat(const __half* a, const __half* b, __half* output,
                    int C_a, int C_b, int H, int W,
                    cudaStream_t stream = nullptr);

// NHWC workspace size computation (used internally by pipeline/model init).
size_t fsr4_nhwc_workspace_size(const FSR4Weights& w, int H_in, int W_in);

} // namespace fsr4
