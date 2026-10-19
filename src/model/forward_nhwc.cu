// FSR 4 All — NHWC Forward Pass Implementation
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// NHWC-throughout pipeline: keeps activations in [H*W, C] layout between layers.
// Eliminates the NCHW↔NHWC transpose overhead that dominates the NCHW pipeline.
//
// Calls Forge's public low-level primitives (int8_gemm, fused_quantize_activation)
// directly, bypassing Forge's int8_conv2d_forward which enforces NCHW I/O.
//
// The fixed activation quantization scales in this file (kConvNextScales,
// kFasterNet32Scales, kFasterNet64Scales) are taken from AMD's FSR4 INT8 quality
// model HLSL (FidelityFX SDK, Copyright (C) 2025 Advanced Micro Devices, Inc.,
// MIT License -- see LICENSE).

#include <fsr4/model.h>
#include <forge/gemm/int8_gemm.cuh>
#include <forge/quant/quantize.cuh>
#include <cstdio>
#include <algorithm>

namespace fsr4 {

// =========================================================================
// NHWC helper kernels
// =========================================================================

// Fused im2col + quantize for NHWC input.
// Reads scattered NHWC input, computes per-row absmax, writes INT8 col + scales.
// Eliminates the intermediate FP16 col buffer (saves M*K*2 bytes workspace).
// One block per output row (M blocks total). Two-pass: absmax then quantize.
// L1 cache handles the re-read efficiently since K is small (144-288 for FSR4).
static __global__ void im2col_quantize_nhwc_kernel(
    const __half* __restrict__ input,    // [H*W, C_in] NHWC
    int8_t*       __restrict__ col_int8, // [M, K] INT8 output
    float*        __restrict__ scales,   // [M] per-row scales
    int C_in, int H, int W,
    int kH, int kW,
    int padH, int padW,
    int strideH, int strideW,
    int H_out, int W_out)
{
    const int K = C_in * kH * kW;
    const int row = blockIdx.x;  // one block per output spatial position
    const int h_out = row / W_out;
    const int w_out = row % W_out;

    // Pass 1: find per-row absmax via im2col reads
    float thread_max = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const int kw = k % kW;
        const int tmp = k / kW;
        const int kh = tmp % kH;
        const int c  = tmp / kH;

        const int h_in = h_out * strideH + kh - padH;
        const int w_in = w_out * strideW + kw - padW;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
            val = __half2float(input[(h_in * W + w_in) * C_in + c]);
        thread_max = fmaxf(thread_max, fabsf(val));
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));

    // Block-level reduction via shared memory
    __shared__ float warp_maxes[32];
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) warp_maxes[warp_id] = thread_max;
    __syncthreads();

    if (warp_id == 0) {
        const int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        float val = (lane_id < num_warps) ? warp_maxes[lane_id] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        if (lane_id == 0) warp_maxes[0] = val;
    }
    __syncthreads();

    const float row_max   = warp_maxes[0];
    const float scale     = fmaxf(row_max, 1e-8f) / 127.0f;
    if (threadIdx.x == 0) scales[row] = scale;
    const float inv_scale = 1.0f / scale;

    // Pass 2: re-read NHWC input, quantize, write INT8 col
    int8_t* out_row = col_int8 + (size_t)row * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const int kw = k % kW;
        const int tmp = k / kW;
        const int kh = tmp % kH;
        const int c  = tmp / kH;

        const int h_in = h_out * strideH + kh - padH;
        const int w_in = w_out * strideW + kw - padW;

        float val = 0.0f;
        if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
            val = __half2float(input[(h_in * W + w_in) * C_in + c]);

        int q = __float2int_rn(val * inv_scale);
        q = max(-128, min(127, q));
        out_row[k] = static_cast<int8_t>(q);
    }
}

// Direct 3×3 conv for small channel counts (C_in=C_out=C).
// Loads entire weight matrix into shared memory. One thread per output pixel,
// all C output channels accumulated in registers. No im2col, no quantize,
// no GEMM — single kernel replaces the entire conv pipeline.
// FP16 input × INT8 weight → FP32 accumulate → FP16 output.
template<int C>
static __global__ void direct_conv3x3_nhwc_kernel(
    const __half* __restrict__ input,    // [H*W, C] NHWC
    const int8_t* __restrict__ weight,   // [C, C*9] INT8 row-major
    const __half* __restrict__ bias,     // [C]
    const __half* __restrict__ w_scale,  // [C]
    __half*       __restrict__ output,   // [H*W, C] NHWC
    int H, int W)
{
    constexpr int K = C * 9;

    // Shared memory: weights [C * K] INT8 + bias [C] FP16 + w_scale [C] FP16
    extern __shared__ char smem_raw[];
    int8_t* w_smem = reinterpret_cast<int8_t*>(smem_raw);
    __half* b_smem = reinterpret_cast<__half*>(w_smem + C * K);
    __half* s_smem = b_smem + C;

    // Cooperative load
    for (int i = threadIdx.x; i < C * K; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H * W) return;

    const int h = pix / W;
    const int w = pix % W;

    // Accumulate all C output channels in registers
    float acc[C];
    #pragma unroll
    for (int co = 0; co < C; co++)
        acc[co] = 0.0f;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int h_in = h + kh - 1;
        if (h_in < 0 || h_in >= H) continue;
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int w_in = w + kw - 1;
            if (w_in < 0 || w_in >= W) continue;

            const int in_base = (h_in * W + w_in) * C;
            const int spatial_off = kh * 3 + kw;

            #pragma unroll
            for (int ci = 0; ci < C; ci++) {
                const float v = __half2float(input[in_base + ci]);
                #pragma unroll
                for (int co = 0; co < C; co++)
                    acc[co] += v * static_cast<float>(w_smem[co * K + ci * 9 + spatial_off]);
            }
        }
    }

    // Apply w_scale + bias, write NHWC output
    __half* out = output + pix * C;
    #pragma unroll
    for (int co = 0; co < C; co++) {
        float result = acc[co] * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        out[co] = __float2half(result);
    }
}

// Channel slice for NHWC: extract first C_first channels.
// src[H*W, C_total] → dst[H*W, C_first]
static __global__ void channel_slice_nhwc_kernel(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    int C_total, int C_first, int spatial)
{
    const int total = C_first * spatial;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        const int s = idx / C_first;
        const int c = idx % C_first;
        dst[idx] = src[s * C_total + c];
    }
}

// =========================================================================
// Direct strided conv kernels for encoder downscale layers
// =========================================================================

// Direct FP16 Conv2d k=2 s=2 (encoder1): one thread per output pixel.
// Input [H*W, 7] NHWC → Output [H/2*W/2, 16] NHWC.
// Replaces the old kernel that used one thread per (spatial × channel) pair.
// All 16 outputs accumulated in registers. Weights (896 B) in shared memory.
static __global__ void __launch_bounds__(256, 4) direct_fp16_conv2d_k2s2_enc1_kernel(
    const __half* __restrict__ input,   // [H*W, 7] NHWC
    const __half* __restrict__ weight,  // [16, 28] FP16
    const __half* __restrict__ bias,    // [16]
    __half*       __restrict__ output,  // [H_out*W_out, 16] NHWC
    int H, int W)
{
    constexpr int C_IN = 7;
    constexpr int C_OUT = 16;
    constexpr int K = C_IN * 4;  // 28

    __shared__ __half w_smem[C_OUT * K];  // 448 halfs = 896 bytes
    __shared__ __half b_smem[C_OUT];

    for (int i = threadIdx.x; i < C_OUT * K; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT)
        b_smem[threadIdx.x] = bias[threadIdx.x];
    __syncthreads();

    const int H_out = H / 2;
    const int W_out = W / 2;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int ho = pix / W_out;
    const int wo = pix % W_out;
    const int hi = ho * 2;
    const int wi = wo * 2;

    float acc[C_OUT];
    #pragma unroll
    for (int co = 0; co < C_OUT; co++)
        acc[co] = __half2float(b_smem[co]);

    // Loop over input channels, reading 2×2 window per channel
    #pragma unroll
    for (int ci = 0; ci < C_IN; ci++) {
        float i00 = __half2float(input[(hi * W + wi) * C_IN + ci]);
        float i01 = __half2float(input[(hi * W + wi + 1) * C_IN + ci]);
        float i10 = __half2float(input[((hi + 1) * W + wi) * C_IN + ci]);
        float i11 = __half2float(input[((hi + 1) * W + wi + 1) * C_IN + ci]);

        #pragma unroll
        for (int co = 0; co < C_OUT; co++) {
            const int w_base = co * K + ci * 4;
            acc[co] += i00 * __half2float(w_smem[w_base + 0])
                     + i01 * __half2float(w_smem[w_base + 1])
                     + i10 * __half2float(w_smem[w_base + 2])
                     + i11 * __half2float(w_smem[w_base + 3]);
        }
    }

    __half* out_ptr = output + pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++)
        out_ptr[co] = __float2half(acc[co]);
}

// Direct INT8 Conv2d k=2 s=2 (encoder2 downscale): one thread per output pixel.
// Input [H*W, 16] NHWC → Output [H/2*W/2, 32] NHWC.
// Non-template to avoid NVCC code gen issues. Replaces im2col+GEMM (2 kernels → 1).
// Weights (2048 B) in shared memory. Peak: 36 floats/thread.
static __global__ void __launch_bounds__(256, 4) direct_int8_conv2d_k2s2_enc2_kernel(
    const __half* __restrict__ input,    // [H*W, 16] NHWC
    const int8_t* __restrict__ weight,   // [32, 64] INT8
    const __half* __restrict__ bias,     // [32]
    const __half* __restrict__ w_scale,  // [32]
    __half*       __restrict__ output,   // [H_out*W_out, 32] NHWC
    int H, int W)
{
    constexpr int C_IN = 16;
    constexpr int C_OUT = 32;
    constexpr int K = C_IN * 4;  // 64

    __shared__ int8_t w_smem[C_OUT * K];  // 2048 bytes
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    for (int i = threadIdx.x; i < C_OUT * K; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int H_out = H / 2;
    const int W_out = W / 2;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int ho = pix / W_out;
    const int wo = pix % W_out;
    const int hi = ho * 2;
    const int wi = wo * 2;

    float acc[C_OUT];
    #pragma unroll
    for (int co = 0; co < C_OUT; co++)
        acc[co] = 0.0f;

    // Loop over input channels, reading 2×2 window per channel
    #pragma unroll
    for (int ci = 0; ci < C_IN; ci++) {
        float i00 = __half2float(input[(hi * W + wi) * C_IN + ci]);
        float i01 = __half2float(input[(hi * W + wi + 1) * C_IN + ci]);
        float i10 = __half2float(input[((hi + 1) * W + wi) * C_IN + ci]);
        float i11 = __half2float(input[((hi + 1) * W + wi + 1) * C_IN + ci]);

        #pragma unroll
        for (int co = 0; co < C_OUT; co++) {
            const int w_base = co * K + ci * 4;
            acc[co] += i00 * static_cast<float>(w_smem[w_base + 0])
                     + i01 * static_cast<float>(w_smem[w_base + 1])
                     + i10 * static_cast<float>(w_smem[w_base + 2])
                     + i11 * static_cast<float>(w_smem[w_base + 3]);
        }
    }

    __half* out_ptr = output + pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++) {
        float val = acc[co] * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        out_ptr[co] = __float2half(val);
    }
}

// Direct INT8 Conv2d k=2 s=2 (encoder3 downscale): one thread per output pixel.
// Input [H*W, 32] NHWC → Output [H/2*W/2, 64] NHWC.
// Weights (8192 B) in shared memory. Processes C_OUT in two halves of 32
// to keep register pressure manageable (~40 regs peak vs ~72 without split).
static __global__ void __launch_bounds__(256, 3) direct_int8_conv2d_k2s2_enc3_kernel(
    const __half* __restrict__ input,    // [H*W, 32] NHWC
    const int8_t* __restrict__ weight,   // [64, 128] INT8
    const __half* __restrict__ bias,     // [64]
    const __half* __restrict__ w_scale,  // [64]
    __half*       __restrict__ output,   // [H_out*W_out, 64] NHWC
    int H, int W)
{
    constexpr int C_IN = 32;
    constexpr int C_OUT = 64;
    constexpr int C_HALF = 32;
    constexpr int K = C_IN * 4;  // 128

    __shared__ int8_t w_smem[C_OUT * K];  // 8192 bytes
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    for (int i = threadIdx.x; i < C_OUT * K; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int H_out = H / 2;
    const int W_out = W / 2;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int ho = pix / W_out;
    const int wo = pix % W_out;
    const int hi = ho * 2;
    const int wi = wo * 2;

    __half* out_ptr = output + pix * C_OUT;

    // Process C_OUT in two halves to cap register pressure at ~40 regs.
    // Input reads hit L1 on second pass (same 2×2 neighborhood).
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const int co_base = half * C_HALF;
        float acc[C_HALF];
        #pragma unroll
        for (int co = 0; co < C_HALF; co++)
            acc[co] = 0.0f;

        for (int ci = 0; ci < C_IN; ci++) {
            float i00 = __half2float(input[(hi * W + wi) * C_IN + ci]);
            float i01 = __half2float(input[(hi * W + wi + 1) * C_IN + ci]);
            float i10 = __half2float(input[((hi + 1) * W + wi) * C_IN + ci]);
            float i11 = __half2float(input[((hi + 1) * W + wi + 1) * C_IN + ci]);

            #pragma unroll
            for (int co = 0; co < C_HALF; co++) {
                const int w_base = (co_base + co) * K + ci * 4;
                acc[co] += i00 * static_cast<float>(w_smem[w_base + 0])
                         + i01 * static_cast<float>(w_smem[w_base + 1])
                         + i10 * static_cast<float>(w_smem[w_base + 2])
                         + i11 * static_cast<float>(w_smem[w_base + 3]);
            }
        }

        #pragma unroll
        for (int co = 0; co < C_HALF; co++) {
            float val = acc[co] * __half2float(s_smem[co_base + co])
                      + __half2float(b_smem[co_base + co]);
            out_ptr[co_base + co] = __float2half(val);
        }
    }
}

// =========================================================================
// Elementwise helpers (layout-agnostic, reused from forward.cu)
// =========================================================================

static __global__ void relu_kernel(const __half* __restrict__ in,
                                    __half* __restrict__ out, int n) {
    const __half zero = __float2half(0.0f);
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = __hgt(in[i], zero) ? in[i] : zero;
    }
}

static __global__ void add_kernel(const __half* __restrict__ a,
                                   const __half* __restrict__ b,
                                   __half* __restrict__ out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += blockDim.x * gridDim.x) {
        out[i] = __hadd(a[i], b[i]);
    }
}

// =========================================================================
// NHWC conv output size
// =========================================================================

static inline int conv_out_size(int in_size, int pad, int dil, int kernel, int stride) {
    return (in_size + 2 * pad - dil * (kernel - 1) - 1) / stride + 1;
}

static inline int divup(int a, int b) { return (a + b - 1) / b; }

// =========================================================================
// NHWC INT8 Conv2d
// =========================================================================
//
// Direct PW (1×1, C≤64):  direct_pw_nhwc (weights in smem, ±ReLU)    [1 kernel]
// PW fallback (1×1, C>64): quantize(input) → GEMM → output          [2 kernels]
// Direct 3×3 (small C): direct_conv3x3_nhwc (weights in smem)        [1 kernel]
// Spatial (k>1 other):  fused_im2col_quantize → GEMM → output       [2 kernels]
//
// GEMM writes directly to output buffer (no staging needed — output IS NHWC).

// Launch the direct 3×3 kernel for a given channel count C.
template<int C>
static void launch_direct_conv3x3(
    const __half* input, const ConvLayer& layer, __half* output,
    int H, int W, cudaStream_t stream)
{
    constexpr int K = C * 9;
    constexpr int BLOCK = 256;
    const int M = H * W;
    // Shared memory: weights[C*K] INT8 + bias[C] FP16 + w_scale[C] FP16
    const int smem = C * K * sizeof(int8_t) + C * sizeof(__half) * 2;
    direct_conv3x3_nhwc_kernel<C><<<divup(M, BLOCK), BLOCK, smem, stream>>>(
        input, layer.weight, layer.bias, layer.w_scale, output, H, W);
}

// Direct pointwise (1×1) conv for small channel counts.
// Loads entire weight matrix into shared memory. One thread per pixel,
// all C_out outputs accumulated in registers. No quantization, no GEMM —
// single kernel replaces quantize+GEMM pipeline.
// FP16 input × INT8 weight → FP32 accumulate → FP16 output.
// Optional fused ReLU in epilogue.
template<int C_IN, int C_OUT, bool RELU>
static __global__ void direct_pw_nhwc_kernel(
    const __half* __restrict__ input,    // [M, C_IN] NHWC
    const int8_t* __restrict__ weight,   // [C_OUT, C_IN] INT8 row-major
    const __half* __restrict__ bias,     // [C_OUT]
    const __half* __restrict__ w_scale,  // [C_OUT]
    __half*       __restrict__ output,   // [M, C_OUT] NHWC
    int M)
{
    __shared__ int8_t w_smem[C_OUT * C_IN];
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    // Cooperative load
    for (int i = threadIdx.x; i < C_OUT * C_IN; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= M) return;

    // Load all input channels into registers
    const __half* in_ptr = input + (size_t)pix * C_IN;
    float in_reg[C_IN];
    #pragma unroll
    for (int ci = 0; ci < C_IN; ci++)
        in_reg[ci] = __half2float(in_ptr[ci]);

    // Matrix-vector product: acc[co] = sum_ci(in[ci] * w[co][ci])
    float acc[C_OUT];
    #pragma unroll
    for (int co = 0; co < C_OUT; co++)
        acc[co] = 0.0f;

    #pragma unroll
    for (int ci = 0; ci < C_IN; ci++) {
        const float v = in_reg[ci];
        #pragma unroll
        for (int co = 0; co < C_OUT; co++)
            acc[co] += v * static_cast<float>(w_smem[co * C_IN + ci]);
    }

    // Dequant, bias, optional ReLU, write output
    __half* out_ptr = output + (size_t)pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++) {
        float val = acc[co] * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        if constexpr (RELU) val = fmaxf(val, 0.0f);
        out_ptr[co] = __float2half(val);
    }
}

template<int C_IN, int C_OUT, bool RELU>
static void launch_direct_pw(
    const __half* input, const ConvLayer& layer, __half* output,
    int M, cudaStream_t stream)
{
    constexpr int THREADS = 256;
    direct_pw_nhwc_kernel<C_IN, C_OUT, RELU><<<divup(M, THREADS), THREADS, 0, stream>>>(
        input, layer.weight, layer.bias, layer.w_scale, output, M);
}

// =========================================================================
// Fused ConvNext block: conv3x3 → pw_expand+ReLU → pw_contract + skip_add
// =========================================================================
//
// Entire ConvNext block in a single kernel. One thread per output pixel.
// INT8 weights pre-converted to FP32 in shared memory (~13.3 KB for C=16).
// Eliminates INT8→FP32 cast from every inner loop iteration (33% fewer instructions).
// Intermediate activations stay in registers — no global memory staging.
// Conv3x3 neighborhood reads rely on L1 cache for data reuse across threads.

template<int C>
static __global__ void __launch_bounds__(256, 3) fused_convnext_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, C] NHWC (also skip source)
    const int8_t* __restrict__ conv_w,         // [C, C*9] INT8
    const __half* __restrict__ conv_b,         // [C]
    const __half* __restrict__ conv_s,         // [C]
    const int8_t* __restrict__ pw_exp_w,       // [2C, C] INT8
    const __half* __restrict__ pw_exp_b,       // [2C]
    const __half* __restrict__ pw_exp_s,       // [2C]
    const int8_t* __restrict__ pw_con_w,       // [C, 2C] INT8
    const __half* __restrict__ pw_con_b,       // [C]
    const __half* __restrict__ pw_con_s,       // [C]
    __half*       __restrict__ output,         // [H*W, C] NHWC
    int H, int W)
{
    constexpr int C2 = 2 * C;
    constexpr int K3 = C * 9;  // conv3x3 weight row length

    // Shared memory layout: FP32 weights (pre-converted from INT8) + FP16 bias/scale
    //   conv weights:  C * K3 float    (was int8)
    //   pw_exp weights: C2 * C float   (was int8)
    //   pw_con weights: C * C2 float   (was int8)
    //   bias/scale: FP16 arrays
    extern __shared__ char smem_raw[];

    float* s_conv_w = reinterpret_cast<float*>(smem_raw);
    float* s_exp_w  = s_conv_w + C * K3;
    float* s_con_w  = s_exp_w  + C2 * C;
    __half* s_conv_b = reinterpret_cast<__half*>(s_con_w + C * C2);
    __half* s_conv_s = s_conv_b + C;
    __half* s_exp_b  = s_conv_s + C;
    __half* s_exp_s  = s_exp_b  + C2;
    __half* s_con_b  = s_exp_s  + C2;
    __half* s_con_s  = s_con_b  + C;

    // Cooperative load: INT8 → FP32 conversion during load (done once, not per-pixel)
    for (int i = threadIdx.x; i < C * K3; i += blockDim.x)
        s_conv_w[i] = static_cast<float>(conv_w[i]);
    for (int i = threadIdx.x; i < C2 * C; i += blockDim.x)
        s_exp_w[i] = static_cast<float>(pw_exp_w[i]);
    for (int i = threadIdx.x; i < C * C2; i += blockDim.x)
        s_con_w[i] = static_cast<float>(pw_con_w[i]);

    // Cooperative load: bias + scale
    if (threadIdx.x < C) {
        s_conv_b[threadIdx.x] = conv_b[threadIdx.x];
        s_conv_s[threadIdx.x] = conv_s[threadIdx.x];
        s_con_b[threadIdx.x]  = pw_con_b[threadIdx.x];
        s_con_s[threadIdx.x]  = pw_con_s[threadIdx.x];
    }
    if (threadIdx.x < C2) {
        s_exp_b[threadIdx.x] = pw_exp_b[threadIdx.x];
        s_exp_s[threadIdx.x] = pw_exp_s[threadIdx.x];
    }
    __syncthreads();

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H * W) return;

    const int h = pix / W;
    const int w = pix % W;

    // ---- Stage 1: Conv 3×3 (C→C) ----
    float conv_out[C];
    #pragma unroll
    for (int co = 0; co < C; co++) conv_out[co] = 0.0f;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int h_in = h + kh - 1;
        if (h_in < 0 || h_in >= H) continue;
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int w_in = w + kw - 1;
            if (w_in < 0 || w_in >= W) continue;

            const int in_base = (h_in * W + w_in) * C;
            const int spatial_off = kh * 3 + kw;

            #pragma unroll
            for (int ci = 0; ci < C; ci++) {
                const float v = __half2float(input[in_base + ci]);
                #pragma unroll
                for (int co = 0; co < C; co++)
                    conv_out[co] += v * s_conv_w[co * K3 + ci * 9 + spatial_off];
            }
        }
    }

    // Dequant + bias for conv3x3
    #pragma unroll
    for (int co = 0; co < C; co++)
        conv_out[co] = conv_out[co] * __half2float(s_conv_s[co]) + __half2float(s_conv_b[co]);

    // ---- Stage 2: PW expand (C→2C) + ReLU ----
    float exp_out[C2];
    #pragma unroll
    for (int co = 0; co < C2; co++) {
        float acc = 0.0f;
        #pragma unroll
        for (int ci = 0; ci < C; ci++)
            acc += conv_out[ci] * s_exp_w[co * C + ci];
        float val = acc * __half2float(s_exp_s[co]) + __half2float(s_exp_b[co]);
        exp_out[co] = fmaxf(val, 0.0f);
    }

    // ---- Stage 3: PW contract (2C→C) + residual skip add ----
    const __half* skip = input + pix * C;
    __half* out = output + pix * C;
    #pragma unroll
    for (int co = 0; co < C; co++) {
        float acc = 0.0f;
        #pragma unroll
        for (int ci = 0; ci < C2; ci++)
            acc += exp_out[ci] * s_con_w[co * C2 + ci];
        float val = acc * __half2float(s_con_s[co]) + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        out[co] = __float2half(val);
    }
}

template<int C>
static void launch_fused_convnext(
    const __half* input, __half* output,
    const ConvNextBlock& block,
    int H, int W, cudaStream_t stream)
{
    constexpr int C2 = 2 * C;
    constexpr int K3 = C * 9;
    constexpr int THREADS = 256;
    const int M = H * W;
    // Shared memory: FP32 weights + FP16 bias/scale
    const int smem = (C * K3 + C2 * C + C * C2) * sizeof(float)   // FP32 weights
                   + (C + C + C2 + C2 + C + C) * sizeof(__half);   // bias + scale
    fused_convnext_nhwc_kernel<C><<<divup(M, THREADS), THREADS, smem, stream>>>(
        input,
        block.conv.weight, block.conv.bias, block.conv.w_scale,
        block.pw_expand.weight, block.pw_expand.bias, block.pw_expand.w_scale,
        block.pw_contract.weight, block.pw_contract.bias, block.pw_contract.w_scale,
        output, H, W);
}

// =========================================================================
// dp4a Fused ConvNext: INT8×INT8→INT32 via __dp4a (sm_61+)
// =========================================================================
//
// Same computation as fused_convnext_nhwc_kernel but using dp4a intrinsics
// for all three stages (conv3x3, pw_expand, pw_contract). This matches AMD's
// designed computation model (dot4add_i8packed) and reduces instruction count
// by ~4x vs FP32 scalar FMA.
//
// FP16 input → on-the-fly INT8 quantization → dp4a compute → FP16 output.
// INT8 weights stay in shared memory directly (no FP32 preconversion needed).
// Inter-stage activations requantized in registers using fixed activation scales
// extracted from AMD's quality model HLSL.

// Per-block activation quantization scales for dp4a ConvNext blocks.
struct ConvNextDP4AScales {
    float input_act_scale;     // Input activation quantization scale
    float conv_requant_rcp;    // rcprOutputScale0 (float→INT8 after conv3x3)
    float pw_exp_requant_rcp;  // rcprOutputScale1 (float→INT8 after pw_expand)
};

// Fixed activation scales from AMD FSR4 quality model.
// [0]=enc2_res[0], [1]=enc2_res[1], [2]=dec2_res[0], [3]=dec2_res[1]
static const ConvNextDP4AScales kConvNextScales[4] = {
    {0.014884727075695992f, 19.41786337677181f,  39.03760807273706f},
    {0.01933070458471775f,  29.379312231835634f, 52.675386653772975f},
    {0.027206305414438248f, 35.15770233112976f,  25.313606416359107f},
    {0.035173576325178146f, 16.584852442729886f, 22.7801944733756f},
};

template<int C>
static __global__ void __launch_bounds__(256, 4) fused_convnext_dp4a_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, C] NHWC (also skip source)
    const int8_t* __restrict__ conv_w,         // [C, C*9] INT8 (original layout)
    const __half* __restrict__ conv_b,         // [C]
    const int8_t* __restrict__ pw_exp_w,       // [2C, C] INT8
    const __half* __restrict__ pw_exp_b,       // [2C]
    const int8_t* __restrict__ pw_con_w,       // [C, 2C] INT8
    const __half* __restrict__ pw_con_b,       // [C]
    __half*       __restrict__ output,         // [H*W, C] NHWC
    int H, int W,
    float act_rcp_scale,           // 1/input_act_scale (FP16→INT8)
    float conv_dequant_scale,      // input_act_scale * conv_w_scale
    float conv_requant_scale,      // rcprOutputScale0
    float pw_exp_dequant_scale,    // (1/rcprOut0) * pw_exp_w_scale
    float pw_exp_requant_scale,    // rcprOutputScale1
    float pw_con_dequant_scale)    // (1/rcprOut1) * pw_con_w_scale
{
    constexpr int C2 = 2 * C;
    constexpr int C_G = C / 4;       // dp4a groups for C channels
    constexpr int C2_G = C2 / 4;     // dp4a groups for 2C channels
    constexpr int K3 = C * 9;

    // Shared memory: packed INT8 weights (as int for dp4a) + FP16 bias
    extern __shared__ char smem_raw[];

    int* s_conv_w  = reinterpret_cast<int*>(smem_raw);                       // [C * 9 * C_G]
    int* s_exp_w   = s_conv_w + C * 9 * C_G;                                // [C2 * C_G]
    int* s_con_w   = s_exp_w  + C2 * C_G;                                   // [C * C2_G]
    __half* s_conv_b = reinterpret_cast<__half*>(s_con_w + C * C2_G);        // [C]
    __half* s_exp_b  = s_conv_b + C;                                         // [C2]
    __half* s_con_b  = s_exp_b + C2;                                         // [C]

    // Cooperative load: repack conv3x3 weights for dp4a
    // Original: weight[co * K3 + ci * 9 + spatial_off]
    // dp4a:     packed[co * 9 * C_G + sp * C_G + ci_group]
    const int conv_total = C * 9 * C_G;
    for (int i = threadIdx.x; i < conv_total; i += blockDim.x) {
        const int ci_g = i % C_G;
        const int sp   = (i / C_G) % 9;
        const int co   = i / (C_G * 9);
        int a = (int)conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
        int b = (int)conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
        int c = (int)conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
        int d = (int)conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
        s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                      ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // Cooperative load: PW weights (C_in contiguous → direct dp4a pack)
    const int exp_total = C2 * C_G;
    for (int i = threadIdx.x; i < exp_total; i += blockDim.x) {
        const int ci_g = i % C_G;
        const int co   = i / C_G;
        const int base = co * C + ci_g * 4;
        int a = (int)pw_exp_w[base + 0];
        int b = (int)pw_exp_w[base + 1];
        int c = (int)pw_exp_w[base + 2];
        int d = (int)pw_exp_w[base + 3];
        s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }
    const int con_total = C * C2_G;
    for (int i = threadIdx.x; i < con_total; i += blockDim.x) {
        const int ci_g = i % C2_G;
        const int co   = i / C2_G;
        const int base = co * C2 + ci_g * 4;
        int a = (int)pw_con_w[base + 0];
        int b = (int)pw_con_w[base + 1];
        int c = (int)pw_con_w[base + 2];
        int d = (int)pw_con_w[base + 3];
        s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // Cooperative load: bias
    if (threadIdx.x < C) {
        s_conv_b[threadIdx.x] = conv_b[threadIdx.x];
        s_con_b[threadIdx.x]  = pw_con_b[threadIdx.x];
    }
    if (threadIdx.x < C2) {
        s_exp_b[threadIdx.x] = pw_exp_b[threadIdx.x];
    }
    __syncthreads();

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H * W) return;

    const int h = pix / W;
    const int w = pix % W;

    // ======== Stage 1: Conv 3×3 (C→C) via dp4a ========
    int conv_acc[C];
    #pragma unroll
    for (int co = 0; co < C; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int h_in = h + kh - 1;
        if (h_in < 0 || h_in >= H) continue;
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int w_in = w + kw - 1;
            if (w_in < 0 || w_in >= W) continue;

            // Quantize FP16 → INT8 inline for dp4a
            int in_packed[C_G];
            {
                const __half* in_ptr = input + (h_in * W + w_in) * C;
                #pragma unroll
                for (int cg = 0; cg < C_G; cg++) {
                    int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * act_rcp_scale);
                    int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * act_rcp_scale);
                    int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * act_rcp_scale);
                    int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * act_rcp_scale);
                    qa = max(-127, min(127, qa));
                    qb = max(-127, min(127, qb));
                    qc = max(-127, min(127, qc));
                    qd = max(-127, min(127, qd));
                    in_packed[cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                    ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
                }
            }

            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C; co++) {
                #pragma unroll
                for (int cg = 0; cg < C_G; cg++) {
                    conv_acc[co] = __dp4a(in_packed[cg],
                                          s_conv_w[co * 9 * C_G + sp * C_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // Dequant conv3x3 → float → requant to INT8 for PW expand
    int conv_q[C_G];
    #pragma unroll
    for (int cg = 0; cg < C_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)conv_acc[cg * 4 + 0] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 1] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 2] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 3] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        conv_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                     ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // ======== Stage 2: PW expand (C→2C) + ReLU via dp4a ========
    int exp_acc[C2];
    #pragma unroll
    for (int co = 0; co < C2; co++) exp_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C2; co++) {
        #pragma unroll
        for (int cg = 0; cg < C_G; cg++) {
            exp_acc[co] = __dp4a(conv_q[cg],
                                 s_exp_w[co * C_G + cg],
                                 exp_acc[co]);
        }
    }

    // Dequant + bias + ReLU → requant to INT8 for PW contract
    int exp_q[C2_G];
    #pragma unroll
    for (int cg = 0; cg < C2_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)exp_acc[cg * 4 + 0] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 1] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 2] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 3] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        exp_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                    ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // ======== Stage 3: PW contract (2C→C) + residual skip add ========
    int con_acc[C];
    #pragma unroll
    for (int co = 0; co < C; co++) con_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C; co++) {
        #pragma unroll
        for (int cg = 0; cg < C2_G; cg++) {
            con_acc[co] = __dp4a(exp_q[cg],
                                 s_con_w[co * C2_G + cg],
                                 con_acc[co]);
        }
    }

    // Dequant + bias + skip add → FP16 output
    const __half* skip = input + pix * C;
    __half* out = output + pix * C;
    #pragma unroll
    for (int co = 0; co < C; co++) {
        float val = (float)con_acc[co] * pw_con_dequant_scale
                  + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        out[co] = __float2half(val);
    }
}

template<int C>
static void launch_fused_convnext_dp4a(
    const __half* input, __half* output,
    const ConvNextBlock& block,
    const ConvNextDP4AScales& scales,
    int H, int W, cudaStream_t stream)
{
    constexpr int C2 = 2 * C;
    constexpr int C_G = C / 4;
    constexpr int C2_G = C2 / 4;
    constexpr int THREADS = 256;
    const int M = H * W;

    // Precompute 6 scalar dp4a scale parameters from per-block scales + weight scales
    float act_rcp       = 1.0f / scales.input_act_scale;
    float conv_dequant  = scales.input_act_scale * block.conv.quant_scale;
    float conv_requant  = scales.conv_requant_rcp;
    float pw_exp_in_s   = 1.0f / scales.conv_requant_rcp;
    float pw_exp_dequant= pw_exp_in_s * block.pw_expand.quant_scale;
    float pw_exp_requant= scales.pw_exp_requant_rcp;
    float pw_con_in_s   = 1.0f / scales.pw_exp_requant_rcp;
    float pw_con_dequant= pw_con_in_s * block.pw_contract.quant_scale;

    // Shared memory: packed weights (int) + bias (FP16)
    const int smem = (C * 9 * C_G + C2 * C_G + C * C2_G) * sizeof(int)
                   + (C + C2 + C) * sizeof(__half);

    fused_convnext_dp4a_nhwc_kernel<C><<<divup(M, THREADS), THREADS, smem, stream>>>(
        input,
        block.conv.weight, block.conv.bias,
        block.pw_expand.weight, block.pw_expand.bias,
        block.pw_contract.weight, block.pw_contract.bias,
        output, H, W,
        act_rcp, conv_dequant, conv_requant,
        pw_exp_dequant, pw_exp_requant, pw_con_dequant);
}

// =========================================================================
// Tiled dp4a ConvNext: shared memory tile eliminates redundant global reads
// =========================================================================
//
// The conv3x3 stage reads 9 neighbors per output pixel from global memory.
// Adjacent threads share 6/9 of these neighbors → 9× read amplification.
// Each read also requires FP16→INT8 quantization (~8 instructions per group).
//
// Solution: 2D thread block loads an (TILE_W+2) × (TILE_H+2) input tile into
// shared memory, pre-quantizing to dp4a-packed INT8 once during load. Conv3x3
// then reads entirely from smem with zero global reads and zero bounds checks.
//
// Traffic per ConvNext block (half-res, 720×1280, C=16):
//   Before: ~71 MB (9× neighbor reads + skip + write)
//   After:  ~37 MB (1.27× tile load + skip + write)

template<int C, int TILE_W = 16, int TILE_H = 16>
static __global__ void __launch_bounds__(256, 4) tiled_convnext_dp4a_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, C] NHWC (also skip source)
    const int8_t* __restrict__ conv_w,         // [C, C*9] INT8 (original layout)
    const __half* __restrict__ conv_b,         // [C]
    const int8_t* __restrict__ pw_exp_w,       // [2C, C] INT8
    const __half* __restrict__ pw_exp_b,       // [2C]
    const int8_t* __restrict__ pw_con_w,       // [C, 2C] INT8
    const __half* __restrict__ pw_con_b,       // [C]
    __half*       __restrict__ output,         // [H*W, C] NHWC
    __half*       __restrict__ skip_out,       // [H*W, C] optional dual-write for skip save
    int H, int W,
    float act_rcp_scale,           // 1/input_act_scale (FP16→INT8)
    float conv_dequant_scale,      // input_act_scale * conv_w_scale
    float conv_requant_scale,      // rcprOutputScale0
    float pw_exp_dequant_scale,    // (1/rcprOut0) * pw_exp_w_scale
    float pw_exp_requant_scale,    // rcprOutputScale1
    float pw_con_dequant_scale)    // (1/rcprOut1) * pw_con_w_scale
{
    constexpr int C2 = 2 * C;
    constexpr int C_G = C / 4;       // dp4a groups for C channels
    constexpr int C2_G = C2 / 4;     // dp4a groups for 2C channels
    constexpr int K3 = C * 9;
    constexpr int TILE_W_PAD = TILE_W + 2;  // 18
    constexpr int TILE_H_PAD = TILE_H + 2;  // 18
    constexpr int TILE_PIXELS = TILE_W_PAD * TILE_H_PAD;  // 324

    // --- Shared memory: tile + weights + bias ---
    __shared__ int s_tile[TILE_PIXELS * C_G];     // 18×18×4 int32 = 5184 B
    __shared__ int s_conv_w[C * 9 * C_G];         // 16×9×4 = 2304 B
    __shared__ int s_exp_w[C2 * C_G];             // 32×4 = 512 B
    __shared__ int s_con_w[C * C2_G];             // 16×8 = 512 B
    __shared__ __half s_conv_b[C];                // 32 B
    __shared__ __half s_exp_b[C2];                // 64 B
    __shared__ __half s_con_b[C];                 // 32 B

    const int tile_x0 = blockIdx.x * TILE_W;     // Global x origin
    const int tile_y0 = blockIdx.y * TILE_H;     // Global y origin

    // === Cooperative load: repack conv3x3 weights for dp4a ===
    const int conv_total = C * 9 * C_G;
    for (int i = threadIdx.x; i < conv_total; i += blockDim.x) {
        const int ci_g = i % C_G;
        const int sp   = (i / C_G) % 9;
        const int co   = i / (C_G * 9);
        int a = (int)conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
        int b = (int)conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
        int c = (int)conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
        int d = (int)conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
        s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                      ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // === Cooperative load: PW expand weights ===
    const int exp_total = C2 * C_G;
    for (int i = threadIdx.x; i < exp_total; i += blockDim.x) {
        const int ci_g = i % C_G;
        const int co   = i / C_G;
        const int base = co * C + ci_g * 4;
        int a = (int)pw_exp_w[base + 0];
        int b = (int)pw_exp_w[base + 1];
        int c = (int)pw_exp_w[base + 2];
        int d = (int)pw_exp_w[base + 3];
        s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // === Cooperative load: PW contract weights ===
    const int con_total = C * C2_G;
    for (int i = threadIdx.x; i < con_total; i += blockDim.x) {
        const int ci_g = i % C2_G;
        const int co   = i / C2_G;
        const int base = co * C2 + ci_g * 4;
        int a = (int)pw_con_w[base + 0];
        int b = (int)pw_con_w[base + 1];
        int c = (int)pw_con_w[base + 2];
        int d = (int)pw_con_w[base + 3];
        s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // === Cooperative load: bias ===
    if (threadIdx.x < C) {
        s_conv_b[threadIdx.x] = conv_b[threadIdx.x];
        s_con_b[threadIdx.x]  = pw_con_b[threadIdx.x];
    }
    if (threadIdx.x < C2) {
        s_exp_b[threadIdx.x] = pw_exp_b[threadIdx.x];
    }

    // === Cooperative load: input tile with FP16→INT8 pre-quantization ===
    // Tile covers global coords [tile_y0-1 .. tile_y0+TILE_H] × [tile_x0-1 .. tile_x0+TILE_W]
    for (int i = threadIdx.x; i < TILE_PIXELS; i += blockDim.x) {
        const int ty = i / TILE_W_PAD;
        const int tx = i % TILE_W_PAD;
        const int gy = tile_y0 + ty - 1;
        const int gx = tile_x0 + tx - 1;

        const int tile_base = i * C_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int cg = 0; cg < C_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * act_rcp_scale);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * act_rcp_scale);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * act_rcp_scale);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * act_rcp_scale);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            // Out-of-bounds = zero padding
            #pragma unroll
            for (int cg = 0; cg < C_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();

    // === Per-thread computation ===
    const int tx = threadIdx.x % TILE_W;
    const int ty = threadIdx.x / TILE_W;
    const int gx = tile_x0 + tx;
    const int gy = tile_y0 + ty;
    if (gx >= W || gy >= H) return;

    const int pix = gy * W + gx;

    // ======== Stage 1: Conv 3×3 (C→C) from smem tile — zero global reads ========
    int conv_acc[C];
    #pragma unroll
    for (int co = 0; co < C; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int tile_idx = ((ty + kh) * TILE_W_PAD + (tx + kw)) * C_G;
            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C; co++) {
                #pragma unroll
                for (int cg = 0; cg < C_G; cg++) {
                    conv_acc[co] = __dp4a(s_tile[tile_idx + cg],
                                          s_conv_w[co * 9 * C_G + sp * C_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // Dequant conv3x3 → float → requant to INT8 for PW expand
    int conv_q[C_G];
    #pragma unroll
    for (int cg = 0; cg < C_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)conv_acc[cg * 4 + 0] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 1] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 2] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 3] * conv_dequant_scale
                    + __half2float(s_conv_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(v * conv_requant_scale)));
        }
        conv_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                     ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // ======== Stage 2: PW expand (C→2C) + ReLU via dp4a ========
    int exp_acc[C2];
    #pragma unroll
    for (int co = 0; co < C2; co++) exp_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C2; co++) {
        #pragma unroll
        for (int cg = 0; cg < C_G; cg++) {
            exp_acc[co] = __dp4a(conv_q[cg],
                                 s_exp_w[co * C_G + cg],
                                 exp_acc[co]);
        }
    }

    // Dequant + bias + ReLU → requant to INT8 for PW contract
    int exp_q[C2_G];
    #pragma unroll
    for (int cg = 0; cg < C2_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)exp_acc[cg * 4 + 0] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 1] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 2] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 3] * pw_exp_dequant_scale
                    + __half2float(s_exp_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant_scale)));
        }
        exp_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                    ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // ======== Stage 3: PW contract (2C→C) + residual skip add ========
    int con_acc[C];
    #pragma unroll
    for (int co = 0; co < C; co++) con_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C; co++) {
        #pragma unroll
        for (int cg = 0; cg < C2_G; cg++) {
            con_acc[co] = __dp4a(exp_q[cg],
                                 s_con_w[co * C2_G + cg],
                                 con_acc[co]);
        }
    }

    // Dequant + bias + skip add → FP16 output
    // Skip = input center pixel (likely L1-cached from tile load)
    const __half* skip = input + pix * C;
    __half* out = output + pix * C;
    #pragma unroll
    for (int co = 0; co < C; co++) {
        float val = (float)con_acc[co] * pw_con_dequant_scale
                  + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        __half result = __float2half(val);
        out[co] = result;
        if (skip_out) skip_out[pix * C + co] = result;
    }
}

template<int C>
static void launch_tiled_convnext_dp4a(
    const __half* input, __half* output,
    const ConvNextBlock& block,
    const ConvNextDP4AScales& scales,
    int H, int W, cudaStream_t stream,
    __half* skip_out = nullptr)
{
    constexpr int TILE_W = 16, TILE_H = 16;
    constexpr int THREADS = TILE_W * TILE_H;  // 256

    float act_rcp       = 1.0f / scales.input_act_scale;
    float conv_dequant  = scales.input_act_scale * block.conv.quant_scale;
    float conv_requant  = scales.conv_requant_rcp;
    float pw_exp_in_s   = 1.0f / scales.conv_requant_rcp;
    float pw_exp_dequant= pw_exp_in_s * block.pw_expand.quant_scale;
    float pw_exp_requant= scales.pw_exp_requant_rcp;
    float pw_con_in_s   = 1.0f / scales.pw_exp_requant_rcp;
    float pw_con_dequant= pw_con_in_s * block.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    tiled_convnext_dp4a_nhwc_kernel<C, TILE_W, TILE_H><<<grid, THREADS, 0, stream>>>(
        input,
        block.conv.weight, block.conv.bias,
        block.pw_expand.weight, block.pw_expand.bias,
        block.pw_contract.weight, block.pw_contract.bias,
        output, skip_out, H, W,
        act_rcp, conv_dequant, conv_requant,
        pw_exp_dequant, pw_exp_requant, pw_con_dequant);
}

static void run_int8_conv2d_nhwc(
    const __half* input,        // [H*W, C_in] NHWC
    const ConvLayer& layer,
    __half* output,             // [H_out*W_out, C_out] NHWC
    int H, int W,
    void* workspace,
    cudaStream_t stream,
    bool relu = false)
{
    const int padH = (layer.kH == 3) ? 1 : 0;
    const int padW = (layer.kW == 3) ? 1 : 0;
    const int strH = (layer.kH == 2) ? 2 : 1;
    const int strW = (layer.kW == 2) ? 2 : 1;
    const int H_out = conv_out_size(H, padH, 1, layer.kH, strH);
    const int W_out = conv_out_size(W, padW, 1, layer.kW, strW);
    const int M = H_out * W_out;
    const int K = layer.C_in * layer.kH * layer.kW;
    constexpr int BLOCK = 256;

    const bool pointwise = (layer.kH == 1 && layer.kW == 1 && strH == 1 && strW == 1);

    // Direct 3×3: single kernel, weights in shared memory.
    // Used for same-padding stride-1 convs with small C_in=C_out.
    const bool direct_3x3 = (layer.kH == 3 && layer.kW == 3 && strH == 1
                             && layer.C_in == layer.C_out
                             && (layer.C_in == 16 || layer.C_in == 32));

    if (direct_3x3) {
        if (layer.C_in == 16)
            launch_direct_conv3x3<16>(input, layer, output, H, W, stream);
        else
            launch_direct_conv3x3<32>(input, layer, output, H, W, stream);
    } else if (pointwise) {
        // Try direct pointwise kernel for small channel counts (≤64).
        // Single kernel: FP16 input × INT8 weight → FP16 output.
        // Eliminates quantize + GEMM (2 kernels → 1), fuses optional ReLU.
        const int Ci = layer.C_in, Co = layer.C_out;
        bool used_direct = false;

        // Direct PW limited to ≤32 channels (≤48 float registers per thread).
        // 32→64 and 64→32 have too much register pressure → use GEMM fallback.
        if (Ci <= 32 && Co <= 32) {
            #define DISPATCH_PW(CI, CO) do { \
                if (relu) launch_direct_pw<CI, CO, true>(input, layer, output, M, stream); \
                else      launch_direct_pw<CI, CO, false>(input, layer, output, M, stream); \
                used_direct = true; \
            } while(0)

            if      (Ci == 16 && Co == 32) DISPATCH_PW(16, 32);
            else if (Ci == 32 && Co == 16) DISPATCH_PW(32, 16);

            #undef DISPATCH_PW
        }

        if (!used_direct) {
            // Fallback: quantize + GEMM for larger channels
            int8_t* col_int8  = reinterpret_cast<int8_t*>(workspace);
            float*  act_scale = reinterpret_cast<float*>(col_int8 + (size_t)M * K);

            forge::quant::fused_quantize_activation(
                input, col_int8, act_scale, M, K, stream);

            forge::gemm::int8_gemm(
                col_int8, layer.weight, act_scale, layer.w_scale, layer.bias,
                output, M, K, layer.C_out, stream);

            // Fuse ReLU after GEMM if requested
            if (relu) {
                const int total = M * layer.C_out;
                relu_kernel<<<divup(total, BLOCK), BLOCK, 0, stream>>>(
                    output, output, total);
            }
        }
    } else {
        // Spatial conv: fused im2col+quantize → GEMM  [2 kernels]
        // Workspace: col_int8[M, K] + act_scale[M]  (no col_fp16 needed)
        int8_t* col_int8  = reinterpret_cast<int8_t*>(workspace);
        float*  act_scale = reinterpret_cast<float*>(col_int8 + (size_t)M * K);

        int threads = std::min(BLOCK, ((K + 31) / 32) * 32);
        if (threads < 32) threads = 32;
        im2col_quantize_nhwc_kernel<<<M, threads, 0, stream>>>(
            input, col_int8, act_scale,
            layer.C_in, H, W,
            layer.kH, layer.kW, padH, padW, strH, strW,
            H_out, W_out);

        forge::gemm::int8_gemm(
            col_int8, layer.weight, act_scale, layer.w_scale, layer.bias,
            output, M, K, layer.C_out, stream);
    }
}

// =========================================================================
// NHWC INT8 ConvTranspose2d k=2 s=2
// =========================================================================

// Direct ConvTranspose kernel for small channel counts.
// One thread per output spatial pixel, computes all C_OUT channels.
// Weight [4*C_OUT, C_IN] in shared memory. Single kernel replaces
// replicate_scale + quantize + GEMM + scatter (4 kernels → 1).
template<int C_IN, int C_OUT>
static __global__ void __launch_bounds__(256, 4) direct_conv_transpose_nhwc_kernel(
    const __half* __restrict__ input,    // [H_in*W_in, C_IN] NHWC
    const int8_t* __restrict__ weight,   // [4*C_OUT, C_IN] INT8 repacked
    const __half* __restrict__ bias,     // [C_OUT]
    const __half* __restrict__ w_scale,  // [C_OUT]
    __half*       __restrict__ output,   // [2*H_in*2*W_in, C_OUT] NHWC
    int H_in, int W_in)
{
    constexpr int W_TOTAL = 4 * C_OUT * C_IN;

    // Shared memory: weights + bias + scale
    __shared__ int8_t w_smem[W_TOTAL];
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    // Cooperative load
    for (int i = threadIdx.x; i < W_TOTAL; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int H_out = 2 * H_in;
    const int W_out = 2 * W_in;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int h_out = pix / W_out;
    const int w_out = pix % W_out;

    // Source pixel and sub-pixel position
    const int h_src = h_out / 2;
    const int w_src = w_out / 2;
    const int sub = (h_out % 2) * 2 + (w_out % 2);

    // Load input channels from source pixel
    const __half* in_ptr = input + (h_src * W_in + w_src) * C_IN;
    float in_reg[C_IN];
    #pragma unroll
    for (int ci = 0; ci < C_IN; ci++)
        in_reg[ci] = __half2float(in_ptr[ci]);

    // Compute each output channel
    __half* out_ptr = output + pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++) {
        float acc = 0.0f;
        const int w_row = (co * 4 + sub) * C_IN;
        #pragma unroll
        for (int ci = 0; ci < C_IN; ci++)
            acc += in_reg[ci] * static_cast<float>(w_smem[w_row + ci]);
        float val = acc * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        out_ptr[co] = __float2half(val);
    }
}

// Optimized ConvTranspose for large C_IN (e.g. 64→32).
// Processes input channels in chunks of C_CHUNK to reduce register pressure.
// Input reads hit L1 cache since adjacent output pixels share source pixels.
template<int C_IN, int C_OUT, int C_CHUNK = 16>
static __global__ void __launch_bounds__(256, 3)
direct_conv_transpose_chunked_nhwc_kernel(
    const __half* __restrict__ input,    // [H_in*W_in, C_IN] NHWC
    const int8_t* __restrict__ weight,   // [4*C_OUT, C_IN] INT8
    const __half* __restrict__ bias,     // [C_OUT]
    const __half* __restrict__ w_scale,  // [C_OUT]
    __half*       __restrict__ output,   // [2*H_in*2*W_in, C_OUT] NHWC
    int H_in, int W_in)
{
    constexpr int W_TOTAL = 4 * C_OUT * C_IN;
    constexpr int N_CHUNKS = C_IN / C_CHUNK;

    __shared__ int8_t w_smem[W_TOTAL];
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    for (int i = threadIdx.x; i < W_TOTAL; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int H_out = 2 * H_in;
    const int W_out = 2 * W_in;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int h_out = pix / W_out;
    const int w_out = pix % W_out;
    const int h_src = h_out / 2;
    const int w_src = w_out / 2;
    const int sub = (h_out % 2) * 2 + (w_out % 2);

    const __half* in_ptr = input + (h_src * W_in + w_src) * C_IN;

    // Accumulate outputs in chunks of C_CHUNK input channels
    float acc[C_OUT];
    #pragma unroll
    for (int co = 0; co < C_OUT; co++)
        acc[co] = 0.0f;

    #pragma unroll
    for (int chunk = 0; chunk < N_CHUNKS; chunk++) {
        float in_chunk[C_CHUNK];
        #pragma unroll
        for (int ci = 0; ci < C_CHUNK; ci++)
            in_chunk[ci] = __half2float(in_ptr[chunk * C_CHUNK + ci]);

        #pragma unroll
        for (int co = 0; co < C_OUT; co++) {
            const int w_row = (co * 4 + sub) * C_IN + chunk * C_CHUNK;
            float sum = 0.0f;
            #pragma unroll
            for (int ci = 0; ci < C_CHUNK; ci++)
                sum += in_chunk[ci] * static_cast<float>(w_smem[w_row + ci]);
            acc[co] += sum;
        }
    }

    __half* out_ptr = output + pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++) {
        float val = acc[co] * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        out_ptr[co] = __float2half(val);
    }
}

// Fused ConvTranspose + skip-add: output[i] = conv_transpose(input)[i] + skip[i]
// Eliminates a full read-modify-write round-trip vs separate add_kernel.
template<int C_IN, int C_OUT>
static __global__ void __launch_bounds__(256, 4) direct_conv_transpose_skip_add_nhwc_kernel(
    const __half* __restrict__ input,    // [H_in*W_in, C_IN] NHWC
    const int8_t* __restrict__ weight,   // [4*C_OUT, C_IN] INT8 repacked
    const __half* __restrict__ bias,     // [C_OUT]
    const __half* __restrict__ w_scale,  // [C_OUT]
    const __half* __restrict__ skip,     // [2*H_in*2*W_in, C_OUT] NHWC
    __half*       __restrict__ output,   // [2*H_in*2*W_in, C_OUT] NHWC
    int H_in, int W_in)
{
    constexpr int W_TOTAL = 4 * C_OUT * C_IN;

    __shared__ int8_t w_smem[W_TOTAL];
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    for (int i = threadIdx.x; i < W_TOTAL; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int H_out = 2 * H_in;
    const int W_out = 2 * W_in;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int h_out = pix / W_out;
    const int w_out = pix % W_out;
    const int h_src = h_out / 2;
    const int w_src = w_out / 2;
    const int sub = (h_out % 2) * 2 + (w_out % 2);

    const __half* in_ptr = input + (h_src * W_in + w_src) * C_IN;
    float in_reg[C_IN];
    #pragma unroll
    for (int ci = 0; ci < C_IN; ci++)
        in_reg[ci] = __half2float(in_ptr[ci]);

    const __half* skip_ptr = skip + pix * C_OUT;
    __half* out_ptr = output + pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++) {
        float acc = 0.0f;
        const int w_row = (co * 4 + sub) * C_IN;
        #pragma unroll
        for (int ci = 0; ci < C_IN; ci++)
            acc += in_reg[ci] * static_cast<float>(w_smem[w_row + ci]);
        float val = acc * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        out_ptr[co] = __float2half(val + __half2float(skip_ptr[co]));
    }
}

// Fused chunked ConvTranspose + skip-add (for large C_IN >= 64).
template<int C_IN, int C_OUT, int C_CHUNK = 16>
static __global__ void __launch_bounds__(256, 3)
direct_conv_transpose_chunked_skip_add_nhwc_kernel(
    const __half* __restrict__ input,    // [H_in*W_in, C_IN] NHWC
    const int8_t* __restrict__ weight,   // [4*C_OUT, C_IN] INT8
    const __half* __restrict__ bias,     // [C_OUT]
    const __half* __restrict__ w_scale,  // [C_OUT]
    const __half* __restrict__ skip,     // [2*H_in*2*W_in, C_OUT] NHWC
    __half*       __restrict__ output,   // [2*H_in*2*W_in, C_OUT] NHWC
    int H_in, int W_in)
{
    constexpr int W_TOTAL = 4 * C_OUT * C_IN;
    constexpr int N_CHUNKS = C_IN / C_CHUNK;

    __shared__ int8_t w_smem[W_TOTAL];
    __shared__ __half b_smem[C_OUT];
    __shared__ __half s_smem[C_OUT];

    for (int i = threadIdx.x; i < W_TOTAL; i += blockDim.x)
        w_smem[i] = weight[i];
    if (threadIdx.x < C_OUT) {
        b_smem[threadIdx.x] = bias[threadIdx.x];
        s_smem[threadIdx.x] = w_scale[threadIdx.x];
    }
    __syncthreads();

    const int H_out = 2 * H_in;
    const int W_out = 2 * W_in;
    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H_out * W_out) return;

    const int h_out = pix / W_out;
    const int w_out = pix % W_out;
    const int h_src = h_out / 2;
    const int w_src = w_out / 2;
    const int sub = (h_out % 2) * 2 + (w_out % 2);

    const __half* in_ptr = input + (h_src * W_in + w_src) * C_IN;

    float acc[C_OUT];
    #pragma unroll
    for (int co = 0; co < C_OUT; co++)
        acc[co] = 0.0f;

    #pragma unroll
    for (int chunk = 0; chunk < N_CHUNKS; chunk++) {
        float in_chunk[C_CHUNK];
        #pragma unroll
        for (int ci = 0; ci < C_CHUNK; ci++)
            in_chunk[ci] = __half2float(in_ptr[chunk * C_CHUNK + ci]);

        #pragma unroll
        for (int co = 0; co < C_OUT; co++) {
            const int w_row = (co * 4 + sub) * C_IN + chunk * C_CHUNK;
            float sum = 0.0f;
            #pragma unroll
            for (int ci = 0; ci < C_CHUNK; ci++)
                sum += in_chunk[ci] * static_cast<float>(w_smem[w_row + ci]);
            acc[co] += sum;
        }
    }

    const __half* skip_ptr = skip + pix * C_OUT;
    __half* out_ptr = output + pix * C_OUT;
    #pragma unroll
    for (int co = 0; co < C_OUT; co++) {
        float val = acc[co] * __half2float(s_smem[co]) + __half2float(b_smem[co]);
        out_ptr[co] = __float2half(val + __half2float(skip_ptr[co]));
    }
}

template<int C_IN, int C_OUT>
static void launch_direct_conv_transpose(
    const __half* input, const ConvTransposeLayer& layer, __half* output,
    int H, int W, cudaStream_t stream)
{
    constexpr int THREADS = 256;
    const int M_out = 4 * H * W;  // output spatial = 2H * 2W

    if constexpr (C_IN >= 64) {
        direct_conv_transpose_chunked_nhwc_kernel<C_IN, C_OUT>
            <<<divup(M_out, THREADS), THREADS, 0, stream>>>(
            input, layer.weight, layer.bias, layer.w_scale, output, H, W);
    } else {
        direct_conv_transpose_nhwc_kernel<C_IN, C_OUT>
            <<<divup(M_out, THREADS), THREADS, 0, stream>>>(
            input, layer.weight, layer.bias, layer.w_scale, output, H, W);
    }
}

template<int C_IN, int C_OUT>
static void launch_direct_conv_transpose_skip_add(
    const __half* input, const ConvTransposeLayer& layer,
    const __half* skip, __half* output,
    int H, int W, cudaStream_t stream)
{
    constexpr int THREADS = 256;
    const int M_out = 4 * H * W;

    if constexpr (C_IN >= 64) {
        direct_conv_transpose_chunked_skip_add_nhwc_kernel<C_IN, C_OUT>
            <<<divup(M_out, THREADS), THREADS, 0, stream>>>(
            input, layer.weight, layer.bias, layer.w_scale, skip, output, H, W);
    } else {
        direct_conv_transpose_skip_add_nhwc_kernel<C_IN, C_OUT>
            <<<divup(M_out, THREADS), THREADS, 0, stream>>>(
            input, layer.weight, layer.bias, layer.w_scale, skip, output, H, W);
    }
}

// =========================================================================
// NHWC block forward passes
// =========================================================================

// ConvNextBlock NHWC: conv3x3 → pw_expand → ReLU → pw_contract + skip
// All data in NHWC [H*W, C] format. C=16 only (all FSR4 ConvNext blocks).
// Fully fused dp4a single-kernel path (4 kernels → 1).
static void convnext_forward_nhwc(
    const __half* input, __half* output,
    const ConvNextBlock& block,
    const ConvNextDP4AScales& scales,
    int H, int W,
    cudaStream_t stream,
    __half* skip_out = nullptr)
{
    launch_tiled_convnext_dp4a<16>(input, output, block, scales, H, W, stream, skip_out);
}

// Fused concat: first C_a channels from a[H*W, C_a], remaining C_b from
// b[H*W, C_total] starting at column offset C_offset.
static __global__ void channel_concat_nhwc_strided_kernel(
    const __half* __restrict__ a,     // [spatial, C_a] contiguous
    const __half* __restrict__ b,     // [spatial, C_total] strided source
    __half*       __restrict__ out,   // [spatial, C_a + C_b]
    int C_a, int C_b, int C_total_b, int C_offset_b, int spatial)
{
    const int C_out = C_a + C_b;
    const int total = C_out * spatial;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += blockDim.x * gridDim.x)
    {
        const int s = idx / C_out;
        const int c = idx % C_out;
        if (c < C_a)
            out[idx] = a[s * C_a + c];
        else
            out[idx] = b[s * C_total_b + C_offset_b + (c - C_a)];
    }
}

// =========================================================================
// dp4a Fused FasterNet 32ch: INT8×INT8→INT32 via __dp4a (sm_61+)
// =========================================================================
//
// Fused FasterNet 32ch block using dp4a intrinsics (replaces FP32 version).
// FasterNet partial-conv structure:
//   1. Conv3×3 on lower 16ch → 16ch output (dp4a, requant to match passthrough scale)
//   2. Quantize upper 16ch passthrough to same scale → 32ch INT8 concat
//   3. PW expand (32→64) + ReLU via dp4a
//   4. PW contract (64→32) + skip add via dp4a
//
// Two input quantization scales (quantScale0 for lower half, quantScale1 for upper half)
// reflect AMD's split-channel quantization scheme. After conv+concat, both halves are
// at quantScale1 for uniform PW expand input.

struct FasterNetDP4AScales {
    float quantScale0;           // Input scale for lower half (conv partial channels)
    float quantScale1;           // Spatial conv output / passthrough quantization target
    float activationScaleFactor; // PW expand output scale
};

// Fixed scales from AMD FSR4 quality model.
// [0]=enc3_res[0], [1]=enc3_res[1], [2]=dec3_res[0], [3]=dec3_res[1]
static const FasterNetDP4AScales kFasterNet32Scales[4] = {
    {0.015390855260193348f, 0.018884973600506783f, 0.014567121863365173f},
    {0.017191138118505478f, 0.021955177187919617f, 0.01790967583656311f},
    {0.021412165835499763f, 0.03075222671031952f,  0.022227706387639046f},
    {0.0215474721044302f,   0.02673504687845707f,  0.020062033087015152f},
};

static __global__ void __launch_bounds__(128, 4) fused_fasternet32_dp4a_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, 32] NHWC
    const int8_t* __restrict__ conv_w,         // [16, 16*9] INT8 (spatial conv)
    const __half* __restrict__ conv_b,         // [16]
    const int8_t* __restrict__ pw_exp_w,       // [64, 32] INT8
    const __half* __restrict__ pw_exp_b,       // [64]
    const int8_t* __restrict__ pw_con_w,       // [32, 64] INT8
    const __half* __restrict__ pw_con_b,       // [32]
    __half*       __restrict__ output,         // [H*W, 32] NHWC
    int H, int W,
    float conv_act_rcp,          // 1/quantScale0
    float conv_dequant,          // quantScale0 * spatial_w_scale
    float concat_rcp,            // 1/quantScale1 (requant conv + quantize passthrough)
    float pw_exp_dequant,        // quantScale1 * pw_expand_w_scale
    float pw_exp_requant,        // 1/activationScaleFactor
    float pw_con_dequant)        // activationScaleFactor * pw_contract_w_scale
{
    constexpr int C_PART = 16;
    constexpr int C_STAGE = 32;
    constexpr int C_EXP = 64;
    constexpr int K3 = C_PART * 9;  // 144

    constexpr int PART_G = C_PART / 4;     // 4 dp4a groups for 16ch
    constexpr int STAGE_G = C_STAGE / 4;   // 8 dp4a groups for 32ch
    constexpr int EXP_G = C_EXP / 4;       // 16 dp4a groups for 64ch

    // Shared memory: dp4a-packed weights + FP16 bias
    extern __shared__ char smem_raw[];

    int* s_conv_w  = reinterpret_cast<int*>(smem_raw);                       // [16, 9, 4]
    int* s_exp_w   = s_conv_w + C_PART * 9 * PART_G;                        // [64, 8]
    int* s_con_w   = s_exp_w  + C_EXP * STAGE_G;                            // [32, 16]
    __half* s_conv_b = reinterpret_cast<__half*>(s_con_w + C_STAGE * EXP_G); // [16]
    __half* s_exp_b  = s_conv_b + C_PART;                                    // [64]
    __half* s_con_b  = s_exp_b + C_EXP;                                      // [32]

    // Cooperative load: repack conv3x3 weights for dp4a (same as ConvNext)
    const int conv_total = C_PART * 9 * PART_G;  // 576
    for (int i = threadIdx.x; i < conv_total; i += blockDim.x) {
        const int ci_g = i % PART_G;
        const int sp   = (i / PART_G) % 9;
        const int co   = i / (PART_G * 9);
        int a = (int)conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
        int b = (int)conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
        int c = (int)conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
        int d = (int)conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
        s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                      ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // PW expand: [64, 32] → 64 × 8 packed
    const int exp_total = C_EXP * STAGE_G;  // 512
    for (int i = threadIdx.x; i < exp_total; i += blockDim.x) {
        const int ci_g = i % STAGE_G;
        const int co   = i / STAGE_G;
        const int base = co * C_STAGE + ci_g * 4;
        int a = (int)pw_exp_w[base + 0];
        int b = (int)pw_exp_w[base + 1];
        int c = (int)pw_exp_w[base + 2];
        int d = (int)pw_exp_w[base + 3];
        s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // PW contract: [32, 64] → 32 × 16 packed
    const int con_total = C_STAGE * EXP_G;  // 512
    for (int i = threadIdx.x; i < con_total; i += blockDim.x) {
        const int ci_g = i % EXP_G;
        const int co   = i / EXP_G;
        const int base = co * C_EXP + ci_g * 4;
        int a = (int)pw_con_w[base + 0];
        int b = (int)pw_con_w[base + 1];
        int c = (int)pw_con_w[base + 2];
        int d = (int)pw_con_w[base + 3];
        s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // Bias
    if (threadIdx.x < C_PART) s_conv_b[threadIdx.x] = conv_b[threadIdx.x];
    if (threadIdx.x < C_EXP)  s_exp_b[threadIdx.x]  = pw_exp_b[threadIdx.x];
    if (threadIdx.x < C_STAGE) s_con_b[threadIdx.x] = pw_con_b[threadIdx.x];
    __syncthreads();

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H * W) return;

    const int h = pix / W;
    const int w = pix % W;

    // ======== Stage 1: Conv 3×3 on lower 16ch (16→16) via dp4a ========
    int conv_acc[C_PART];
    #pragma unroll
    for (int co = 0; co < C_PART; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int h_in = h + kh - 1;
        if (h_in < 0 || h_in >= H) continue;
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int w_in = w + kw - 1;
            if (w_in < 0 || w_in >= W) continue;

            // Load ONLY first 16 FP16 channels, quantize with conv_act_rcp
            const __half* in_ptr = input + (h_in * W + w_in) * C_STAGE;
            int in_packed[PART_G];
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * conv_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * conv_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * conv_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * conv_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                in_packed[cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }

            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C_PART; co++) {
                #pragma unroll
                for (int cg = 0; cg < PART_G; cg++) {
                    conv_acc[co] = __dp4a(in_packed[cg],
                                          s_conv_w[co * 9 * PART_G + sp * PART_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // ======== Stage 2: Build 32ch INT8 concat for PW expand ========
    // Conv output: dequant → float → requant to quantScale1
    // Passthrough upper 16ch: quantize to quantScale1
    // Both halves at same scale → uniform PW expand input
    int concat_q[STAGE_G];  // 8 packed uint32 = 32 INT8 channels

    // Lower 16ch: conv output → requant
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)conv_acc[cg * 4 + 0] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 1] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 2] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 3] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        concat_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // Upper 16ch: passthrough from center pixel, quantize to same scale
    const __half* center = input + pix * C_STAGE + C_PART;  // channels 16..31
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int qa = __float2int_rn(__half2float(center[cg * 4 + 0]) * concat_rcp);
        int qb = __float2int_rn(__half2float(center[cg * 4 + 1]) * concat_rcp);
        int qc = __float2int_rn(__half2float(center[cg * 4 + 2]) * concat_rcp);
        int qd = __float2int_rn(__half2float(center[cg * 4 + 3]) * concat_rcp);
        qa = max(-127, min(127, qa));
        qb = max(-127, min(127, qb));
        qc = max(-127, min(127, qc));
        qd = max(-127, min(127, qd));
        concat_q[PART_G + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                 ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
    }

    // ======== Stage 3: PW expand (32→64) + ReLU via dp4a ========
    int exp_acc[C_EXP];
    #pragma unroll
    for (int co = 0; co < C_EXP; co++) exp_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C_EXP; co++) {
        #pragma unroll
        for (int cg = 0; cg < STAGE_G; cg++) {
            exp_acc[co] = __dp4a(concat_q[cg],
                                 s_exp_w[co * STAGE_G + cg],
                                 exp_acc[co]);
        }
    }

    // Dequant + bias + ReLU → requant to INT8
    int exp_q[EXP_G];  // 16 packed uint32 = 64 INT8 channels
    #pragma unroll
    for (int cg = 0; cg < EXP_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)exp_acc[cg * 4 + 0] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 1] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 2] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 3] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        exp_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                    ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // ======== Stage 4: PW contract (64→32) + residual skip add ========
    int con_acc[C_STAGE];
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) con_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        #pragma unroll
        for (int cg = 0; cg < EXP_G; cg++) {
            con_acc[co] = __dp4a(exp_q[cg],
                                 s_con_w[co * EXP_G + cg],
                                 con_acc[co]);
        }
    }

    // Dequant + bias + skip add → FP16 output
    const __half* skip = input + pix * C_STAGE;
    __half* out = output + pix * C_STAGE;
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        float val = (float)con_acc[co] * pw_con_dequant
                  + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        out[co] = __float2half(val);
    }
}

static void launch_fused_fasternet32_dp4a(
    const __half* input, __half* output,
    const FasterNetBlock& block,
    const FasterNetDP4AScales& scales,
    int H, int W, cudaStream_t stream)
{
    constexpr int C_PART = 16;
    constexpr int C_STAGE = 32;
    constexpr int C_EXP = 64;
    constexpr int PART_G = C_PART / 4;
    constexpr int STAGE_G = C_STAGE / 4;
    constexpr int EXP_G = C_EXP / 4;
    constexpr int THREADS = 128;
    const int M = H * W;

    // Precompute 6 scale parameters
    float conv_act_rcp   = 1.0f / scales.quantScale0;
    float conv_dequant   = scales.quantScale0 * block.spatial.quant_scale;
    float concat_rcp     = 1.0f / scales.quantScale1;
    float pw_exp_dequant = scales.quantScale1 * block.pw_expand.quant_scale;
    float pw_exp_requant = 1.0f / scales.activationScaleFactor;
    float pw_con_dequant = scales.activationScaleFactor * block.pw_contract.quant_scale;

    // Shared memory: packed weights + bias
    const int smem = (C_PART * 9 * PART_G + C_EXP * STAGE_G + C_STAGE * EXP_G) * sizeof(int)
                   + (C_PART + C_EXP + C_STAGE) * sizeof(__half);

    fused_fasternet32_dp4a_nhwc_kernel<<<divup(M, THREADS), THREADS, smem, stream>>>(
        input,
        block.spatial.weight, block.spatial.bias,
        block.pw_expand.weight, block.pw_expand.bias,
        block.pw_contract.weight, block.pw_contract.bias,
        output, H, W,
        conv_act_rcp, conv_dequant, concat_rcp,
        pw_exp_dequant, pw_exp_requant, pw_con_dequant);
}

// =========================================================================
// Tiled FasterNet32 dp4a: shared memory tile for conv3x3 on lower 16ch
// =========================================================================
//
// Same concept as tiled ConvNext, but only the lower 16 of 32 channels go
// through conv3x3. The tile stores pre-quantized INT8 for lower 16ch only.
// Upper 16ch passthrough reads center pixel from global (no redundancy).
//
// TILE_W=16, TILE_H=8 → 128 threads (matches original for register budget).

template<int TILE_W = 16, int TILE_H = 8>
static __global__ void __launch_bounds__(128, 4) tiled_fasternet32_dp4a_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, 32] NHWC
    const int8_t* __restrict__ conv_w,         // [16, 16*9] INT8 (spatial conv)
    const __half* __restrict__ conv_b,         // [16]
    const int8_t* __restrict__ pw_exp_w,       // [64, 32] INT8
    const __half* __restrict__ pw_exp_b,       // [64]
    const int8_t* __restrict__ pw_con_w,       // [32, 64] INT8
    const __half* __restrict__ pw_con_b,       // [32]
    __half*       __restrict__ output,         // [H*W, 32] NHWC
    __half*       __restrict__ skip_out,       // [H*W, 32] optional dual-write for skip save
    int H, int W,
    float conv_act_rcp,          // 1/quantScale0
    float conv_dequant,          // quantScale0 * spatial_w_scale
    float concat_rcp,            // 1/quantScale1
    float pw_exp_dequant,        // quantScale1 * pw_expand_w_scale
    float pw_exp_requant,        // 1/activationScaleFactor
    float pw_con_dequant)        // activationScaleFactor * pw_contract_w_scale
{
    constexpr int C_PART = 16;
    constexpr int C_STAGE = 32;
    constexpr int C_EXP = 64;
    constexpr int K3 = C_PART * 9;  // 144

    constexpr int PART_G = C_PART / 4;     // 4
    constexpr int STAGE_G = C_STAGE / 4;   // 8
    constexpr int EXP_G = C_EXP / 4;       // 16

    constexpr int TILE_W_PAD = TILE_W + 2;  // 18
    constexpr int TILE_H_PAD = TILE_H + 2;  // 10
    constexpr int TILE_PIXELS = TILE_W_PAD * TILE_H_PAD;  // 180

    // --- Shared memory ---
    __shared__ int s_tile[TILE_PIXELS * PART_G];          // 180×4 = 2880 B
    __shared__ int s_conv_w[C_PART * 9 * PART_G];        // 2304 B
    __shared__ int s_exp_w[C_EXP * STAGE_G];             // 2048 B
    __shared__ int s_con_w[C_STAGE * EXP_G];             // 2048 B
    __shared__ __half s_conv_b[C_PART];                   // 32 B
    __shared__ __half s_exp_b[C_EXP];                     // 128 B
    __shared__ __half s_con_b[C_STAGE];                   // 64 B

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;

    // === Cooperative load: repack conv3x3 weights for dp4a ===
    const int conv_total = C_PART * 9 * PART_G;
    for (int i = threadIdx.x; i < conv_total; i += blockDim.x) {
        const int ci_g = i % PART_G;
        const int sp   = (i / PART_G) % 9;
        const int co   = i / (PART_G * 9);
        int a = (int)conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
        int b = (int)conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
        int c = (int)conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
        int d = (int)conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
        s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                      ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // PW expand: [64, 32] → 64 × 8 packed
    const int exp_total = C_EXP * STAGE_G;
    for (int i = threadIdx.x; i < exp_total; i += blockDim.x) {
        const int ci_g = i % STAGE_G;
        const int co   = i / STAGE_G;
        const int base = co * C_STAGE + ci_g * 4;
        int a = (int)pw_exp_w[base + 0];
        int b = (int)pw_exp_w[base + 1];
        int c = (int)pw_exp_w[base + 2];
        int d = (int)pw_exp_w[base + 3];
        s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // PW contract: [32, 64] → 32 × 16 packed
    const int con_total = C_STAGE * EXP_G;
    for (int i = threadIdx.x; i < con_total; i += blockDim.x) {
        const int ci_g = i % EXP_G;
        const int co   = i / EXP_G;
        const int base = co * C_EXP + ci_g * 4;
        int a = (int)pw_con_w[base + 0];
        int b = (int)pw_con_w[base + 1];
        int c = (int)pw_con_w[base + 2];
        int d = (int)pw_con_w[base + 3];
        s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // Bias
    if (threadIdx.x < C_PART) s_conv_b[threadIdx.x] = conv_b[threadIdx.x];
    if (threadIdx.x < C_EXP)  s_exp_b[threadIdx.x]  = pw_exp_b[threadIdx.x];
    if (threadIdx.x < C_STAGE) s_con_b[threadIdx.x] = pw_con_b[threadIdx.x];

    // === Cooperative load: lower 16ch tile with FP16→INT8 pre-quantization ===
    for (int i = threadIdx.x; i < TILE_PIXELS; i += blockDim.x) {
        const int ty = i / TILE_W_PAD;
        const int tx = i % TILE_W_PAD;
        const int gy = tile_y0 + ty - 1;
        const int gx = tile_x0 + tx - 1;

        const int tile_base = i * PART_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            // Load ONLY first 16ch from the 32ch pixel, quantize
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * conv_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * conv_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * conv_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * conv_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();

    // === Per-thread computation ===
    const int tx = threadIdx.x % TILE_W;
    const int ty = threadIdx.x / TILE_W;
    const int gx = tile_x0 + tx;
    const int gy = tile_y0 + ty;
    if (gx >= W || gy >= H) return;

    const int pix = gy * W + gx;

    // ======== Stage 1: Conv 3×3 on lower 16ch from smem tile ========
    int conv_acc[C_PART];
    #pragma unroll
    for (int co = 0; co < C_PART; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int tile_idx = ((ty + kh) * TILE_W_PAD + (tx + kw)) * PART_G;
            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C_PART; co++) {
                #pragma unroll
                for (int cg = 0; cg < PART_G; cg++) {
                    conv_acc[co] = __dp4a(s_tile[tile_idx + cg],
                                          s_conv_w[co * 9 * PART_G + sp * PART_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // ======== Stage 2: Build 32ch INT8 concat for PW expand ========
    int concat_q[STAGE_G];

    // Lower 16ch: conv output → dequant → requant
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)conv_acc[cg * 4 + 0] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 1] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 2] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 3] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        concat_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // Upper 16ch: passthrough from center pixel, quantize to same scale
    const __half* center = input + pix * C_STAGE + C_PART;
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int qa = __float2int_rn(__half2float(center[cg * 4 + 0]) * concat_rcp);
        int qb = __float2int_rn(__half2float(center[cg * 4 + 1]) * concat_rcp);
        int qc = __float2int_rn(__half2float(center[cg * 4 + 2]) * concat_rcp);
        int qd = __float2int_rn(__half2float(center[cg * 4 + 3]) * concat_rcp);
        qa = max(-127, min(127, qa));
        qb = max(-127, min(127, qb));
        qc = max(-127, min(127, qc));
        qd = max(-127, min(127, qd));
        concat_q[PART_G + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                 ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
    }

    // ======== Stage 3: PW expand (32→64) + ReLU via dp4a ========
    int exp_acc[C_EXP];
    #pragma unroll
    for (int co = 0; co < C_EXP; co++) exp_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C_EXP; co++) {
        #pragma unroll
        for (int cg = 0; cg < STAGE_G; cg++) {
            exp_acc[co] = __dp4a(concat_q[cg],
                                 s_exp_w[co * STAGE_G + cg],
                                 exp_acc[co]);
        }
    }

    // Dequant + bias + ReLU → requant to INT8
    int exp_q[EXP_G];
    #pragma unroll
    for (int cg = 0; cg < EXP_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)exp_acc[cg * 4 + 0] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 1] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 2] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        {
            float v = (float)exp_acc[cg * 4 + 3] * pw_exp_dequant
                    + __half2float(s_exp_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
        }
        exp_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                    ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // ======== Stage 4: PW contract (64→32) + residual skip add ========
    int con_acc[C_STAGE];
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) con_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        #pragma unroll
        for (int cg = 0; cg < EXP_G; cg++) {
            con_acc[co] = __dp4a(exp_q[cg],
                                 s_con_w[co * EXP_G + cg],
                                 con_acc[co]);
        }
    }

    // Dequant + bias + skip add → FP16 output
    const __half* skip = input + pix * C_STAGE;
    __half* out = output + pix * C_STAGE;
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        float val = (float)con_acc[co] * pw_con_dequant
                  + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        __half result = __float2half(val);
        out[co] = result;
        if (skip_out) skip_out[pix * C_STAGE + co] = result;
    }
}

static void launch_tiled_fasternet32_dp4a(
    const __half* input, __half* output,
    const FasterNetBlock& block,
    const FasterNetDP4AScales& scales,
    int H, int W, cudaStream_t stream,
    __half* skip_out = nullptr)
{
    constexpr int TILE_W = 16, TILE_H = 8;
    constexpr int THREADS = TILE_W * TILE_H;  // 128

    float conv_act_rcp   = 1.0f / scales.quantScale0;
    float conv_dequant   = scales.quantScale0 * block.spatial.quant_scale;
    float concat_rcp     = 1.0f / scales.quantScale1;
    float pw_exp_dequant = scales.quantScale1 * block.pw_expand.quant_scale;
    float pw_exp_requant = 1.0f / scales.activationScaleFactor;
    float pw_con_dequant = scales.activationScaleFactor * block.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    tiled_fasternet32_dp4a_nhwc_kernel<TILE_W, TILE_H><<<grid, THREADS, 0, stream>>>(
        input,
        block.spatial.weight, block.spatial.bias,
        block.pw_expand.weight, block.pw_expand.bias,
        block.pw_contract.weight, block.pw_contract.bias,
        output, skip_out, H, W,
        conv_act_rcp, conv_dequant, concat_rcp,
        pw_exp_dequant, pw_exp_requant, pw_con_dequant);
}

// =========================================================================
// FasterNet 64ch dp4a kernel (bottleneck blocks)
// =========================================================================
//
// Same structure as 32ch dp4a but with doubled channel counts:
//   Conv 3×3: lower 32ch → 32ch (C_PART=32)
//   Concat: 32ch conv output + 32ch passthrough → 64ch
//   PW expand: 64→128 + ReLU (split into two halves for register pressure)
//   PW contract: 128→64 + skip add
//
// Shared memory: ~26 KB (weights + bias). Fits in 48 KB limit.
// PW expand split: avoids 128 live accumulators — peak ~96 regs + working.

// Fixed scales from AMD FSR4 quality model.
// [0]=bn_res[0], [1]=bn_res[1], [2]=bn_res[2]
static const FasterNetDP4AScales kFasterNet64Scales[3] = {
    {0.017320655286312103f, 0.02951941266655922f,  0.020330490544438362f},
    {0.015502181835472584f, 0.035003241151571274f, 0.017967987805604935f},
    {0.0190073624253273f,   0.038449302315711975f, 0.024075137451291084f},
};

static __global__ void __launch_bounds__(128, 3) fused_fasternet64_dp4a_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, 64] NHWC
    const int8_t* __restrict__ conv_w,         // [32, 32*9] INT8 (spatial conv)
    const __half* __restrict__ conv_b,         // [32]
    const int8_t* __restrict__ pw_exp_w,       // [128, 64] INT8
    const __half* __restrict__ pw_exp_b,       // [128]
    const int8_t* __restrict__ pw_con_w,       // [64, 128] INT8
    const __half* __restrict__ pw_con_b,       // [64]
    __half*       __restrict__ output,         // [H*W, 64] NHWC
    int H, int W,
    float conv_act_rcp,          // 1/quantScale0
    float conv_dequant,          // quantScale0 * spatial_w_scale
    float concat_rcp,            // 1/quantScale1
    float pw_exp_dequant,        // quantScale1 * pw_expand_w_scale
    float pw_exp_requant,        // 1/activationScaleFactor
    float pw_con_dequant)        // activationScaleFactor * pw_contract_w_scale
{
    constexpr int C_PART = 32;
    constexpr int C_STAGE = 64;
    constexpr int C_EXP = 128;
    constexpr int K3 = C_PART * 9;  // 288

    constexpr int PART_G = C_PART / 4;     // 8 dp4a groups
    constexpr int STAGE_G = C_STAGE / 4;   // 16 dp4a groups
    constexpr int EXP_G = C_EXP / 4;       // 32 dp4a groups
    constexpr int HALF_EXP = C_EXP / 2;    // 64 (split PW expand)
    constexpr int HALF_EXP_G = HALF_EXP / 4; // 16 groups per half

    // Shared memory: dp4a-packed weights + FP16 bias
    extern __shared__ char smem_raw[];

    int* s_conv_w  = reinterpret_cast<int*>(smem_raw);                        // [32, 9, 8]
    int* s_exp_w   = s_conv_w + C_PART * 9 * PART_G;                         // [128, 16]
    int* s_con_w   = s_exp_w  + C_EXP * STAGE_G;                             // [64, 32]
    __half* s_conv_b = reinterpret_cast<__half*>(s_con_w + C_STAGE * EXP_G); // [32]
    __half* s_exp_b  = s_conv_b + C_PART;                                     // [128]
    __half* s_con_b  = s_exp_b + C_EXP;                                       // [64]

    // Cooperative load: repack conv3x3 weights for dp4a
    const int conv_total = C_PART * 9 * PART_G;  // 2304
    for (int i = threadIdx.x; i < conv_total; i += blockDim.x) {
        const int ci_g = i % PART_G;
        const int sp   = (i / PART_G) % 9;
        const int co   = i / (PART_G * 9);
        int a = (int)conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
        int b = (int)conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
        int c = (int)conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
        int d = (int)conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
        s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                      ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // PW expand: [128, 64] → 128 × 16 packed
    const int exp_total = C_EXP * STAGE_G;  // 2048
    for (int i = threadIdx.x; i < exp_total; i += blockDim.x) {
        const int ci_g = i % STAGE_G;
        const int co   = i / STAGE_G;
        const int base = co * C_STAGE + ci_g * 4;
        int a = (int)pw_exp_w[base + 0];
        int b = (int)pw_exp_w[base + 1];
        int c = (int)pw_exp_w[base + 2];
        int d = (int)pw_exp_w[base + 3];
        s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // PW contract: [64, 128] → 64 × 32 packed
    const int con_total = C_STAGE * EXP_G;  // 2048
    for (int i = threadIdx.x; i < con_total; i += blockDim.x) {
        const int ci_g = i % EXP_G;
        const int co   = i / EXP_G;
        const int base = co * C_EXP + ci_g * 4;
        int a = (int)pw_con_w[base + 0];
        int b = (int)pw_con_w[base + 1];
        int c = (int)pw_con_w[base + 2];
        int d = (int)pw_con_w[base + 3];
        s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    // Bias
    for (int i = threadIdx.x; i < C_PART; i += blockDim.x)
        s_conv_b[i] = conv_b[i];
    for (int i = threadIdx.x; i < C_EXP; i += blockDim.x)
        s_exp_b[i] = pw_exp_b[i];
    for (int i = threadIdx.x; i < C_STAGE; i += blockDim.x)
        s_con_b[i] = pw_con_b[i];
    __syncthreads();

    const int pix = blockIdx.x * blockDim.x + threadIdx.x;
    if (pix >= H * W) return;

    const int h = pix / W;
    const int w = pix % W;

    // ======== Stage 1: Conv 3×3 on lower 32ch (32→32) via dp4a ========
    int conv_acc[C_PART];
    #pragma unroll
    for (int co = 0; co < C_PART; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        const int h_in = h + kh - 1;
        if (h_in < 0 || h_in >= H) continue;
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int w_in = w + kw - 1;
            if (w_in < 0 || w_in >= W) continue;

            // Load first 32 FP16 channels, quantize with conv_act_rcp
            const __half* in_ptr = input + (h_in * W + w_in) * C_STAGE;
            int in_packed[PART_G];
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * conv_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * conv_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * conv_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * conv_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                in_packed[cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }

            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C_PART; co++) {
                #pragma unroll
                for (int cg = 0; cg < PART_G; cg++) {
                    conv_acc[co] = __dp4a(in_packed[cg],
                                          s_conv_w[co * 9 * PART_G + sp * PART_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // ======== Stage 2: Build 64ch INT8 concat for PW expand ========
    // Conv output: dequant → float → requant to quantScale1
    // Passthrough upper 32ch: quantize to quantScale1
    int concat_q[STAGE_G];  // 16 packed uint32 = 64 INT8 channels

    // Lower 32ch: conv output → requant
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)conv_acc[cg * 4 + 0] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 1] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 2] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 3] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        concat_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // Upper 32ch: passthrough from center pixel, quantize to same scale
    const __half* center = input + pix * C_STAGE + C_PART;
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int qa = __float2int_rn(__half2float(center[cg * 4 + 0]) * concat_rcp);
        int qb = __float2int_rn(__half2float(center[cg * 4 + 1]) * concat_rcp);
        int qc = __float2int_rn(__half2float(center[cg * 4 + 2]) * concat_rcp);
        int qd = __float2int_rn(__half2float(center[cg * 4 + 3]) * concat_rcp);
        qa = max(-127, min(127, qa));
        qb = max(-127, min(127, qb));
        qc = max(-127, min(127, qc));
        qd = max(-127, min(127, qd));
        concat_q[PART_G + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                 ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
    }

    // ======== Stage 3: PW expand (64→128) + ReLU via dp4a ========
    // Split into two halves (64 outputs each) to cap register pressure.
    int exp_q[EXP_G];  // 32 packed uint32 = 128 INT8 channels

    // --- Half 1: output channels 0..63 ---
    {
        int acc[HALF_EXP];
        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) acc[co] = 0;

        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) {
            #pragma unroll
            for (int cg = 0; cg < STAGE_G; cg++) {
                acc[co] = __dp4a(concat_q[cg],
                                 s_exp_w[co * STAGE_G + cg],
                                 acc[co]);
            }
        }

        #pragma unroll
        for (int cg = 0; cg < HALF_EXP_G; cg++) {
            int q0, q1, q2, q3;
            {
                float v = (float)acc[cg * 4 + 0] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 0]);
                q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 1] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 1]);
                q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 2] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 2]);
                q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 3] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 3]);
                q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            exp_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
    }

    // --- Half 2: output channels 64..127 ---
    {
        int acc[HALF_EXP];
        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) acc[co] = 0;

        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) {
            #pragma unroll
            for (int cg = 0; cg < STAGE_G; cg++) {
                acc[co] = __dp4a(concat_q[cg],
                                 s_exp_w[(co + HALF_EXP) * STAGE_G + cg],
                                 acc[co]);
            }
        }

        #pragma unroll
        for (int cg = 0; cg < HALF_EXP_G; cg++) {
            int q0, q1, q2, q3;
            {
                float v = (float)acc[cg * 4 + 0] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 0]);
                q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 1] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 1]);
                q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 2] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 2]);
                q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 3] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 3]);
                q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            exp_q[HALF_EXP_G + cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                                       ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
    }

    // ======== Stage 4: PW contract (128→64) + residual skip add ========
    int con_acc[C_STAGE];
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) con_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        #pragma unroll
        for (int cg = 0; cg < EXP_G; cg++) {
            con_acc[co] = __dp4a(exp_q[cg],
                                 s_con_w[co * EXP_G + cg],
                                 con_acc[co]);
        }
    }

    // Dequant + bias + skip add → FP16 output
    const __half* skip = input + pix * C_STAGE;
    __half* out = output + pix * C_STAGE;
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        float val = (float)con_acc[co] * pw_con_dequant
                  + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        out[co] = __float2half(val);
    }
}

static void launch_fused_fasternet64_dp4a(
    const __half* input, __half* output,
    const FasterNetBlock& block,
    const FasterNetDP4AScales& scales,
    int H, int W, cudaStream_t stream)
{
    constexpr int C_PART = 32;
    constexpr int C_STAGE = 64;
    constexpr int C_EXP = 128;
    constexpr int PART_G = C_PART / 4;
    constexpr int STAGE_G = C_STAGE / 4;
    constexpr int EXP_G = C_EXP / 4;
    constexpr int THREADS = 128;
    const int M = H * W;

    // Precompute 6 scale parameters
    float conv_act_rcp   = 1.0f / scales.quantScale0;
    float conv_dequant   = scales.quantScale0 * block.spatial.quant_scale;
    float concat_rcp     = 1.0f / scales.quantScale1;
    float pw_exp_dequant = scales.quantScale1 * block.pw_expand.quant_scale;
    float pw_exp_requant = 1.0f / scales.activationScaleFactor;
    float pw_con_dequant = scales.activationScaleFactor * block.pw_contract.quant_scale;

    // Shared memory: packed INT8 weights + FP16 bias
    const int smem = (C_PART * 9 * PART_G + C_EXP * STAGE_G + C_STAGE * EXP_G) * sizeof(int)
                   + (C_PART + C_EXP + C_STAGE) * sizeof(__half);

    fused_fasternet64_dp4a_nhwc_kernel<<<divup(M, THREADS), THREADS, smem, stream>>>(
        input,
        block.spatial.weight, block.spatial.bias,
        block.pw_expand.weight, block.pw_expand.bias,
        block.pw_contract.weight, block.pw_contract.bias,
        output, H, W,
        conv_act_rcp, conv_dequant, concat_rcp,
        pw_exp_dequant, pw_exp_requant, pw_con_dequant);
}

// =========================================================================
// Tiled FasterNet64 dp4a: shared memory tile for conv3x3 on lower 32ch
// =========================================================================
//
// Same tiling as FasterNet32 but for 64ch bottleneck blocks. Conv3x3 operates
// on lower 32ch (PART_G=8), tile stores pre-quantized INT8. Upper 32ch pass
// through. PW expand split in halves for register pressure management.

template<int TILE_W = 16, int TILE_H = 8>
static __global__ void __launch_bounds__(128, 3) tiled_fasternet64_dp4a_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, 64] NHWC
    const int8_t* __restrict__ conv_w,         // [32, 32*9] INT8
    const __half* __restrict__ conv_b,         // [32]
    const int8_t* __restrict__ pw_exp_w,       // [128, 64] INT8
    const __half* __restrict__ pw_exp_b,       // [128]
    const int8_t* __restrict__ pw_con_w,       // [64, 128] INT8
    const __half* __restrict__ pw_con_b,       // [64]
    __half*       __restrict__ output,         // [H*W, 64] NHWC
    int H, int W,
    float conv_act_rcp,
    float conv_dequant,
    float concat_rcp,
    float pw_exp_dequant,
    float pw_exp_requant,
    float pw_con_dequant)
{
    constexpr int C_PART = 32;
    constexpr int C_STAGE = 64;
    constexpr int C_EXP = 128;
    constexpr int K3 = C_PART * 9;

    constexpr int PART_G = C_PART / 4;     // 8
    constexpr int STAGE_G = C_STAGE / 4;   // 16
    constexpr int EXP_G = C_EXP / 4;       // 32
    constexpr int HALF_EXP = C_EXP / 2;    // 64
    constexpr int HALF_EXP_G = HALF_EXP / 4; // 16

    constexpr int TILE_W_PAD = TILE_W + 2;
    constexpr int TILE_H_PAD = TILE_H + 2;
    constexpr int TILE_PIXELS = TILE_W_PAD * TILE_H_PAD;

    // --- Shared memory ---
    __shared__ int s_tile[TILE_PIXELS * PART_G];          // 180×8 = 5760 B
    __shared__ int s_conv_w[C_PART * 9 * PART_G];        // 9216 B
    __shared__ int s_exp_w[C_EXP * STAGE_G];             // 8192 B
    __shared__ int s_con_w[C_STAGE * EXP_G];             // 8192 B
    __shared__ __half s_conv_b[C_PART];
    __shared__ __half s_exp_b[C_EXP];
    __shared__ __half s_con_b[C_STAGE];

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;

    // === Cooperative load: repack conv3x3 weights for dp4a ===
    const int conv_total = C_PART * 9 * PART_G;
    for (int i = threadIdx.x; i < conv_total; i += blockDim.x) {
        const int ci_g = i % PART_G;
        const int sp   = (i / PART_G) % 9;
        const int co   = i / (PART_G * 9);
        int a = (int)conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
        int b = (int)conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
        int c = (int)conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
        int d = (int)conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
        s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                      ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    const int exp_total = C_EXP * STAGE_G;
    for (int i = threadIdx.x; i < exp_total; i += blockDim.x) {
        const int ci_g = i % STAGE_G;
        const int co   = i / STAGE_G;
        const int base = co * C_STAGE + ci_g * 4;
        int a = (int)pw_exp_w[base + 0];
        int b = (int)pw_exp_w[base + 1];
        int c = (int)pw_exp_w[base + 2];
        int d = (int)pw_exp_w[base + 3];
        s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    const int con_total = C_STAGE * EXP_G;
    for (int i = threadIdx.x; i < con_total; i += blockDim.x) {
        const int ci_g = i % EXP_G;
        const int co   = i / EXP_G;
        const int base = co * C_EXP + ci_g * 4;
        int a = (int)pw_con_w[base + 0];
        int b = (int)pw_con_w[base + 1];
        int c = (int)pw_con_w[base + 2];
        int d = (int)pw_con_w[base + 3];
        s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                     ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
    }

    for (int i = threadIdx.x; i < C_PART; i += blockDim.x)
        s_conv_b[i] = conv_b[i];
    for (int i = threadIdx.x; i < C_EXP; i += blockDim.x)
        s_exp_b[i] = pw_exp_b[i];
    for (int i = threadIdx.x; i < C_STAGE; i += blockDim.x)
        s_con_b[i] = pw_con_b[i];

    // === Cooperative load: lower 32ch tile with pre-quantization ===
    for (int i = threadIdx.x; i < TILE_PIXELS; i += blockDim.x) {
        const int ty = i / TILE_W_PAD;
        const int tx = i % TILE_W_PAD;
        const int gy = tile_y0 + ty - 1;
        const int gx = tile_x0 + tx - 1;

        const int tile_base = i * PART_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * conv_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * conv_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * conv_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * conv_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();

    // === Per-thread computation ===
    const int tx = threadIdx.x % TILE_W;
    const int ty = threadIdx.x / TILE_W;
    const int gx = tile_x0 + tx;
    const int gy = tile_y0 + ty;
    if (gx >= W || gy >= H) return;

    const int pix = gy * W + gx;

    // ======== Stage 1: Conv 3×3 on lower 32ch from smem tile ========
    int conv_acc[C_PART];
    #pragma unroll
    for (int co = 0; co < C_PART; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int tile_idx = ((ty + kh) * TILE_W_PAD + (tx + kw)) * PART_G;
            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C_PART; co++) {
                #pragma unroll
                for (int cg = 0; cg < PART_G; cg++) {
                    conv_acc[co] = __dp4a(s_tile[tile_idx + cg],
                                          s_conv_w[co * 9 * PART_G + sp * PART_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // ======== Stage 2: 64ch INT8 concat ========
    int concat_q[STAGE_G];

    // Lower 32ch: conv output → requant
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int q0, q1, q2, q3;
        {
            float v = (float)conv_acc[cg * 4 + 0] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 0]);
            q0 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 1] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 1]);
            q1 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 2] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 2]);
            q2 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        {
            float v = (float)conv_acc[cg * 4 + 3] * conv_dequant
                    + __half2float(s_conv_b[cg * 4 + 3]);
            q3 = max(-127, min(127, __float2int_rn(v * concat_rcp)));
        }
        concat_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // Upper 32ch: passthrough from center pixel
    const __half* center = input + pix * C_STAGE + C_PART;
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int qa = __float2int_rn(__half2float(center[cg * 4 + 0]) * concat_rcp);
        int qb = __float2int_rn(__half2float(center[cg * 4 + 1]) * concat_rcp);
        int qc = __float2int_rn(__half2float(center[cg * 4 + 2]) * concat_rcp);
        int qd = __float2int_rn(__half2float(center[cg * 4 + 3]) * concat_rcp);
        qa = max(-127, min(127, qa));
        qb = max(-127, min(127, qb));
        qc = max(-127, min(127, qc));
        qd = max(-127, min(127, qd));
        concat_q[PART_G + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                 ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
    }

    // ======== Stage 3: PW expand (64→128) + ReLU — split halves ========
    int exp_q[EXP_G];

    // Half 1: output channels 0..63
    {
        int acc[HALF_EXP];
        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) acc[co] = 0;

        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) {
            #pragma unroll
            for (int cg = 0; cg < STAGE_G; cg++) {
                acc[co] = __dp4a(concat_q[cg],
                                 s_exp_w[co * STAGE_G + cg],
                                 acc[co]);
            }
        }

        #pragma unroll
        for (int cg = 0; cg < HALF_EXP_G; cg++) {
            int q0, q1, q2, q3;
            {
                float v = (float)acc[cg * 4 + 0] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 0]);
                q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 1] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 1]);
                q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 2] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 2]);
                q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 3] * pw_exp_dequant
                        + __half2float(s_exp_b[cg * 4 + 3]);
                q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            exp_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
    }

    // Half 2: output channels 64..127
    {
        int acc[HALF_EXP];
        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) acc[co] = 0;

        #pragma unroll
        for (int co = 0; co < HALF_EXP; co++) {
            #pragma unroll
            for (int cg = 0; cg < STAGE_G; cg++) {
                acc[co] = __dp4a(concat_q[cg],
                                 s_exp_w[(co + HALF_EXP) * STAGE_G + cg],
                                 acc[co]);
            }
        }

        #pragma unroll
        for (int cg = 0; cg < HALF_EXP_G; cg++) {
            int q0, q1, q2, q3;
            {
                float v = (float)acc[cg * 4 + 0] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 0]);
                q0 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 1] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 1]);
                q1 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 2] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 2]);
                q2 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            {
                float v = (float)acc[cg * 4 + 3] * pw_exp_dequant
                        + __half2float(s_exp_b[HALF_EXP + cg * 4 + 3]);
                q3 = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_requant)));
            }
            exp_q[HALF_EXP_G + cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                                       ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
        }
    }

    // ======== Stage 4: PW contract (128→64) + skip add ========
    int con_acc[C_STAGE];
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) con_acc[co] = 0;

    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        #pragma unroll
        for (int cg = 0; cg < EXP_G; cg++) {
            con_acc[co] = __dp4a(exp_q[cg],
                                 s_con_w[co * EXP_G + cg],
                                 con_acc[co]);
        }
    }

    const __half* skip = input + pix * C_STAGE;
    __half* out = output + pix * C_STAGE;
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        float val = (float)con_acc[co] * pw_con_dequant
                  + __half2float(s_con_b[co]);
        val += __half2float(skip[co]);
        out[co] = __float2half(val);
    }
}

static void launch_tiled_fasternet64_dp4a(
    const __half* input, __half* output,
    const FasterNetBlock& block,
    const FasterNetDP4AScales& scales,
    int H, int W, cudaStream_t stream)
{
    constexpr int TILE_W = 16, TILE_H = 8;
    constexpr int THREADS = TILE_W * TILE_H;  // 128

    float conv_act_rcp   = 1.0f / scales.quantScale0;
    float conv_dequant   = scales.quantScale0 * block.spatial.quant_scale;
    float concat_rcp     = 1.0f / scales.quantScale1;
    float pw_exp_dequant = scales.quantScale1 * block.pw_expand.quant_scale;
    float pw_exp_requant = 1.0f / scales.activationScaleFactor;
    float pw_con_dequant = scales.activationScaleFactor * block.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    tiled_fasternet64_dp4a_nhwc_kernel<TILE_W, TILE_H><<<grid, THREADS, 0, stream>>>(
        input,
        block.spatial.weight, block.spatial.bias,
        block.pw_expand.weight, block.pw_expand.bias,
        block.pw_contract.weight, block.pw_contract.bias,
        output, H, W,
        conv_act_rcp, conv_dequant, concat_rcp,
        pw_exp_dequant, pw_exp_requant, pw_con_dequant);
}

// 2x slower than separated path with optimized INT8 GEMM). Separated path kept.
static void fasternet_forward_nhwc_impl(
    const __half* input, __half* output,
    const FasterNetBlock& block,
    const FasterNetDP4AScales* scales,  // non-null for 32ch dp4a, null for 64ch
    int C_stage, int H, int W,
    __half* tmp_a,
    __half* tmp_b,
    void* workspace,
    cudaStream_t stream,
    __half* skip_out = nullptr)
{
    // dp4a tiled path for 32ch stage (smem tile eliminates 9× conv3x3 redundancy)
    if (C_stage == 32 && scales) {
        launch_tiled_fasternet32_dp4a(input, output, block, *scales, H, W, stream, skip_out);
        return;
    }

    // dp4a tiled path for 64ch bottleneck
    if (C_stage == 64 && scales) {
        launch_tiled_fasternet64_dp4a(input, output, block, *scales, H, W, stream);
        return;
    }

    // Separated path fallback
    const int C_partial = block.spatial.C_in;
    const int C_spatial_out = block.spatial.C_out;
    const int C_remaining = C_stage - C_partial;
    const int spatial = H * W;
    constexpr int BLOCK = 256;

    channel_slice_nhwc_kernel<<<divup(C_partial * spatial, BLOCK), BLOCK, 0, stream>>>(
        input, tmp_a, C_stage, C_partial, spatial);

    run_int8_conv2d_nhwc(tmp_a, block.spatial, tmp_b, H, W, workspace, stream);

    channel_concat_nhwc_strided_kernel<<<divup(C_stage * spatial, BLOCK), BLOCK, 0, stream>>>(
        tmp_b, input, tmp_a,
        C_spatial_out, C_remaining, C_stage, C_partial, spatial);

    run_int8_conv2d_nhwc(tmp_a, block.pw_expand, tmp_b, H, W, workspace, stream,
                         /*relu=*/true);

    run_int8_conv2d_nhwc(tmp_b, block.pw_contract, tmp_a, H, W, workspace, stream);

    add_kernel<<<divup(C_stage * spatial, BLOCK), BLOCK, 0, stream>>>(
        tmp_a, input, output, C_stage * spatial);
}

// =========================================================================
// Workspace computation (NHWC)
// =========================================================================

static size_t compute_max_workspace_nhwc(const FSR4Weights& w, int H_in, int W_in) {
    // All layers now use fused dp4a kernels or direct kernels with shared
    // memory — no workspace buffers required in the NHWC path.
    //
    // - ConvNext 16ch: fused dp4a kernel (shared memory)
    // - FasterNet 32ch: fused dp4a kernel (shared memory)
    // - FasterNet 64ch: fused dp4a kernel (shared memory)
    // - enc1_downscale: direct FP16 kernel
    // - enc2_downscale: direct INT8 kernel
    // - enc3_downscale: direct INT8 kernel
    // - bn_upscale:     direct ConvTranspose<64,32>
    // - dec3_upscale:   direct ConvTranspose<32,16>
    // - dec2_upscale:   direct ConvTranspose<16,8>
    (void)w; (void)H_in; (void)W_in;
    return 0;
}

// =========================================================================
// FSR4Model::forward_nhwc — CUDA Graph wrapper
// =========================================================================
//
// First non-profiled call: captures all kernel launches into a CUDA graph.
// Subsequent calls with same input/output/resolution: replays the graph,
// eliminating ~20 kernel launch overheads (~5µs each = ~100µs total).
// Profiled calls bypass the graph (event recording is incompatible with capture).

void FSR4Model::forward_nhwc(const __half* input, __half* output,
                              int H_in, int W_in, cudaStream_t stream,
                              FSR4Profile* prof)
{
    // Fast path: replay cached CUDA graph
    if (!prof && nhwc_graph_exec_
        && nhwc_graph_H_ == H_in && nhwc_graph_W_ == W_in
        && nhwc_graph_input_ == input && nhwc_graph_output_ == output) {
        cudaGraphLaunch(nhwc_graph_exec_, stream);
        return;
    }

    // Profiled path: run kernels directly (event recording incompatible with capture)
    if (prof) {
        forward_nhwc_kernels(input, output, H_in, W_in, stream, prof);
        return;
    }

    // Capture new CUDA graph (first non-profiled call or parameter change)
    if (nhwc_graph_exec_) {
        cudaGraphExecDestroy(nhwc_graph_exec_);
        nhwc_graph_exec_ = nullptr;
    }

    // Stream capture requires a non-default stream
    if (!nhwc_capture_stream_)
        cudaStreamCreate(&nhwc_capture_stream_);

    cudaStreamBeginCapture(nhwc_capture_stream_, cudaStreamCaptureModeGlobal);
    forward_nhwc_kernels(input, output, H_in, W_in, nhwc_capture_stream_, nullptr);

    cudaGraph_t graph;
    cudaStreamEndCapture(nhwc_capture_stream_, &graph);
    cudaGraphInstantiate(&nhwc_graph_exec_, graph, nullptr, nullptr, 0);
    cudaGraphDestroy(graph);

    nhwc_graph_H_ = H_in;
    nhwc_graph_W_ = W_in;
    nhwc_graph_input_ = input;
    nhwc_graph_output_ = output;

    // Launch the newly captured graph on the caller's stream
    cudaGraphLaunch(nhwc_graph_exec_, stream);
}

// =========================================================================
// ConvNext dp4a pixel helper (used by mega-kernels)
// Computes conv3x3 → PW expand + ReLU → PW contract from tiled INT8 data.
// Returns float result[C] WITHOUT residual skip (caller adds skip).
// =========================================================================

template<int C>
static __device__ __forceinline__ void convnext_block_dp4a(
    const int* __restrict__ tile, int tile_stride,
    int tile_r, int tile_c,
    const int* __restrict__ s_cw, const __half* __restrict__ s_cb,
    const int* __restrict__ s_ew, const __half* __restrict__ s_eb,
    const int* __restrict__ s_pw, const __half* __restrict__ s_pb,
    float conv_dq, float conv_rq,
    float exp_dq, float exp_rq, float con_dq,
    float* __restrict__ result)
{
    constexpr int C2 = 2 * C;
    constexpr int C_G = C / 4;
    constexpr int C2_G = C2 / 4;

    // Stage 1: Conv 3×3 (C→C) from tile
    int conv_acc[C];
    #pragma unroll
    for (int co = 0; co < C; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int tidx = ((tile_r + kh) * tile_stride + (tile_c + kw)) * C_G;
            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C; co++) {
                #pragma unroll
                for (int cg = 0; cg < C_G; cg++) {
                    conv_acc[co] = __dp4a(tile[tidx + cg],
                                          s_cw[co * 9 * C_G + sp * C_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // Dequant conv3x3 → float → requant to INT8
    int conv_q[C_G];
    #pragma unroll
    for (int cg = 0; cg < C_G; cg++) {
        int q0 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+0] * conv_dq + __half2float(s_cb[cg*4+0])) * conv_rq)));
        int q1 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+1] * conv_dq + __half2float(s_cb[cg*4+1])) * conv_rq)));
        int q2 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+2] * conv_dq + __half2float(s_cb[cg*4+2])) * conv_rq)));
        int q3 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+3] * conv_dq + __half2float(s_cb[cg*4+3])) * conv_rq)));
        conv_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                      ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }

    // Stage 2: PW expand (C→2C) + ReLU
    int exp_q[C2_G];
    #pragma unroll
    for (int cg = 0; cg < C2_G; cg++) {
        int q[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int co = cg * 4 + j;
            int acc = 0;
            #pragma unroll
            for (int g = 0; g < C_G; g++)
                acc = __dp4a(conv_q[g], s_ew[co * C_G + g], acc);
            float v = (float)acc * exp_dq + __half2float(s_eb[co]);
            q[j] = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * exp_rq)));
        }
        exp_q[cg] = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) |
                     ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
    }

    // Stage 3: PW contract (2C→C) — caller adds skip
    #pragma unroll
    for (int co = 0; co < C; co++) {
        int acc = 0;
        #pragma unroll
        for (int cg = 0; cg < C2_G; cg++)
            acc = __dp4a(exp_q[cg], s_pw[co * C2_G + cg], acc);
        result[co] = (float)acc * con_dq + __half2float(s_pb[co]);
    }
}

// =========================================================================
// FasterNet32 dp4a pixel helper (used by mega-kernels)
// Computes partial conv3x3(16ch) → concat(32ch) → PW expand(64ch)+ReLU
// → PW contract(32ch). Returns float result[32] WITHOUT residual skip.
// tile: INT8-packed lower 16ch data. upper_16ch: FP16 upper 16ch passthrough.
// =========================================================================

static __device__ __forceinline__ void fasternet32_block_dp4a(
    const int* __restrict__ tile, int tile_stride,
    int tile_r, int tile_c,
    const __half* __restrict__ upper_16ch,
    const int* __restrict__ s_cw, const __half* __restrict__ s_cb,
    const int* __restrict__ s_ew, const __half* __restrict__ s_eb,
    const int* __restrict__ s_pw, const __half* __restrict__ s_pb,
    float conv_dq, float concat_rcp,
    float pw_exp_dq, float pw_exp_rq, float pw_con_dq,
    float* __restrict__ result)
{
    constexpr int C_PART = 16;
    constexpr int C_STAGE = 32;
    constexpr int C_EXP = 64;
    constexpr int PART_G = C_PART / 4;     // 4
    constexpr int STAGE_G = C_STAGE / 4;   // 8
    constexpr int EXP_G = C_EXP / 4;       // 16

    // Stage 1: Conv 3×3 on lower 16ch from tile
    int conv_acc[C_PART];
    #pragma unroll
    for (int co = 0; co < C_PART; co++) conv_acc[co] = 0;

    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            const int tidx = ((tile_r + kh) * tile_stride + (tile_c + kw)) * PART_G;
            const int sp = kh * 3 + kw;
            #pragma unroll
            for (int co = 0; co < C_PART; co++) {
                #pragma unroll
                for (int cg = 0; cg < PART_G; cg++) {
                    conv_acc[co] = __dp4a(tile[tidx + cg],
                                          s_cw[co * 9 * PART_G + sp * PART_G + cg],
                                          conv_acc[co]);
                }
            }
        }
    }

    // Stage 2: Build 32ch INT8 concat (lower=conv dequant+requant, upper=passthrough requant)
    int concat_q[STAGE_G];
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int q0 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+0] * conv_dq + __half2float(s_cb[cg*4+0])) * concat_rcp)));
        int q1 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+1] * conv_dq + __half2float(s_cb[cg*4+1])) * concat_rcp)));
        int q2 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+2] * conv_dq + __half2float(s_cb[cg*4+2])) * concat_rcp)));
        int q3 = max(-127, min(127, __float2int_rn(
            ((float)conv_acc[cg*4+3] * conv_dq + __half2float(s_cb[cg*4+3])) * concat_rcp)));
        concat_q[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) |
                        ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24);
    }
    #pragma unroll
    for (int cg = 0; cg < PART_G; cg++) {
        int qa = max(-127, min(127, __float2int_rn(__half2float(upper_16ch[cg*4+0]) * concat_rcp)));
        int qb = max(-127, min(127, __float2int_rn(__half2float(upper_16ch[cg*4+1]) * concat_rcp)));
        int qc = max(-127, min(127, __float2int_rn(__half2float(upper_16ch[cg*4+2]) * concat_rcp)));
        int qd = max(-127, min(127, __float2int_rn(__half2float(upper_16ch[cg*4+3]) * concat_rcp)));
        concat_q[PART_G + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                 ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
    }

    // Stage 3: PW expand (32→64) + ReLU, 4 outputs at a time for register pressure
    int exp_q[EXP_G];
    #pragma unroll
    for (int cg = 0; cg < EXP_G; cg++) {
        int q[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int co = cg * 4 + j;
            int acc = 0;
            #pragma unroll
            for (int g = 0; g < STAGE_G; g++)
                acc = __dp4a(concat_q[g], s_ew[co * STAGE_G + g], acc);
            float v = (float)acc * pw_exp_dq + __half2float(s_eb[co]);
            q[j] = max(-127, min(127, __float2int_rn(fmaxf(v, 0.0f) * pw_exp_rq)));
        }
        exp_q[cg] = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) |
                     ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
    }

    // Stage 4: PW contract (64→32) — caller adds skip
    #pragma unroll
    for (int co = 0; co < C_STAGE; co++) {
        int acc = 0;
        #pragma unroll
        for (int cg = 0; cg < EXP_G; cg++)
            acc = __dp4a(exp_q[cg], s_pw[co * EXP_G + cg], acc);
        result[co] = (float)acc * pw_con_dq + __half2float(s_pb[co]);
    }
}

// =========================================================================
// Enc2 Mega-Kernel: 2× ConvNextBlock(16ch) + downscale(16→32)
//
// Fuses three kernel launches into one. All intermediate data stays in
// shared memory — only reads the global input tile once and writes skip2
// + downscale output once.
//
// Phase A: Load 20×20 input tile + block 0 weights into smem.
// Phase B: Compute ConvNext block 0 for 18×18 region (inner + halo),
//          save skip for block 1, write INT8 to s_inter.
// Phase C: Reload block 1 weights.
// Phase D: Compute ConvNext block 1 for 16×16, write skip2 to global.
// Phase E: 64 threads compute k2s2 downscale (read skip2 from L1 cache).
//
// Traffic per tile: ~23 MB (was ~44 MB for 3 separate kernels)
// =========================================================================

template<int C = 16, int TILE_W = 16, int TILE_H = 16>
static __global__ void __launch_bounds__(256, 3) enc2_mega_nhwc_kernel(
    const __half* __restrict__ input,          // [H*W, C] buf A
    // Block 0 weights
    const int8_t* __restrict__ b0_conv_w,
    const __half* __restrict__ b0_conv_b,
    const int8_t* __restrict__ b0_exp_w,
    const __half* __restrict__ b0_exp_b,
    const int8_t* __restrict__ b0_con_w,
    const __half* __restrict__ b0_con_b,
    // Block 1 weights
    const int8_t* __restrict__ b1_conv_w,
    const __half* __restrict__ b1_conv_b,
    const int8_t* __restrict__ b1_exp_w,
    const __half* __restrict__ b1_exp_b,
    const int8_t* __restrict__ b1_con_w,
    const __half* __restrict__ b1_con_b,
    // Downscale weights
    const int8_t* __restrict__ ds_w,           // [32, 64] INT8
    const __half* __restrict__ ds_bias,        // [32]
    const __half* __restrict__ ds_wscale,      // [32]
    // Outputs
    __half*       __restrict__ skip2_out,      // [H*W, C] buf B
    __half*       __restrict__ ds_out,         // [H/2*W/2, 32] buf C
    int H, int W,
    // Block 0 scales
    float b0_act_rcp, float b0_conv_dq, float b0_conv_rq,
    float b0_exp_dq, float b0_exp_rq, float b0_con_dq,
    // Block 1 scales
    float b1_act_rcp, float b1_conv_dq, float b1_conv_rq,
    float b1_exp_dq, float b1_exp_rq, float b1_con_dq)
{
    constexpr int C2 = 2 * C;         // 32
    constexpr int C_G = C / 4;        // 4
    constexpr int C2_G = C2 / 4;      // 8
    constexpr int K3 = C * 9;
    constexpr int DS_C_OUT = 32;
    constexpr int DS_K = C * 4;       // 64

    constexpr int TILE_W_EXP = TILE_W + 4;  // 20
    constexpr int TILE_H_EXP = TILE_H + 4;  // 20
    constexpr int TILE_PIXELS_EXP = TILE_W_EXP * TILE_H_EXP;  // 400

    constexpr int INTER_W = TILE_W + 2;  // 18
    constexpr int INTER_H = TILE_H + 2;  // 18
    constexpr int INTER_PIXELS = INTER_W * INTER_H;  // 324

    // --- Shared memory ---
    // s_tile + s_inter must be separate (read tile, write inter in Phase B)
    __shared__ int s_tile[TILE_PIXELS_EXP * C_G];  // 1600 ints = 6400 B
    __shared__ int s_inter[INTER_PIXELS * C_G];     // 1296 ints = 5184 B
    __shared__ int s_conv_w[C * 9 * C_G];           // 576 ints = 2304 B
    __shared__ int s_exp_w[C2 * C_G];               // 128 ints = 512 B
    __shared__ int s_con_w[C * C2_G];               // 128 ints = 512 B
    __shared__ __half s_conv_b[C];                  // 32 B
    __shared__ __half s_exp_b[C2];                  // 64 B
    __shared__ __half s_con_b[C];                   // 32 B
    // Total: 6400 + 5184 + 2304 + 512 + 512 + 32 + 64 + 32 = 15,040 B

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;
    const int tx = threadIdx.x % TILE_W;  // 0..15
    const int ty = threadIdx.x / TILE_W;  // 0..15

    // =================================================================
    // PHASE A: Load 20×20 input tile + block 0 weights
    // =================================================================

    // Block 0 conv3x3 weights (dp4a packed)
    {
        const int total = C * 9 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int sp   = (i / C_G) % 9;
            const int co   = i / (C_G * 9);
            int a = (int)b0_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b0_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b0_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b0_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW expand weights
    {
        const int total = C2 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int co   = i / C_G;
            const int base = co * C + ci_g * 4;
            int a = (int)b0_exp_w[base + 0];
            int b = (int)b0_exp_w[base + 1];
            int c = (int)b0_exp_w[base + 2];
            int d = (int)b0_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW contract weights
    {
        const int total = C * C2_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C2_G;
            const int co   = i / C2_G;
            const int base = co * C2 + ci_g * 4;
            int a = (int)b0_con_w[base + 0];
            int b = (int)b0_con_w[base + 1];
            int c = (int)b0_con_w[base + 2];
            int d = (int)b0_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 biases
    if (threadIdx.x < C)  s_conv_b[threadIdx.x] = b0_conv_b[threadIdx.x];
    if (threadIdx.x < C)  s_con_b[threadIdx.x]  = b0_con_b[threadIdx.x];
    if (threadIdx.x < C2) s_exp_b[threadIdx.x]  = b0_exp_b[threadIdx.x];

    // 20×20 input tile with FP16→INT8 pre-quantization
    for (int i = threadIdx.x; i < TILE_PIXELS_EXP; i += blockDim.x) {
        const int tr = i / TILE_W_EXP;
        const int tc = i % TILE_W_EXP;
        const int gy = tile_y0 + tr - 2;  // -2 for double halo
        const int gx = tile_x0 + tc - 2;

        const int tile_base = i * C_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int cg = 0; cg < C_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * b0_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * b0_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * b0_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * b0_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < C_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();  // Barrier 1: tile + block 0 weights ready

    // =================================================================
    // PHASE B: Compute ConvNext block 0 for 18×18 region
    //   - Inner pixel (ty+1, tx+1) → save skip, write s_inter
    //   - Halo pixels (threads 0-67) → write s_inter only
    // =================================================================

    // Block 0's 18×18 position maps to tile position for conv3x3 center:
    //   s_tile row/col = (r, c) where conv3x3 reads (r..r+2, c..c+2)
    //   Global pos = (tile_y0 - 1 + r, tile_x0 - 1 + c)

    __half my_skip[C];  // Block 0 output for this thread's inner pixel

    // B.1: Each thread computes its inner pixel at 18×18 pos (ty+1, tx+1)
    {
        const int r = ty + 1, c = tx + 1;  // 18×18 grid position
        float result[C];
        convnext_block_dp4a<C>(
            s_tile, TILE_W_EXP, r, c,
            s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
            b0_conv_dq, b0_conv_rq, b0_exp_dq, b0_exp_rq, b0_con_dq,
            result);

        // Add skip from global input (L1 cached from tile load)
        const int gy = tile_y0 + ty;
        const int gx = tile_x0 + tx;
        if (gy < H && gx < W) {
            const __half* skip_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int co = 0; co < C; co++)
                result[co] += __half2float(skip_ptr[co]);
        }

        // Save as FP16 for block 1's skip connection
        #pragma unroll
        for (int co = 0; co < C; co++)
            my_skip[co] = __float2half(result[co]);

        // Quantize to INT8 for block 1's conv3x3 input (uses block 1's act scale)
        const int inter_base = (r * INTER_W + c) * C_G;
        #pragma unroll
        for (int cg = 0; cg < C_G; cg++) {
            int qa = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 0]) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 1]) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 2]) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 3]) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }

    // B.2: Threads 0-67 also compute one halo pixel each
    if (threadIdx.x < 68) {
        int hr, hc;
        if (threadIdx.x < 18) {
            hr = 0;  hc = threadIdx.x;           // top row
        } else if (threadIdx.x < 36) {
            hr = 17; hc = threadIdx.x - 18;      // bottom row
        } else if (threadIdx.x < 52) {
            hr = threadIdx.x - 36 + 1; hc = 0;   // left col (rows 1-16)
        } else {
            hr = threadIdx.x - 52 + 1; hc = 17;  // right col (rows 1-16)
        }

        float result[C];
        convnext_block_dp4a<C>(
            s_tile, TILE_W_EXP, hr, hc,
            s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
            b0_conv_dq, b0_conv_rq, b0_exp_dq, b0_exp_rq, b0_con_dq,
            result);

        // Add skip from global input (bounds-checked)
        const int gy = tile_y0 - 1 + hr;
        const int gx = tile_x0 - 1 + hc;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* skip_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int co = 0; co < C; co++)
                result[co] += __half2float(skip_ptr[co]);
        }

        // Quantize to INT8, write to s_inter (no skip save needed)
        const int inter_base = (hr * INTER_W + hc) * C_G;
        #pragma unroll
        for (int cg = 0; cg < C_G; cg++) {
            __half h0 = __float2half(result[cg * 4 + 0]);
            __half h1 = __float2half(result[cg * 4 + 1]);
            __half h2 = __float2half(result[cg * 4 + 2]);
            __half h3 = __float2half(result[cg * 4 + 3]);
            int qa = max(-127, min(127, __float2int_rn(__half2float(h0) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(__half2float(h1) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(__half2float(h2) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(__half2float(h3) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }
    __syncthreads();  // Barrier 2: s_inter complete

    // =================================================================
    // PHASE C: Reload block 1 weights
    // =================================================================

    // Block 1 conv3x3 weights
    {
        const int total = C * 9 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int sp   = (i / C_G) % 9;
            const int co   = i / (C_G * 9);
            int a = (int)b1_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b1_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b1_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b1_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 PW expand weights
    {
        const int total = C2 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int co   = i / C_G;
            const int base = co * C + ci_g * 4;
            int a = (int)b1_exp_w[base + 0];
            int b = (int)b1_exp_w[base + 1];
            int c = (int)b1_exp_w[base + 2];
            int d = (int)b1_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 PW contract weights
    {
        const int total = C * C2_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C2_G;
            const int co   = i / C2_G;
            const int base = co * C2 + ci_g * 4;
            int a = (int)b1_con_w[base + 0];
            int b = (int)b1_con_w[base + 1];
            int c = (int)b1_con_w[base + 2];
            int d = (int)b1_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 biases
    if (threadIdx.x < C)  s_conv_b[threadIdx.x] = b1_conv_b[threadIdx.x];
    if (threadIdx.x < C)  s_con_b[threadIdx.x]  = b1_con_b[threadIdx.x];
    if (threadIdx.x < C2) s_exp_b[threadIdx.x]  = b1_exp_b[threadIdx.x];
    __syncthreads();  // Barrier 3: block 1 weights ready

    // =================================================================
    // PHASE D: Compute ConvNext block 1 for 16×16 inner region
    //   Conv3x3 from s_inter, skip from my_skip registers.
    //   Write result to global skip2_out.
    // =================================================================
    {
        const int gx = tile_x0 + tx;
        const int gy = tile_y0 + ty;

        if (gx < W && gy < H) {
            float result[C];
            convnext_block_dp4a<C>(
                s_inter, INTER_W, ty, tx,
                s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
                b1_conv_dq, b1_conv_rq, b1_exp_dq, b1_exp_rq, b1_con_dq,
                result);

            // Add skip (block 0 output, saved in my_skip)
            const int pix = gy * W + gx;
            __half* out_ptr = skip2_out + pix * C;
            #pragma unroll
            for (int co = 0; co < C; co++) {
                result[co] += __half2float(my_skip[co]);
                out_ptr[co] = __float2half(result[co]);
            }
        }
    }
    __syncthreads();  // Barrier 4: skip2 global writes visible to block

    // =================================================================
    // PHASE E: Downscale k=2 s=2 (16ch → 32ch)
    //   64 threads read 2×2 from skip2_out (L1 cached), compute
    //   downscale, write 32ch output to ds_out.
    // =================================================================

    constexpr int DS_TILE = TILE_W / 2;  // 8
    const int ds_pix = threadIdx.x;
    if (ds_pix < DS_TILE * DS_TILE) {
        const int ds_row = ds_pix / DS_TILE;
        const int ds_col = ds_pix % DS_TILE;

        // Block 1 source positions in half-res
        const int b1_gy = tile_y0 + ds_row * 2;
        const int b1_gx = tile_x0 + ds_col * 2;

        // Quarter-res output position
        const int H4 = H / 2, W4 = W / 2;
        const int ds_gy = tile_y0 / 2 + ds_row;
        const int ds_gx = tile_x0 / 2 + ds_col;

        if (ds_gy < H4 && ds_gx < W4 &&
            b1_gy + 1 < H && b1_gx + 1 < W) {
            // Read 2×2 neighborhood from skip2_out (just written, L1 hot)
            const __half* p00 = skip2_out + (b1_gy       * W + b1_gx)     * C;
            const __half* p01 = skip2_out + (b1_gy       * W + b1_gx + 1) * C;
            const __half* p10 = skip2_out + ((b1_gy + 1) * W + b1_gx)     * C;
            const __half* p11 = skip2_out + ((b1_gy + 1) * W + b1_gx + 1) * C;

            float acc[DS_C_OUT];
            #pragma unroll
            for (int co = 0; co < DS_C_OUT; co++) acc[co] = 0.0f;

            #pragma unroll
            for (int ci = 0; ci < C; ci++) {
                float i00 = __half2float(p00[ci]);
                float i01 = __half2float(p01[ci]);
                float i10 = __half2float(p10[ci]);
                float i11 = __half2float(p11[ci]);

                #pragma unroll
                for (int co = 0; co < DS_C_OUT; co++) {
                    const int w_base = co * DS_K + ci * 4;
                    acc[co] += i00 * (float)ds_w[w_base + 0]
                             + i01 * (float)ds_w[w_base + 1]
                             + i10 * (float)ds_w[w_base + 2]
                             + i11 * (float)ds_w[w_base + 3];
                }
            }

            __half* out_ptr = ds_out + (ds_gy * W4 + ds_gx) * DS_C_OUT;
            #pragma unroll
            for (int co = 0; co < DS_C_OUT; co++) {
                float val = acc[co] * __half2float(ds_wscale[co])
                          + __half2float(ds_bias[co]);
                out_ptr[co] = __float2half(val);
            }
        }
    }
}

// Launcher for enc2_mega kernel
static void launch_enc2_mega(
    const __half* input,
    const ConvNextBlock& block0, const ConvNextDP4AScales& scales0,
    const ConvNextBlock& block1, const ConvNextDP4AScales& scales1,
    const ConvLayer& downscale,
    __half* skip2_out, __half* ds_out,
    int H, int W, cudaStream_t stream)
{
    constexpr int C = 16;
    constexpr int TILE_W = 16, TILE_H = 16;

    // Block 0 quantization scales
    float b0_act_rcp    = 1.0f / scales0.input_act_scale;
    float b0_conv_dq    = scales0.input_act_scale * block0.conv.quant_scale;
    float b0_conv_rq    = scales0.conv_requant_rcp;
    float b0_exp_in_s   = 1.0f / scales0.conv_requant_rcp;
    float b0_exp_dq     = b0_exp_in_s * block0.pw_expand.quant_scale;
    float b0_exp_rq     = scales0.pw_exp_requant_rcp;
    float b0_con_in_s   = 1.0f / scales0.pw_exp_requant_rcp;
    float b0_con_dq     = b0_con_in_s * block0.pw_contract.quant_scale;

    // Block 1 quantization scales
    float b1_act_rcp    = 1.0f / scales1.input_act_scale;
    float b1_conv_dq    = scales1.input_act_scale * block1.conv.quant_scale;
    float b1_conv_rq    = scales1.conv_requant_rcp;
    float b1_exp_in_s   = 1.0f / scales1.conv_requant_rcp;
    float b1_exp_dq     = b1_exp_in_s * block1.pw_expand.quant_scale;
    float b1_exp_rq     = scales1.pw_exp_requant_rcp;
    float b1_con_in_s   = 1.0f / scales1.pw_exp_requant_rcp;
    float b1_con_dq     = b1_con_in_s * block1.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    enc2_mega_nhwc_kernel<C, TILE_W, TILE_H><<<grid, 256, 0, stream>>>(
        input,
        block0.conv.weight, block0.conv.bias,
        block0.pw_expand.weight, block0.pw_expand.bias,
        block0.pw_contract.weight, block0.pw_contract.bias,
        block1.conv.weight, block1.conv.bias,
        block1.pw_expand.weight, block1.pw_expand.bias,
        block1.pw_contract.weight, block1.pw_contract.bias,
        downscale.weight, downscale.bias, downscale.w_scale,
        skip2_out, ds_out, H, W,
        b0_act_rcp, b0_conv_dq, b0_conv_rq,
        b0_exp_dq, b0_exp_rq, b0_con_dq,
        b1_act_rcp, b1_conv_dq, b1_conv_rq,
        b1_exp_dq, b1_exp_rq, b1_con_dq);
}

// =========================================================================
// Enc3 Mega-Kernel: 2× FasterNetBlock(32ch) + downscale(32→64)
//
// Fuses three kernel launches into one. Block 1's conv3x3 reads from
// s_inter (shared memory), passthrough and skip from registers.
//
// Phase A: Load 20×12 input tile (lower 16ch) + block 0 weights.
// Phase B: Compute FasterNet block 0 for 18×10 region (128 inner + 52 halo),
//          save full 32ch in registers, write lower 16ch INT8 to s_inter.
// Phase C: Reload block 1 weights.
// Phase D: Compute FasterNet block 1 for 16×8 inner, write skip3 to global.
// Phase E: 32 threads compute k2s2 downscale 32→64 (read skip3 from L1).
//
// Traffic savings: eliminates block 0→1 global round-trip (~7 MB at quarter-res).
// =========================================================================

template<int TILE_W = 16, int TILE_H = 8>
static __global__ void __launch_bounds__(128, 4) enc3_mega_nhwc_kernel(
    const __half* __restrict__ input,           // [H*W, 32] NHWC
    // Block 0 weights (spatial conv partial 16ch)
    const int8_t* __restrict__ b0_conv_w,
    const __half* __restrict__ b0_conv_b,
    const int8_t* __restrict__ b0_exp_w,
    const __half* __restrict__ b0_exp_b,
    const int8_t* __restrict__ b0_con_w,
    const __half* __restrict__ b0_con_b,
    // Block 1 weights
    const int8_t* __restrict__ b1_conv_w,
    const __half* __restrict__ b1_conv_b,
    const int8_t* __restrict__ b1_exp_w,
    const __half* __restrict__ b1_exp_b,
    const int8_t* __restrict__ b1_con_w,
    const __half* __restrict__ b1_con_b,
    // Downscale weights
    const int8_t* __restrict__ ds_w,            // [64, 128] INT8
    const __half* __restrict__ ds_bias,         // [64]
    const __half* __restrict__ ds_wscale,       // [64]
    // Outputs
    __half*       __restrict__ skip3_out,       // [H*W, 32] skip3
    __half*       __restrict__ ds_out,          // [H/2*W/2, 64]
    int H, int W,
    // Block 0 scales
    float b0_act_rcp, float b0_conv_dq, float b0_concat_rcp,
    float b0_pw_exp_dq, float b0_pw_exp_rq, float b0_pw_con_dq,
    // Block 1 scales
    float b1_act_rcp, float b1_conv_dq, float b1_concat_rcp,
    float b1_pw_exp_dq, float b1_pw_exp_rq, float b1_pw_con_dq)
{
    constexpr int C_PART = 16;
    constexpr int C_STAGE = 32;
    constexpr int C_EXP = 64;
    constexpr int PART_G = C_PART / 4;     // 4
    constexpr int STAGE_G = C_STAGE / 4;   // 8
    constexpr int EXP_G = C_EXP / 4;       // 16
    constexpr int K3 = C_PART * 9;         // 144

    constexpr int DS_C_OUT = 64;
    constexpr int DS_K = C_STAGE * 4;      // 128

    constexpr int TILE_W_EXP = TILE_W + 4;  // 20
    constexpr int TILE_H_EXP = TILE_H + 4;  // 12
    constexpr int TILE_PIXELS_EXP = TILE_W_EXP * TILE_H_EXP;  // 240

    constexpr int INTER_W = TILE_W + 2;  // 18
    constexpr int INTER_H = TILE_H + 2;  // 10
    constexpr int INTER_PIXELS = INTER_W * INTER_H;  // 180

    // --- Shared memory ---
    __shared__ int s_tile[TILE_PIXELS_EXP * PART_G];  // 960 ints = 3840 B
    __shared__ int s_inter[INTER_PIXELS * PART_G];    // 720 ints = 2880 B
    __shared__ int s_conv_w[C_PART * 9 * PART_G];     // 576 ints = 2304 B
    __shared__ int s_exp_w[C_EXP * STAGE_G];          // 512 ints = 2048 B
    __shared__ int s_con_w[C_STAGE * EXP_G];          // 512 ints = 2048 B
    __shared__ __half s_conv_b[C_PART];               // 32 B
    __shared__ __half s_exp_b[C_EXP];                 // 128 B
    __shared__ __half s_con_b[C_STAGE];               // 64 B
    // Total: 3840 + 2880 + 2304 + 2048 + 2048 + 32 + 128 + 64 = 13,344 B

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;
    const int tx = threadIdx.x % TILE_W;  // 0..15
    const int ty = threadIdx.x / TILE_W;  // 0..7

    // =================================================================
    // PHASE A: Load 20×12 input tile (lower 16ch) + block 0 weights
    // =================================================================

    // Block 0 conv3x3 weights (dp4a packed)
    {
        const int total = C_PART * 9 * PART_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % PART_G;
            const int sp   = (i / PART_G) % 9;
            const int co   = i / (PART_G * 9);
            int a = (int)b0_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b0_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b0_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b0_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW expand weights [64, 32] → packed
    {
        const int total = C_EXP * STAGE_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % STAGE_G;
            const int co   = i / STAGE_G;
            const int base = co * C_STAGE + ci_g * 4;
            int a = (int)b0_exp_w[base + 0];
            int b = (int)b0_exp_w[base + 1];
            int c = (int)b0_exp_w[base + 2];
            int d = (int)b0_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW contract weights [32, 64] → packed
    {
        const int total = C_STAGE * EXP_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % EXP_G;
            const int co   = i / EXP_G;
            const int base = co * C_EXP + ci_g * 4;
            int a = (int)b0_con_w[base + 0];
            int b = (int)b0_con_w[base + 1];
            int c = (int)b0_con_w[base + 2];
            int d = (int)b0_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 biases
    if (threadIdx.x < C_PART) s_conv_b[threadIdx.x] = b0_conv_b[threadIdx.x];
    if (threadIdx.x < C_EXP)  s_exp_b[threadIdx.x]  = b0_exp_b[threadIdx.x];
    if (threadIdx.x < C_STAGE) s_con_b[threadIdx.x] = b0_con_b[threadIdx.x];

    // 20×12 input tile: lower 16ch with FP16→INT8 pre-quantization
    for (int i = threadIdx.x; i < TILE_PIXELS_EXP; i += blockDim.x) {
        const int tr = i / TILE_W_EXP;
        const int tc = i % TILE_W_EXP;
        const int gy = tile_y0 + tr - 2;  // -2 for double halo
        const int gx = tile_x0 + tc - 2;

        const int tile_base = i * PART_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * b0_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * b0_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * b0_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * b0_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();  // Barrier 1: tile + block 0 weights ready

    // =================================================================
    // PHASE B: Compute FasterNet block 0 for 18×10 region
    //   - Inner pixel (ty+1, tx+1) → save 32ch in registers, write lower 16ch to s_inter
    //   - Halo pixels (threads 0-51) → write lower 16ch to s_inter only
    // =================================================================

    __half my_res0[C_STAGE];  // Block 0 output for this thread's inner pixel

    // B.1: Each thread computes its inner pixel at 18×10 pos (ty+1, tx+1)
    {
        const int ir = ty + 1, ic = tx + 1;  // 18×10 grid position
        const int gy = tile_y0 + ty;
        const int gx = tile_x0 + tx;

        float result[C_STAGE];
        if (gy < H && gx < W) {
            // Upper 16ch passthrough from global
            const __half* upper = input + (gy * W + gx) * C_STAGE + C_PART;
            fasternet32_block_dp4a(
                s_tile, TILE_W_EXP, ir, ic,
                upper,
                s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
                b0_conv_dq, b0_concat_rcp,
                b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
                result);

            // Add skip from global input (all 32ch)
            const __half* skip_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int co = 0; co < C_STAGE; co++) {
                result[co] += __half2float(skip_ptr[co]);
                my_res0[co] = __float2half(result[co]);
            }
        } else {
            #pragma unroll
            for (int co = 0; co < C_STAGE; co++)
                my_res0[co] = __float2half(0.0f);
        }

        // Quantize lower 16ch to INT8 for block 1's conv3x3
        const int inter_base = (ir * INTER_W + ic) * PART_G;
        #pragma unroll
        for (int cg = 0; cg < PART_G; cg++) {
            int qa = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 0]) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 1]) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 2]) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 3]) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }

    // B.2: Threads 0-51 also compute one halo pixel each
    if (threadIdx.x < 52) {
        int hr, hc;
        if (threadIdx.x < 18) {
            hr = 0;  hc = threadIdx.x;               // top row (18 pixels)
        } else if (threadIdx.x < 36) {
            hr = 9;  hc = threadIdx.x - 18;          // bottom row (18 pixels)
        } else if (threadIdx.x < 44) {
            hr = threadIdx.x - 36 + 1; hc = 0;       // left col rows 1-8
        } else {
            hr = threadIdx.x - 44 + 1; hc = 17;      // right col rows 1-8
        }

        const int gy = tile_y0 + hr - 1;
        const int gx = tile_x0 + hc - 1;

        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* upper = input + (gy * W + gx) * C_STAGE + C_PART;
            float result[C_STAGE];
            fasternet32_block_dp4a(
                s_tile, TILE_W_EXP, hr, hc,
                upper,
                s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
                b0_conv_dq, b0_concat_rcp,
                b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
                result);

            // Add skip, quantize lower 16ch to s_inter (no register save needed)
            const __half* skip_ptr = input + (gy * W + gx) * C_STAGE;
            const int inter_base = (hr * INTER_W + hc) * PART_G;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                __half h0 = __float2half(result[cg * 4 + 0] + __half2float(skip_ptr[cg * 4 + 0]));
                __half h1 = __float2half(result[cg * 4 + 1] + __half2float(skip_ptr[cg * 4 + 1]));
                __half h2 = __float2half(result[cg * 4 + 2] + __half2float(skip_ptr[cg * 4 + 2]));
                __half h3 = __float2half(result[cg * 4 + 3] + __half2float(skip_ptr[cg * 4 + 3]));
                int qa = max(-127, min(127, __float2int_rn(__half2float(h0) * b1_act_rcp)));
                int qb = max(-127, min(127, __float2int_rn(__half2float(h1) * b1_act_rcp)));
                int qc = max(-127, min(127, __float2int_rn(__half2float(h2) * b1_act_rcp)));
                int qd = max(-127, min(127, __float2int_rn(__half2float(h3) * b1_act_rcp)));
                s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                            ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            // Out-of-bounds: zero padding
            const int inter_base = (hr * INTER_W + hc) * PART_G;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++)
                s_inter[inter_base + cg] = 0;
        }
    }
    __syncthreads();  // Barrier 2: s_inter complete

    // =================================================================
    // PHASE C: Reload block 1 weights
    // =================================================================

    {
        const int total = C_PART * 9 * PART_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % PART_G;
            const int sp   = (i / PART_G) % 9;
            const int co   = i / (PART_G * 9);
            int a = (int)b1_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b1_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b1_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b1_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }
    {
        const int total = C_EXP * STAGE_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % STAGE_G;
            const int co   = i / STAGE_G;
            const int base = co * C_STAGE + ci_g * 4;
            int a = (int)b1_exp_w[base + 0];
            int b = (int)b1_exp_w[base + 1];
            int c = (int)b1_exp_w[base + 2];
            int d = (int)b1_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }
    {
        const int total = C_STAGE * EXP_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % EXP_G;
            const int co   = i / EXP_G;
            const int base = co * C_EXP + ci_g * 4;
            int a = (int)b1_con_w[base + 0];
            int b = (int)b1_con_w[base + 1];
            int c = (int)b1_con_w[base + 2];
            int d = (int)b1_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }
    if (threadIdx.x < C_PART) s_conv_b[threadIdx.x] = b1_conv_b[threadIdx.x];
    if (threadIdx.x < C_EXP)  s_exp_b[threadIdx.x]  = b1_exp_b[threadIdx.x];
    if (threadIdx.x < C_STAGE) s_con_b[threadIdx.x] = b1_con_b[threadIdx.x];
    __syncthreads();  // Barrier 3: block 1 weights ready

    // =================================================================
    // PHASE D: Compute FasterNet block 1 for 16×8 inner region
    //   Conv3x3 from s_inter, upper passthrough + skip from my_res0 registers.
    //   Write result to global skip3_out.
    // =================================================================
    {
        const int gx = tile_x0 + tx;
        const int gy = tile_y0 + ty;

        if (gx < W && gy < H) {
            // Upper 16ch passthrough from block 0's output (registers)
            const __half* upper = my_res0 + C_PART;
            float result[C_STAGE];
            fasternet32_block_dp4a(
                s_inter, INTER_W, ty, tx,
                upper,
                s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
                b1_conv_dq, b1_concat_rcp,
                b1_pw_exp_dq, b1_pw_exp_rq, b1_pw_con_dq,
                result);

            // Add skip (block 0 output from my_res0 registers)
            const int pix = gy * W + gx;
            __half* out_ptr = skip3_out + pix * C_STAGE;
            #pragma unroll
            for (int co = 0; co < C_STAGE; co++) {
                result[co] += __half2float(my_res0[co]);
                out_ptr[co] = __float2half(result[co]);
            }
        }
    }
    __syncthreads();  // Barrier 4: skip3 global writes visible to block

    // =================================================================
    // PHASE E: Downscale k=2 s=2 (32ch → 64ch)
    //   32 threads read 2×2 from skip3_out (L1 cached), compute
    //   downscale with two-half split for register pressure, write to ds_out.
    // =================================================================

    constexpr int DS_TILE_W = TILE_W / 2;  // 8
    constexpr int DS_TILE_H = TILE_H / 2;  // 4
    constexpr int DS_HALF = DS_C_OUT / 2;  // 32
    const int ds_pix = threadIdx.x;
    if (ds_pix < DS_TILE_W * DS_TILE_H) {
        const int ds_row = ds_pix / DS_TILE_W;
        const int ds_col = ds_pix % DS_TILE_W;

        // Source positions in quarter-res
        const int src_gy = tile_y0 + ds_row * 2;
        const int src_gx = tile_x0 + ds_col * 2;

        // Eighth-res output position
        const int H_out = H / 2, W_out = W / 2;
        const int ds_gy = tile_y0 / 2 + ds_row;
        const int ds_gx = tile_x0 / 2 + ds_col;

        if (ds_gy < H_out && ds_gx < W_out &&
            src_gy + 1 < H && src_gx + 1 < W) {
            // Read 2×2 neighborhood from skip3_out (just written, L1 hot)
            const __half* p00 = skip3_out + (src_gy       * W + src_gx)     * C_STAGE;
            const __half* p01 = skip3_out + (src_gy       * W + src_gx + 1) * C_STAGE;
            const __half* p10 = skip3_out + ((src_gy + 1) * W + src_gx)     * C_STAGE;
            const __half* p11 = skip3_out + ((src_gy + 1) * W + src_gx + 1) * C_STAGE;

            __half* out_ptr = ds_out + (ds_gy * W_out + ds_gx) * DS_C_OUT;

            // Two-half split to cap register pressure at ~32 floats
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                const int co_base = half * DS_HALF;
                float acc[DS_HALF];
                #pragma unroll
                for (int co = 0; co < DS_HALF; co++) acc[co] = 0.0f;

                #pragma unroll
                for (int ci = 0; ci < C_STAGE; ci++) {
                    float i00 = __half2float(p00[ci]);
                    float i01 = __half2float(p01[ci]);
                    float i10 = __half2float(p10[ci]);
                    float i11 = __half2float(p11[ci]);

                    #pragma unroll
                    for (int co = 0; co < DS_HALF; co++) {
                        const int w_base = (co_base + co) * DS_K + ci * 4;
                        acc[co] += i00 * (float)ds_w[w_base + 0]
                                 + i01 * (float)ds_w[w_base + 1]
                                 + i10 * (float)ds_w[w_base + 2]
                                 + i11 * (float)ds_w[w_base + 3];
                    }
                }

                #pragma unroll
                for (int co = 0; co < DS_HALF; co++) {
                    float val = acc[co] * __half2float(ds_wscale[co_base + co])
                              + __half2float(ds_bias[co_base + co]);
                    out_ptr[co_base + co] = __float2half(val);
                }
            }
        }
    }
}

// Launcher for enc3_mega kernel
static void launch_enc3_mega(
    const __half* input,
    const FasterNetBlock& block0, const FasterNetDP4AScales& scales0,
    const FasterNetBlock& block1, const FasterNetDP4AScales& scales1,
    const ConvLayer& downscale,
    __half* skip3_out, __half* ds_out,
    int H, int W, cudaStream_t stream)
{
    constexpr int TILE_W = 16, TILE_H = 8;

    // Block 0 quantization scales
    float b0_act_rcp     = 1.0f / scales0.quantScale0;
    float b0_conv_dq     = scales0.quantScale0 * block0.spatial.quant_scale;
    float b0_concat_rcp  = 1.0f / scales0.quantScale1;
    float b0_pw_exp_dq   = scales0.quantScale1 * block0.pw_expand.quant_scale;
    float b0_pw_exp_rq   = 1.0f / scales0.activationScaleFactor;
    float b0_pw_con_dq   = scales0.activationScaleFactor * block0.pw_contract.quant_scale;

    // Block 1 quantization scales
    float b1_act_rcp     = 1.0f / scales1.quantScale0;
    float b1_conv_dq     = scales1.quantScale0 * block1.spatial.quant_scale;
    float b1_concat_rcp  = 1.0f / scales1.quantScale1;
    float b1_pw_exp_dq   = scales1.quantScale1 * block1.pw_expand.quant_scale;
    float b1_pw_exp_rq   = 1.0f / scales1.activationScaleFactor;
    float b1_pw_con_dq   = scales1.activationScaleFactor * block1.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    enc3_mega_nhwc_kernel<TILE_W, TILE_H><<<grid, 128, 0, stream>>>(
        input,
        block0.spatial.weight, block0.spatial.bias,
        block0.pw_expand.weight, block0.pw_expand.bias,
        block0.pw_contract.weight, block0.pw_contract.bias,
        block1.spatial.weight, block1.spatial.bias,
        block1.pw_expand.weight, block1.pw_expand.bias,
        block1.pw_contract.weight, block1.pw_contract.bias,
        downscale.weight, downscale.bias, downscale.w_scale,
        skip3_out, ds_out, H, W,
        b0_act_rcp, b0_conv_dq, b0_concat_rcp,
        b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
        b1_act_rcp, b1_conv_dq, b1_concat_rcp,
        b1_pw_exp_dq, b1_pw_exp_rq, b1_pw_con_dq);
}

// =========================================================================
// BN Mega-Kernel: 2× FasterNetBlock(64ch) fused
//
// Fuses bn_res0 + bn_res1 into a single kernel. Block 0's output stays
// in shared memory (lower 32ch tiled) and registers (full 64ch per inner
// thread). Block 1 reads conv3x3 input from smem and passthrough/skip
// from registers — zero global traffic for block 1 inner pixels.
//
// Phase A: Load 20×12 input tile (lower 32ch, 2-pixel halo) + block 0 weights.
// Phase B: Compute FasterNet64 block 0 for 18×10 region (128 inner + 52 halo).
//          Inner: save full 64ch in registers, write lower 32ch to s_inter.
//          Halo: write lower 32ch to s_inter only.
// Phase C: Reload block 1 weights.
// Phase D: Compute FasterNet64 block 1 for 16×8 inner only.
//          Conv3x3 from s_inter, passthrough+skip from registers.
//          Write output to global.
//
// Traffic savings: ~27 KB per tile (eliminates block 1 tile load + passthrough
// + skip reads from global). ~3 MB at 720p, ~12 MB at 1440p.
// =========================================================================

template<int TILE_W = 16, int TILE_H = 8>
static __global__ void __launch_bounds__(128, 3) bn_mega_nhwc_kernel(
    const __half* __restrict__ input,           // [H*W, 64] NHWC
    // Block 0 weights
    const int8_t* __restrict__ b0_conv_w,       // [32, 32*9] INT8
    const __half* __restrict__ b0_conv_b,       // [32]
    const int8_t* __restrict__ b0_exp_w,        // [128, 64] INT8
    const __half* __restrict__ b0_exp_b,        // [128]
    const int8_t* __restrict__ b0_con_w,        // [64, 128] INT8
    const __half* __restrict__ b0_con_b,        // [64]
    // Block 1 weights
    const int8_t* __restrict__ b1_conv_w,
    const __half* __restrict__ b1_conv_b,
    const int8_t* __restrict__ b1_exp_w,
    const __half* __restrict__ b1_exp_b,
    const int8_t* __restrict__ b1_con_w,
    const __half* __restrict__ b1_con_b,
    __half*       __restrict__ output,          // [H*W, 64] NHWC
    int H, int W,
    // Block 0 scales
    float b0_conv_act_rcp, float b0_conv_dq, float b0_concat_rcp,
    float b0_pw_exp_dq, float b0_pw_exp_rq, float b0_pw_con_dq,
    // Block 1 scales
    float b1_conv_act_rcp, float b1_conv_dq, float b1_concat_rcp,
    float b1_pw_exp_dq, float b1_pw_exp_rq, float b1_pw_con_dq)
{
    constexpr int C_PART = 32;
    constexpr int C_STAGE = 64;
    constexpr int C_EXP = 128;
    constexpr int K3 = C_PART * 9;

    constexpr int PART_G = C_PART / 4;     // 8
    constexpr int STAGE_G = C_STAGE / 4;   // 16
    constexpr int EXP_G = C_EXP / 4;       // 32
    constexpr int HALF_EXP = C_EXP / 2;    // 64
    constexpr int HALF_EXP_G = HALF_EXP / 4; // 16

    constexpr int TILE_W_EXP = TILE_W + 4;  // 20
    constexpr int TILE_H_EXP = TILE_H + 4;  // 12
    constexpr int TILE_PIXELS_EXP = TILE_W_EXP * TILE_H_EXP;  // 240

    constexpr int INTER_W = TILE_W + 2;  // 18
    constexpr int INTER_H = TILE_H + 2;  // 10
    constexpr int INTER_PIXELS = INTER_W * INTER_H;  // 180

    // --- Shared memory (3 separate weight arrays, all loaded at once per block) ---
    __shared__ int s_tile[TILE_PIXELS_EXP * PART_G];   // 240*8 = 7680 B
    __shared__ int s_inter[INTER_PIXELS * PART_G];      // 180*8 = 5760 B
    __shared__ int s_conv_w[C_PART * 9 * PART_G];       // 32*9*8 = 9216 B
    __shared__ int s_exp_w[C_EXP * STAGE_G];             // 128*16 = 8192 B
    __shared__ int s_con_w[C_STAGE * EXP_G];             // 64*32 = 8192 B
    __shared__ __half s_conv_b[C_PART];                   // 64 B
    __shared__ __half s_exp_b[C_EXP];                     // 256 B
    __shared__ __half s_con_b[C_STAGE];                   // 128 B
    // Total: 7680 + 5760 + 9216 + 8192 + 8192 + 64 + 256 + 128 = 39,488 B

    // Load ALL 3 weight arrays + biases for one FasterNet block at once
    #define BN_LOAD_ALL_WEIGHTS(CONV_SRC, CONV_BIAS, EXP_SRC, EXP_BIAS, CON_SRC, CON_BIAS) do { \
        for (int i = threadIdx.x; i < C_PART * 9 * PART_G; i += blockDim.x) { \
            const int ci_g = i % PART_G; \
            const int sp   = (i / PART_G) % 9; \
            const int co   = i / (PART_G * 9); \
            int a = (int)(CONV_SRC)[co * K3 + (ci_g * 4 + 0) * 9 + sp]; \
            int b = (int)(CONV_SRC)[co * K3 + (ci_g * 4 + 1) * 9 + sp]; \
            int c = (int)(CONV_SRC)[co * K3 + (ci_g * 4 + 2) * 9 + sp]; \
            int d = (int)(CONV_SRC)[co * K3 + (ci_g * 4 + 3) * 9 + sp]; \
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) | \
                           ((c & 0xFF) << 16) | ((d & 0xFF) << 24); \
        } \
        for (int i = threadIdx.x; i < C_EXP * STAGE_G; i += blockDim.x) { \
            const int ci_g = i % STAGE_G; \
            const int co   = i / STAGE_G; \
            const int base = co * C_STAGE + ci_g * 4; \
            int a = (int)(EXP_SRC)[base + 0]; int b = (int)(EXP_SRC)[base + 1]; \
            int c = (int)(EXP_SRC)[base + 2]; int d = (int)(EXP_SRC)[base + 3]; \
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) | \
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24); \
        } \
        for (int i = threadIdx.x; i < C_STAGE * EXP_G; i += blockDim.x) { \
            const int ci_g = i % EXP_G; \
            const int co   = i / EXP_G; \
            const int base = co * C_EXP + ci_g * 4; \
            int a = (int)(CON_SRC)[base + 0]; int b = (int)(CON_SRC)[base + 1]; \
            int c = (int)(CON_SRC)[base + 2]; int d = (int)(CON_SRC)[base + 3]; \
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) | \
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24); \
        } \
        for (int i = threadIdx.x; i < C_PART; i += blockDim.x) \
            s_conv_b[i] = (CONV_BIAS)[i]; \
        for (int i = threadIdx.x; i < C_EXP; i += blockDim.x) \
            s_exp_b[i] = (EXP_BIAS)[i]; \
        for (int i = threadIdx.x; i < C_STAGE; i += blockDim.x) \
            s_con_b[i] = (CON_BIAS)[i]; \
    } while(0)

    // Computation macros (each reads from its own named weight array)
    #define BN_COMPUTE_CONV3x3_CONCAT(TILE_PTR, TILE_STRIDE, TR, TC, \
        PIX_PTR, CONV_DQ, CONCAT_RCP, OUT_CQ) \
    do { \
        int conv_acc_m[C_PART]; \
        _Pragma("unroll") \
        for (int co = 0; co < C_PART; co++) conv_acc_m[co] = 0; \
        _Pragma("unroll") \
        for (int kh = 0; kh < 3; kh++) { \
            _Pragma("unroll") \
            for (int kw = 0; kw < 3; kw++) { \
                const int tidx = (((TR) + kh) * (TILE_STRIDE) + ((TC) + kw)) * PART_G; \
                const int sp = kh * 3 + kw; \
                _Pragma("unroll") \
                for (int co = 0; co < C_PART; co++) { \
                    _Pragma("unroll") \
                    for (int cg = 0; cg < PART_G; cg++) \
                        conv_acc_m[co] = __dp4a((TILE_PTR)[tidx + cg], \
                            s_conv_w[co * 9 * PART_G + sp * PART_G + cg], conv_acc_m[co]); \
                } \
            } \
        } \
        _Pragma("unroll") \
        for (int cg = 0; cg < PART_G; cg++) { \
            float v0 = (float)conv_acc_m[cg*4+0] * (CONV_DQ) + __half2float(s_conv_b[cg*4+0]); \
            float v1 = (float)conv_acc_m[cg*4+1] * (CONV_DQ) + __half2float(s_conv_b[cg*4+1]); \
            float v2 = (float)conv_acc_m[cg*4+2] * (CONV_DQ) + __half2float(s_conv_b[cg*4+2]); \
            float v3 = (float)conv_acc_m[cg*4+3] * (CONV_DQ) + __half2float(s_conv_b[cg*4+3]); \
            int q0 = max(-127, min(127, __float2int_rn(v0 * (CONCAT_RCP)))); \
            int q1 = max(-127, min(127, __float2int_rn(v1 * (CONCAT_RCP)))); \
            int q2 = max(-127, min(127, __float2int_rn(v2 * (CONCAT_RCP)))); \
            int q3 = max(-127, min(127, __float2int_rn(v3 * (CONCAT_RCP)))); \
            (OUT_CQ)[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | \
                             ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24); \
        } \
        const __half* upper_m = (PIX_PTR) + C_PART; \
        _Pragma("unroll") \
        for (int cg = 0; cg < PART_G; cg++) { \
            int qa = max(-127, min(127, __float2int_rn(__half2float(upper_m[cg*4+0]) * (CONCAT_RCP)))); \
            int qb = max(-127, min(127, __float2int_rn(__half2float(upper_m[cg*4+1]) * (CONCAT_RCP)))); \
            int qc = max(-127, min(127, __float2int_rn(__half2float(upper_m[cg*4+2]) * (CONCAT_RCP)))); \
            int qd = max(-127, min(127, __float2int_rn(__half2float(upper_m[cg*4+3]) * (CONCAT_RCP)))); \
            (OUT_CQ)[PART_G + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) | \
                                      ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24); \
        } \
    } while(0)

    #define BN_COMPUTE_PW_EXPAND(CQ, EXP_DQ, EXP_RQ, OUT_EQ) \
    do { \
        { \
            int acc_m[HALF_EXP]; \
            _Pragma("unroll") \
            for (int co = 0; co < HALF_EXP; co++) { acc_m[co] = 0; \
                _Pragma("unroll") \
                for (int cg = 0; cg < STAGE_G; cg++) \
                    acc_m[co] = __dp4a((CQ)[cg], s_exp_w[co * STAGE_G + cg], acc_m[co]); \
            } \
            _Pragma("unroll") \
            for (int cg = 0; cg < HALF_EXP_G; cg++) { \
                float v0 = (float)acc_m[cg*4+0] * (EXP_DQ) + __half2float(s_exp_b[cg*4+0]); \
                float v1 = (float)acc_m[cg*4+1] * (EXP_DQ) + __half2float(s_exp_b[cg*4+1]); \
                float v2 = (float)acc_m[cg*4+2] * (EXP_DQ) + __half2float(s_exp_b[cg*4+2]); \
                float v3 = (float)acc_m[cg*4+3] * (EXP_DQ) + __half2float(s_exp_b[cg*4+3]); \
                int q0 = max(-127, min(127, __float2int_rn(fmaxf(v0, 0.0f) * (EXP_RQ)))); \
                int q1 = max(-127, min(127, __float2int_rn(fmaxf(v1, 0.0f) * (EXP_RQ)))); \
                int q2 = max(-127, min(127, __float2int_rn(fmaxf(v2, 0.0f) * (EXP_RQ)))); \
                int q3 = max(-127, min(127, __float2int_rn(fmaxf(v3, 0.0f) * (EXP_RQ)))); \
                (OUT_EQ)[cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | \
                                ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24); \
            } \
        } \
        { \
            int acc_m[HALF_EXP]; \
            _Pragma("unroll") \
            for (int co = 0; co < HALF_EXP; co++) { acc_m[co] = 0; \
                _Pragma("unroll") \
                for (int cg = 0; cg < STAGE_G; cg++) \
                    acc_m[co] = __dp4a((CQ)[cg], s_exp_w[(co + HALF_EXP) * STAGE_G + cg], acc_m[co]); \
            } \
            _Pragma("unroll") \
            for (int cg = 0; cg < HALF_EXP_G; cg++) { \
                float v0 = (float)acc_m[cg*4+0] * (EXP_DQ) + __half2float(s_exp_b[HALF_EXP+cg*4+0]); \
                float v1 = (float)acc_m[cg*4+1] * (EXP_DQ) + __half2float(s_exp_b[HALF_EXP+cg*4+1]); \
                float v2 = (float)acc_m[cg*4+2] * (EXP_DQ) + __half2float(s_exp_b[HALF_EXP+cg*4+2]); \
                float v3 = (float)acc_m[cg*4+3] * (EXP_DQ) + __half2float(s_exp_b[HALF_EXP+cg*4+3]); \
                int q0 = max(-127, min(127, __float2int_rn(fmaxf(v0, 0.0f) * (EXP_RQ)))); \
                int q1 = max(-127, min(127, __float2int_rn(fmaxf(v1, 0.0f) * (EXP_RQ)))); \
                int q2 = max(-127, min(127, __float2int_rn(fmaxf(v2, 0.0f) * (EXP_RQ)))); \
                int q3 = max(-127, min(127, __float2int_rn(fmaxf(v3, 0.0f) * (EXP_RQ)))); \
                (OUT_EQ)[HALF_EXP_G + cg] = (q0 & 0xFF) | ((q1 & 0xFF) << 8) | \
                                              ((q2 & 0xFF) << 16) | ((q3 & 0xFF) << 24); \
            } \
        } \
    } while(0)

    #define BN_COMPUTE_PW_CONTRACT(EXP_Q, CON_DQ, RESULT_ARR) \
    do { \
        _Pragma("unroll") \
        for (int co = 0; co < C_STAGE; co++) { \
            int acc = 0; \
            _Pragma("unroll") \
            for (int cg = 0; cg < EXP_G; cg++) \
                acc = __dp4a((EXP_Q)[cg], s_con_w[co * EXP_G + cg], acc); \
            (RESULT_ARR)[co] = (float)acc * (CON_DQ) + __half2float(s_con_b[co]); \
        } \
    } while(0)

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;
    const int tx = threadIdx.x % TILE_W;  // 0..15
    const int ty = threadIdx.x / TILE_W;  // 0..7

    // Precompute inner pixel coordinates
    const int inner_gy = tile_y0 + ty;
    const int inner_gx = tile_x0 + tx;
    const bool inner_valid = (inner_gy < H && inner_gx < W);

    // Precompute halo pixel coordinates (threads 0-51)
    int halo_hr = 0, halo_hc = 0;
    const bool halo_active = (threadIdx.x < 52);
    if (halo_active) {
        if (threadIdx.x < 18)      { halo_hr = 0; halo_hc = threadIdx.x; }
        else if (threadIdx.x < 36) { halo_hr = 9; halo_hc = threadIdx.x - 18; }
        else if (threadIdx.x < 44) { halo_hr = threadIdx.x - 36 + 1; halo_hc = 0; }
        else                       { halo_hr = threadIdx.x - 44 + 1; halo_hc = 17; }
    }
    const int halo_gy = tile_y0 + halo_hr - 1;
    const int halo_gx = tile_x0 + halo_hc - 1;
    const bool halo_valid = halo_active && (halo_gy >= 0 && halo_gy < H &&
                                             halo_gx >= 0 && halo_gx < W);

    // Persistent state
    __half my_res0[C_STAGE];  // Block 0 output for this thread's inner pixel

    // =================================================================
    // PHASE A: Load 20x12 input tile + ALL block 0 weights (parallel)
    // =================================================================
    for (int i = threadIdx.x; i < TILE_PIXELS_EXP; i += blockDim.x) {
        const int tr = i / TILE_W_EXP;
        const int tc = i % TILE_W_EXP;
        const int gy = tile_y0 + tr - 2;
        const int gx = tile_x0 + tc - 2;

        const int tile_base = i * PART_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * b0_conv_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * b0_conv_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * b0_conv_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * b0_conv_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    BN_LOAD_ALL_WEIGHTS(b0_conv_w, b0_conv_b, b0_exp_w, b0_exp_b, b0_con_w, b0_con_b);
    __syncthreads();  // tile + all block 0 weights ready

    // =================================================================
    // PHASE B: Block 0 -- all 3 stages (no inter-stage syncs needed)
    //   Inner (128 threads): conv->expand->contract -> my_res0 + s_inter
    //   Halo (threads 0-51): conv->expand->contract -> s_inter only
    // =================================================================

    // Inner pixel: full pipeline
    if (inner_valid) {
        const __half* pix = input + (inner_gy * W + inner_gx) * C_STAGE;
        int concat_q_inner[STAGE_G];
        BN_COMPUTE_CONV3x3_CONCAT(s_tile, TILE_W_EXP, ty + 1, tx + 1,
            pix, b0_conv_dq, b0_concat_rcp, concat_q_inner);
        int exp_q_inner[EXP_G];
        BN_COMPUTE_PW_EXPAND(concat_q_inner, b0_pw_exp_dq, b0_pw_exp_rq, exp_q_inner);
        float result[C_STAGE];
        BN_COMPUTE_PW_CONTRACT(exp_q_inner, b0_pw_con_dq, result);
        #pragma unroll
        for (int co = 0; co < C_STAGE; co++) {
            result[co] += __half2float(pix[co]);  // skip from global input
            my_res0[co] = __float2half(result[co]);
        }
    } else {
        #pragma unroll
        for (int co = 0; co < C_STAGE; co++)
            my_res0[co] = __float2half(0.0f);
    }

    // Quantize inner pixel's lower 32ch to s_inter for block 1
    {
        const int inner_ir = ty + 1;
        const int inner_ic = tx + 1;
        const int inter_base = (inner_ir * INTER_W + inner_ic) * PART_G;
        #pragma unroll
        for (int cg = 0; cg < PART_G; cg++) {
            int qa = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 0]) * b1_conv_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 1]) * b1_conv_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 2]) * b1_conv_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 3]) * b1_conv_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }

    // Halo pixels: full pipeline -> s_inter only
    if (halo_valid) {
        const __half* pix = input + (halo_gy * W + halo_gx) * C_STAGE;
        int concat_q_halo[STAGE_G];
        BN_COMPUTE_CONV3x3_CONCAT(s_tile, TILE_W_EXP, halo_hr, halo_hc,
            pix, b0_conv_dq, b0_concat_rcp, concat_q_halo);
        int exp_q_halo[EXP_G];
        BN_COMPUTE_PW_EXPAND(concat_q_halo, b0_pw_exp_dq, b0_pw_exp_rq, exp_q_halo);
        float result[C_STAGE];
        BN_COMPUTE_PW_CONTRACT(exp_q_halo, b0_pw_con_dq, result);
        const int inter_base = (halo_hr * INTER_W + halo_hc) * PART_G;
        #pragma unroll
        for (int cg = 0; cg < PART_G; cg++) {
            __half h0 = __float2half(result[cg*4+0] + __half2float(pix[cg*4+0]));
            __half h1 = __float2half(result[cg*4+1] + __half2float(pix[cg*4+1]));
            __half h2 = __float2half(result[cg*4+2] + __half2float(pix[cg*4+2]));
            __half h3 = __float2half(result[cg*4+3] + __half2float(pix[cg*4+3]));
            int qa = max(-127, min(127, __float2int_rn(__half2float(h0) * b1_conv_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(__half2float(h1) * b1_conv_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(__half2float(h2) * b1_conv_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(__half2float(h3) * b1_conv_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    } else if (halo_active) {
        const int inter_base = (halo_hr * INTER_W + halo_hc) * PART_G;
        #pragma unroll
        for (int cg = 0; cg < PART_G; cg++)
            s_inter[inter_base + cg] = 0;
    }
    __syncthreads();  // s_inter complete, safe to overwrite weight arrays

    // =================================================================
    // PHASE C: Block 1 -- reload weights, compute from s_inter
    // =================================================================
    BN_LOAD_ALL_WEIGHTS(b1_conv_w, b1_conv_b, b1_exp_w, b1_exp_b, b1_con_w, b1_con_b);
    __syncthreads();  // block 1 weights ready

    if (inner_valid) {
        int concat_q_blk1[STAGE_G];
        // Conv3x3 from s_inter. Passthrough upper from my_res0 registers.
        BN_COMPUTE_CONV3x3_CONCAT(s_inter, INTER_W, ty, tx,
            my_res0, b1_conv_dq, b1_concat_rcp, concat_q_blk1);
        int exp_q_blk1[EXP_G];
        BN_COMPUTE_PW_EXPAND(concat_q_blk1, b1_pw_exp_dq, b1_pw_exp_rq, exp_q_blk1);
        float result[C_STAGE];
        BN_COMPUTE_PW_CONTRACT(exp_q_blk1, b1_pw_con_dq, result);

        // Add skip (block 0 output from my_res0), write to global
        const int pix = inner_gy * W + inner_gx;
        __half* out_ptr = output + pix * C_STAGE;
        #pragma unroll
        for (int co = 0; co < C_STAGE; co++) {
            result[co] += __half2float(my_res0[co]);
            out_ptr[co] = __float2half(result[co]);
        }
    }

    #undef BN_LOAD_ALL_WEIGHTS
    #undef BN_COMPUTE_CONV3x3_CONCAT
    #undef BN_COMPUTE_PW_EXPAND
    #undef BN_COMPUTE_PW_CONTRACT
}

static void launch_bn_mega(
    const __half* input,
    const FasterNetBlock& block0, const FasterNetDP4AScales& scales0,
    const FasterNetBlock& block1, const FasterNetDP4AScales& scales1,
    __half* output,
    int H, int W, cudaStream_t stream)
{
    constexpr int TILE_W = 16, TILE_H = 8;

    // Block 0 quantization scales
    float b0_act_rcp     = 1.0f / scales0.quantScale0;
    float b0_conv_dq     = scales0.quantScale0 * block0.spatial.quant_scale;
    float b0_concat_rcp  = 1.0f / scales0.quantScale1;
    float b0_pw_exp_dq   = scales0.quantScale1 * block0.pw_expand.quant_scale;
    float b0_pw_exp_rq   = 1.0f / scales0.activationScaleFactor;
    float b0_pw_con_dq   = scales0.activationScaleFactor * block0.pw_contract.quant_scale;

    // Block 1 quantization scales
    float b1_act_rcp     = 1.0f / scales1.quantScale0;
    float b1_conv_dq     = scales1.quantScale0 * block1.spatial.quant_scale;
    float b1_concat_rcp  = 1.0f / scales1.quantScale1;
    float b1_pw_exp_dq   = scales1.quantScale1 * block1.pw_expand.quant_scale;
    float b1_pw_exp_rq   = 1.0f / scales1.activationScaleFactor;
    float b1_pw_con_dq   = scales1.activationScaleFactor * block1.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    bn_mega_nhwc_kernel<TILE_W, TILE_H><<<grid, 128, 0, stream>>>(
        input,
        block0.spatial.weight, block0.spatial.bias,
        block0.pw_expand.weight, block0.pw_expand.bias,
        block0.pw_contract.weight, block0.pw_contract.bias,
        block1.spatial.weight, block1.spatial.bias,
        block1.pw_expand.weight, block1.pw_expand.bias,
        block1.pw_contract.weight, block1.pw_contract.bias,
        output, H, W,
        b0_act_rcp, b0_conv_dq, b0_concat_rcp,
        b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
        b1_act_rcp, b1_conv_dq, b1_concat_rcp,
        b1_pw_exp_dq, b1_pw_exp_rq, b1_pw_con_dq);
}

// =========================================================================
// Dec2 Mega-Kernel: 2× ConvNextBlock(16ch) + ConvTranspose upscale(16→8)
//
// Fuses three kernel launches into one. Block 1's output stays in
// registers and feeds directly to ConvTranspose — no global write/read
// round-trip between block 1 and upscale.
//
// Phase A: Load 20×20 input tile (16ch, 2-pixel halo) + all weights.
// Phase B: Compute ConvNext block 0 for 18×18 region (256 inner + 68 halo),
//          save skip in registers, write INT8 to s_inter.
// Phase C: Reload block 1 ConvNext weights.
// Phase D: Compute ConvNext block 1 for 16×16 inner, keep in registers.
// Phase E: ConvTranspose 16→8 from registers, write directly to output.
//
// Traffic savings vs 3 separate kernels: ~100 MB at half-res (1440p).
// =========================================================================

template<int C = 16, int TILE_W = 16, int TILE_H = 16>
static __global__ void __launch_bounds__(256, 3) dec2_mega_nhwc_kernel(
    const __half* __restrict__ input,       // [H*W, 16] NHWC (half-res)
    // Block 0 ConvNext weights
    const int8_t* __restrict__ b0_conv_w,
    const __half* __restrict__ b0_conv_b,
    const int8_t* __restrict__ b0_exp_w,
    const __half* __restrict__ b0_exp_b,
    const int8_t* __restrict__ b0_con_w,
    const __half* __restrict__ b0_con_b,
    // Block 1 ConvNext weights
    const int8_t* __restrict__ b1_conv_w,
    const __half* __restrict__ b1_conv_b,
    const int8_t* __restrict__ b1_exp_w,
    const __half* __restrict__ b1_exp_b,
    const int8_t* __restrict__ b1_con_w,
    const __half* __restrict__ b1_con_b,
    // ConvTranspose weights (16→8)
    const int8_t* __restrict__ ct_w,        // [4*8, 16] = [32, 16] INT8
    const __half* __restrict__ ct_bias,     // [8]
    const __half* __restrict__ ct_scale,    // [8]
    // Output
    __half*       __restrict__ output,      // [4*H*W, 8] NHWC (full-res)
    int H, int W,                           // half-res dimensions
    // Block 0 scales
    float b0_act_rcp, float b0_conv_dq, float b0_conv_rq,
    float b0_exp_dq, float b0_exp_rq, float b0_con_dq,
    // Block 1 scales
    float b1_act_rcp, float b1_conv_dq, float b1_conv_rq,
    float b1_exp_dq, float b1_exp_rq, float b1_con_dq)
{
    constexpr int C2 = 2 * C;         // 32
    constexpr int C_G = C / 4;        // 4
    constexpr int C2_G = C2 / 4;      // 8
    constexpr int K3 = C * 9;

    constexpr int CT_C_OUT = 8;       // ConvTranspose output channels
    constexpr int CT_W_ROWS = 4 * CT_C_OUT;  // 32 weight rows
    constexpr int CT_W_TOTAL = CT_W_ROWS * C; // 512 bytes

    constexpr int TILE_W_EXP = TILE_W + 4;  // 20
    constexpr int TILE_H_EXP = TILE_H + 4;  // 20
    constexpr int TILE_PIXELS_EXP = TILE_W_EXP * TILE_H_EXP;  // 400

    constexpr int INTER_W = TILE_W + 2;  // 18
    constexpr int INTER_H = TILE_H + 2;  // 18
    constexpr int INTER_PIXELS = INTER_W * INTER_H;  // 324

    // --- Shared memory ---
    __shared__ int s_tile[TILE_PIXELS_EXP * C_G];  // 1600 ints = 6400 B
    __shared__ int s_inter[INTER_PIXELS * C_G];     // 1296 ints = 5184 B
    __shared__ int s_conv_w[C * 9 * C_G];           // 576 ints = 2304 B
    __shared__ int s_exp_w[C2 * C_G];               // 128 ints = 512 B
    __shared__ int s_con_w[C * C2_G];               // 128 ints = 512 B
    __shared__ __half s_conv_b[C];                   // 32 B
    __shared__ __half s_exp_b[C2];                   // 64 B
    __shared__ __half s_con_b[C];                    // 32 B
    // ConvTranspose weights (persist from Phase A through Phase E)
    __shared__ int8_t s_ct_w[CT_W_TOTAL];           // 512 B
    __shared__ __half s_ct_b[CT_C_OUT];              // 16 B
    __shared__ __half s_ct_s[CT_C_OUT];              // 16 B
    // Total: 6400+5184+2304+512+512+32+64+32+512+16+16 = 15,584 B

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;
    const int tx = threadIdx.x % TILE_W;  // 0..15
    const int ty = threadIdx.x / TILE_W;  // 0..15

    // =================================================================
    // PHASE A: Load 20×20 input tile + block 0 weights + CT weights
    // =================================================================

    // Block 0 conv3x3 weights (dp4a packed)
    {
        const int total = C * 9 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int sp   = (i / C_G) % 9;
            const int co   = i / (C_G * 9);
            int a = (int)b0_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b0_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b0_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b0_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW expand weights
    {
        const int total = C2 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int co   = i / C_G;
            const int base = co * C + ci_g * 4;
            int a = (int)b0_exp_w[base + 0];
            int b = (int)b0_exp_w[base + 1];
            int c = (int)b0_exp_w[base + 2];
            int d = (int)b0_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW contract weights
    {
        const int total = C * C2_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C2_G;
            const int co   = i / C2_G;
            const int base = co * C2 + ci_g * 4;
            int a = (int)b0_con_w[base + 0];
            int b = (int)b0_con_w[base + 1];
            int c = (int)b0_con_w[base + 2];
            int d = (int)b0_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 biases
    if (threadIdx.x < C)  s_conv_b[threadIdx.x] = b0_conv_b[threadIdx.x];
    if (threadIdx.x < C)  s_con_b[threadIdx.x]  = b0_con_b[threadIdx.x];
    if (threadIdx.x < C2) s_exp_b[threadIdx.x]  = b0_exp_b[threadIdx.x];

    // ConvTranspose weights (persist through all phases)
    for (int i = threadIdx.x; i < CT_W_TOTAL; i += blockDim.x)
        s_ct_w[i] = ct_w[i];
    if (threadIdx.x < CT_C_OUT) {
        s_ct_b[threadIdx.x] = ct_bias[threadIdx.x];
        s_ct_s[threadIdx.x] = ct_scale[threadIdx.x];
    }

    // 20×20 input tile with FP16→INT8 pre-quantization
    for (int i = threadIdx.x; i < TILE_PIXELS_EXP; i += blockDim.x) {
        const int tr = i / TILE_W_EXP;
        const int tc = i % TILE_W_EXP;
        const int gy = tile_y0 + tr - 2;  // -2 for double halo
        const int gx = tile_x0 + tc - 2;

        const int tile_base = i * C_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int cg = 0; cg < C_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * b0_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * b0_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * b0_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * b0_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < C_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();  // Barrier 1: tile + block 0 weights + CT weights ready

    // =================================================================
    // PHASE B: Compute ConvNext block 0 for 18×18 region
    //   - Inner pixel (ty+1, tx+1) → save skip, write s_inter
    //   - Halo pixels (threads 0-67) → write s_inter only
    // =================================================================

    __half my_skip[C];  // Block 0 output for this thread's inner pixel

    // B.1: Each thread computes its inner pixel at 18×18 pos (ty+1, tx+1)
    {
        const int r = ty + 1, c = tx + 1;  // 18×18 grid position
        float result[C];
        convnext_block_dp4a<C>(
            s_tile, TILE_W_EXP, r, c,
            s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
            b0_conv_dq, b0_conv_rq, b0_exp_dq, b0_exp_rq, b0_con_dq,
            result);

        // Add skip from global input (L1 cached from tile load)
        const int gy = tile_y0 + ty;
        const int gx = tile_x0 + tx;
        if (gy < H && gx < W) {
            const __half* skip_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int co = 0; co < C; co++)
                result[co] += __half2float(skip_ptr[co]);
        }

        // Save as FP16 for block 1's skip connection
        #pragma unroll
        for (int co = 0; co < C; co++)
            my_skip[co] = __float2half(result[co]);

        // Quantize to INT8 for block 1's conv3x3 input (uses block 1's act scale)
        const int inter_base = (r * INTER_W + c) * C_G;
        #pragma unroll
        for (int cg = 0; cg < C_G; cg++) {
            int qa = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 0]) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 1]) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 2]) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(
                __half2float(my_skip[cg * 4 + 3]) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }

    // B.2: Threads 0-67 also compute one halo pixel each
    if (threadIdx.x < 68) {
        int hr, hc;
        if (threadIdx.x < 18) {
            hr = 0;  hc = threadIdx.x;           // top row
        } else if (threadIdx.x < 36) {
            hr = 17; hc = threadIdx.x - 18;      // bottom row
        } else if (threadIdx.x < 52) {
            hr = threadIdx.x - 36 + 1; hc = 0;   // left col (rows 1-16)
        } else {
            hr = threadIdx.x - 52 + 1; hc = 17;  // right col (rows 1-16)
        }

        float result[C];
        convnext_block_dp4a<C>(
            s_tile, TILE_W_EXP, hr, hc,
            s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
            b0_conv_dq, b0_conv_rq, b0_exp_dq, b0_exp_rq, b0_con_dq,
            result);

        // Add skip from global input (bounds-checked)
        const int gy = tile_y0 - 1 + hr;
        const int gx = tile_x0 - 1 + hc;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* skip_ptr = input + (gy * W + gx) * C;
            #pragma unroll
            for (int co = 0; co < C; co++)
                result[co] += __half2float(skip_ptr[co]);
        }

        // Quantize to INT8, write to s_inter (no skip save needed)
        const int inter_base = (hr * INTER_W + hc) * C_G;
        #pragma unroll
        for (int cg = 0; cg < C_G; cg++) {
            __half h0 = __float2half(result[cg * 4 + 0]);
            __half h1 = __float2half(result[cg * 4 + 1]);
            __half h2 = __float2half(result[cg * 4 + 2]);
            __half h3 = __float2half(result[cg * 4 + 3]);
            int qa = max(-127, min(127, __float2int_rn(__half2float(h0) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(__half2float(h1) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(__half2float(h2) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(__half2float(h3) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }
    __syncthreads();  // Barrier 2: s_inter complete

    // =================================================================
    // PHASE C: Reload block 1 ConvNext weights (CT weights persist)
    // =================================================================

    // Block 1 conv3x3 weights
    {
        const int total = C * 9 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int sp   = (i / C_G) % 9;
            const int co   = i / (C_G * 9);
            int a = (int)b1_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b1_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b1_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b1_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 PW expand weights
    {
        const int total = C2 * C_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C_G;
            const int co   = i / C_G;
            const int base = co * C + ci_g * 4;
            int a = (int)b1_exp_w[base + 0];
            int b = (int)b1_exp_w[base + 1];
            int c = (int)b1_exp_w[base + 2];
            int d = (int)b1_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 PW contract weights
    {
        const int total = C * C2_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % C2_G;
            const int co   = i / C2_G;
            const int base = co * C2 + ci_g * 4;
            int a = (int)b1_con_w[base + 0];
            int b = (int)b1_con_w[base + 1];
            int c = (int)b1_con_w[base + 2];
            int d = (int)b1_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 biases
    if (threadIdx.x < C)  s_conv_b[threadIdx.x] = b1_conv_b[threadIdx.x];
    if (threadIdx.x < C)  s_con_b[threadIdx.x]  = b1_con_b[threadIdx.x];
    if (threadIdx.x < C2) s_exp_b[threadIdx.x]  = b1_exp_b[threadIdx.x];
    __syncthreads();  // Barrier 3: block 1 weights ready

    // =================================================================
    // PHASE D+E: Compute ConvNext block 1 → ConvTranspose 16→8
    //   Block 1 output stays in registers. Each thread computes its
    //   pixel, adds skip from my_skip, then directly writes 4 output
    //   sub-pixels via ConvTranspose. Zero global reads for upscale.
    // =================================================================
    {
        const int gx = tile_x0 + tx;
        const int gy = tile_y0 + ty;

        if (gx < W && gy < H) {
            // Phase D: ConvNext block 1
            float blk1[C];
            convnext_block_dp4a<C>(
                s_inter, INTER_W, ty, tx,
                s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
                b1_conv_dq, b1_conv_rq, b1_exp_dq, b1_exp_rq, b1_con_dq,
                blk1);

            // Add skip (block 0 output, saved in my_skip)
            #pragma unroll
            for (int co = 0; co < C; co++)
                blk1[co] += __half2float(my_skip[co]);

            // Phase E: ConvTranspose 16→8 directly from registers
            // Each input pixel produces 4 output sub-pixels (2×2).
            // Weight layout: [4*C_OUT, C_IN] where row = co*4+sub
            const int H_out = 2 * H;
            const int W_out = 2 * W;

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                const int sh = sub / 2;  // 0 or 1
                const int sw = sub % 2;  // 0 or 1
                const int oy = 2 * gy + sh;
                const int ox = 2 * gx + sw;

                __half* out_ptr = output + (oy * W_out + ox) * CT_C_OUT;
                #pragma unroll
                for (int co = 0; co < CT_C_OUT; co++) {
                    float acc = 0.0f;
                    const int w_row = (co * 4 + sub) * C;
                    #pragma unroll
                    for (int ci = 0; ci < C; ci++)
                        acc += blk1[ci] * static_cast<float>(s_ct_w[w_row + ci]);
                    float val = acc * __half2float(s_ct_s[co]) + __half2float(s_ct_b[co]);
                    out_ptr[co] = __float2half(val);
                }
            }
        }
    }
}

// Launcher for dec2_mega kernel
static void launch_dec2_mega(
    const __half* input,
    const ConvNextBlock& block0, const ConvNextDP4AScales& scales0,
    const ConvNextBlock& block1, const ConvNextDP4AScales& scales1,
    const ConvTransposeLayer& upscale,
    __half* output,
    int H, int W, cudaStream_t stream)
{
    constexpr int C = 16;
    constexpr int TILE_W = 16, TILE_H = 16;

    // Block 0 quantization scales
    float b0_act_rcp    = 1.0f / scales0.input_act_scale;
    float b0_conv_dq    = scales0.input_act_scale * block0.conv.quant_scale;
    float b0_conv_rq    = scales0.conv_requant_rcp;
    float b0_exp_in_s   = 1.0f / scales0.conv_requant_rcp;
    float b0_exp_dq     = b0_exp_in_s * block0.pw_expand.quant_scale;
    float b0_exp_rq     = scales0.pw_exp_requant_rcp;
    float b0_con_in_s   = 1.0f / scales0.pw_exp_requant_rcp;
    float b0_con_dq     = b0_con_in_s * block0.pw_contract.quant_scale;

    // Block 1 quantization scales
    float b1_act_rcp    = 1.0f / scales1.input_act_scale;
    float b1_conv_dq    = scales1.input_act_scale * block1.conv.quant_scale;
    float b1_conv_rq    = scales1.conv_requant_rcp;
    float b1_exp_in_s   = 1.0f / scales1.conv_requant_rcp;
    float b1_exp_dq     = b1_exp_in_s * block1.pw_expand.quant_scale;
    float b1_exp_rq     = scales1.pw_exp_requant_rcp;
    float b1_con_in_s   = 1.0f / scales1.pw_exp_requant_rcp;
    float b1_con_dq     = b1_con_in_s * block1.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    dec2_mega_nhwc_kernel<C, TILE_W, TILE_H><<<grid, 256, 0, stream>>>(
        input,
        block0.conv.weight, block0.conv.bias,
        block0.pw_expand.weight, block0.pw_expand.bias,
        block0.pw_contract.weight, block0.pw_contract.bias,
        block1.conv.weight, block1.conv.bias,
        block1.pw_expand.weight, block1.pw_expand.bias,
        block1.pw_contract.weight, block1.pw_contract.bias,
        upscale.weight, upscale.bias, upscale.w_scale,
        output, H, W,
        b0_act_rcp, b0_conv_dq, b0_conv_rq,
        b0_exp_dq, b0_exp_rq, b0_con_dq,
        b1_act_rcp, b1_conv_dq, b1_conv_rq,
        b1_exp_dq, b1_exp_rq, b1_con_dq);
}

// =========================================================================
// Dec3 Mega-Kernel: 2× FasterNetBlock(32ch) + ConvTranspose upscale(32→16)
//
// Fuses three kernel launches into one. Block 1's output stays in
// registers and feeds directly to ConvTranspose with skip_add.
//
// Phase A: Load 20×12 input tile (lower 16ch) + block 0 weights + CT weights.
// Phase B: Compute FasterNet block 0 for 18×10 region (128 inner + 52 halo),
//          save full 32ch in registers, write lower 16ch INT8 to s_inter.
// Phase C: Reload block 1 FasterNet weights (CT weights persist).
// Phase D: Compute FasterNet block 1 for 16×8 inner, keep in registers.
// Phase E: ConvTranspose 32→16 + skip_add from registers, write to output.
//
// Traffic savings: eliminates block 0→1 global round-trip + block 1→upscale
// round-trip (~14 MB at quarter-res).
// =========================================================================

template<int TILE_W = 16, int TILE_H = 8>
static __global__ void __launch_bounds__(128, 3) dec3_mega_nhwc_kernel(
    const __half* __restrict__ input,           // [H*W, 32] NHWC (quarter-res)
    // Block 0 weights (spatial conv partial 16ch)
    const int8_t* __restrict__ b0_conv_w,
    const __half* __restrict__ b0_conv_b,
    const int8_t* __restrict__ b0_exp_w,
    const __half* __restrict__ b0_exp_b,
    const int8_t* __restrict__ b0_con_w,
    const __half* __restrict__ b0_con_b,
    // Block 1 weights
    const int8_t* __restrict__ b1_conv_w,
    const __half* __restrict__ b1_conv_b,
    const int8_t* __restrict__ b1_exp_w,
    const __half* __restrict__ b1_exp_b,
    const int8_t* __restrict__ b1_con_w,
    const __half* __restrict__ b1_con_b,
    // ConvTranspose weights (32→16)
    const int8_t* __restrict__ ct_w,            // [4*16, 32] = [64, 32] INT8
    const __half* __restrict__ ct_bias,         // [16]
    const __half* __restrict__ ct_scale,        // [16]
    // Skip connection (half-res)
    const __half* __restrict__ skip,            // [4*H*W, 16] skip2 at half-res
    // Output (half-res, same buffer as skip for in-place consumption)
    __half*       __restrict__ output,          // [4*H*W, 16]
    int H, int W,                               // quarter-res dimensions
    // Block 0 scales
    float b0_act_rcp, float b0_conv_dq, float b0_concat_rcp,
    float b0_pw_exp_dq, float b0_pw_exp_rq, float b0_pw_con_dq,
    // Block 1 scales
    float b1_act_rcp, float b1_conv_dq, float b1_concat_rcp,
    float b1_pw_exp_dq, float b1_pw_exp_rq, float b1_pw_con_dq)
{
    constexpr int C_PART = 16;
    constexpr int C_STAGE = 32;
    constexpr int C_EXP = 64;
    constexpr int PART_G = C_PART / 4;     // 4
    constexpr int STAGE_G = C_STAGE / 4;   // 8
    constexpr int EXP_G = C_EXP / 4;       // 16
    constexpr int K3 = C_PART * 9;         // 144

    constexpr int CT_C_OUT = 16;           // ConvTranspose output channels
    constexpr int CT_C_IN = C_STAGE;       // 32
    constexpr int CT_W_ROWS = 4 * CT_C_OUT;  // 64
    constexpr int CT_W_TOTAL = CT_W_ROWS * CT_C_IN; // 2048 bytes

    constexpr int TILE_W_EXP = TILE_W + 4;  // 20
    constexpr int TILE_H_EXP = TILE_H + 4;  // 12
    constexpr int TILE_PIXELS_EXP = TILE_W_EXP * TILE_H_EXP;  // 240

    constexpr int INTER_W = TILE_W + 2;  // 18
    constexpr int INTER_H = TILE_H + 2;  // 10
    constexpr int INTER_PIXELS = INTER_W * INTER_H;  // 180

    // --- Shared memory ---
    __shared__ int s_tile[TILE_PIXELS_EXP * PART_G];  // 960 ints = 3840 B
    __shared__ int s_inter[INTER_PIXELS * PART_G];     // 720 ints = 2880 B
    __shared__ int s_conv_w[C_PART * 9 * PART_G];      // 576 ints = 2304 B
    __shared__ int s_exp_w[C_EXP * STAGE_G];            // 512 ints = 2048 B
    __shared__ int s_con_w[C_STAGE * EXP_G];            // 512 ints = 2048 B
    __shared__ __half s_conv_b[C_PART];                 // 32 B
    __shared__ __half s_exp_b[C_EXP];                   // 128 B
    __shared__ __half s_con_b[C_STAGE];                 // 64 B
    // ConvTranspose weights (persist from Phase A through Phase E)
    __shared__ int8_t s_ct_w[CT_W_TOTAL];              // 2048 B
    __shared__ __half s_ct_b[CT_C_OUT];                 // 32 B
    __shared__ __half s_ct_s[CT_C_OUT];                 // 32 B
    // Total: 3840+2880+2304+2048+2048+32+128+64+2048+32+32 = 15,456 B

    const int tile_x0 = blockIdx.x * TILE_W;
    const int tile_y0 = blockIdx.y * TILE_H;
    const int tx = threadIdx.x % TILE_W;  // 0..15
    const int ty = threadIdx.x / TILE_W;  // 0..7

    // =================================================================
    // PHASE A: Load 20×12 input tile (lower 16ch) + block 0 weights + CT weights
    // =================================================================

    // Block 0 conv3x3 weights (dp4a packed)
    {
        const int total = C_PART * 9 * PART_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % PART_G;
            const int sp   = (i / PART_G) % 9;
            const int co   = i / (PART_G * 9);
            int a = (int)b0_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b0_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b0_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b0_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW expand weights
    {
        const int total = C_EXP * STAGE_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % STAGE_G;
            const int co   = i / STAGE_G;
            const int base = co * C_STAGE + ci_g * 4;
            int a = (int)b0_exp_w[base + 0];
            int b = (int)b0_exp_w[base + 1];
            int c = (int)b0_exp_w[base + 2];
            int d = (int)b0_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 PW contract weights
    {
        const int total = C_STAGE * EXP_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % EXP_G;
            const int co   = i / EXP_G;
            const int base = co * C_EXP + ci_g * 4;
            int a = (int)b0_con_w[base + 0];
            int b = (int)b0_con_w[base + 1];
            int c = (int)b0_con_w[base + 2];
            int d = (int)b0_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 0 biases
    if (threadIdx.x < C_PART)  s_conv_b[threadIdx.x] = b0_conv_b[threadIdx.x];
    if (threadIdx.x < C_STAGE) s_con_b[threadIdx.x]  = b0_con_b[threadIdx.x];
    if (threadIdx.x < C_EXP)   s_exp_b[threadIdx.x]  = b0_exp_b[threadIdx.x];

    // ConvTranspose weights (persist through all phases)
    for (int i = threadIdx.x; i < CT_W_TOTAL; i += blockDim.x)
        s_ct_w[i] = ct_w[i];
    if (threadIdx.x < CT_C_OUT) {
        s_ct_b[threadIdx.x] = ct_bias[threadIdx.x];
        s_ct_s[threadIdx.x] = ct_scale[threadIdx.x];
    }

    // 20×12 input tile: lower 16ch with FP16→INT8 pre-quantization
    for (int i = threadIdx.x; i < TILE_PIXELS_EXP; i += blockDim.x) {
        const int tr = i / TILE_W_EXP;
        const int tc = i % TILE_W_EXP;
        const int gy = tile_y0 + tr - 2;  // -2 for double halo
        const int gx = tile_x0 + tc - 2;

        const int tile_base = i * PART_G;
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++) {
                int qa = __float2int_rn(__half2float(in_ptr[cg * 4 + 0]) * b0_act_rcp);
                int qb = __float2int_rn(__half2float(in_ptr[cg * 4 + 1]) * b0_act_rcp);
                int qc = __float2int_rn(__half2float(in_ptr[cg * 4 + 2]) * b0_act_rcp);
                int qd = __float2int_rn(__half2float(in_ptr[cg * 4 + 3]) * b0_act_rcp);
                qa = max(-127, min(127, qa));
                qb = max(-127, min(127, qb));
                qc = max(-127, min(127, qc));
                qd = max(-127, min(127, qd));
                s_tile[tile_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                          ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
            }
        } else {
            #pragma unroll
            for (int cg = 0; cg < PART_G; cg++)
                s_tile[tile_base + cg] = 0;
        }
    }
    __syncthreads();  // Barrier 1: tile + block 0 weights + CT weights ready

    // =================================================================
    // PHASE B: Compute FasterNet block 0 for 18×10 region
    //   - Inner pixel (ty+1, tx+1) → save full 32ch in registers + s_inter
    //   - Halo pixels (threads 0-51) → write s_inter only
    // =================================================================

    __half my_res0[C_STAGE];  // Block 0 output for this thread's inner pixel

    // B.1: Each thread computes its inner pixel at 18×10 pos (ty+1, tx+1)
    {
        const int r = ty + 1, c = tx + 1;  // 18×10 grid position
        const int gy = tile_y0 + ty;
        const int gx = tile_x0 + tx;

        // Upper 16ch passthrough from global input
        __half upper[C_PART];
        if (gy < H && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int ci = 0; ci < C_PART; ci++)
                upper[ci] = in_ptr[C_PART + ci];
        } else {
            #pragma unroll
            for (int ci = 0; ci < C_PART; ci++)
                upper[ci] = __float2half(0.0f);
        }

        float result[C_STAGE];
        fasternet32_block_dp4a(
            s_tile, TILE_W_EXP, r, c,
            upper,
            s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
            b0_conv_dq, b0_concat_rcp,
            b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
            result);

        // Add skip from global input (full 32ch)
        if (gy < H && gx < W) {
            const __half* skip_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int co = 0; co < C_STAGE; co++)
                result[co] += __half2float(skip_ptr[co]);
        }

        // Save as FP16 for block 1's skip + passthrough
        #pragma unroll
        for (int co = 0; co < C_STAGE; co++)
            my_res0[co] = __float2half(result[co]);

        // Quantize lower 16ch to INT8 for block 1's conv3x3 input
        const int inter_base = (r * INTER_W + c) * PART_G;
        #pragma unroll
        for (int cg = 0; cg < PART_G; cg++) {
            int qa = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 0]) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 1]) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 2]) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(
                __half2float(my_res0[cg * 4 + 3]) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }

    // B.2: Threads 0-51 also compute one halo pixel each
    if (threadIdx.x < 52) {
        int hr, hc;
        if (threadIdx.x < 18) {
            hr = 0;  hc = threadIdx.x;           // top row (18 pixels)
        } else if (threadIdx.x < 36) {
            hr = 9;  hc = threadIdx.x - 18;      // bottom row (18 pixels)
        } else if (threadIdx.x < 44) {
            hr = threadIdx.x - 36 + 1; hc = 0;   // left col (rows 1-8)
        } else {
            hr = threadIdx.x - 44 + 1; hc = 17;  // right col (rows 1-8)
        }

        // Upper 16ch passthrough from global input
        const int gy = tile_y0 - 1 + hr;
        const int gx = tile_x0 - 1 + hc;
        __half upper[C_PART];
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* in_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int ci = 0; ci < C_PART; ci++)
                upper[ci] = in_ptr[C_PART + ci];
        } else {
            #pragma unroll
            for (int ci = 0; ci < C_PART; ci++)
                upper[ci] = __float2half(0.0f);
        }

        float result[C_STAGE];
        fasternet32_block_dp4a(
            s_tile, TILE_W_EXP, hr, hc,
            upper,
            s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
            b0_conv_dq, b0_concat_rcp,
            b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
            result);

        // Add skip from global input (bounds-checked)
        if (gy >= 0 && gy < H && gx >= 0 && gx < W) {
            const __half* skip_ptr = input + (gy * W + gx) * C_STAGE;
            #pragma unroll
            for (int co = 0; co < C_STAGE; co++)
                result[co] += __half2float(skip_ptr[co]);
        }

        // Quantize lower 16ch to INT8, write to s_inter
        const int inter_base = (hr * INTER_W + hc) * PART_G;
        #pragma unroll
        for (int cg = 0; cg < PART_G; cg++) {
            __half h0 = __float2half(result[cg * 4 + 0]);
            __half h1 = __float2half(result[cg * 4 + 1]);
            __half h2 = __float2half(result[cg * 4 + 2]);
            __half h3 = __float2half(result[cg * 4 + 3]);
            int qa = max(-127, min(127, __float2int_rn(__half2float(h0) * b1_act_rcp)));
            int qb = max(-127, min(127, __float2int_rn(__half2float(h1) * b1_act_rcp)));
            int qc = max(-127, min(127, __float2int_rn(__half2float(h2) * b1_act_rcp)));
            int qd = max(-127, min(127, __float2int_rn(__half2float(h3) * b1_act_rcp)));
            s_inter[inter_base + cg] = (qa & 0xFF) | ((qb & 0xFF) << 8) |
                                        ((qc & 0xFF) << 16) | ((qd & 0xFF) << 24);
        }
    }
    __syncthreads();  // Barrier 2: s_inter complete

    // =================================================================
    // PHASE C: Reload block 1 FasterNet weights (CT weights persist)
    // =================================================================

    // Block 1 conv3x3 weights
    {
        const int total = C_PART * 9 * PART_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % PART_G;
            const int sp   = (i / PART_G) % 9;
            const int co   = i / (PART_G * 9);
            int a = (int)b1_conv_w[co * K3 + (ci_g * 4 + 0) * 9 + sp];
            int b = (int)b1_conv_w[co * K3 + (ci_g * 4 + 1) * 9 + sp];
            int c = (int)b1_conv_w[co * K3 + (ci_g * 4 + 2) * 9 + sp];
            int d = (int)b1_conv_w[co * K3 + (ci_g * 4 + 3) * 9 + sp];
            s_conv_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                          ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 PW expand weights
    {
        const int total = C_EXP * STAGE_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % STAGE_G;
            const int co   = i / STAGE_G;
            const int base = co * C_STAGE + ci_g * 4;
            int a = (int)b1_exp_w[base + 0];
            int b = (int)b1_exp_w[base + 1];
            int c = (int)b1_exp_w[base + 2];
            int d = (int)b1_exp_w[base + 3];
            s_exp_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 PW contract weights
    {
        const int total = C_STAGE * EXP_G;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {
            const int ci_g = i % EXP_G;
            const int co   = i / EXP_G;
            const int base = co * C_EXP + ci_g * 4;
            int a = (int)b1_con_w[base + 0];
            int b = (int)b1_con_w[base + 1];
            int c = (int)b1_con_w[base + 2];
            int d = (int)b1_con_w[base + 3];
            s_con_w[i] = (a & 0xFF) | ((b & 0xFF) << 8) |
                         ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
        }
    }

    // Block 1 biases
    if (threadIdx.x < C_PART)  s_conv_b[threadIdx.x] = b1_conv_b[threadIdx.x];
    if (threadIdx.x < C_STAGE) s_con_b[threadIdx.x]  = b1_con_b[threadIdx.x];
    if (threadIdx.x < C_EXP)   s_exp_b[threadIdx.x]  = b1_exp_b[threadIdx.x];
    __syncthreads();  // Barrier 3: block 1 weights ready

    // =================================================================
    // PHASE D+E: Compute FasterNet block 1 → ConvTranspose 32→16 + skip_add
    //   Block 1 output stays in registers. Each thread computes its pixel,
    //   adds skip from my_res0, then writes 4 output sub-pixels via
    //   ConvTranspose with skip2 addition.
    // =================================================================
    {
        const int gx = tile_x0 + tx;
        const int gy = tile_y0 + ty;

        if (gx < W && gy < H) {
            // Phase D: FasterNet block 1
            const __half* upper = my_res0 + C_PART;  // upper 16ch from block 0
            float blk1[C_STAGE];
            fasternet32_block_dp4a(
                s_inter, INTER_W, ty, tx,
                upper,
                s_conv_w, s_conv_b, s_exp_w, s_exp_b, s_con_w, s_con_b,
                b1_conv_dq, b1_concat_rcp,
                b1_pw_exp_dq, b1_pw_exp_rq, b1_pw_con_dq,
                blk1);

            // Add skip (block 0 output from my_res0 registers)
            #pragma unroll
            for (int co = 0; co < C_STAGE; co++)
                blk1[co] += __half2float(my_res0[co]);

            // Phase E: ConvTranspose 32→16 + skip_add directly from registers
            // Each input pixel produces 4 output sub-pixels (2×2).
            // Weight layout: [4*C_OUT, C_IN] where row = co*4+sub
            const int H_out = 2 * H;
            const int W_out = 2 * W;

            #pragma unroll
            for (int sub = 0; sub < 4; sub++) {
                const int sh = sub / 2;  // 0 or 1
                const int sw = sub % 2;  // 0 or 1
                const int oy = 2 * gy + sh;
                const int ox = 2 * gx + sw;

                const __half* skip_ptr = skip + (oy * W_out + ox) * CT_C_OUT;
                __half* out_ptr = output + (oy * W_out + ox) * CT_C_OUT;
                #pragma unroll
                for (int co = 0; co < CT_C_OUT; co++) {
                    float acc = 0.0f;
                    const int w_row = (co * 4 + sub) * CT_C_IN;
                    #pragma unroll
                    for (int ci = 0; ci < CT_C_IN; ci++)
                        acc += blk1[ci] * static_cast<float>(s_ct_w[w_row + ci]);
                    float val = acc * __half2float(s_ct_s[co]) + __half2float(s_ct_b[co]);
                    out_ptr[co] = __float2half(val + __half2float(skip_ptr[co]));
                }
            }
        }
    }
}

// Launcher for dec3_mega kernel
static void launch_dec3_mega(
    const __half* input,
    const FasterNetBlock& block0, const FasterNetDP4AScales& scales0,
    const FasterNetBlock& block1, const FasterNetDP4AScales& scales1,
    const ConvTransposeLayer& upscale,
    const __half* skip, __half* output,
    int H, int W, cudaStream_t stream)
{
    constexpr int TILE_W = 16, TILE_H = 8;

    // Block 0 quantization scales
    float b0_act_rcp     = 1.0f / scales0.quantScale0;
    float b0_conv_dq     = scales0.quantScale0 * block0.spatial.quant_scale;
    float b0_concat_rcp  = 1.0f / scales0.quantScale1;
    float b0_pw_exp_dq   = scales0.quantScale1 * block0.pw_expand.quant_scale;
    float b0_pw_exp_rq   = 1.0f / scales0.activationScaleFactor;
    float b0_pw_con_dq   = scales0.activationScaleFactor * block0.pw_contract.quant_scale;

    // Block 1 quantization scales
    float b1_act_rcp     = 1.0f / scales1.quantScale0;
    float b1_conv_dq     = scales1.quantScale0 * block1.spatial.quant_scale;
    float b1_concat_rcp  = 1.0f / scales1.quantScale1;
    float b1_pw_exp_dq   = scales1.quantScale1 * block1.pw_expand.quant_scale;
    float b1_pw_exp_rq   = 1.0f / scales1.activationScaleFactor;
    float b1_pw_con_dq   = scales1.activationScaleFactor * block1.pw_contract.quant_scale;

    dim3 grid(divup(W, TILE_W), divup(H, TILE_H));
    dec3_mega_nhwc_kernel<TILE_W, TILE_H><<<grid, 128, 0, stream>>>(
        input,
        block0.spatial.weight, block0.spatial.bias,
        block0.pw_expand.weight, block0.pw_expand.bias,
        block0.pw_contract.weight, block0.pw_contract.bias,
        block1.spatial.weight, block1.spatial.bias,
        block1.pw_expand.weight, block1.pw_expand.bias,
        block1.pw_contract.weight, block1.pw_contract.bias,
        upscale.weight, upscale.bias, upscale.w_scale,
        skip, output, H, W,
        b0_act_rcp, b0_conv_dq, b0_concat_rcp,
        b0_pw_exp_dq, b0_pw_exp_rq, b0_pw_con_dq,
        b1_act_rcp, b1_conv_dq, b1_concat_rcp,
        b1_pw_exp_dq, b1_pw_exp_rq, b1_pw_con_dq);
}

// =========================================================================
// FSR4Model::forward_nhwc_kernels — actual kernel dispatch
// =========================================================================

void FSR4Model::forward_nhwc_kernels(const __half* input, __half* output,
                                      int H_in, int W_in, cudaStream_t stream,
                                      FSR4Profile* prof)
{
    #define PROF_BEGIN(name) if (prof) prof->begin(name, stream)
    #define PROF_END()       if (prof) prof->end(stream)

    // Buffer assignment for mega-kernel dispatch:
    //   A/B/C = working buffers. S = bottleneck scratch.
    //   skip2 = B (written by enc2_mega), skip3 = A (written by enc3_mega)
    //   Bottleneck uses S↔C ping-pong (preserves skip2=B, skip3=A).
    //   In-place skip consumption for upscale kernels.
    __half* A = buffers.buf_a;
    __half* B = buffers.buf_b;
    __half* C = buffers.buf_c;
    __half* S = buffers.bn_scratch;

    int H2 = H_in / 2, W2 = W_in / 2;
    int H4 = H_in / 4, W4 = W_in / 4;
    int H8 = H_in / 8, W8 = W_in / 8;

    constexpr int BLOCK = 256;

    // =====================================================================
    // Encoder 1: FP16 Conv2d(7→16, k=2, s=2) — input → A
    // =====================================================================
    PROF_BEGIN("enc1_downscale (7->16, k2s2 FP16)");
    {
        const int M_out = H2 * W2;
        direct_fp16_conv2d_k2s2_enc1_kernel<<<divup(M_out, BLOCK), BLOCK, 0, stream>>>(
            input, weights.enc1_downscale.weight, weights.enc1_downscale.bias,
            A, H_in, W_in);
    }
    PROF_END();

    // =====================================================================
    // Encoder 2 (MEGA): 2× ConvNextBlock(16ch) + downscale(16→32)
    //   Single kernel: A → B(skip2, half-res 16ch) + C(quarter-res 32ch)
    // =====================================================================
    PROF_BEGIN("enc2_mega (2×ConvNext + downscale)");
    launch_enc2_mega(A,
                     weights.enc2_res[0], kConvNextScales[0],
                     weights.enc2_res[1], kConvNextScales[1],
                     weights.enc2_downscale,
                     B, C, H2, W2, stream);
    PROF_END();
    // B = skip2 (half-res 16ch). A is stale.

    // =====================================================================
    // Encoder 3 (MEGA): 2× FasterNetBlock(32ch) + downscale(32→64)
    //   Single kernel: C → A(skip3, quarter-res 32ch) + S(eighth-res 64ch)
    // =====================================================================
    PROF_BEGIN("enc3_mega (2×FasterNet32 + downscale)");
    launch_enc3_mega(C,
                     weights.enc3_res[0], kFasterNet32Scales[0],
                     weights.enc3_res[1], kFasterNet32Scales[1],
                     weights.enc3_downscale,
                     A, S, H4, W4, stream);
    PROF_END();
    // A = skip3 (quarter-res 32ch). S = eighth-res 64ch. C is stale.

    // =====================================================================
    // Bottleneck: bn_mega(res0+res1) + standalone res2 + upscale(64→32)
    //   S→C (bn_mega), C→S (res2), S+A→A (upscale).
    // =====================================================================
    PROF_BEGIN("bn_mega (2×FasterNet64)");
    launch_bn_mega(S,
                   weights.bn_res[0], kFasterNet64Scales[0],
                   weights.bn_res[1], kFasterNet64Scales[1],
                   C, H8, W8, stream);
    PROF_END();

    PROF_BEGIN("bn_res2 (FasterNet 64ch)");
    fasternet_forward_nhwc_impl(C, S, weights.bn_res[2], &kFasterNet64Scales[2], 64, H8, W8, nullptr, nullptr, nullptr, stream);
    PROF_END();

    PROF_BEGIN("bn_upscale (64->32, CT k2s2 +skip)");
    launch_direct_conv_transpose_skip_add<64, 32>(S, weights.bn_upscale, A, A, H8, W8, stream);
    PROF_END();
    // skip3 (A) consumed in-place. A = quarter-res 32ch.

    // =====================================================================
    // Decoder 3 (MEGA): 2× FasterNetBlock(32ch) + upscale(32→16)
    //   Single kernel: A → B (half-res 16ch). Upscale adds skip2=B in-place.
    //   Block 1 output stays in registers for ConvTranspose (zero global RT).
    // =====================================================================
    PROF_BEGIN("dec3_mega (2×FasterNet32 + upscale)");
    launch_dec3_mega(A,
                     weights.dec3_res[0], kFasterNet32Scales[2],
                     weights.dec3_res[1], kFasterNet32Scales[3],
                     weights.dec3_upscale,
                     B, B, H4, W4, stream);
    PROF_END();
    // skip2 (B) consumed in-place. B = half-res 16ch.

    // =====================================================================
    // Decoder 2 (MEGA): 2× ConvNextBlock(16ch) + upscale(16→8)
    //   Single kernel: B → output (full-res 8ch).
    //   Block 1 output stays in registers for ConvTranspose (zero global RT).
    // =====================================================================
    PROF_BEGIN("dec2_mega (2×ConvNext + upscale)");
    launch_dec2_mega(B,
                     weights.dec2_res[0], kConvNextScales[2],
                     weights.dec2_res[1], kConvNextScales[3],
                     weights.dec2_upscale,
                     output, H2, W2, stream);
    PROF_END();

    #undef PROF_BEGIN
    #undef PROF_END
}

// =========================================================================
// NHWC workspace size query (used by pipeline init)
// =========================================================================

size_t fsr4_nhwc_workspace_size(const FSR4Weights& w, int H_in, int W_in) {
    return compute_max_workspace_nhwc(w, H_in, W_in);
}

} // namespace fsr4
