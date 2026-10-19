// FSR 4 All — Pre/Post-processing Pipeline (FEAT-5)
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Implements the full FSR4 upscaling pipeline:
//   1. sRGB→linear conversion
//   2. Preprocessing: bilinear upsample + mu-law + YCbCr → 7ch model input
//   3. U-Net forward pass (INT8)
//   4. Postprocessing: anisotropic Gaussian filter on low-res input → upscaled RGB
//   5. linear→sRGB conversion
//
// Algorithm faithfully ported from AMD FSR4 HLSL (post_common.hlsli, pre_common.hlsli).

#include <fsr4/pipeline.h>
#include <cstdio>
#include <cmath>
#include <algorithm>

namespace fsr4 {

// =========================================================================
// Constants (from FSR4 HLSL source)
// =========================================================================

// Mu-law compression (ffx_fsr4_gpu_colorspace_common.h)
static constexpr float MU_LAW_MU     = 150.0f;
static constexpr float MU_LAW_ENCODE = 0.1174f;       // 1/mu * ln coefficient
static constexpr float MU_LAW_DECODE = 8.51788f;      // 1/0.1174

// Gaussian filter sigma (post_common.hlsli)
// GAUSS_SIGMA used via GAUSS_SIGMA_INV2 only; kept for documentation.
static constexpr float GAUSS_SIGMA_INV2 = -0.5f / (0.47f * 0.47f);  // -2.2637

// BT.709 YCbCr coefficients (mlsr_optimized_includes.hlsli)
static constexpr float KA = 0.2126f;   // R weight for Y
static constexpr float KB = 0.7152f;   // G weight for Y
static constexpr float KC = 0.0722f;   // B weight for Y
// KD (1.8556, Cb scale) and KE (1.5748, Cr scale) reserved for chrominance distance.


// =========================================================================
// Device helper functions
// =========================================================================

// sRGB → linear: degree-3 polynomial approximation (no SFU, pure ALU).
// Replaces __powf(x, 2.4) which serializes on the SFU (~400 cycles/element).
// Max error < 0.4% vs exact IEC 61966-2-1 — invisible at FP16 precision.
// Coefficients: standard GPU fast-sRGB (sum = 1.0, correct at 0 and 1).
static __device__ __forceinline__ float d_srgb_to_linear(float x) {
    return x * (x * (x * 0.305306011f + 0.682171111f) + 0.012522878f);
}

// linear → sRGB: sqrt decomposition (3 hardware sqrtf, no __powf).
// x^(1/2.4) ≈ a*x^(1/2) + b*x^(1/4) + c*x^(1/8).
// sqrtf is a single-cycle hardware instruction on SM_20+.
// Max error < 1% vs exact IEC 61966-2-1 — invisible at FP16 precision.
static __device__ __forceinline__ float d_linear_to_srgb(float x) {
    x = fmaxf(x, 0.0f);
    float s1 = sqrtf(x);       // x^(1/2)
    float s2 = sqrtf(s1);      // x^(1/4)
    float s3 = sqrtf(s2);      // x^(1/8)
    return 0.585122381f * s1 + 0.783140355f * s2 - 0.368262736f * s3;
}

// Mu-law forward: linear → compressed
static __device__ __forceinline__ float d_mu_law_encode(float x) {
    return MU_LAW_ENCODE * __logf(1.0f + MU_LAW_MU * fmaxf(x, 0.0f));
}

// Mu-law inverse: compressed → linear
static __device__ __forceinline__ float d_mu_law_decode(float x) {
    return (__expf(MU_LAW_DECODE * x) - 1.0f) / MU_LAW_MU;
}

// MODEL_COLOR_NORMALIZE: [0,1] → [-1,1]
static __device__ __forceinline__ float d_normalize(float x) {
    return 2.0f * x - 1.0f;
}

// Sigmoid
static __device__ __forceinline__ float d_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

// tanh via exp
static __device__ __forceinline__ float d_tanh(float x) {
    float e2x = __expf(2.0f * x);
    return (e2x - 1.0f) / (e2x + 1.0f);
}

// Gaussian weight: exp(sigma_inv2 * (scale * dist)^2)
static __device__ __forceinline__ float d_gauss_weight(float dist, float scale) {
    float sd = scale * dist;
    return __expf(GAUSS_SIGMA_INV2 * sd * sd);
}

// Clamp integer to [0, max]
static __device__ __forceinline__ int d_clamp(int v, int lo, int hi) {
    return max(lo, min(v, hi));
}

// Sample from pre-converted NHWC mu-law buffer [H, W, 3] (clamped coords)
static __device__ __forceinline__ float d_sample_mulaw(const __half* lr_mulaw_nhwc,
                                                        int c, int y, int x,
                                                        int H, int W) {
    y = d_clamp(y, 0, H - 1);
    x = d_clamp(x, 0, W - 1);
    return __half2float(lr_mulaw_nhwc[(y * W + x) * 3 + c]);
}


// =========================================================================
// Kernel: sRGB ↔ linear conversion
// =========================================================================

// Vectorized sRGB kernels: 4 elements per thread via half2 loads for ILP + coalescing.
// Each thread processes 4 independent FP16 values, hiding ALU latency.

static __global__ void srgb_to_linear_kernel(const __half* __restrict__ in,
                                              __half* __restrict__ out, int n) {
    const int n4 = n & ~3;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int stride = blockDim.x * gridDim.x * 4;

    for (; idx < n4; idx += stride) {
        const half2 v0 = *reinterpret_cast<const half2*>(in + idx);
        const half2 v1 = *reinterpret_cast<const half2*>(in + idx + 2);

        float a = __half2float(__low2half(v0));
        float b = __half2float(__high2half(v0));
        float c = __half2float(__low2half(v1));
        float d = __half2float(__high2half(v1));

        a = d_srgb_to_linear(a);
        b = d_srgb_to_linear(b);
        c = d_srgb_to_linear(c);
        d = d_srgb_to_linear(d);

        *reinterpret_cast<half2*>(out + idx)     = __halves2half2(__float2half(a), __float2half(b));
        *reinterpret_cast<half2*>(out + idx + 2) = __halves2half2(__float2half(c), __float2half(d));
    }
    // Tail: handle remaining 0-3 elements
    for (int i = n4 + threadIdx.x; i < n; i += blockDim.x)
        out[i] = __float2half(d_srgb_to_linear(__half2float(in[i])));
}

static __global__ void linear_to_srgb_kernel(const __half* __restrict__ in,
                                              __half* __restrict__ out, int n) {
    const int n4 = n & ~3;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int stride = blockDim.x * gridDim.x * 4;

    for (; idx < n4; idx += stride) {
        const half2 v0 = *reinterpret_cast<const half2*>(in + idx);
        const half2 v1 = *reinterpret_cast<const half2*>(in + idx + 2);

        float a = __half2float(__low2half(v0));
        float b = __half2float(__high2half(v0));
        float c = __half2float(__low2half(v1));
        float d = __half2float(__high2half(v1));

        a = d_linear_to_srgb(a);
        b = d_linear_to_srgb(b);
        c = d_linear_to_srgb(c);
        d = d_linear_to_srgb(d);

        *reinterpret_cast<half2*>(out + idx)     = __halves2half2(__float2half(a), __float2half(b));
        *reinterpret_cast<half2*>(out + idx + 2) = __halves2half2(__float2half(c), __float2half(d));
    }
    for (int i = n4 + threadIdx.x; i < n; i += blockDim.x)
        out[i] = __float2half(d_linear_to_srgb(__half2float(in[i])));
}

void srgb_to_linear(const __half* input, __half* output, int count, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (count / 4 + BLOCK - 1) / BLOCK;
    grid = max(grid, 1);
    srgb_to_linear_kernel<<<grid, BLOCK, 0, stream>>>(input, output, count);
}

void linear_to_srgb(const __half* input, __half* output, int count, cudaStream_t stream) {
    constexpr int BLOCK = 256;
    int grid = (count / 4 + BLOCK - 1) / BLOCK;
    grid = max(grid, 1);
    linear_to_srgb_kernel<<<grid, BLOCK, 0, stream>>>(input, output, count);
}


// =========================================================================
// Kernel: LR Prepare (pre-convert low-res sRGB to NHWC mu-law)
// =========================================================================
//
// Converts lr_srgb [3, H, W] NCHW to lr_mulaw_nhwc [H, W, 3] NHWC with
// fused sRGB→linear + mu-law encode. Run once per frame at LR resolution.
// Eliminates 39 __logf calls per output pixel in preprocess+postprocess
// (replaced by 3 per LR pixel at ¼ resolution), and converts scattered
// NCHW channel reads to coalesced NHWC access pattern.

static __global__ void lr_prepare_kernel(
    const __half* __restrict__ lr_srgb,       // [3, H, W] NCHW sRGB (or linear if input_linear)
    __half* __restrict__ lr_mulaw_nhwc,       // [H, W, 3] NHWC mu-law linear
    const float* __restrict__ d_exposure,     // [1] exposure value (computed by auto_exposure)
    int H, int W, bool input_linear)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float exposure = *d_exposure;

    int spatial = H * W;
    int pixel = y * W + x;
    int nhwc_base = pixel * 3;

    for (int c = 0; c < 3; c++) {
        float val = __half2float(lr_srgb[c * spatial + pixel]);
        float linear = input_linear ? val : d_srgb_to_linear(val);
        // AMD: ApplyMuLaw(LoadInputColor() * exposureValue)
        float mulaw = d_mu_law_encode(linear * exposure);
        lr_mulaw_nhwc[nhwc_base + c] = __float2half(mulaw);
    }
}


// =========================================================================
// Kernel: Auto-exposure (geometric mean luminance → exposure value)
// =========================================================================
//
// Matches AMD FSR4's SPD auto-exposure: compute geometric mean luminance of
// the input, then derive exposure = 1 / (9.6 * geo_mean). This normalizes
// HDR input into the range the model was trained on.
//
// Two-stage reduction:
//   1. auto_exposure_reduce: per-block shared-memory reduction of log(lum),
//      atomicAdd block sums to global accumulator.
//   2. auto_exposure_finalize: reads accumulator, computes exposure, writes
//      to d_exposure_value for subsequent kernels to read.

static __global__ void auto_exposure_reduce(
    const __half* __restrict__ input,   // [3, H, W] NCHW linear
    float* __restrict__ d_log_sum,      // [1] atomic accumulator
    int* __restrict__ d_count,          // [1] atomic counter
    int H, int W, bool input_is_srgb)
{
    __shared__ float s_sum[256];
    __shared__ int s_count[256];

    int tid = threadIdx.x;
    int spatial = H * W;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    float local_sum = 0.0f;
    int local_count = 0;

    // Grid-stride loop over all pixels
    for (int i = idx; i < spatial; i += blockDim.x * gridDim.x) {
        float r = __half2float(input[0 * spatial + i]);
        float g = __half2float(input[1 * spatial + i]);
        float b = __half2float(input[2 * spatial + i]);

        if (input_is_srgb) {
            r = d_srgb_to_linear(r);
            g = d_srgb_to_linear(g);
            b = d_srgb_to_linear(b);
        }

        float lum = 0.2126f * r + 0.7152f * g + 0.0722f * b;
        local_sum += __logf(fmaxf(lum, 1e-6f));
        local_count++;
    }

    s_sum[tid] = local_sum;
    s_count[tid] = local_count;
    __syncthreads();

    // Block-level reduction
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_count[tid] += s_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(d_log_sum, s_sum[0]);
        atomicAdd(d_count, s_count[0]);
    }
}

static __global__ void auto_exposure_finalize(
    const float* __restrict__ d_log_sum,
    const int* __restrict__ d_count,
    float* __restrict__ d_exposure,
    bool first_frame)
{
    float avg_log = *d_log_sum / fmaxf((float)*d_count, 1.0f);
    float geo_mean = __expf(avg_log);

    // Standard photographic exposure: 1 / (9.6 * avg_luminance)
    // Derived from EV100 = log2(L * 100 / 12.5), exposure = 1 / (1.2 * 2^EV100)
    float new_exposure = 1.0f / (9.6f * fmaxf(geo_mean, 1e-6f));
    new_exposure = fminf(fmaxf(new_exposure, 0.001f), 100.0f);

    // Temporal EMA smoothing (matches AMD SPD auto-exposure behavior).
    // Prevents wild per-frame exposure jumps that corrupt temporal history.
    float prev = *d_exposure;
    if (first_frame || prev < 0.0001f) {
        *d_exposure = new_exposure;  // First frame: instant lock
    } else {
        // Slow adaptation: ~1 stop/second at 30fps (alpha=0.05 per frame)
        *d_exposure = prev + 0.05f * (new_exposure - prev);
    }
}


// =========================================================================
// Kernel: Reprojection (warp history using motion vectors)
// =========================================================================
//
// Catmull-Rom 4x4 interpolation for history color, bilinear for recurrent.
// Matches AMD pre_common.hlsli: SampleTextureCatmullRom for color,
// SampleLevel (linear sampler) for recurrent state.

// Catmull-Rom 1D weight (B=0, C=0.5)
static __device__ __forceinline__ float d_catmull_rom(float t) {
    t = fabsf(t);
    float t2 = t * t;
    float t3 = t2 * t;
    if (t < 1.0f)
        return 1.5f * t3 - 2.5f * t2 + 1.0f;
    else if (t < 2.0f)
        return -0.5f * t3 + 2.5f * t2 - 4.0f * t + 2.0f;
    return 0.0f;
}

static __global__ void reprojection_kernel(
    const __half* __restrict__ history_color,      // [3, H, W] mu-law (prev frame)
    const __half* __restrict__ recurrent_in,       // [4, H, W] (prev frame)
    const __half* __restrict__ motion_vectors,     // [2, H, W] pixel-space MVs (nullable)
    __half* __restrict__ reprojected_color,        // [3, H, W] output
    __half* __restrict__ reprojected_recurrent,    // [4, H, W] output
    int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    int spatial = H * W;
    int pixel_idx = y * W + x;

    // Compute reprojected position
    float prev_x = (float)x;
    float prev_y = (float)y;
    if (motion_vectors != nullptr) {
        float mv_x = __half2float(motion_vectors[0 * spatial + pixel_idx]);
        float mv_y = __half2float(motion_vectors[1 * spatial + pixel_idx]);

        // MV sanity check (matches AMD local_disocclusion_user):
        // NaN, Inf, or absurdly large MVs → treat as disocclusion (zero history)
        if (isnan(mv_x) || isnan(mv_y) || isinf(mv_x) || isinf(mv_y)
            || fabsf(mv_x) > 1000000.f || fabsf(mv_y) > 1000000.f) {
            for (int c = 0; c < 3; c++)
                reprojected_color[c * spatial + pixel_idx] = __float2half(0.0f);
            for (int r = 0; r < 4; r++)
                reprojected_recurrent[r * spatial + pixel_idx] = __float2half(0.0f);
            return;
        }

        prev_x += mv_x;
        prev_y += mv_y;
    }

    // On-screen check
    bool onscreen = (prev_x >= 0.0f && prev_x <= (float)(W - 1) &&
                     prev_y >= 0.0f && prev_y <= (float)(H - 1));

    if (onscreen) {
        // --- Catmull-Rom 4x4 for history color ---
        int ix = (int)floorf(prev_x);
        int iy = (int)floorf(prev_y);
        float fx = prev_x - (float)ix;
        float fy = prev_y - (float)iy;

        float color_sum[3] = {0.0f, 0.0f, 0.0f};
        float weight_sum = 0.0f;

        for (int ty = -1; ty <= 2; ty++) {
            float wy = d_catmull_rom(fy - (float)ty);
            for (int tx = -1; tx <= 2; tx++) {
                float wx = d_catmull_rom(fx - (float)tx);
                float w = wx * wy;

                int sx = ix + tx;
                int sy = iy + ty;
                // Zero weight for out-of-bounds (don't contribute)
                if (sx < 0 || sx >= W || sy < 0 || sy >= H) {
                    // Don't add to weight_sum either, for proper normalization
                    continue;
                }
                weight_sum += w;
                int sidx = sy * W + sx;
                for (int c = 0; c < 3; c++)
                    color_sum[c] += w * __half2float(history_color[c * spatial + sidx]);
            }
        }

        if (weight_sum > 1e-6f) {
            float inv_w = 1.0f / weight_sum;
            for (int c = 0; c < 3; c++)
                reprojected_color[c * spatial + pixel_idx] =
                    __float2half(fmaxf(color_sum[c] * inv_w, 0.0f));
        } else {
            for (int c = 0; c < 3; c++)
                reprojected_color[c * spatial + pixel_idx] = __float2half(0.0f);
        }

        // --- Bilinear for recurrent state ---
        int x0 = ix, y0 = iy;
        int x1 = x0 + 1, y1 = y0 + 1;
        x0 = d_clamp(x0, 0, W - 1); x1 = d_clamp(x1, 0, W - 1);
        y0 = d_clamp(y0, 0, H - 1); y1 = d_clamp(y1, 0, H - 1);
        for (int r = 0; r < 4; r++) {
            float tl = __half2float(recurrent_in[r * spatial + y0 * W + x0]);
            float tr = __half2float(recurrent_in[r * spatial + y0 * W + x1]);
            float bl = __half2float(recurrent_in[r * spatial + y1 * W + x0]);
            float br = __half2float(recurrent_in[r * spatial + y1 * W + x1]);
            float val = tl * (1.0f - fx) * (1.0f - fy) + tr * fx * (1.0f - fy)
                      + bl * (1.0f - fx) * fy            + br * fx * fy;
            reprojected_recurrent[r * spatial + pixel_idx] = __float2half(val);
        }
    } else {
        // Off-screen: zero → triggers reset-like behavior in preprocess
        for (int c = 0; c < 3; c++)
            reprojected_color[c * spatial + pixel_idx] = __float2half(0.0f);
        for (int r = 0; r < 4; r++)
            reprojected_recurrent[r * spatial + pixel_idx] = __float2half(0.0f);
    }
}


// =========================================================================
// Kernel: Preprocessing (assemble 7-channel model input at output resolution)
// =========================================================================
//
// For each output pixel:
//   1. Map to low-res position (subpixel)
//   2. Gaussian-weighted 2×2 bilinear from low-res linear RGB
//   3. Mu-law encode
//   4. Convert to YCbCr
//   5. Assemble 7 channels: [Y_curr, Y_hist, color_dist, rfm0, rfm1, rfm2, rfm3]
//   6. Normalize to [-1,1]

static __global__ void preprocess_kernel(
    const __half* __restrict__ lr_mulaw_nhwc,  // [H_lr, W_lr, 3] pre-converted NHWC mu-law
    const __half* __restrict__ recurrent,       // [4, H_out, W_out] reprojected recurrent (or nullptr)
    __half* __restrict__ model_input,           // [7, H_out, W_out]
    __half* __restrict__ reprojected_mulaw,     // [3, H_out, W_out] mu-law (reprojected or bilinear)
    int H_lr, int W_lr, int H_out, int W_out,
    float inv_scale_x, float inv_scale_y,
    float jitter_x, float jitter_y,
    bool first_frame, float rect_alpha_scale = 0.1f)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_out || y >= H_out) return;

    int spatial_out = H_out * W_out;
    int pixel_idx = y * W_out + x;

    // Map output pixel center to low-res position (with jitter offset)
    // Matches AMD HLSL: lr_pos_f = (-0.5 + inv_scale/2 + float2(x,y)*inv_scale) + jitter
    float lr_xf = ((float)x + 0.5f) * inv_scale_x - 0.5f + jitter_x;
    float lr_yf = ((float)y + 0.5f) * inv_scale_y - 0.5f + jitter_y;

    // 2×2 bilinear sample positions
    int x0 = (int)floorf(lr_xf);
    int y0 = (int)floorf(lr_yf);
    int x1 = x0 + 1;
    int y1 = y0 + 1;

    // Gaussian weights for the 4 samples (sigma=0.47)
    float scale_x = 1.0f / inv_scale_x;  // upscale factor
    float scale_y = 1.0f / inv_scale_y;
    float gx0 = d_gauss_weight(fabsf((float)x0 - lr_xf), scale_x);
    float gx1 = d_gauss_weight(fabsf((float)x1 - lr_xf), scale_x);
    float gy0 = d_gauss_weight(fabsf((float)y0 - lr_yf), scale_y);
    float gy1 = d_gauss_weight(fabsf((float)y1 - lr_yf), scale_y);
    float w_tl = gx0 * gy0;
    float w_tr = gx1 * gy0;
    float w_bl = gx0 * gy1;
    float w_br = gx1 * gy1;
    float w_sum = w_tl + w_tr + w_bl + w_br;
    float w_inv = (w_sum > 1e-8f) ? (1.0f / w_sum) : 0.0f;

    // Sample pre-computed mu-law from NHWC buffer (no sRGB/logf needed)
    float rgb_mulaw[3] = {0.0f, 0.0f, 0.0f};
    for (int c = 0; c < 3; c++) {
        float s_tl = d_sample_mulaw(lr_mulaw_nhwc, c, y0, x0, H_lr, W_lr);
        float s_tr = d_sample_mulaw(lr_mulaw_nhwc, c, y0, x1, H_lr, W_lr);
        float s_bl = d_sample_mulaw(lr_mulaw_nhwc, c, y1, x0, H_lr, W_lr);
        float s_br = d_sample_mulaw(lr_mulaw_nhwc, c, y1, x1, H_lr, W_lr);
        rgb_mulaw[c] = (s_tl * w_tl + s_tr * w_tr + s_bl * w_bl + s_br * w_br) * w_inv;
    }

    // First frame: store mu-law bilinear upscale as initial "reprojected color".
    // Subsequent frames: inject rect_alpha (2-10% of current Gaussian upscale into
    // reprojected history). Matches AMD pre_common.hlsli:
    //   temporal_sample_weight = center_weight * 0.1;  // rect_alpha = 0.1
    //   history = weight * current + (1 - weight) * reprojected;
    if (first_frame) {
        for (int c = 0; c < 3; c++) {
            reprojected_mulaw[c * spatial_out + pixel_idx] = __float2half(rgb_mulaw[c]);
        }
    } else {
        // rect_alpha injection: blend current Gaussian upscale into reprojected history
        float center_weight = fmaxf(fmaxf(w_tl, w_tr), fmaxf(w_bl, w_br));
        float rect_alpha = center_weight * rect_alpha_scale;  // AMD default: 0.1
        for (int c = 0; c < 3; c++) {
            float hist = __half2float(reprojected_mulaw[c * spatial_out + pixel_idx]);
            float blended = rect_alpha * rgb_mulaw[c] + (1.0f - rect_alpha) * hist;
            reprojected_mulaw[c * spatial_out + pixel_idx] = __float2half(blended);
        }
    }

    // RGB → YCbCr (BT.709)
    float Y  = rgb_mulaw[0] * KA + rgb_mulaw[1] * KB + rgb_mulaw[2] * KC;

    // NHWC output: model_input[pixel_idx * 7 + ch]
    int nhwc_base = pixel_idx * 7;

    // Ch 0: Current luminance (normalized)
    model_input[nhwc_base + 0] = __float2half(d_normalize(Y));

    // Ch 1: History luminance + Ch 2: Color distance
    if (first_frame) {
        // First frame: history = current, distance = 0 (AMD reset path)
        model_input[nhwc_base + 1] = __float2half(d_normalize(Y));
        model_input[nhwc_base + 2] = __float2half(0.0f);
    } else {
        // Subsequent frames: read reprojected history (stored from previous postprocess)

        float hist_r = __half2float(reprojected_mulaw[0 * spatial_out + pixel_idx]);
        float hist_g = __half2float(reprojected_mulaw[1 * spatial_out + pixel_idx]);
        float hist_b = __half2float(reprojected_mulaw[2 * spatial_out + pixel_idx]);
        float Y_hist = hist_r * KA + hist_g * KB + hist_b * KC;
        model_input[nhwc_base + 1] = __float2half(d_normalize(Y_hist));

        // Color distance = CbCr chrominance Euclidean distance (matches AMD)
        // BT.709: Cb = (B - Y) / 1.8556, Cr = (R - Y) / 1.5748
        float Cb_curr = (rgb_mulaw[2] - Y) / 1.8556f;
        float Cr_curr = (rgb_mulaw[0] - Y) / 1.5748f;
        float Cb_hist = (hist_b - Y_hist) / 1.8556f;
        float Cr_hist = (hist_r - Y_hist) / 1.5748f;
        float dCb = Cb_curr - Cb_hist;
        float dCr = Cr_curr - Cr_hist;
        model_input[nhwc_base + 2] = __float2half(sqrtf(dCb*dCb + dCr*dCr + 1e-6f));
    }

    // Ch 3-6: Recurrent feedback (recurrent state stays NCHW internally)
    if (first_frame || recurrent == nullptr) {
        // MODEL_RFM_NORMALIZE(0) = 2*0 - 1 = -1
        model_input[nhwc_base + 3] = __float2half(-1.0f);
        model_input[nhwc_base + 4] = __float2half(-1.0f);
        model_input[nhwc_base + 5] = __float2half(-1.0f);
        model_input[nhwc_base + 6] = __float2half(-1.0f);
    } else {
        for (int r = 0; r < 4; r++) {
            float rfm = __half2float(recurrent[r * spatial_out + pixel_idx]);
            model_input[nhwc_base + 3 + r] = __float2half(d_normalize(rfm));
        }
    }
}


// =========================================================================
// Kernel: Postprocessing (anisotropic Gaussian filter → upscaled RGB)
// =========================================================================
//
// For each output pixel:
//   1. Read model output ch0-3, apply activations
//   2. Map to low-res position
//   3. Sample 3×3 neighborhood with anisotropic Gaussian weights
//   4. Mu-law weighted average, then remove mu-law
//   5. Temporal blend (skip for first frame)
//   6. Store recurrent state (ch4-7 sigmoid)

static __global__ void postprocess_kernel(
    const __half* __restrict__ model_output,       // [H*W, 8] NHWC
    const __half* __restrict__ lr_mulaw_nhwc,      // [H_lr, W_lr, 3] pre-converted NHWC mu-law
    const __half* __restrict__ reprojected_mulaw,  // [3, H_out, W_out] mu-law (reprojected history)
    __half* __restrict__ history_color_out,        // [3, H_out, W_out] blended mu-law → persistent
    __half* __restrict__ output_srgb,              // [3, H_out, W_out] output (sRGB or linear)
    __half* __restrict__ recurrent_out,            // [4, H_out, W_out] recurrent state → persistent
    const float* __restrict__ d_exposure,          // [1] exposure value for decode
    const __half* __restrict__ d_motion_vectors,   // [2, H_out, W_out] NCHW (or nullptr)
    int H_lr, int W_lr, int H_out, int W_out,
    float inv_scale_x, float inv_scale_y,
    float jitter_x, float jitter_y,
    bool first_frame, bool output_linear = false,
    float max_blend = 1.0f)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W_out || y >= H_out) return;

    int spatial_out = H_out * W_out;
    int pixel_idx = y * W_out + x;

    // --- Read filter parameters from model output (NHWC: [H*W, 8]) ---
    int nhwc_base = pixel_idx * 8;
    float raw0 = __half2float(model_output[nhwc_base + 0]);
    float raw1 = __half2float(model_output[nhwc_base + 1]);
    float raw2 = __half2float(model_output[nhwc_base + 2]);
    float raw3 = __half2float(model_output[nhwc_base + 3]);

    // Apply activations (from post_common.hlsli)
    float correlation   = d_tanh(raw0);               // [-1, 1]
    float scale_x       = 2.0f * d_sigmoid(raw1);     // [0, 2]
    float scale_y       = 2.0f * d_sigmoid(raw2);     // [0, 2]
    float blend_factor  = d_sigmoid(raw3);

    // Per-pixel motion-adaptive blend cap: reduce temporal blending for fast-moving
    // pixels. Compensates for 1-2 frame display lag in our deferred pipeline —
    // AMD's zero-lag implementation doesn't need this.
    // Ramp: full blend at <=10px, linearly reduce to 0.05 at 100px+ of MV magnitude.
    if (d_motion_vectors != nullptr) {
        int mv_spatial = H_out * W_out;
        float mvx = __half2float(d_motion_vectors[0 * mv_spatial + pixel_idx]);
        float mvy = __half2float(d_motion_vectors[1 * mv_spatial + pixel_idx]);
        float mv_mag = sqrtf(mvx * mvx + mvy * mvy);
        float motion_cap = fmaxf(0.05f, 1.0f - (mv_mag - 10.0f) * (1.0f / 90.0f));
        blend_factor = fminf(blend_factor, motion_cap);
    }

    blend_factor = fminf(blend_factor, max_blend);  // global cap from config

    float sx2 = scale_x * scale_x;
    float sy2 = scale_y * scale_y;
    float sxy = scale_x * scale_y;

    // --- Map output pixel to low-res position (with jitter offset) ---
    // Matches AMD HLSL: lr_pos_f = (-0.5 + inv_scale/2 + float2(x,y)*inv_scale) + jitter
    float lr_xf = ((float)x + 0.5f) * inv_scale_x - 0.5f + jitter_x;
    float lr_yf = ((float)y + 0.5f) * inv_scale_y - 0.5f + jitter_y;

    // Nearest integer position (center of 3×3 kernel)
    int cx = (int)roundf(lr_xf);
    int cy = (int)roundf(lr_yf);

    // Upscale factors for distance computation
    float up_x = 1.0f / inv_scale_x;
    float up_y = 1.0f / inv_scale_y;

    // --- 3×3 anisotropic Gaussian filter ---
    float weighted_rgb[3] = {0.0f, 0.0f, 0.0f};
    float total_weight = 0.0f;

    for (int dy = -1; dy <= 1; dy++) {
        int sy = cy + dy;
        float y_dist = ((float)sy - lr_yf) * up_y;
        float y_dist2 = y_dist * y_dist;

        for (int dx = -1; dx <= 1; dx++) {
            int sx = cx + dx;
            float x_dist = ((float)sx - lr_xf) * up_x;
            float x_dist2 = x_dist * x_dist;

            // Anisotropic Gaussian: exp(sigma_inv2 * (sx2*x2 + 2*corr*xy*sxy + sy2*y2))
            float exponent = GAUSS_SIGMA_INV2 * (
                x_dist2 * sx2
                + 2.0f * correlation * x_dist * y_dist * sxy
                + y_dist2 * sy2
            );
            float w = __expf(exponent);
            total_weight += w;

            // Sample pre-computed mu-law from NHWC buffer (no sRGB/logf needed)
            for (int c = 0; c < 3; c++) {
                float val = d_sample_mulaw(lr_mulaw_nhwc, c, sy, sx, H_lr, W_lr);
                weighted_rgb[c] += val * w;
            }
        }
    }

    // Normalize weights (result is in mu-law space)
    float w_inv = (total_weight > 1e-8f) ? (1.0f / total_weight) : 0.0f;
    float upscaled_mulaw[3];
    for (int c = 0; c < 3; c++) {
        upscaled_mulaw[c] = weighted_rgb[c] * w_inv;
    }

    // --- Temporal blend in mu-law space ---
    // Matches AMD post_common.hlsli:
    //   res = upscaled * (1 - blend) + reprojected * blend
    // For first frame: reprojected = mu-law bilinear upscale (from preprocess).
    // For subsequent frames: reprojected = Catmull-Rom warped history (via reprojection_kernel).
    float blended_mulaw[3];
    for (int c = 0; c < 3; c++) {
        float reprojected = __half2float(
            reprojected_mulaw[c * H_out * W_out + pixel_idx]);
        blended_mulaw[c] = upscaled_mulaw[c] * (1.0f - blend_factor)
                         + reprojected * blend_factor;
    }

    // Store blended mu-law as history for next frame (matches AMD rw_history_color).
    // Persistent buffer — reprojection kernel warps this into d_reprojected_mulaw next frame.
    for (int c = 0; c < 3; c++) {
        history_color_out[c * H_out * W_out + pixel_idx] = __float2half(blended_mulaw[c]);
    }

    // Decode mu-law → linear, then undo exposure normalization
    // AMD: RemoveMuLaw(blended) / exposureValue
    float inv_exposure = 1.0f / fmaxf(*d_exposure, 1e-6f);
    float final_rgb[3];
    for (int c = 0; c < 3; c++) {
        final_rgb[c] = d_mu_law_decode(blended_mulaw[c]) * inv_exposure;
    }

    // --- Write output (sRGB or linear depending on output_linear flag) ---
    for (int c = 0; c < 3; c++) {
        if (output_linear) {
            // HDR: clamp to >=0 only, preserve highlights
            output_srgb[c * spatial_out + pixel_idx] = __float2half(fmaxf(0.0f, final_rgb[c]));
        } else {
            // SDR: clamp to [0,1] then apply sRGB curve
            float v = fmaxf(0.0f, fminf(final_rgb[c], 1.0f));
            output_srgb[c * spatial_out + pixel_idx] = __float2half(d_linear_to_srgb(v));
        }
    }

    // --- Store recurrent state (ch4-7 with sigmoid activation) ---
    // Model output is NHWC, recurrent state stays NCHW (only used by pipeline)
    if (recurrent_out != nullptr) {
        for (int r = 0; r < 4; r++) {
            float raw = __half2float(model_output[nhwc_base + 4 + r]);
            recurrent_out[r * spatial_out + pixel_idx] = __float2half(d_sigmoid(raw));
        }
    }
}


// =========================================================================
// FSR4Pipeline methods
// =========================================================================

bool FSR4Pipeline::init(const std::string& weight_dir, int h_lr, int w_lr, int s) {
    H_lr = h_lr;
    W_lr = w_lr;
    scale = s;
    H_out = H_lr * scale;
    W_out = W_lr * scale;
    first_frame = true;

    // Align output to 8 (for 3 downsample stages)
    H_out = (H_out / 8) * 8;
    W_out = (W_out / 8) * 8;

    std::printf("Pipeline: input %dx%d → output %dx%d (%dx)\n",
                W_lr, H_lr, W_out, H_out, scale);

    // Initialize model at OUTPUT resolution
    if (!model.init(weight_dir, H_out, W_out, /*nhwc_only=*/true)) {
        std::fprintf(stderr, "Pipeline: Failed to initialize model at %dx%d\n",
                     W_out, H_out);
        return false;
    }

    size_t lr_spatial  = (size_t)H_lr * W_lr;
    size_t out_spatial  = (size_t)H_out * W_out;

    // Allocate pipeline buffers
    cudaMalloc(&d_lr_mulaw_nhwc, 3 * lr_spatial * sizeof(__half));
    cudaMalloc(&d_model_input,  7 * out_spatial * sizeof(__half));
    cudaMalloc(&d_model_output, 8 * out_spatial * sizeof(__half));
    cudaMalloc(&d_recurrent,    4 * out_spatial * sizeof(__half));
    cudaMalloc(&d_history_color,          3 * out_spatial * sizeof(__half));
    cudaMalloc(&d_reprojected_mulaw,      3 * out_spatial * sizeof(__half));
    cudaMalloc(&d_reprojected_recurrent,  4 * out_spatial * sizeof(__half));

    // Auto-exposure buffers
    cudaMalloc(&d_exposure_value, sizeof(float));
    cudaMalloc(&d_log_lum_accum, sizeof(float));
    cudaMalloc(&d_pixel_count, sizeof(int));
    // Initialize exposure to 1.0 (neutral) for first frame
    float one = 1.0f;
    cudaMemcpy(d_exposure_value, &one, sizeof(float), cudaMemcpyHostToDevice);

    // Zero persistent state
    cudaMemset(d_recurrent, 0, 4 * out_spatial * sizeof(__half));
    cudaMemset(d_history_color, 0, 3 * out_spatial * sizeof(__half));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::fprintf(stderr, "Pipeline: CUDA alloc error: %s\n", cudaGetErrorString(err));
        return false;
    }

    size_t pipeline_bytes = (3 * lr_spatial
                             + 7 * out_spatial + 8 * out_spatial
                             + 4 * out_spatial
                             + 3 * out_spatial + 3 * out_spatial
                             + 4 * out_spatial) * sizeof(__half);
    std::printf("Pipeline buffers: %.1f MB\n", pipeline_bytes / (1024.0 * 1024.0));

    return true;
}

void FSR4Pipeline::upscale(const __half* d_input_srgb, __half* d_out_srgb,
                            const __half* d_motion_vectors,
                            float jitter_x, float jitter_y,
                            cudaStream_t stream, FSR4Profile* prof) {
    #define PIPE_PROF_BEGIN(name) if (prof) prof->begin(name, stream)
    #define PIPE_PROF_END()       if (prof) prof->end(stream)

    size_t out_pixels = (size_t)H_out * W_out;

    float inv_scale_x = (float)W_lr / (float)W_out;
    float inv_scale_y = (float)H_lr / (float)H_out;

    dim3 block(16, 16);
    dim3 grid_out((W_out + block.x - 1) / block.x,
                  (H_out + block.y - 1) / block.y);

    // --- Step 0a: Auto-exposure (compute geometric mean luminance → exposure) ---
    // Matches AMD FSR4 SPD auto-exposure: normalizes HDR input range for mu-law.
    PIPE_PROF_BEGIN("auto_exposure");
    cudaMemsetAsync(d_log_lum_accum, 0, sizeof(float), stream);
    cudaMemsetAsync(d_pixel_count, 0, sizeof(int), stream);
    {
        int lr_pixels = H_lr * W_lr;
        int ae_block = 256;
        int ae_grid = min((lr_pixels + ae_block - 1) / ae_block, 1024);
        auto_exposure_reduce<<<ae_grid, ae_block, 0, stream>>>(
            d_input_srgb, d_log_lum_accum, d_pixel_count,
            H_lr, W_lr, !input_linear);
        auto_exposure_finalize<<<1, 1, 0, stream>>>(
            d_log_lum_accum, d_pixel_count, d_exposure_value, first_frame);
    }
    PIPE_PROF_END();

    // --- Step 0b: LR Prepare (sRGB→linear + exposure + mu-law, NCHW→NHWC) ---
    // Pre-convert low-res input once at LR resolution (¼ output pixels).
    dim3 block_lr(16, 16);
    dim3 grid_lr((W_lr + block_lr.x - 1) / block_lr.x,
                 (H_lr + block_lr.y - 1) / block_lr.y);
    PIPE_PROF_BEGIN("lr_prepare");
    lr_prepare_kernel<<<grid_lr, block_lr, 0, stream>>>(
        d_input_srgb, d_lr_mulaw_nhwc, d_exposure_value,
        H_lr, W_lr, input_linear);
    PIPE_PROF_END();

    // --- Step 1: Reprojection (warp history using motion vectors) ---
    // On first frame: no history to reproject. Preprocess writes bilinear to reprojected_mulaw.
    // On subsequent frames: warp d_history_color → d_reprojected_mulaw,
    //                       warp d_recurrent → d_reprojected_recurrent.
    if (!first_frame) {
        PIPE_PROF_BEGIN("reprojection");
        reprojection_kernel<<<grid_out, block, 0, stream>>>(
            d_history_color,
            d_recurrent,
            d_motion_vectors,     // nullptr = identity reprojection
            d_reprojected_mulaw,
            d_reprojected_recurrent,
            H_out, W_out
        );
        PIPE_PROF_END();
    }

    // --- Step 2: Preprocessing ---
    PIPE_PROF_BEGIN("preprocess");
    preprocess_kernel<<<grid_out, block, 0, stream>>>(
        d_lr_mulaw_nhwc,
        first_frame ? nullptr : d_reprojected_recurrent,
        d_model_input,
        d_reprojected_mulaw,
        H_lr, W_lr, H_out, W_out,
        inv_scale_x, inv_scale_y,
        jitter_x, jitter_y,
        first_frame, rect_alpha
    );
    PIPE_PROF_END();

    // --- Step 3: Model forward pass (NHWC) ---
    PIPE_PROF_BEGIN("model_forward");
    if (bypass_model) {
        cudaMemsetAsync(d_model_output, 0, 8 * out_pixels * sizeof(__half), stream);
    } else {
        model.forward_nhwc(d_model_input, d_model_output, H_out, W_out, stream, prof);
    }
    PIPE_PROF_END();

    // --- Step 4: Postprocessing ---
    // Reads reprojected_mulaw for temporal blend.
    // Writes history_color (persistent) and recurrent (persistent).
    PIPE_PROF_BEGIN("postprocess");
    postprocess_kernel<<<grid_out, block, 0, stream>>>(
        d_model_output,
        d_lr_mulaw_nhwc,       // pre-converted NHWC mu-law (has exposure baked in)
        d_reprojected_mulaw,   // read: reprojected history for blend
        d_history_color,       // write: blended output → persistent
        d_out_srgb,            // write sRGB or linear depending on output_linear
        d_recurrent,           // write: recurrent state → persistent
        d_exposure_value,      // exposure for mu-law decode inversion
        d_motion_vectors,      // per-pixel MV for motion-adaptive blend cap (or nullptr)
        H_lr, W_lr, H_out, W_out,
        inv_scale_x, inv_scale_y,
        jitter_x, jitter_y,
        first_frame, output_linear, max_blend
    );
    PIPE_PROF_END();

    first_frame = false;

    #undef PIPE_PROF_BEGIN
    #undef PIPE_PROF_END
}

void FSR4Pipeline::free() {
    model.free();
    if (d_lr_mulaw_nhwc){ cudaFree(d_lr_mulaw_nhwc); d_lr_mulaw_nhwc = nullptr; }
    if (d_model_input)  { cudaFree(d_model_input);  d_model_input = nullptr; }
    if (d_model_output) { cudaFree(d_model_output); d_model_output = nullptr; }
    if (d_recurrent)    { cudaFree(d_recurrent);     d_recurrent = nullptr; }
    if (d_history_color){ cudaFree(d_history_color); d_history_color = nullptr; }
    if (d_reprojected_mulaw) { cudaFree(d_reprojected_mulaw); d_reprojected_mulaw = nullptr; }
    if (d_reprojected_recurrent) { cudaFree(d_reprojected_recurrent); d_reprojected_recurrent = nullptr; }
    if (d_exposure_value) { cudaFree(d_exposure_value); d_exposure_value = nullptr; }
    if (d_log_lum_accum) { cudaFree(d_log_lum_accum); d_log_lum_accum = nullptr; }
    if (d_pixel_count) { cudaFree(d_pixel_count); d_pixel_count = nullptr; }
}

} // namespace fsr4
