// FSR 4 All - Format Conversion Kernels (D3D12 ↔ Pipeline)
// Copyright (c) 2026 TheWizardsForge. MIT License.

#include "format_convert.cuh"
#include <cstdio>

namespace fsr4 {

// =========================================================================
// RGBA ↔ RGB NCHW
// =========================================================================

static __global__ void rgba_to_rgb_nchw_kernel(
    const __half* __restrict__ src,  // [H*W*4] interleaved
    __half* __restrict__ dst,        // [3, H, W] NCHW
    int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial) return;

    int src_base = idx * 4;
    dst[0 * spatial + idx] = src[src_base + 0];  // R
    dst[1 * spatial + idx] = src[src_base + 1];  // G
    dst[2 * spatial + idx] = src[src_base + 2];  // B
    // Alpha discarded
}

void rgba_to_rgb_nchw(const __half* src, __half* dst,
                      int H, int W, cudaStream_t stream) {
    int spatial = H * W;
    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;
    rgba_to_rgb_nchw_kernel<<<blocks, threads, 0, stream>>>(src, dst, spatial);
}

static __global__ void rgb_nchw_to_rgba_kernel(
    const __half* __restrict__ src,  // [3, H, W] NCHW
    __half* __restrict__ dst,        // [H*W*4] interleaved
    int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial) return;

    int dst_base = idx * 4;
    dst[dst_base + 0] = src[0 * spatial + idx];  // R
    dst[dst_base + 1] = src[1 * spatial + idx];  // G
    dst[dst_base + 2] = src[2 * spatial + idx];  // B
    dst[dst_base + 3] = __float2half(1.0f);      // A = 1.0
}

void rgb_nchw_to_rgba(const __half* src, __half* dst,
                      int H, int W, cudaStream_t stream) {
    int spatial = H * W;
    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;
    rgb_nchw_to_rgba_kernel<<<blocks, threads, 0, stream>>>(src, dst, spatial);
}

// =========================================================================
// Motion Vector Extraction
// =========================================================================

static __global__ void extract_mv_kernel(
    const __half* __restrict__ src,  // [H*W*mv_channels] interleaved FP16
    __half* __restrict__ dst,        // [2, H_out, W_out] NCHW
    int H, int W,
    float mv_scale_x, float mv_scale_y,
    float jitter_x, float jitter_y,
    bool jitter_cancel,
    bool display_res_mvs,
    int H_out, int W_out,
    int mv_channels)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = H * W;
    if (idx >= spatial) return;

    float mv_x = __half2float(src[idx * mv_channels + 0]) * mv_scale_x;
    float mv_y = __half2float(src[idx * mv_channels + 1]) * mv_scale_y;

    if (jitter_cancel) {
        mv_x -= jitter_x;
        mv_y -= jitter_y;
    }

    // If MVs are at render resolution but pipeline expects output resolution,
    // scale them up
    if (!display_res_mvs && (H != H_out || W != W_out)) {
        mv_x *= (float)W_out / (float)W;
        mv_y *= (float)H_out / (float)H;
    }

    // For render-res MVs: write at render resolution, pipeline will handle
    // For display-res MVs: write directly
    int out_spatial = display_res_mvs ? (H_out * W_out) : spatial;
    if (idx < out_spatial) {
        dst[0 * out_spatial + idx] = __float2half(mv_x);
        dst[1 * out_spatial + idx] = __float2half(mv_y);
    }
}

void extract_motion_vectors(const __half* src, __half* dst,
                            int H, int W,
                            float mv_scale_x, float mv_scale_y,
                            float jitter_x, float jitter_y,
                            bool jitter_cancel,
                            bool display_res_mvs,
                            int H_out, int W_out,
                            int mv_channels,
                            cudaStream_t stream) {
    int spatial = H * W;
    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;
    extract_mv_kernel<<<blocks, threads, 0, stream>>>(
        src, dst, H, W, mv_scale_x, mv_scale_y,
        jitter_x, jitter_y, jitter_cancel,
        display_res_mvs, H_out, W_out, mv_channels);
}

// =========================================================================
// R11G11B10_FLOAT pack/unpack
// =========================================================================

// R11G11B10_FLOAT: 11-bit R (5e+6m), 11-bit G (5e+6m), 10-bit B (5e+5m)
static __device__ __forceinline__ float unpack_float11(uint32_t bits) {
    uint32_t mantissa = bits & 0x3F;       // 6-bit mantissa
    uint32_t exponent = (bits >> 6) & 0x1F; // 5-bit exponent
    if (exponent == 0) return (mantissa == 0) ? 0.0f : (float)mantissa / 64.0f * powf(2.0f, -14.0f);
    if (exponent == 31) return (mantissa == 0) ? __int_as_float(0x7F800000) : __int_as_float(0x7FC00000);
    return (1.0f + (float)mantissa / 64.0f) * powf(2.0f, (float)exponent - 15.0f);
}

static __device__ __forceinline__ float unpack_float10(uint32_t bits) {
    uint32_t mantissa = bits & 0x1F;       // 5-bit mantissa
    uint32_t exponent = (bits >> 5) & 0x1F; // 5-bit exponent
    if (exponent == 0) return (mantissa == 0) ? 0.0f : (float)mantissa / 32.0f * powf(2.0f, -14.0f);
    if (exponent == 31) return (mantissa == 0) ? __int_as_float(0x7F800000) : __int_as_float(0x7FC00000);
    return (1.0f + (float)mantissa / 32.0f) * powf(2.0f, (float)exponent - 15.0f);
}

static __global__ void r11g11b10_to_rgb_kernel(
    const uint32_t* __restrict__ src, // [H*W] packed
    __half* __restrict__ dst,         // [3, H, W] NCHW
    int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial) return;

    uint32_t packed = src[idx];
    float r = unpack_float11(packed & 0x7FF);
    float g = unpack_float11((packed >> 11) & 0x7FF);
    float b = unpack_float10((packed >> 22) & 0x3FF);

    dst[0 * spatial + idx] = __float2half(r);
    dst[1 * spatial + idx] = __float2half(g);
    dst[2 * spatial + idx] = __float2half(b);
}

void r11g11b10_to_rgb_nchw(const uint32_t* src, __half* dst,
                            int H, int W, cudaStream_t stream) {
    int spatial = H * W;
    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;
    r11g11b10_to_rgb_kernel<<<blocks, threads, 0, stream>>>(src, dst, spatial);
}

static __device__ __forceinline__ uint32_t pack_float11(float val) {
    if (val <= 0.0f) return 0;
    uint32_t bits = __float_as_uint(val);
    uint32_t sign = bits >> 31;
    if (sign) return 0; // Clamp negative to 0
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = (bits >> 17) & 0x3F;
    if (exp >= 31) return (31 << 6); // Infinity
    if (exp <= 0) return 0;
    return ((uint32_t)exp << 6) | man;
}

static __device__ __forceinline__ uint32_t pack_float10(float val) {
    if (val <= 0.0f) return 0;
    uint32_t bits = __float_as_uint(val);
    uint32_t sign = bits >> 31;
    if (sign) return 0;
    int exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = (bits >> 18) & 0x1F;
    if (exp >= 31) return (31 << 5);
    if (exp <= 0) return 0;
    return ((uint32_t)exp << 5) | man;
}

static __global__ void rgb_to_r11g11b10_kernel(
    const __half* __restrict__ src, // [3, H, W] NCHW
    uint32_t* __restrict__ dst,     // [H*W] packed
    int spatial)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial) return;

    float r = __half2float(src[0 * spatial + idx]);
    float g = __half2float(src[1 * spatial + idx]);
    float b = __half2float(src[2 * spatial + idx]);

    dst[idx] = pack_float11(r) | (pack_float11(g) << 11) | (pack_float10(b) << 22);
}

void rgb_nchw_to_r11g11b10(const __half* src, uint32_t* dst,
                            int H, int W, cudaStream_t stream) {
    int spatial = H * W;
    int threads = 256;
    int blocks = (spatial + threads - 1) / threads;
    rgb_to_r11g11b10_kernel<<<blocks, threads, 0, stream>>>(src, dst, spatial);
}

// =========================================================================
// sRGB in-place conversion helpers
// =========================================================================

static __device__ __forceinline__ float d_srgb_to_linear(float x) {
    return x * (x * (x * 0.305306011f + 0.682171111f) + 0.012522878f);
}

static __device__ __forceinline__ float d_linear_to_srgb(float x) {
    x = fmaxf(x, 0.0f);
    float s1 = sqrtf(x);
    float s2 = sqrtf(s1);
    float s3 = sqrtf(s2);
    return 0.585122381f * s1 + 0.783140355f * s2 - 0.368262736f * s3;
}

static __global__ void srgb_to_linear_kernel(__half* data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float v = __half2float(data[idx]);
    data[idx] = __float2half(d_srgb_to_linear(v));
}

static __global__ void linear_to_srgb_kernel(__half* data, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    float v = __half2float(data[idx]);
    data[idx] = __float2half(d_linear_to_srgb(v));
}

void srgb_to_linear_inplace(__half* data, int count, cudaStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    srgb_to_linear_kernel<<<blocks, threads, 0, stream>>>(data, count);
}

void linear_to_srgb_inplace(__half* data, int count, cudaStream_t stream) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    linear_to_srgb_kernel<<<blocks, threads, 0, stream>>>(data, count);
}

// =========================================================================
// Nearest-neighbor MV extraction at output resolution
// =========================================================================
// Replaces extract_motion_vectors + bilinear_resize_nchw for MVs.
// Bilinear interpolation of motion vectors at depth discontinuities creates
// averaged MVs that point to neither foreground nor background, causing
// pervasive ghosting during camera motion.  Nearest-neighbor avoids this.
//
// Optional depth-based dilation (AMD's dilate_mv): for each output pixel,
// examine a 3x3 neighborhood in render space and pick the MV from the
// closest-to-camera sample.  This ensures foreground objects expand their
// MVs into neighboring pixels, preventing background bleed.

static __global__ void extract_mv_nearest_kernel(
    const __half* __restrict__ mv_src,     // [H_render * W_render * mv_channels] interleaved FP16
    const float*  __restrict__ depth_src,  // [H_render * W_render] FP32 depth, or nullptr
    __half* __restrict__ dst,              // [2, H_out, W_out] NCHW
    int H_render, int W_render,
    int H_out, int W_out,
    float mv_scale_x, float mv_scale_y,
    float jitter_cancel_x, float jitter_cancel_y,
    bool jitter_cancel,
    bool depth_inverted,  // true = larger depth = closer to camera
    int mv_channels)      // FP16 values per pixel (2=RG16F, 4=RGBA16F)
{
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= W_out || oy >= H_out) return;

    float inv_scale_x = (float)W_render / (float)W_out;
    float inv_scale_y = (float)H_render / (float)H_out;

    // Map output pixel center to render-res coordinate
    // AMD: center_pos = round(float2(tid) * inv_scale + origin_offset)
    // where origin_offset = 0.5 * inv_scale - 0.5
    float center_fx = ((float)ox + 0.5f) * inv_scale_x - 0.5f;
    float center_fy = ((float)oy + 0.5f) * inv_scale_y - 0.5f;
    int cx = (int)roundf(center_fx);
    int cy = (int)roundf(center_fy);
    cx = max(0, min(cx, W_render - 1));
    cy = max(0, min(cy, H_render - 1));

    int selected_x = cx;
    int selected_y = cy;

    // Depth-based dilation: 3x3 closest-depth in output space, mapped to render space
    if (depth_src != nullptr) {
        float center_depth = depth_src[cy * W_render + cx];
        float closest_depth = center_depth;

        // 8 neighbors in output space (AMD's kDepthOffsets, excluding center)
        const int offsets[8][2] = {
            {-1,-1}, {-1,0}, {-1,1},
            { 0,-1},         { 0,1},
            { 1,-1}, { 1,0}, { 1,1}
        };

        for (int i = 0; i < 8; i++) {
            int nx = ox + offsets[i][1];
            int ny = oy + offsets[i][0];
            // Map neighbor output pixel to render space
            float nfx = ((float)nx + 0.5f) * inv_scale_x - 0.5f;
            float nfy = ((float)ny + 0.5f) * inv_scale_y - 0.5f;
            int nrx = (int)roundf(nfx);
            int nry = (int)roundf(nfy);
            nrx = max(0, min(nrx, W_render - 1));
            nry = max(0, min(nry, H_render - 1));

            float d = depth_src[nry * W_render + nrx];
            bool is_closer = depth_inverted ? (d > closest_depth) : (d < closest_depth);
            if (is_closer) {
                closest_depth = d;
                selected_x = nrx;
                selected_y = nry;
            }
        }
    }

    // Read MV at selected render-res position
    int render_idx = selected_y * W_render + selected_x;
    float mv_x = __half2float(mv_src[render_idx * mv_channels + 0]) * mv_scale_x;
    float mv_y = __half2float(mv_src[render_idx * mv_channels + 1]) * mv_scale_y;

    if (jitter_cancel) {
        mv_x -= jitter_cancel_x;
        mv_y -= jitter_cancel_y;
    }

    // NOTE: No internal render→output scaling here.
    // The caller passes mv_scale_x/y that already include the correct
    // total conversion factor (motionVectorScale * upscale ratio when needed).
    // This avoids double-scaling with UE5 games where motionVectorScale
    // already gives display-resolution pixel values.

    int out_spatial = H_out * W_out;
    int out_idx = oy * W_out + ox;
    dst[0 * out_spatial + out_idx] = __float2half(mv_x);
    dst[1 * out_spatial + out_idx] = __float2half(mv_y);
}

void extract_mv_nearest(const __half* mv_src, const float* depth_src,
                         __half* dst,
                         int H_render, int W_render,
                         int H_out, int W_out,
                         float mv_scale_x, float mv_scale_y,
                         float jitter_cancel_x, float jitter_cancel_y,
                         bool jitter_cancel, bool depth_inverted,
                         int mv_channels,
                         cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((W_out + 15) / 16, (H_out + 15) / 16);
    extract_mv_nearest_kernel<<<grid, block, 0, stream>>>(
        mv_src, depth_src, dst,
        H_render, W_render, H_out, W_out,
        mv_scale_x, mv_scale_y,
        jitter_cancel_x, jitter_cancel_y,
        jitter_cancel, depth_inverted, mv_channels);
}

// =========================================================================
// Bilinear resize (NCHW)
// =========================================================================

static __global__ void bilinear_resize_nchw_kernel(
    const __half* __restrict__ src,
    __half* __restrict__ dst,
    int src_h, int src_w, int dst_h, int dst_w, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= dst_w || y >= dst_h) return;

    float sx = ((float)x + 0.5f) * (float)src_w / (float)dst_w - 0.5f;
    float sy = ((float)y + 0.5f) * (float)src_h / (float)dst_h - 0.5f;

    int x0 = (int)floorf(sx);
    int y0 = (int)floorf(sy);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float fx = sx - (float)x0;
    float fy = sy - (float)y0;

    x0 = max(x0, 0); x1 = min(x1, src_w - 1);
    y0 = max(y0, 0); y1 = min(y1, src_h - 1);

    int src_spatial = src_h * src_w;
    int dst_spatial = dst_h * dst_w;

    for (int c = 0; c < channels; c++) {
        float v00 = __half2float(src[c * src_spatial + y0 * src_w + x0]);
        float v01 = __half2float(src[c * src_spatial + y0 * src_w + x1]);
        float v10 = __half2float(src[c * src_spatial + y1 * src_w + x0]);
        float v11 = __half2float(src[c * src_spatial + y1 * src_w + x1]);

        float v = v00 * (1.0f - fx) * (1.0f - fy) + v01 * fx * (1.0f - fy)
                + v10 * (1.0f - fx) * fy + v11 * fx * fy;

        dst[c * dst_spatial + y * dst_w + x] = __float2half(v);
    }
}

void bilinear_resize_nchw(const __half* src, __half* dst,
                           int src_h, int src_w, int dst_h, int dst_w,
                           int channels, cudaStream_t stream) {
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    bilinear_resize_nchw_kernel<<<grid, block, 0, stream>>>(
        src, dst, src_h, src_w, dst_h, dst_w, channels);
}

} // namespace fsr4
