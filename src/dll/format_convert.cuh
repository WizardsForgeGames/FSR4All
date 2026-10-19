// FSR 4 All - Format Conversion Kernels (D3D12 ↔ Pipeline)
// Copyright (c) 2026 TheWizardsForge. MIT License.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

namespace fsr4 {

// RGBA16F interleaved → RGB NCHW FP16 (drop alpha, deinterleave)
// src: [H*W*4] interleaved, dst: [3, H, W] NCHW
void rgba_to_rgb_nchw(const __half* src, __half* dst,
                      int H, int W, cudaStream_t stream = nullptr);

// RGB NCHW FP16 → RGBA16F interleaved (add alpha=1.0, interleave)
// src: [3, H, W] NCHW, dst: [H*W*4] interleaved
void rgb_nchw_to_rgba(const __half* src, __half* dst,
                      int H, int W, cudaStream_t stream = nullptr);

// Interleaved FP16 MVs → [2, H, W] NCHW, apply scale and optional jitter cancellation
// src: [H*W*mv_channels] interleaved FP16 (mv_channels=2 for RG16F, 4 for RGBA16F)
// dst: [2, H, W] NCHW pixel-space MVs
// mv_scale_x/y: from ffxDispatchDescUpscale.motionVectorScale
// jitter_x/y:   from ffxDispatchDescUpscale.jitterOffset (subtracted if jitter_cancel)
// mv_channels: number of FP16 values per pixel in source (2=RG16F, 4=RGBA16F)
void extract_motion_vectors(const __half* src, __half* dst,
                            int H, int W,
                            float mv_scale_x, float mv_scale_y,
                            float jitter_x, float jitter_y,
                            bool jitter_cancel,
                            bool display_res_mvs,
                            int H_out, int W_out,
                            int mv_channels = 2,
                            cudaStream_t stream = nullptr);

// R11G11B10_FLOAT → RGB NCHW FP16 (unpack 32-bit packed format)
// src: [H*W] uint32 packed, dst: [3, H, W] NCHW FP16
void r11g11b10_to_rgb_nchw(const uint32_t* src, __half* dst,
                            int H, int W, cudaStream_t stream = nullptr);

// RGB NCHW FP16 → R11G11B10_FLOAT (pack to 32-bit)
// src: [3, H, W] NCHW FP16, dst: [H*W] uint32 packed
void rgb_nchw_to_r11g11b10(const __half* src, uint32_t* dst,
                            int H, int W, cudaStream_t stream = nullptr);

// sRGB→linear for pipeline input (when game provides non-linear color)
void srgb_to_linear_inplace(__half* data, int count, cudaStream_t stream = nullptr);

// linear→sRGB for pipeline output (when game expects non-linear output)
void linear_to_srgb_inplace(__half* data, int count, cudaStream_t stream = nullptr);

// Nearest-neighbor MV extraction at output resolution with optional depth dilation.
// Replaces extract_motion_vectors + bilinear_resize_nchw for MVs.
// mv_src: [H_render*W_render*mv_channels] interleaved FP16 at render resolution
// depth_src: [H_render*W_render] FP32 depth at render resolution, or nullptr
// dst: [2, H_out, W_out] NCHW pixel-space MVs at output resolution
// jitter_cancel_x/y: jitter delta (prevJitter - curJitter) when jitter_cancel=true
// mv_channels: number of FP16 values per pixel in source (2=RG16F, 4=RGBA16F)
void extract_mv_nearest(const __half* mv_src, const float* depth_src,
                         __half* dst,
                         int H_render, int W_render,
                         int H_out, int W_out,
                         float mv_scale_x, float mv_scale_y,
                         float jitter_cancel_x, float jitter_cancel_y,
                         bool jitter_cancel, bool depth_inverted,
                         int mv_channels = 2,
                         cudaStream_t stream = nullptr);

// Bilinear resize NCHW FP16 data (works for any channel count)
// src: [channels, src_h, src_w], dst: [channels, dst_h, dst_w]
void bilinear_resize_nchw(const __half* src, __half* dst,
                           int src_h, int src_w, int dst_h, int dst_w,
                           int channels, cudaStream_t stream = nullptr);

} // namespace fsr4
