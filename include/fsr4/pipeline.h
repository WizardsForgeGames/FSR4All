// FSR 4 All — Full Upscaling Pipeline (FEAT-5)
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Pre-processing: bilinear upsample + mu-law + YCbCr + normalization → 7ch model input
// Model: U-Net INT8 inference at output resolution
// Post-processing: anisotropic Gaussian filter on low-res input → upscaled RGB
//
// For first-frame / static image: no history, zero motion vectors.
#pragma once

#include <fsr4/model.h>
#include <fsr4/profile.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>

namespace fsr4 {

struct FSR4Pipeline {
    FSR4Model model;

    // Model I/O at output resolution
    __half* d_model_input  = nullptr;  // [1, 7, H_out, W_out]
    __half* d_model_output = nullptr;  // [1, 8, H_out, W_out]

    // Recurrent state from previous frame [1, 4, H_out, W_out]
    __half* d_recurrent = nullptr;

    // Persistent history buffers (written by postprocess, survive across frames)
    __half* d_history_color = nullptr;      // [3, H_out, W_out] mu-law blended output
    // Note: d_recurrent also persists across frames (written by postprocess)

    // Pre-converted low-res input (sRGB→linear+mu-law, NCHW→NHWC, per-frame)
    __half* d_lr_mulaw_nhwc = nullptr;         // [H_lr, W_lr, 3] NHWC mu-law linear

    // Per-frame reprojected buffers (written by reprojection kernel, consumed within frame)
    __half* d_reprojected_mulaw = nullptr;     // [3, H_out, W_out] warped history color
    __half* d_reprojected_recurrent = nullptr;  // [4, H_out, W_out] warped recurrent state

    // Auto-exposure (matches AMD SPD auto-exposure pipeline)
    float* d_exposure_value = nullptr;         // [1] computed exposure on device
    float* d_log_lum_accum = nullptr;          // [1] atomic accumulator for log-lum sum
    int*   d_pixel_count   = nullptr;          // [1] atomic pixel counter

    int H_lr = 0, W_lr = 0;
    int H_out = 0, W_out = 0;
    int scale = 2;
    bool first_frame = true;
    bool bypass_model = false;  // Skip model, zero output → neutral filter params
    bool input_linear = false;   // Input is already linear (skip sRGB→linear in lr_prepare)
    bool output_linear = false;  // Output should be linear (skip linear→sRGB in postprocess)
    float rect_alpha = 0.1f;    // History blend strength (lower = less ghosting, less temporal detail)
    float max_blend = 1.0f;     // Cap on temporal blend factor (1.0 = no cap, 0.0 = pure Gaussian)

    // Initialize model and allocate all pipeline buffers.
    // H_lr, W_lr = input (low-res) dimensions. Must be divisible by 8.
    // scale = upscaling factor (default 2).
    bool init(const std::string& weight_dir, int H_lr, int W_lr, int scale = 2);

    // Run the full upscaling pipeline on one frame.
    // d_input_srgb:      [1, 3, H_lr, W_lr] sRGB FP16 on GPU (values in [0,1])
    // d_out_srgb:        [1, 3, H_out, W_out] sRGB FP16 on GPU (output)
    // d_motion_vectors:  [2, H_out, W_out] FP16 pixel-space MVs at output resolution, or nullptr.
    //                    Convention: current_pos + mv = previous_pos.
    //                    nullptr = identity reprojection (no motion).
    // jitter_x/y:        Sub-pixel jitter offset (in low-res pixel units) from the game.
    //                    Used to shift the reconstruction filter so temporal samples
    //                    resolve into a stable output image.
    // For static images, call once with first_frame=true (set automatically).
    void upscale(const __half* d_input_srgb, __half* d_out_srgb,
                 const __half* d_motion_vectors = nullptr,
                 float jitter_x = 0.0f, float jitter_y = 0.0f,
                 cudaStream_t stream = nullptr, FSR4Profile* profile = nullptr);

    void free();
};

// Standalone kernel launchers (exposed for testing)

// sRGB→linear conversion (in-place or out-of-place)
void srgb_to_linear(const __half* input, __half* output, int count,
                    cudaStream_t stream = nullptr);

// linear→sRGB conversion (in-place or out-of-place)
void linear_to_srgb(const __half* input, __half* output, int count,
                    cudaStream_t stream = nullptr);

} // namespace fsr4
