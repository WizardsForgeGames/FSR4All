// FSR 4 All - DLL Context Manager
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Internal context for the FFX-compatible DLL replacement.
// All compute dispatches recorded inline on the game's command list.
// No CUDA, no staging buffers, no cross-frame delay.
#pragma once

#include <fsr4/ffx_api_compat.h>
#include "fsr4_d3d12_pipeline.h"
#include <d3d12.h>
#include <cstdio>
#include <string>

namespace fsr4 {

// Runtime debug toggles (loaded from fsr4_config.json next to DLL).
// Live-reloaded every ~2 seconds — no rebuild required.
struct DebugConfig {
    bool no_temporal   = false;  // Force reset every frame
    bool negate_mvs    = false;  // Flip MV direction
    bool zero_mvs      = false;  // Disable all MVs
    float mv_scale     = 1.0f;   // Multiply MV magnitude
    bool zero_jitter   = false;  // Zero jitter values
    bool negate_jitter = false;  // Flip jitter sign
    bool dump_frame    = false;  // Dump diagnostics (single shot)
    int  dump_delay    = 0;      // Dump after N frames
    bool use_original  = false;  // Route dispatches to original DLL (A/B comparison)
    bool fast_exp      = true;   // Use fast_exp Post shader (false = AMD original)
    bool debug_blend   = false;  // Show blend factor heatmap overlay
    bool loaded        = false;
};

DebugConfig& get_debug_config();

struct FSR4DLLContext {
    // D3D12 device (from game)
    ID3D12Device* device = nullptr;

    // D3D12 compute pipeline (root sigs, PSOs, descriptors, internal resources)
    FSR4D3D12Pipeline pipeline;

    // Resolution from context creation
    uint32_t max_render_w = 0, max_render_h = 0;
    uint32_t max_upscale_w = 0, max_upscale_h = 0;
    uint32_t create_flags = 0;

    // Debug log
    FILE* log = nullptr;

    // Lifecycle
    bool create(ID3D12Device* dev,
                const ffxCreateContextDescUpscale* upscale_desc,
                const std::string& weight_base);
    void destroy();

    // Per-frame dispatch
    ffxReturnCode_t dispatch(const ffxDispatchDescUpscale* desc);

    // Logging helper
    void logf(const char* fmt, ...);
};

// Quality mode scale factors
inline float get_upscale_ratio(uint32_t quality_mode) {
    switch (quality_mode) {
        case FFX_UPSCALE_QUALITY_MODE_NATIVEAA:          return 1.0f;
        case FFX_UPSCALE_QUALITY_MODE_QUALITY:           return 1.5f;
        case FFX_UPSCALE_QUALITY_MODE_BALANCED:          return 1.7f;
        case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE:       return 2.0f;
        case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
        default: return 1.5f;
    }
}

// Halton sequence for jitter (base 2 and 3)
inline float halton(int index, int base) {
    float result = 0.0f;
    float f = 1.0f / (float)base;
    int i = index;
    while (i > 0) {
        result += f * (float)(i % base);
        i /= base;
        f /= (float)base;
    }
    return result;
}

} // namespace fsr4
