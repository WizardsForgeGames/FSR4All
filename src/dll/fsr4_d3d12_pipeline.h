// FSR 4 All - D3D12 Compute Pipeline
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Manages root signatures, PSOs, descriptor heaps, and internal GPU resources
// for dispatching AMD's FSR4 HLSL compute shaders directly on the game's
// command list. No CUDA, no staging buffers, no cross-frame delay.
#pragma once

#include <d3d12.h>
#include <dxgi1_4.h>
#include <cstdint>
#include <array>
#include <cstdio>

struct ffxDispatchDescUpscale;

namespace fsr4 {

// Matches HLSL cbuffer MLSR_Optimized_Constants at register(b0)
// Defined in ffx_fsr4_upscale_resources.h via mlsr_optimized_includes.hlsli
struct alignas(256) OptimizedConstants {
    float inv_size[2];
    float scale[2];
    float inv_scale[2];
    float jitter[2];
    float mv_scale[2];
    float tex_size[2];
    float max_renderSize[2];
    float fMotionVectorJitterCancellation[2];

    uint32_t width;
    uint32_t height;
    uint32_t reset;
    uint32_t width_lr;

    uint32_t height_lr;
    float preExposure;
    float previous_preExposure;
    uint32_t rcas_enabled;

    float rcas_sharpness;
    float _pad1;
};

// Matches HLSL cbuffer cbRCAS at register(b0)
// NOTE: RCAS shader overrides bind to b0, NOT b2. We bind this to b0 during RCAS pass.
struct alignas(256) RcasConstants {
    uint32_t config[4];
    float    preExposure;
    uint32_t _pad[3];
};

// Internal texture indices
enum InternalTexture {
    TEX_HISTORY = 0,            // R16G16B16A16_FLOAT, upscale res
    TEX_RECURRENT_0,            // R8G8B8A8_UNORM, upscale res
    TEX_HISTORY_REPROJECTED,    // R16G16B16A16_FLOAT, upscale res
    TEX_DILATED_MV,             // R16G16_FLOAT, upscale res
    TEX_RCAS_TEMP,              // R16G16B16A16_FLOAT, upscale res
    TEX_DEFAULT_EXPOSURE,       // R32G32_FLOAT, 1x1
    TEX_DEBUG_BLEND,            // R32_FLOAT, upscale res (blend factor debug viz)
    TEX_COUNT
};

// PSO indices (matches shader blob order)
enum PSOIndex {
    PSO_PRE = 0,
    PSO_PASS_1,  PSO_PASS_2,  PSO_PASS_3,  PSO_PASS_4,
    PSO_PASS_5,  PSO_PASS_6,  PSO_PASS_7,  PSO_PASS_8,
    PSO_PASS_9,  PSO_PASS_10, PSO_PASS_11, PSO_PASS_12,
    PSO_POST,
    PSO_RCAS,
    PSO_POST_DEBUG,         // Post with FFX_DEBUG_VISUALIZE=1 (writes blend factor to u12)
    PSO_DEBUG_OVERLAY,      // Heatmap overlay: reads debug tex, composites onto output
    PSO_COUNT
};

// Pre/post root signature parameter indices
enum PrePostRootParam {
    PP_ROOT_CBV_B0 = 0,    // OptimizedConstants (pre/post) or RcasConstants (rcas)
    PP_ROOT_CBV_B1,         // cbPass_Weights (encoder1 conv weights, 256 uint32_t)
    PP_SRV_TABLE,           // t0-t20 (21 descriptors)
    PP_UAV_TABLE,           // u0-u12 (13 descriptors)
    PP_ROOT_PARAM_COUNT
};

// Model root signature parameter indices
enum ModelRootParam {
    MODEL_ROOT_SRV_T0 = 0,  // Scratch buffer (read region)
    MODEL_ROOT_SRV_T1,      // Initializer buffer
    MODEL_ROOT_UAV_U0,      // Scratch buffer (write region)
    MODEL_ROOT_UAV_U1,      // Scratch buffer
    MODEL_ROOT_PARAM_COUNT
};

// Descriptor heap layout
static const uint32_t DESC_SRV_TABLE_START = 0;     // t0-t20 for pre/post
static const uint32_t DESC_SRV_COUNT       = 21;
static const uint32_t DESC_UAV_TABLE_START = 21;     // u0-u12 for pre/post
static const uint32_t DESC_UAV_COUNT       = 13;
// RCAS uses a second set of SRV/UAV descriptors with different aliases
static const uint32_t DESC_RCAS_SRV_START  = 34;     // t0-t20 for RCAS
static const uint32_t DESC_RCAS_UAV_START  = 55;     // u0-u12 for RCAS
// Debug overlay uses a third set with t19=debug_blend SRV, u2=game output UAV
static const uint32_t DESC_OVERLAY_SRV_START = 68;    // t0-t20 for overlay
static const uint32_t DESC_OVERLAY_UAV_START = 89;    // u0-u12 for overlay
static const uint32_t DESC_TOTAL           = 102;

struct FSR4D3D12Pipeline {
    // Root signatures
    ID3D12RootSignature* pre_post_root_sig = nullptr;   // Pre, Post, RCAS
    ID3D12RootSignature* model_root_sig = nullptr;      // Model passes 1-12

    // Pipeline state objects
    std::array<ID3D12PipelineState*, PSO_COUNT> pso = {};
    ID3D12PipelineState* pso_post_normal = nullptr;  // AMD original Post (A/B toggle)

    // Internal GPU resources (DEFAULT heap)
    std::array<ID3D12Resource*, TEX_COUNT> textures = {};
    ID3D12Resource* scratch_buffer = nullptr;        // Raw buffer ~20MB
    ID3D12Resource* initializer_buffer = nullptr;    // Raw buffer ~87KB

    // Upload heap resources
    ID3D12Resource* cb_upload = nullptr;              // Constant buffers
    ID3D12Resource* initializer_upload = nullptr;     // One-time init upload
    void* cb_mapped = nullptr;                        // Persistently mapped

    // Descriptor heap (CBV/SRV/UAV)
    ID3D12DescriptorHeap* srv_uav_heap = nullptr;
    uint32_t descriptor_increment = 0;

    // Resolution and quality
    uint32_t max_upscale_w = 0, max_upscale_h = 0;
    uint32_t max_render_w = 0, max_render_h = 0;
    int resolution_tier = 1;   // 0=1080p, 1=2160p, 2=4320p
    int quality_preset = 2;    // 0=quality, 1=balanced, 2=performance, 3=ultraperf

    // Constant buffer offsets in upload buffer
    static const uint32_t CB_MAIN_OFFSET = 0;        // OptimizedConstants (256 bytes)
    static const uint32_t CB_RCAS_OFFSET = 256;       // RcasConstants (256 bytes)
    static const uint32_t CB_WEIGHTS_OFFSET = 512;    // cbPass_Weights (1024 bytes)
    static const uint32_t CB_EXPOSURE_OFFSET = 1536;    // Default exposure data (8 bytes, 512-aligned for texture copy)
    static const uint32_t CB_UPLOAD_SIZE = 2048;       // Total upload buffer

    // State tracking
    float previous_jitter[2] = {};
    bool initializer_uploaded = false;
    bool resources_created = false;
    bool use_fast_exp = true;  // Toggle: true=fast_exp Post, false=AMD original Post
    int frame_index = 0;

    // Log file (shared with context)
    FILE* log = nullptr;

#ifndef FSR4_RELEASE
    // GPU readback diagnostics (debug only)
    ID3D12Resource* diag_readback = nullptr;
    bool diag_pending = false;
    uint32_t diag_center_x = 0, diag_center_y = 0;
    static const uint32_t DIAG_BUFFER_SIZE = 3328;
    static const uint32_t DIAG_HISTORY_OFFSET = 0;
    static const uint32_t DIAG_REPROJECTED_OFFSET = 256;
    static const uint32_t DIAG_MV_CENTER_OFFSET = 512;
    static const uint32_t DIAG_RECURRENT_OFFSET = 768;
    static const uint32_t DIAG_MV_QUARTER_OFFSET = 1024;
    static const uint32_t DIAG_MV_3QUARTER_OFFSET = 1280;
    static const uint32_t DIAG_MV_CORNER_OFFSET = 1536;
    static const uint32_t DIAG_OUTPUT_OFFSET = 1792;
    static const uint32_t DIAG_INPUT_COLOR_OFFSET = 2048;
    static const uint32_t DIAG_EXPOSURE_OFFSET = 2304;
    static const uint32_t DIAG_DILATED_MV_OFFSET = 2560;
    static const uint32_t DIAG_RCAS_TEMP_OFFSET = 2816;

    // GPU timestamp profiling
    // 17 timestamps: before Pre, after Pre, after Model 1-12, after Post, after RCAS
    static const uint32_t TIMESTAMP_COUNT = 17;
    ID3D12QueryHeap* timestamp_heap = nullptr;
    ID3D12Resource* timestamp_readback = nullptr;
    bool timestamp_pending = false;
    uint32_t timestamp_pass_count = 0;   // How many timestamps were written last frame
    uint64_t gpu_timestamp_frequency = 0; // Ticks per second from command queue

    void resolve_timestamps(ID3D12GraphicsCommandList* cmd, uint32_t count);
    void read_timestamp_results();
#endif

    // Lifecycle
    bool create(ID3D12Device* device,
                uint32_t max_render_w, uint32_t max_render_h,
                uint32_t max_upscale_w, uint32_t max_upscale_h,
                uint32_t create_flags);
    void destroy();

    // Setup descriptors for current frame's game resources
    void bind_game_resources(ID3D12Device* device,
                             ID3D12Resource* color_res, DXGI_FORMAT color_fmt,
                             ID3D12Resource* depth_res, DXGI_FORMAT depth_fmt,
                             ID3D12Resource* mv_res, DXGI_FORMAT mv_fmt,
                             ID3D12Resource* exposure_res,
                             ID3D12Resource* output_res, DXGI_FORMAT output_fmt,
                             bool enable_sharpening);

    // Update constant buffer from dispatch descriptor
    void update_constants(const ffxDispatchDescUpscale* desc,
                          uint32_t upscale_w, uint32_t upscale_h,
                          bool force_reset, uint32_t create_flags);

    // Record all compute dispatches on game's command list
    void dispatch(ID3D12GraphicsCommandList* cmd,
                  uint32_t render_w, uint32_t render_h,
                  uint32_t upscale_w, uint32_t upscale_h,
                  bool enable_sharpening,
                  bool debug_blend = false);

    // Upload initializer data (call once on first dispatch)
    void ensure_initializer_uploaded(ID3D12GraphicsCommandList* cmd);

    // Update default exposure texture with current preExposure value
    // Call each frame when no game exposure resource is provided
    void update_default_exposure(ID3D12GraphicsCommandList* cmd, float preExposure);

#ifndef FSR4_RELEASE
    // GPU readback diagnostics: dump center pixel of internal textures
    void issue_diagnostic_readback(ID3D12GraphicsCommandList* cmd,
                                    ID3D12Resource* mv_res, DXGI_FORMAT mv_fmt,
                                    ID3D12Resource* output_res, DXGI_FORMAT output_fmt,
                                    ID3D12Resource* color_res, DXGI_FORMAT color_fmt,
                                    ID3D12Resource* exposure_res, DXGI_FORMAT exposure_fmt,
                                    uint32_t render_w, uint32_t render_h);
    void read_diagnostic_results();
#endif

    void logf(const char* fmt, ...);

private:
    bool create_root_signatures(ID3D12Device* device);
    bool create_psos(ID3D12Device* device);
    bool create_internal_resources(ID3D12Device* device);
    bool create_descriptor_heap(ID3D12Device* device);
    void setup_internal_descriptors(ID3D12Device* device);

    void uav_barrier(ID3D12GraphicsCommandList* cmd, ID3D12Resource* resource);
    void uav_barrier_all(ID3D12GraphicsCommandList* cmd);

    D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle(uint32_t index) const;
    D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle(uint32_t index) const;
};

// Helper: RoundUpDiv
template<typename T>
inline T round_up_div(T a, T b) { return (a + b - 1) / b; }

// Helper: AlignUp to multiple
template<typename T>
inline T align_up(T val, T align) { return round_up_div(val, align) * align; }

} // namespace fsr4
