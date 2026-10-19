// FSR 4 All - DLL Context Manager Implementation
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// D3D12 compute pipeline — all dispatches inline on the game's command list.
// No CUDA, no staging buffers, no background threads.

#include "fsr4_context.h"
#include <cstdarg>
#include <cmath>
#include <cstring>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// High-resolution timing
static double qpc_freq_inv = 0.0;
static void init_timer() {
    if (qpc_freq_inv == 0.0) {
        LARGE_INTEGER freq;
        QueryPerformanceFrequency(&freq);
        qpc_freq_inv = 1000.0 / (double)freq.QuadPart;
    }
}
static double now_ms() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * qpc_freq_inv;
}

namespace fsr4 {

// =========================================================================
// DXGI format name (for logging)
// =========================================================================

static const char* dxgi_format_name(DXGI_FORMAT fmt) {
    switch (fmt) {
        case DXGI_FORMAT_UNKNOWN:               return "NONE(0)";
        case DXGI_FORMAT_R16G16_FLOAT:          return "R16G16_FLOAT";
        case DXGI_FORMAT_R16G16_UNORM:          return "R16G16_UNORM";
        case DXGI_FORMAT_R16G16_SNORM:          return "R16G16_SNORM";
        case DXGI_FORMAT_R32G32_FLOAT:          return "R32G32_FLOAT";
        case DXGI_FORMAT_R16G16B16A16_FLOAT:    return "R16G16B16A16_FLOAT";
        case DXGI_FORMAT_R11G11B10_FLOAT:       return "R11G11B10_FLOAT";
        case DXGI_FORMAT_R10G10B10A2_UNORM:     return "R10G10B10A2_UNORM";
        case DXGI_FORMAT_R32_FLOAT:             return "R32_FLOAT";
        case DXGI_FORMAT_D32_FLOAT:             return "D32_FLOAT";
        case DXGI_FORMAT_R32_TYPELESS:          return "R32_TYPELESS";
        case DXGI_FORMAT_R24G8_TYPELESS:        return "R24G8_TYPELESS";
        case DXGI_FORMAT_D24_UNORM_S8_UINT:     return "D24_UNORM_S8_UINT";
        case DXGI_FORMAT_R32G8X24_TYPELESS:     return "R32G8X24_TYPELESS";
        case DXGI_FORMAT_D32_FLOAT_S8X24_UINT:  return "D32_FLOAT_S8X24_UINT";
        case DXGI_FORMAT_R32_FLOAT_X8X24_TYPELESS: return "R32_FLOAT_X8X24_TYPELESS";
        case DXGI_FORMAT_R16_FLOAT:             return "R16_FLOAT";
        case DXGI_FORMAT_R16_UNORM:             return "R16_UNORM";
        case DXGI_FORMAT_R16_TYPELESS:          return "R16_TYPELESS";
        case DXGI_FORMAT_D16_UNORM:             return "D16_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM:        return "R8G8B8A8_UNORM";
        case DXGI_FORMAT_B8G8R8A8_UNORM:        return "B8G8R8A8_UNORM";
        case DXGI_FORMAT_R8G8B8A8_UNORM_SRGB:   return "R8G8B8A8_UNORM_SRGB";
        default: {
            // Return raw format number for anything we missed
            static thread_local char buf[32];
            snprintf(buf, sizeof(buf), "FMT_%d", (int)fmt);
            return buf;
        }
    }
}

// =========================================================================
// FFX API → D3D12 resource state conversion
// =========================================================================

static D3D12_RESOURCE_STATES ffx_api_to_d3d12_state(uint32_t ffx_state) {
    if (ffx_state & FFX_API_RESOURCE_STATE_UNORDERED_ACCESS)
        return D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
    if (ffx_state & FFX_API_RESOURCE_STATE_RENDER_TARGET)
        return D3D12_RESOURCE_STATE_RENDER_TARGET;
    if (ffx_state & FFX_API_RESOURCE_STATE_COPY_SRC)
        return D3D12_RESOURCE_STATE_COPY_SOURCE;
    if (ffx_state & FFX_API_RESOURCE_STATE_COPY_DEST)
        return D3D12_RESOURCE_STATE_COPY_DEST;

    D3D12_RESOURCE_STATES d3d = (D3D12_RESOURCE_STATES)0;
    if (ffx_state & FFX_API_RESOURCE_STATE_COMPUTE_READ)
        d3d |= D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    if (ffx_state & FFX_API_RESOURCE_STATE_PIXEL_READ)
        d3d |= D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
    if (d3d != (D3D12_RESOURCE_STATES)0) return d3d;

    return D3D12_RESOURCE_STATE_COMMON;
}

// =========================================================================
// Resource barrier helper
// =========================================================================

static D3D12_RESOURCE_BARRIER make_transition(
    ID3D12Resource* res, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after)
{
    D3D12_RESOURCE_BARRIER b = {};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition.pResource = res;
    b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    b.Transition.StateBefore = before;
    b.Transition.StateAfter = after;
    return b;
}

// =========================================================================
// Debug config (fsr4_config.json next to DLL, live-reloaded every 2s)
// In release mode: returns static default config (no file I/O)
// =========================================================================

static DebugConfig g_debug_config;

#ifdef FSR4_RELEASE

DebugConfig& get_debug_config() {
    return g_debug_config;
}

#else  // Debug mode: full config file polling

static std::string g_config_path;
static double g_config_last_check = 0;
static FILETIME g_config_last_write = {};

static std::string get_dll_directory() {
    char dll_path[MAX_PATH] = {};
    HMODULE hmod = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)get_dll_directory, &hmod);
    GetModuleFileNameA(hmod, dll_path, MAX_PATH);
    std::string dir(dll_path);
    size_t sep = dir.find_last_of("\\/");
    return (sep != std::string::npos) ? dir.substr(0, sep) : ".";
}

static bool json_find_bool(const std::string& json, const char* key, bool def) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos < json.size() && json[pos] == 't') return true;
    if (pos < json.size() && json[pos] == 'f') return false;
    return def;
}

static float json_find_float(const std::string& json, const char* key, float def) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return def;
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return def;
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    char* end = nullptr;
    float val = strtof(json.c_str() + pos, &end);
    return (end && end != json.c_str() + pos) ? val : def;
}

static void write_default_config(const std::string& path) {
    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f,
        "{\n"
        "    \"no_temporal\":   false,\n"
        "    \"negate_mvs\":    false,\n"
        "    \"zero_mvs\":      false,\n"
        "    \"mv_scale\":      1.0,\n"
        "    \"zero_jitter\":   false,\n"
        "    \"negate_jitter\": false,\n"
        "    \"dump_frame\":    false,\n"
        "    \"dump_delay\":    0,\n"
        "    \"use_original\":  false,\n"
        "    \"fast_exp\":      true,\n"
        "    \"debug_blend\":   false\n"
        "}\n");
    fclose(f);
}

static void load_config_from_file(const std::string& path) {
    FILE* f = fopen(path.c_str(), "r");
    if (!f) {
        write_default_config(path);
        g_debug_config.loaded = true;
        return;
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json(sz, '\0');
    fread(&json[0], 1, sz, f);
    fclose(f);

    g_debug_config.no_temporal   = json_find_bool(json, "no_temporal", false);
    g_debug_config.negate_mvs    = json_find_bool(json, "negate_mvs", false);
    g_debug_config.zero_mvs      = json_find_bool(json, "zero_mvs", false);
    g_debug_config.mv_scale      = json_find_float(json, "mv_scale", 1.0f);
    g_debug_config.zero_jitter   = json_find_bool(json, "zero_jitter", false);
    g_debug_config.negate_jitter = json_find_bool(json, "negate_jitter", false);
    g_debug_config.dump_frame    = json_find_bool(json, "dump_frame", false);
    g_debug_config.dump_delay    = (int)json_find_float(json, "dump_delay", 0.0f);
    g_debug_config.use_original  = json_find_bool(json, "use_original", false);
    g_debug_config.fast_exp      = json_find_bool(json, "fast_exp", true);
    g_debug_config.debug_blend   = json_find_bool(json, "debug_blend", false);
    g_debug_config.loaded = true;
}

DebugConfig& get_debug_config() {
    init_timer();

    if (g_config_path.empty()) {
        g_config_path = get_dll_directory() + "\\fsr4_config.json";
        load_config_from_file(g_config_path);
        g_config_last_check = now_ms();
        HANDLE hf = CreateFileA(g_config_path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE,
                                nullptr, OPEN_EXISTING, 0, nullptr);
        if (hf != INVALID_HANDLE_VALUE) {
            GetFileTime(hf, nullptr, nullptr, &g_config_last_write);
            CloseHandle(hf);
        }
        return g_debug_config;
    }

    double now = now_ms();
    if (now - g_config_last_check > 2000.0) {
        g_config_last_check = now;
        HANDLE hf = CreateFileA(g_config_path.c_str(), GENERIC_READ,
                                FILE_SHARE_READ | FILE_SHARE_WRITE,
                                nullptr, OPEN_EXISTING, 0, nullptr);
        if (hf != INVALID_HANDLE_VALUE) {
            FILETIME ft;
            GetFileTime(hf, nullptr, nullptr, &ft);
            CloseHandle(hf);
            if (CompareFileTime(&ft, &g_config_last_write) != 0) {
                g_config_last_write = ft;
                load_config_from_file(g_config_path);
                printf("[FSR4] Config reloaded: dump=%d no_temporal=%d zero_mvs=%d negate_mvs=%d mv_scale=%.2f\n",
                       g_debug_config.dump_frame, g_debug_config.no_temporal,
                       g_debug_config.zero_mvs, g_debug_config.negate_mvs,
                       g_debug_config.mv_scale);
            }
        }
    }
    return g_debug_config;
}

#endif  // FSR4_RELEASE

// =========================================================================
// Logging
// =========================================================================

#ifdef FSR4_RELEASE
void FSR4DLLContext::logf(const char*, ...) {}
#else
void FSR4DLLContext::logf(const char* fmt, ...) {
    if (!log) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(log, fmt, args);
    va_end(args);
    fflush(log);
}
#endif

// =========================================================================
// Context lifecycle
// =========================================================================

bool FSR4DLLContext::create(ID3D12Device* dev,
                             const ffxCreateContextDescUpscale* upscale_desc,
                             const std::string& weight_base) {
    init_timer();
    device = dev;
    create_flags = upscale_desc->flags;
    max_render_w = upscale_desc->maxRenderSize.width;
    max_render_h = upscale_desc->maxRenderSize.height;
    max_upscale_w = upscale_desc->maxUpscaleSize.width;
    max_upscale_h = upscale_desc->maxUpscaleSize.height;

    logf("Creating context: render=%ux%u upscale=%ux%u flags=0x%x\n",
         max_render_w, max_render_h, max_upscale_w, max_upscale_h, create_flags);

    // Select quality preset based on scaling ratio (matches AMD's ffx_provider_fsr4_dx12.cpp)
    float scaling_ratio = (max_render_w > 0) ?
        static_cast<float>(max_upscale_w) / static_cast<float>(max_render_w) : 2.0f;
    int preset = 0;  // Quality (default for ratio < 1.69)
    if (scaling_ratio >= 2.99f) preset = 3;       // UltraPerformance
    else if (scaling_ratio >= 1.99f) preset = 2;  // Performance
    else if (scaling_ratio >= 1.69f) preset = 1;  // Balanced

    const char* preset_names[] = { "Quality", "Balanced", "Performance", "UltraPerformance" };
    logf("Scaling ratio: %.2f -> preset: %s (%d)\n", scaling_ratio, preset_names[preset], preset);

    // Create D3D12 compute pipeline (root sigs, PSOs, textures, descriptors)
    pipeline.log = log;
    pipeline.quality_preset = preset;
    if (!pipeline.create(device, max_render_w, max_render_h,
                         max_upscale_w, max_upscale_h, create_flags)) {
        logf("ERROR: D3D12 pipeline creation failed\n");
        return false;
    }

#ifndef FSR4_RELEASE
    // Log config state
    auto& dbg = get_debug_config();
    logf("Config: %s (live-reload every 2s)\n", g_config_path.c_str());
    if (dbg.no_temporal || dbg.negate_mvs || dbg.zero_mvs || dbg.mv_scale != 1.0f ||
        dbg.zero_jitter || dbg.negate_jitter) {
        logf("Active toggles:\n");
        if (dbg.no_temporal)      logf("  no_temporal = true\n");
        if (dbg.negate_mvs)       logf("  negate_mvs = true\n");
        if (dbg.zero_mvs)         logf("  zero_mvs = true\n");
        if (dbg.mv_scale != 1.0f) logf("  mv_scale = %.3f\n", dbg.mv_scale);
        if (dbg.zero_jitter)      logf("  zero_jitter = true\n");
        if (dbg.negate_jitter)    logf("  negate_jitter = true\n");
    }
#endif

#ifndef FSR4_RELEASE
    // Query GPU timestamp frequency (needed for profiling)
    // Create a temporary direct queue just to get the frequency, then release it
    {
        D3D12_COMMAND_QUEUE_DESC qd = {};
        qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        ID3D12CommandQueue* tmp_queue = nullptr;
        if (SUCCEEDED(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&tmp_queue)))) {
            uint64_t freq = 0;
            if (SUCCEEDED(tmp_queue->GetTimestampFrequency(&freq))) {
                pipeline.gpu_timestamp_frequency = freq;
                logf("GPU timestamp frequency: %llu Hz (%.1f MHz)\n",
                     (unsigned long long)freq, freq / 1e6);
            }
            tmp_queue->Release();
        }
    }
#endif

    logf("Context created (D3D12 compute, %d PSOs, inline dispatch)\n", PSO_COUNT);
    return true;
}

void FSR4DLLContext::destroy() {
    logf("Destroying context\n");
    pipeline.destroy();
    device = nullptr;
}

// =========================================================================
// Per-frame dispatch (all compute passes inline on game's command list)
// =========================================================================

ffxReturnCode_t FSR4DLLContext::dispatch(const ffxDispatchDescUpscale* desc) {
#ifndef FSR4_RELEASE
    init_timer();
    double t0 = now_ms();
#endif

    auto* cmd = static_cast<ID3D12GraphicsCommandList*>(desc->commandList);
    if (!cmd) {
        logf("ERROR: null commandList\n");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    auto* color_res = static_cast<ID3D12Resource*>(desc->color.resource);
    auto* depth_res = static_cast<ID3D12Resource*>(desc->depth.resource);
    auto* mv_res = static_cast<ID3D12Resource*>(desc->motionVectors.resource);
    auto* exposure_res = static_cast<ID3D12Resource*>(desc->exposure.resource);
    auto* output_res = static_cast<ID3D12Resource*>(desc->output.resource);

    if (!color_res || !output_res) {
        logf("ERROR: null color or output resource\n");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    DXGI_FORMAT color_fmt = color_res->GetDesc().Format;
    DXGI_FORMAT depth_fmt = depth_res ? depth_res->GetDesc().Format : DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT mv_fmt = mv_res ? mv_res->GetDesc().Format : DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT exposure_fmt = exposure_res ? exposure_res->GetDesc().Format : DXGI_FORMAT_UNKNOWN;
    DXGI_FORMAT output_fmt = output_res->GetDesc().Format;

    uint32_t render_w = desc->renderSize.width;
    uint32_t render_h = desc->renderSize.height;
    uint32_t upscale_w = desc->upscaleSize.width > 0 ? desc->upscaleSize.width : max_upscale_w;
    uint32_t upscale_h = desc->upscaleSize.height > 0 ? desc->upscaleSize.height : max_upscale_h;

#ifdef FSR4_RELEASE
    bool force_reset = desc->reset;
#else
    // Apply debug config overrides
    auto& dbg = get_debug_config();
    bool force_reset = desc->reset || dbg.no_temporal;
#endif

    // =====================================================================
    // STEP 1: Transition game resources to compute states
    // =====================================================================
    D3D12_RESOURCE_BARRIER barriers[5];
    int n = 0;

    D3D12_RESOURCE_STATES color_before = ffx_api_to_d3d12_state(desc->color.state);
    if (color_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        barriers[n++] = make_transition(color_res, color_before,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);

    D3D12_RESOURCE_STATES depth_before = (D3D12_RESOURCE_STATES)0;
    if (depth_res) {
        depth_before = ffx_api_to_d3d12_state(desc->depth.state);
        if (depth_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            barriers[n++] = make_transition(depth_res, depth_before,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    D3D12_RESOURCE_STATES mv_before = (D3D12_RESOURCE_STATES)0;
    if (mv_res) {
        mv_before = ffx_api_to_d3d12_state(desc->motionVectors.state);
        if (mv_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            barriers[n++] = make_transition(mv_res, mv_before,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    D3D12_RESOURCE_STATES output_before = ffx_api_to_d3d12_state(desc->output.state);
    if (output_before != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        barriers[n++] = make_transition(output_res, output_before,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

    D3D12_RESOURCE_STATES exposure_before = (D3D12_RESOURCE_STATES)0;
    if (exposure_res) {
        exposure_before = ffx_api_to_d3d12_state(desc->exposure.state);
        if (exposure_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
            barriers[n++] = make_transition(exposure_res, exposure_before,
                                            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    }

    if (n > 0) cmd->ResourceBarrier(n, barriers);

    // =====================================================================
    // STEP 2: Bind game resources as SRV/UAV descriptors
    // =====================================================================
    pipeline.bind_game_resources(device,
                                  color_res, color_fmt,
                                  depth_res, depth_fmt,
                                  mv_res, mv_fmt,
                                  exposure_res,
                                  output_res, output_fmt,
                                  desc->enableSharpening);

    // =====================================================================
    // STEP 3: Update constant buffer
    // =====================================================================
    pipeline.update_constants(desc, upscale_w, upscale_h,
                               force_reset, create_flags);

#ifndef FSR4_RELEASE
    // Apply debug overrides to the mapped constant buffer
    auto* main_cb = reinterpret_cast<OptimizedConstants*>(
        static_cast<uint8_t*>(pipeline.cb_mapped) + FSR4D3D12Pipeline::CB_MAIN_OFFSET);

    if (dbg.zero_mvs) {
        main_cb->mv_scale[0] = 0;
        main_cb->mv_scale[1] = 0;
        main_cb->fMotionVectorJitterCancellation[0] = 0;
        main_cb->fMotionVectorJitterCancellation[1] = 0;
    }
    if (dbg.negate_mvs) {
        main_cb->mv_scale[0] = -main_cb->mv_scale[0];
        main_cb->mv_scale[1] = -main_cb->mv_scale[1];
    }
    if (dbg.mv_scale != 1.0f) {
        main_cb->mv_scale[0] *= dbg.mv_scale;
        main_cb->mv_scale[1] *= dbg.mv_scale;
    }
    if (dbg.zero_jitter) {
        main_cb->jitter[0] = 0;
        main_cb->jitter[1] = 0;
    }
    if (dbg.negate_jitter) {
        main_cb->jitter[0] = -main_cb->jitter[0];
        main_cb->jitter[1] = -main_cb->jitter[1];
    }
#endif

    // =====================================================================
    // STEP 4: Upload initializer data (first frame only)
    // =====================================================================
    pipeline.ensure_initializer_uploaded(cmd);

    // =====================================================================
    // STEP 4b: Update default exposure texture with preExposure
    // When no game exposure resource: shader computes exposure_value = tex.r / preExposure.
    // Writing preExposure to texture gives exposure_value = preExposure/preExposure = 1.0,
    // keeping colors in the model's expected range.
    // =====================================================================
    if (!exposure_res) {
        pipeline.update_default_exposure(cmd, desc->preExposure);
    }

    // =====================================================================
    // STEP 5: Dispatch all compute passes
    // =====================================================================
#ifndef FSR4_RELEASE
    double t1 = now_ms();
#endif

    pipeline.use_fast_exp = get_debug_config().fast_exp;
#ifndef FSR4_RELEASE
    bool debug_blend = get_debug_config().debug_blend;
#else
    bool debug_blend = false;
#endif
    pipeline.dispatch(cmd, render_w, render_h,
                       upscale_w, upscale_h,
                       desc->enableSharpening,
                       debug_blend);

#ifndef FSR4_RELEASE
    double t2 = now_ms();

    // =====================================================================
    // STEP 5b: GPU timestamp profiling (read previous frame, log every 120th)
    // =====================================================================
    if (pipeline.timestamp_pending) {
        // Only log first 5 frames + every 120th (matches existing debug log cadence)
        bool should_log = (pipeline.frame_index <= 5) || (pipeline.frame_index % 120 == 1);
        if (should_log) {
            pipeline.read_timestamp_results();
        } else {
            pipeline.timestamp_pending = false;
        }
    }

    // =====================================================================
    // STEP 5c: GPU diagnostic readback (when dump_frame is enabled)
    // Issue copy on this frame, read results on next frame
    // =====================================================================
    pipeline.read_diagnostic_results();  // Read previous frame's data
    if (dbg.dump_frame) {
        pipeline.issue_diagnostic_readback(cmd, mv_res, mv_fmt,
                                            output_res, output_fmt,
                                            color_res, color_fmt,
                                            exposure_res, exposure_fmt,
                                            render_w, render_h);
        logf("DIAG: Readback issued for frame %d (will read next frame)\n",
             pipeline.frame_index + 1);
        printf("[FSR4] DUMP frame %d | %.1fms | %s%s%s\n",
               pipeline.frame_index + 1, t2 - t1,
               dbg.no_temporal ? "no_temporal " : "",
               dbg.zero_mvs ? "zero_mvs " : "",
               dbg.negate_mvs ? "negate_mvs " : "");
    }
#endif

    // =====================================================================
    // STEP 6: Transition game resources back to original states
    // =====================================================================
    n = 0;
    if (color_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        barriers[n++] = make_transition(color_res,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        color_before);
    if (depth_res && depth_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        barriers[n++] = make_transition(depth_res,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        depth_before);
    if (mv_res && mv_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        barriers[n++] = make_transition(mv_res,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        mv_before);
    if (output_before != D3D12_RESOURCE_STATE_UNORDERED_ACCESS)
        barriers[n++] = make_transition(output_res,
                                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                        output_before);
    if (exposure_res && exposure_before != D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)
        barriers[n++] = make_transition(exposure_res,
                                        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                        exposure_before);

    if (n > 0) cmd->ResourceBarrier(n, barriers);

    pipeline.frame_index++;

#ifndef FSR4_RELEASE
    double t3 = now_ms();

    // Logging: first 5 frames (detailed) + every 120th (periodic)
    bool detail_log = (pipeline.frame_index <= 5);
    bool periodic_log = (pipeline.frame_index % 120 == 1);

    if (detail_log || periodic_log) {
        // Read final CB values (after debug overrides)
        auto* cb = reinterpret_cast<OptimizedConstants*>(
            static_cast<uint8_t*>(pipeline.cb_mapped) + FSR4D3D12Pipeline::CB_MAIN_OFFSET);

        logf("Dispatch #%d: %.1fms total (setup=%.1f dispatch=%.1f barriers=%.1f)\n",
             pipeline.frame_index, t3 - t0, t1 - t0, t2 - t1, t3 - t2);
        logf("  render=%ux%u upscale=%ux%u reset=%d\n",
             render_w, render_h, upscale_w, upscale_h, cb->reset);
        logf("  CB.jitter=(%.4f, %.4f)  CB.mv_scale=(%.6f, %.6f)\n",
             cb->jitter[0], cb->jitter[1], cb->mv_scale[0], cb->mv_scale[1]);
        logf("  CB.jitter_cancel=(%.6f, %.6f)\n",
             cb->fMotionVectorJitterCancellation[0], cb->fMotionVectorJitterCancellation[1]);
        logf("  CB.scale=(%.3f, %.3f)  CB.inv_scale=(%.3f, %.3f)\n",
             cb->scale[0], cb->scale[1], cb->inv_scale[0], cb->inv_scale[1]);
        logf("  CB.inv_size=(%.6f, %.6f)  CB.tex_size=(%.0f, %.0f)\n",
             cb->inv_size[0], cb->inv_size[1], cb->tex_size[0], cb->tex_size[1]);
        logf("  CB.max_renderSize=(%.0f, %.0f)  CB.preExposure=%.4f  CB.prev_preExposure=%.4f\n",
             cb->max_renderSize[0], cb->max_renderSize[1],
             cb->preExposure, cb->previous_preExposure);
        logf("  game: mvScale=(%.1f, %.1f)  jitter=(%.4f, %.4f)\n",
             desc->motionVectorScale.x, desc->motionVectorScale.y,
             desc->jitterOffset.x, desc->jitterOffset.y);
        logf("  color=%s depth=%s mv=%s output=%s exposure=%s sharpening=%s\n",
             dxgi_format_name(color_fmt),
             depth_res ? dxgi_format_name(depth_fmt) : "none",
             mv_res ? dxgi_format_name(mv_fmt) : "none",
             dxgi_format_name(output_fmt),
             exposure_res ? dxgi_format_name(exposure_fmt) : "default(preExposure-matched)",
             desc->enableSharpening ? "yes" : "no");
        if (desc->enableSharpening) {
            auto* rcas_cb = reinterpret_cast<RcasConstants*>(
                static_cast<uint8_t*>(pipeline.cb_mapped) + FSR4D3D12Pipeline::CB_RCAS_OFFSET);
            float sharp_linear;
            memcpy(&sharp_linear, &rcas_cb->config[0], sizeof(float));
            logf("  RCAS: desc->sharpness=%.6f  config[0]=0x%08x (%.6f)  preExposure=%.4f\n",
                 desc->sharpness, rcas_cb->config[0], sharp_linear, rcas_cb->preExposure);
        }

        // MV resource details (for diagnosing zero-MV issue)
        if (mv_res) {
            auto mv_desc = mv_res->GetDesc();
            logf("  mv_res=%p  mv_dims=%llux%llu  mv_state=0x%x  mv_flags=0x%x\n",
                 (void*)mv_res, mv_desc.Width, (unsigned long long)mv_desc.Height,
                 desc->motionVectors.state, mv_desc.Flags);
        }

        if (dbg.no_temporal || dbg.negate_mvs || dbg.zero_mvs ||
            dbg.mv_scale != 1.0f || dbg.zero_jitter || dbg.negate_jitter) {
            logf("  DEBUG OVERRIDES:");
            if (dbg.no_temporal)      logf(" no_temporal");
            if (dbg.negate_mvs)       logf(" negate_mvs");
            if (dbg.zero_mvs)         logf(" zero_mvs");
            if (dbg.mv_scale != 1.0f) logf(" mv_scale=%.2f", dbg.mv_scale);
            if (dbg.zero_jitter)      logf(" zero_jitter");
            if (dbg.negate_jitter)    logf(" negate_jitter");
            logf("\n");
        }
    }
#endif  // !FSR4_RELEASE

    return FFX_API_RETURN_OK;
}

} // namespace fsr4
