// FSR 4 All - FSR2 Legacy API Shim
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Drop-in replacement for ffx_fsr2_api_x64.dll.
// Accepts old-style FSR2 API calls (ffxFsr2ContextCreate, ffxFsr2ContextDispatch, etc.)
// and routes them through our FSR4 D3D12 compute pipeline.
//
// This enables FSR4 upscaling in ANY game that supports FSR2 — which is the
// vast majority of modern games with temporal upscaling support.
//
// Usage: rename this DLL to ffx_fsr2_api_x64.dll and place in game directory.
// The game's DX12 backend DLL (ffx_fsr2_api_dx12_x64.dll) stays in place.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include "fsr2_compat.h"
#include "fsr4_context.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

// =========================================================================
// Globals
// =========================================================================

static FILE* g_log = nullptr;
static std::string g_dll_dir;

static void open_log() {
    if (g_log) return;

    char dll_path[MAX_PATH] = {};
    HMODULE hmod = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)open_log, &hmod);
    GetModuleFileNameA(hmod, dll_path, MAX_PATH);

    std::string dir(dll_path);
    size_t last_sep = dir.find_last_of("\\/");
    if (last_sep != std::string::npos)
        dir = dir.substr(0, last_sep);
    g_dll_dir = dir;

    std::string log_path = dir + "\\fsr4_debug.log";
    g_log = fopen(log_path.c_str(), "w");
    if (g_log) {
        fprintf(g_log, "=== FSR4 All — FSR2 Shim Loaded ===\n");
        fprintf(g_log, "DLL path: %s\n", dll_path);
        fprintf(g_log, "Mode: FSR2 API compatibility (old-style ffxFsr2* functions)\n");
        fflush(g_log);
    }
}

static void log_msg(const char* fmt, ...) {
    if (!g_log) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(g_log, fmt, args);
    va_end(args);
    fflush(g_log);
}

// =========================================================================
// Context pointer storage in opaque FfxFsr2Context
// =========================================================================

// Store our FSR4DLLContext* in the first 8 bytes of the opaque data blob.
// The game allocated this blob — we just use it as storage.
static fsr4::FSR4DLLContext** get_ctx_slot(FfxFsr2Context* ctx) {
    return reinterpret_cast<fsr4::FSR4DLLContext**>(&ctx->data[0]);
}

static fsr4::FSR4DLLContext* get_ctx(FfxFsr2Context* ctx) {
    return *get_ctx_slot(ctx);
}

// =========================================================================
// Flag translation (FSR2 → unified)
// Bits 0-6 are identical. Bit 7 differs (FSR2=TEXTURE1D, unified=DEBUG).
// We map FSR2 flags to our internal format, keeping what matters.
// =========================================================================

static uint32_t fsr2_flags_to_unified(uint32_t fsr2_flags) {
    // Keep bits 0-6 (HDR, display-res MV, jitter cancel, depth inverted,
    // depth infinite, auto exposure, dynamic resolution)
    uint32_t unified = fsr2_flags & 0x7F;

    // FSR2 bit 8 = DEBUG_CHECKING → unified bit 7 = DEBUG_CHECKING
    if (fsr2_flags & FFX_FSR2_ENABLE_DEBUG_CHECKING)
        unified |= (1 << 7);  // FFX_UPSCALE_ENABLE_DEBUG_CHECKING

    return unified;
}

// =========================================================================
// Halton sequence (for jitter — identical to FSR2's implementation)
// =========================================================================

static float halton_seq(int index, int base) {
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

// =========================================================================
// DLL exports — FSR2 old-style API
// =========================================================================

extern "C" {

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
            open_log();
            log_msg("DllMain: PROCESS_ATTACH\n");
            break;
        case DLL_PROCESS_DETACH:
            log_msg("DllMain: PROCESS_DETACH\n");
            if (g_log) { fclose(g_log); g_log = nullptr; }
            break;
    }
    return TRUE;
}

// ---- Context lifecycle ----

__declspec(dllexport) FfxErrorCode ffxFsr2ContextCreate(
    FfxFsr2Context* pContext,
    const FfxFsr2ContextDescription* pContextDescription)
{
    open_log();
    log_msg("\n=== ffxFsr2ContextCreate ===\n");

    if (!pContext || !pContextDescription) {
        log_msg("ERROR: null context or description\n");
        return FFX_ERROR_INVALID_POINTER;
    }

    // Extract D3D12 device from the backend interface
    // The game called ffxGetInterfaceDX12() which stored the device pointer
    // in backendInterface.device (last field of FfxInterface)
    ID3D12Device* device = static_cast<ID3D12Device*>(
        pContextDescription->backendInterface.device);

    if (!device) {
        log_msg("ERROR: null device in backendInterface\n");
        return FFX_ERROR_NULL_DEVICE;
    }

    uint32_t flags = pContextDescription->flags;
    uint32_t max_render_w = pContextDescription->maxRenderSize.width;
    uint32_t max_render_h = pContextDescription->maxRenderSize.height;
    uint32_t display_w = pContextDescription->displaySize.width;
    uint32_t display_h = pContextDescription->displaySize.height;

    log_msg("Device: %p\n", device);
    log_msg("Max render: %ux%u, display: %ux%u, flags: 0x%x\n",
            max_render_w, max_render_h, display_w, display_h, flags);
    log_msg("FSR2 flags: HDR=%d DispResMV=%d JitterCancel=%d DepthInv=%d "
            "DepthInf=%d AutoExposure=%d DynRes=%d Tex1D=%d Debug=%d\n",
            !!(flags & FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE),
            !!(flags & FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS),
            !!(flags & FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION),
            !!(flags & FFX_FSR2_ENABLE_DEPTH_INVERTED),
            !!(flags & FFX_FSR2_ENABLE_DEPTH_INFINITE),
            !!(flags & FFX_FSR2_ENABLE_AUTO_EXPOSURE),
            !!(flags & FFX_FSR2_ENABLE_DYNAMIC_RESOLUTION),
            !!(flags & FFX_FSR2_ENABLE_TEXTURE1D_USAGE),
            !!(flags & FFX_FSR2_ENABLE_DEBUG_CHECKING));

    // Create our internal FSR4 context
    auto* ctx = new fsr4::FSR4DLLContext();
    ctx->log = g_log;

    // Build a fake ffxCreateContextDescUpscale for our pipeline
    // (we reuse FSR4DLLContext::create which expects this struct)
    ffxCreateContextDescUpscale upscale_desc = {};
    upscale_desc.flags = fsr2_flags_to_unified(flags);
    upscale_desc.maxRenderSize.width = max_render_w;
    upscale_desc.maxRenderSize.height = max_render_h;
    upscale_desc.maxUpscaleSize.width = display_w;
    upscale_desc.maxUpscaleSize.height = display_h;

    // Weight base dir (not used by D3D12 pipeline, weights are embedded in shaders)
    std::string weight_base = g_dll_dir + "\\weights";

    if (!ctx->create(device, &upscale_desc, weight_base)) {
        log_msg("ERROR: FSR4 context creation failed\n");
        delete ctx;
        return FFX_ERROR_BACKEND_API_ERROR;
    }

    // Store context pointer in opaque blob
    memset(pContext->data, 0, sizeof(pContext->data));
    *get_ctx_slot(pContext) = ctx;

    log_msg("FSR2 shim context created: %p (FSR4 pipeline active)\n", ctx);
    return FFX_OK;
}

__declspec(dllexport) FfxErrorCode ffxFsr2ContextDestroy(FfxFsr2Context* pContext) {
    log_msg("\n=== ffxFsr2ContextDestroy ===\n");

    if (!pContext) {
        log_msg("null context\n");
        return FFX_ERROR_INVALID_POINTER;
    }

    auto* ctx = get_ctx(pContext);
    if (!ctx) {
        log_msg("no internal context, nothing to destroy\n");
        return FFX_OK;
    }

    FILE* saved_log = ctx->log;
    ctx->log = nullptr;
    ctx->destroy();
    delete ctx;
    *get_ctx_slot(pContext) = nullptr;

    if (saved_log) {
        fprintf(saved_log, "FSR2 shim context destroyed\n");
        fflush(saved_log);
    }
    return FFX_OK;
}

// ---- Per-frame dispatch ----

__declspec(dllexport) FfxErrorCode ffxFsr2ContextDispatch(
    FfxFsr2Context* pContext,
    const FfxFsr2DispatchDescription* pDispatchDescription)
{
    if (!pContext || !pDispatchDescription) {
        log_msg("ffxFsr2ContextDispatch: null params\n");
        return FFX_ERROR_INVALID_POINTER;
    }

    auto* ctx = get_ctx(pContext);
    if (!ctx) {
        log_msg("ffxFsr2ContextDispatch: no internal context\n");
        return FFX_ERROR_NULL_DEVICE;
    }

    // Translate FSR2 dispatch → unified ffxDispatchDescUpscale
    // The resource layouts are binary-compatible (both use void* + description + state)
    ffxDispatchDescUpscale desc = {};
    desc.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    desc.commandList = pDispatchDescription->commandList;

    // Copy resources (binary-compatible structs)
    static_assert(sizeof(FfxApiResource_FSR2) == sizeof(FfxApiResource),
                  "FSR2 and unified FfxApiResource must have identical layout");
    memcpy(&desc.color, &pDispatchDescription->color, sizeof(FfxApiResource));
    memcpy(&desc.depth, &pDispatchDescription->depth, sizeof(FfxApiResource));
    memcpy(&desc.motionVectors, &pDispatchDescription->motionVectors, sizeof(FfxApiResource));
    memcpy(&desc.exposure, &pDispatchDescription->exposure, sizeof(FfxApiResource));
    memcpy(&desc.reactive, &pDispatchDescription->reactive, sizeof(FfxApiResource));
    memcpy(&desc.transparencyAndComposition, &pDispatchDescription->transparencyAndComposition, sizeof(FfxApiResource));
    memcpy(&desc.output, &pDispatchDescription->output, sizeof(FfxApiResource));

    // Copy scalar fields
    desc.jitterOffset.x = pDispatchDescription->jitterOffset.x;
    desc.jitterOffset.y = pDispatchDescription->jitterOffset.y;
    desc.motionVectorScale.x = pDispatchDescription->motionVectorScale.x;
    desc.motionVectorScale.y = pDispatchDescription->motionVectorScale.y;
    desc.renderSize.width = pDispatchDescription->renderSize.width;
    desc.renderSize.height = pDispatchDescription->renderSize.height;

    // FSR2 has no per-dispatch upscale size — use context's display size
    desc.upscaleSize.width = ctx->max_upscale_w;
    desc.upscaleSize.height = ctx->max_upscale_h;

    desc.enableSharpening = pDispatchDescription->enableSharpening;
    desc.sharpness = pDispatchDescription->sharpness;
    desc.frameTimeDelta = pDispatchDescription->frameTimeDelta;
    desc.preExposure = pDispatchDescription->preExposure;
    desc.reset = pDispatchDescription->reset;
    desc.cameraNear = pDispatchDescription->cameraNear;
    desc.cameraFar = pDispatchDescription->cameraFar;
    desc.cameraFovAngleVertical = pDispatchDescription->cameraFovAngleVertical;
    desc.viewSpaceToMetersFactor = pDispatchDescription->viewSpaceToMetersFactor;

    // Dispatch through our FSR4 pipeline
    uint32_t ret = ctx->dispatch(&desc);

    // Convert return code
    return (ret == FFX_API_RETURN_OK) ? FFX_OK : FFX_ERROR_BACKEND_API_ERROR;
}

// ---- Reactive mask generation (no-op) ----

__declspec(dllexport) FfxErrorCode ffxFsr2ContextGenerateReactiveMask(
    FfxFsr2Context* pContext,
    const FfxFsr2GenerateReactiveDescription* pParams)
{
    log_msg("ffxFsr2ContextGenerateReactiveMask: no-op\n");
    return FFX_OK;
}

// ---- GPU memory usage query ----

__declspec(dllexport) FfxErrorCode ffxFsr2ContextGetGpuMemoryUsage(
    FfxFsr2Context* pContext,
    FfxApiEffectMemoryUsage_FSR2* pVramUsage)
{
    if (!pVramUsage) return FFX_ERROR_INVALID_POINTER;
    pVramUsage->totalUsageInBytes = 128 * 1024 * 1024;  // ~128 MB estimate (includes scratch)
    pVramUsage->aliasableUsageInBytes = 0;
    return FFX_OK;
}

__declspec(dllexport) FfxErrorCode ffxFsr2GetGpuMemoryUsage(
    FfxDevice device,
    FfxApiDimensions2D_FSR2* maxRenderSize,
    FfxApiDimensions2D_FSR2* maxUpscaleSize,
    FfxApiEffectMemoryUsage_FSR2* pVramUsage)
{
    if (!pVramUsage) return FFX_ERROR_INVALID_POINTER;
    pVramUsage->totalUsageInBytes = 128 * 1024 * 1024;
    pVramUsage->aliasableUsageInBytes = 0;
    return FFX_OK;
}

// ---- Quality mode helpers ----

__declspec(dllexport) float ffxFsr2GetUpscaleRatioFromQualityMode(
    uint32_t qualityMode)
{
    switch (qualityMode) {
        case FFX_FSR2_QUALITY_MODE_NATIVEAA:          return 1.0f;
        case FFX_FSR2_QUALITY_MODE_QUALITY:           return 1.5f;
        case FFX_FSR2_QUALITY_MODE_BALANCED:          return 1.7f;
        case FFX_FSR2_QUALITY_MODE_PERFORMANCE:       return 2.0f;
        case FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
        default: return 0.0f;
    }
}

__declspec(dllexport) FfxErrorCode ffxFsr2GetRenderResolutionFromQualityMode(
    uint32_t* pRenderWidth,
    uint32_t* pRenderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    uint32_t qualityMode)
{
    if (!pRenderWidth || !pRenderHeight) return FFX_ERROR_INVALID_POINTER;

    float ratio = ffxFsr2GetUpscaleRatioFromQualityMode(qualityMode);
    if (ratio <= 0.0f) return FFX_ERROR_INVALID_ENUM;

    *pRenderWidth = (uint32_t)((float)displayWidth / ratio);
    *pRenderHeight = (uint32_t)((float)displayHeight / ratio);
    return FFX_OK;
}

// ---- Jitter helpers ----

__declspec(dllexport) int32_t ffxFsr2GetJitterPhaseCount(
    int32_t renderWidth,
    int32_t displayWidth)
{
    float ratio = (float)displayWidth / (float)renderWidth;
    int32_t count = (int32_t)ceilf(8.0f * ratio * ratio);
    return count > 0 ? count : 1;
}

__declspec(dllexport) FfxErrorCode ffxFsr2GetJitterOffset(
    float* pOutX,
    float* pOutY,
    int32_t index,
    int32_t phaseCount)
{
    if (!pOutX || !pOutY) return FFX_ERROR_INVALID_POINTER;
    if (phaseCount <= 0) return FFX_ERROR_INVALID_ARGUMENT;

    // Halton(2,3) sequence, centered to [-0.5, 0.5]
    int idx = (index % phaseCount) + 1;  // 1-based for Halton
    *pOutX = halton_seq(idx, 2) - 0.5f;
    *pOutY = halton_seq(idx, 3) - 0.5f;
    return FFX_OK;
}

// ---- Misc helpers ----

__declspec(dllexport) bool ffxFsr2ResourceIsNull(FfxApiResource_FSR2 resource) {
    return resource.resource == nullptr;
}

__declspec(dllexport) FfxVersionNumber ffxFsr2GetEffectVersion() {
    // Return FSR2 version 2.3.4 (what games expect)
    return (2 << 22) | (3 << 12) | 4;
}

__declspec(dllexport) FfxErrorCode ffxFsr2SetGlobalDebugMessage(
    ffxMessageCallback fpMessage,
    uint32_t debugLevel)
{
    // No-op — we use our own logging
    return FFX_OK;
}

} // extern "C"
