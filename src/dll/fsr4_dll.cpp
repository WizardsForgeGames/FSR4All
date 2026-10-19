// FSR 4 All - FidelityFX API DLL Entry Point
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Drop-in replacement for amd_fidelityfx_upscaler_dx12.dll.
// Exports the 5 FidelityFX C API functions, routes upscale dispatch
// through our CUDA pipeline.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <fsr4/ffx_api_compat.h>
#include "fsr4_context.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

// Forward declarations
#ifdef FSR4_RELEASE
static inline void log_msg(const char*, ...) {}
#else
static void log_msg(const char* fmt, ...);
#endif

// =========================================================================
// Original DLL proxy (for A/B comparison with FSR 3.x)
// =========================================================================

// Function pointer types matching our 5 exports
typedef ffxReturnCode_t (*PFN_ffxCreateContext)(ffxContext*, ffxCreateContextDescHeader*, const ffxAllocationCallbacks*);
typedef ffxReturnCode_t (*PFN_ffxDestroyContext)(ffxContext*, const ffxAllocationCallbacks*);
typedef ffxReturnCode_t (*PFN_ffxConfigure)(ffxContext*, const ffxConfigureDescHeader*);
typedef ffxReturnCode_t (*PFN_ffxQuery)(ffxContext*, ffxQueryDescHeader*);
typedef ffxReturnCode_t (*PFN_ffxDispatch)(ffxContext*, const ffxDispatchDescHeader*);

static HMODULE g_original_dll = nullptr;
static PFN_ffxCreateContext  g_orig_CreateContext  = nullptr;
static PFN_ffxDestroyContext g_orig_DestroyContext = nullptr;
static PFN_ffxConfigure      g_orig_Configure      = nullptr;
static PFN_ffxQuery           g_orig_Query          = nullptr;
static PFN_ffxDispatch        g_orig_Dispatch       = nullptr;

// Side table: maps FSR4DLLContext* → original DLL's context for A/B comparison.
// The game always receives FSR4DLLContext* directly as ffxContext (no wrapper).
#include <unordered_map>
static std::unordered_map<fsr4::FSR4DLLContext*, ffxContext> g_orig_contexts;

// Contexts we created (for identification in destroy/dispatch)
#include <unordered_set>
static std::unordered_set<void*> g_our_contexts;

// =========================================================================
// Globals
// =========================================================================

static FILE* g_log = nullptr;
static std::string g_weight_base_dir;

static void init_dll_paths() {
    // Find DLL location for weight directory resolution
    char dll_path[MAX_PATH] = {};
    HMODULE hmod = nullptr;
    GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                       GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       (LPCSTR)init_dll_paths, &hmod);
    GetModuleFileNameA(hmod, dll_path, MAX_PATH);

    std::string dir(dll_path);
    size_t last_sep = dir.find_last_of("\\/");
    if (last_sep != std::string::npos)
        dir = dir.substr(0, last_sep);

    // Weight directory: weights/ next to the DLL. The D3D12 path embeds its
    // model data at build time; this value is only logged and passed through.
    g_weight_base_dir = dir + "\\weights";

    // Open log file next to DLL
#ifdef FSR4_RELEASE
    std::string log_path = dir + "\\fsr4_release.log";
#else
    std::string log_path = dir + "\\fsr4_debug.log";
#endif
    g_log = fopen(log_path.c_str(), "w");
    if (g_log) {
        // Get timestamp
        SYSTEMTIME st;
        GetLocalTime(&st);
#ifdef FSR4_RELEASE
        fprintf(g_log, "=== FSR4 All DLL (RELEASE) ===\n");
#else
        fprintf(g_log, "=== FSR4 All DLL (DEBUG) ===\n");
#endif
        fprintf(g_log, "Loaded: %04d-%02d-%02d %02d:%02d:%02d\n",
                st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);
        fprintf(g_log, "DLL path: %s\n", dll_path);
        fprintf(g_log, "Weight base: %s\n", g_weight_base_dir.c_str());
        fflush(g_log);
    }
}

#ifndef FSR4_RELEASE
static void open_log() {
    static bool initialized = false;
    if (!initialized) { init_dll_paths(); initialized = true; }
}
#endif

// Load original DLL for A/B comparison (called lazily, NOT from DllMain)
static bool g_original_load_attempted = false;
static std::string g_dll_dir;

static void load_original_dll() {
    if (g_original_load_attempted) return;
    g_original_load_attempted = true;

    if (g_dll_dir.empty()) {
        char dll_path[MAX_PATH] = {};
        HMODULE hmod = nullptr;
        GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           (LPCSTR)load_original_dll, &hmod);
        GetModuleFileNameA(hmod, dll_path, MAX_PATH);
        g_dll_dir = std::string(dll_path);
        size_t sep = g_dll_dir.find_last_of("\\/");
        if (sep != std::string::npos) g_dll_dir = g_dll_dir.substr(0, sep);
    }

    // Search for original DLL under several naming conventions:
    // 1. amd_fidelityfx_upscaler_dx12_original.dll  (upscaler-only backup)
    // 2. amd_fidelityfx_upscaler_dx12.dll.bak        (upscaler .bak)
    // 3. amd_fidelityfx_dx12_original.dll             (monolithic backup)
    // 4. amd_fidelityfx_dx12.dll.bak                  (monolithic .bak)
    // 5. amd_fidelityfx_loader_dx12_original.dll      (loader backup)
    const char* candidates[] = {
        "\\amd_fidelityfx_upscaler_dx12_original.dll",
        "\\amd_fidelityfx_upscaler_dx12.dll.bak",
        "\\amd_fidelityfx_dx12_original.dll",
        "\\amd_fidelityfx_dx12.dll.bak",
        "\\amd_fidelityfx_loader_dx12_original.dll",
    };
    std::string orig_path;
    for (const char* suffix : candidates) {
        orig_path = g_dll_dir + suffix;
        g_original_dll = LoadLibraryA(orig_path.c_str());
        if (g_original_dll) break;
    }
    if (g_original_dll) {
        g_orig_CreateContext  = (PFN_ffxCreateContext) GetProcAddress(g_original_dll, "ffxCreateContext");
        g_orig_DestroyContext = (PFN_ffxDestroyContext)GetProcAddress(g_original_dll, "ffxDestroyContext");
        g_orig_Configure      = (PFN_ffxConfigure)     GetProcAddress(g_original_dll, "ffxConfigure");
        g_orig_Query           = (PFN_ffxQuery)          GetProcAddress(g_original_dll, "ffxQuery");
        g_orig_Dispatch        = (PFN_ffxDispatch)       GetProcAddress(g_original_dll, "ffxDispatch");
        log_msg("Original DLL loaded: %s\n", orig_path.c_str());
        log_msg("  CreateContext=%p Dispatch=%p Query=%p\n",
                g_orig_CreateContext, g_orig_Dispatch, g_orig_Query);
    } else {
        log_msg("No original DLL found (A/B comparison disabled)\n");
    }
}

#ifndef FSR4_RELEASE
static void log_msg(const char* fmt, ...) {
    if (!g_log) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(g_log, fmt, args);
    va_end(args);
    fflush(g_log);
}
#endif

// =========================================================================
// Descriptor chain walker
// =========================================================================

template<typename T>
static const T* find_desc(const ffxApiHeader* chain, uint64_t type) {
    for (const ffxApiHeader* h = chain; h; h = h->pNext) {
        if (h->type == type)
            return reinterpret_cast<const T*>(h);
    }
    return nullptr;
}

template<typename T>
static T* find_desc_mut(ffxApiHeader* chain, uint64_t type) {
    for (ffxApiHeader* h = chain; h; h = h->pNext) {
        if (h->type == type)
            return reinterpret_cast<T*>(h);
    }
    return nullptr;
}

// =========================================================================
// Jitter helper (Halton 2,3 sequence)
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
// DLL exports
// =========================================================================

extern "C" {

#ifndef FSR4_RELEASE
static bool g_console_allocated = false;
#endif

BOOL WINAPI DllMain(HINSTANCE hinstDLL, DWORD fdwReason, LPVOID lpReserved) {
    switch (fdwReason) {
        case DLL_PROCESS_ATTACH:
#ifdef FSR4_RELEASE
            init_dll_paths();
#else
            open_log();
            log_msg("DllMain: PROCESS_ATTACH\n");
            // Allocate debug console window
            if (AllocConsole()) {
                freopen("CONOUT$", "w", stdout);
                freopen("CONOUT$", "w", stderr);
                SetConsoleTitleA("FSR4 Debug Console");
                g_console_allocated = true;
                printf("[FSR4] Debug console attached\n");
            }
#endif
            break;
        case DLL_PROCESS_DETACH:
            log_msg("DllMain: PROCESS_DETACH\n");
            if (g_original_dll) { FreeLibrary(g_original_dll); g_original_dll = nullptr; }
            if (g_log) { fclose(g_log); g_log = nullptr; }
#ifndef FSR4_RELEASE
            if (g_console_allocated) { FreeConsole(); g_console_allocated = false; }
#endif
            break;
    }
    return TRUE;
}

__declspec(dllexport) ffxReturnCode_t ffxCreateContext(
    ffxContext* context,
    ffxCreateContextDescHeader* desc,
    const ffxAllocationCallbacks* memCb)
{
#ifndef FSR4_RELEASE
    open_log();
#endif
    log_msg("\n=== ffxCreateContext ===\n");

    if (!context || !desc) {
        log_msg("ERROR: null context or desc\n");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    // Find DX12 backend descriptor (provides ID3D12Device*)
    auto* dx12_desc = find_desc<ffxCreateBackendDX12Desc>(
        desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12);
    if (!dx12_desc || !dx12_desc->device) {
        log_msg("ERROR: no DX12 backend descriptor or null device\n");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    // Find upscale create descriptor
    auto* upscale_desc = find_desc<ffxCreateContextDescUpscale>(
        desc, FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE);
    if (!upscale_desc) {
        // Not an upscaler context (frame gen, denoiser, etc.) — forward to original DLL
        load_original_dll();
        if (g_orig_CreateContext) {
            log_msg("Non-upscale context (type=0x%llx), forwarding to original DLL\n", desc->type);
            return g_orig_CreateContext(context, desc, memCb);
        }
        log_msg("WARNING: non-upscale context but no original DLL to forward to\n");
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    log_msg("DX12 Device: %p\n", dx12_desc->device);
    log_msg("Max render: %ux%u, max upscale: %ux%u, flags: 0x%x\n",
            upscale_desc->maxRenderSize.width, upscale_desc->maxRenderSize.height,
            upscale_desc->maxUpscaleSize.width, upscale_desc->maxUpscaleSize.height,
            upscale_desc->flags);

    // Create our FSR4 context (returned directly as the game's ffxContext)
    auto* ctx = new fsr4::FSR4DLLContext();
#ifndef FSR4_RELEASE
    ctx->log = g_log;
#endif

    if (!ctx->create(dx12_desc->device, upscale_desc, g_weight_base_dir)) {
        if (g_log) { fprintf(g_log, "ERROR: FSR4 context creation failed\n"); fflush(g_log); }
        delete ctx;
        return FFX_API_RETURN_ERROR_RUNTIME_ERROR;
    }

    // Write basic context info to log (both release and debug)
    if (g_log) {
        float ratio = (upscale_desc->maxRenderSize.width > 0) ?
            (float)upscale_desc->maxUpscaleSize.width / (float)upscale_desc->maxRenderSize.width : 0;
        const char* preset = "Unknown";
        if (ratio >= 2.99f) preset = "UltraPerformance";
        else if (ratio >= 1.99f) preset = "Performance";
        else if (ratio >= 1.69f) preset = "Balanced";
        else preset = "Quality";
        fprintf(g_log, "Context created: render=%ux%u upscale=%ux%u ratio=%.2f preset=%s\n",
                upscale_desc->maxRenderSize.width, upscale_desc->maxRenderSize.height,
                upscale_desc->maxUpscaleSize.width, upscale_desc->maxUpscaleSize.height,
                ratio, preset);
        fflush(g_log);
    }
    log_msg("FSR4 context ptr: %p\n", ctx);
    g_our_contexts.insert(static_cast<void*>(ctx));

#ifndef FSR4_RELEASE
    // Also create original DLL's context (for A/B comparison)
    // Wrapped in SEH to survive crashes in the original DLL
    load_original_dll();
    if (g_orig_CreateContext) {
        ffxContext orig_ctx = nullptr;
        __try {
            ffxReturnCode_t orig_ret = g_orig_CreateContext(&orig_ctx, desc, memCb);
            if (orig_ret == FFX_API_RETURN_OK && orig_ctx) {
                g_orig_contexts[ctx] = orig_ctx;
                log_msg("Original context created: %p (A/B comparison enabled)\n", orig_ctx);
            } else {
                log_msg("Original context creation failed (ret=%u), A/B disabled\n", orig_ret);
            }
        } __except(EXCEPTION_EXECUTE_HANDLER) {
            log_msg("Original context creation CRASHED (exception 0x%lx), A/B disabled\n",
                    GetExceptionCode());
        }
    }
#endif

    *context = static_cast<ffxContext>(ctx);
    return FFX_API_RETURN_OK;
}

__declspec(dllexport) ffxReturnCode_t ffxDestroyContext(
    ffxContext* context,
    const ffxAllocationCallbacks* memCb)
{
    log_msg("\n=== ffxDestroyContext ===\n");

    if (!context || !*context) {
        log_msg("null context, nothing to destroy\n");
        return FFX_API_RETURN_OK;
    }

    // Check if this is one of our contexts or a forwarded non-upscale context
    if (g_our_contexts.find(*context) == g_our_contexts.end()) {
        if (g_orig_DestroyContext) {
            log_msg("Forwarding destroy to original DLL\n");
            return g_orig_DestroyContext(context, memCb);
        }
        *context = nullptr;
        return FFX_API_RETURN_OK;
    }

    auto* ctx = static_cast<fsr4::FSR4DLLContext*>(*context);

#ifndef FSR4_RELEASE
    // Destroy original DLL's context (A/B comparison)
    auto it = g_orig_contexts.find(ctx);
    if (it != g_orig_contexts.end()) {
        if (g_orig_DestroyContext) {
            g_orig_DestroyContext(&it->second, memCb);
            log_msg("Original context destroyed\n");
        }
        g_orig_contexts.erase(it);
    }

    // Destroy our FSR4 context
    ctx->log = nullptr;
#endif
    ctx->destroy();
    g_our_contexts.erase(*context);
    delete ctx;
    *context = nullptr;

    log_msg("Context destroyed\n");
    return FFX_API_RETURN_OK;
}

__declspec(dllexport) ffxReturnCode_t ffxConfigure(
    ffxContext* context,
    const ffxConfigureDescHeader* desc)
{
    log_msg("ffxConfigure: type=0x%llx context=%p\n", desc ? desc->type : 0,
            context ? *context : nullptr);

    // Forward non-upscale contexts to original DLL
    if (context && *context && g_our_contexts.find(*context) == g_our_contexts.end()) {
        load_original_dll();
        if (g_orig_Configure) {
            log_msg("  Forwarding configure to original DLL (non-upscale context)\n");
            return g_orig_Configure(context, desc);
        }
    }

    // Forward context-less configure calls for non-upscale effects to original DLL
    if (desc && (!context || !*context)) {
        constexpr uint64_t EFFECT_MASK = 0xffff0000ULL;
        constexpr uint64_t EFFECT_UPSCALE = 0x00010000ULL;
        uint64_t effect = desc->type & EFFECT_MASK;
        if (effect != EFFECT_UPSCALE && effect != 0) {
            load_original_dll();
            if (g_orig_Configure) {
                log_msg("  Forwarding context-less configure 0x%llx to original DLL\n", desc->type);
                return g_orig_Configure(context, desc);
            }
        }
    }

    return FFX_API_RETURN_OK;
}

__declspec(dllexport) ffxReturnCode_t ffxQuery(
    ffxContext* context,
    ffxQueryDescHeader* desc)
{
    if (!desc) return FFX_API_RETURN_ERROR_PARAMETER;

    log_msg("ffxQuery: type=0x%llx\n", desc->type);

    // Check if query is for a non-upscale effect — forward to original DLL
    constexpr uint64_t FFX_API_EFFECT_MASK = 0xffff0000ULL;
    constexpr uint64_t FFX_API_EFFECT_UPSCALE = 0x00010000ULL;

    switch (desc->type) {
        // ---- Core API: Version enumeration ----
        case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
            auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
            log_msg("  GetVersions: createDescType=0x%llx device=%p\n",
                    q->createDescType, q->device);

            // If querying versions for a non-upscale effect, forward to original DLL
            if ((q->createDescType & FFX_API_EFFECT_MASK) != FFX_API_EFFECT_UPSCALE) {
                load_original_dll();
                if (g_orig_Query) {
                    log_msg("  Non-upscale version query, forwarding to original DLL\n");
                    return g_orig_Query(context, desc);
                }
                // No original DLL — report 0 versions for this effect
                if (q->outputCount) *q->outputCount = 0;
                log_msg("  Non-upscale version query, no original DLL, reporting 0 versions\n");
                return FFX_API_RETURN_OK;
            }

            if (!q->outputCount) {
                log_msg("  ERROR: null outputCount\n");
                return FFX_API_RETURN_ERROR_PARAMETER;
            }

            // We provide 1 version: FSR4 All
            uint64_t capacity = *q->outputCount;
            *q->outputCount = 1;

            if (capacity > 0) {
                // AMD version format: 0xF5A5'CA1E << 32 | FFX_MAKE_VERSION(major,minor,patch)
                // FFX_MAKE_VERSION(m,n,p) = (m<<22)|(n<<12)|p
                // Report as FSR 4.0.2 — games check major version to decide whether
                // to show "FSR 4" vs "FSR 3" in their UI.
                static const char* s_version_name = "4.0.2";
                static const uint64_t s_version_id = 0xF5A5CA1EULL << 32 | ((4 << 22) | (0 << 12) | 2);

                if (q->versionIds)
                    q->versionIds[0] = s_version_id;
                if (q->versionNames)
                    q->versionNames[0] = s_version_name;

                log_msg("  Reported 1 version: id=0x%llx name='%s'\n",
                        s_version_id, s_version_name);
            } else {
                log_msg("  Capacity query: reported count=1\n");
            }
            return FFX_API_RETURN_OK;
        }

        case FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION: {
            auto* q = reinterpret_cast<ffxQueryGetProviderVersion*>(desc);
            static const char* s_provider_name = "4.0.2";
            q->versionId = 0xF5A5CA1EULL << 32 | ((4 << 22) | (0 << 12) | 2);
            q->versionName = s_provider_name;
            log_msg("  GetProviderVersion: id=0x%llx name='%s'\n",
                    q->versionId, q->versionName);
            return FFX_API_RETURN_OK;
        }

        // ---- Upscale queries ----
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(desc);
            if (q->pOutUpscaleRatio) {
                *q->pOutUpscaleRatio = fsr4::get_upscale_ratio(q->qualityMode);
                log_msg("  GetUpscaleRatio: mode=%u ratio=%.1f\n",
                        q->qualityMode, *q->pOutUpscaleRatio);
            }
            return FFX_API_RETURN_OK;
        }

        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(desc);
            float ratio = fsr4::get_upscale_ratio(q->qualityMode);
            if (q->pOutRenderWidth)
                *q->pOutRenderWidth = (uint32_t)((float)q->displayWidth / ratio);
            if (q->pOutRenderHeight)
                *q->pOutRenderHeight = (uint32_t)((float)q->displayHeight / ratio);
            log_msg("  GetRenderRes: display=%ux%u mode=%u → render=%ux%u\n",
                    q->displayWidth, q->displayHeight, q->qualityMode,
                    q->pOutRenderWidth ? *q->pOutRenderWidth : 0,
                    q->pOutRenderHeight ? *q->pOutRenderHeight : 0);
            return FFX_API_RETURN_OK;
        }

        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(desc);
            if (q->pOutPhaseCount) {
                // Phase count = 8 * (upscale_ratio^2), standard FSR formula
                float ratio = (float)q->displayWidth / (float)q->renderWidth;
                *q->pOutPhaseCount = (int32_t)(8.0f * ratio * ratio);
                if (*q->pOutPhaseCount < 1) *q->pOutPhaseCount = 1;
                log_msg("  GetJitterPhaseCount: %d (ratio=%.2f)\n",
                        *q->pOutPhaseCount, ratio);
            }
            return FFX_API_RETURN_OK;
        }

        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(desc);
            if (q->pOutX && q->pOutY) {
                // Halton(2,3) sequence, centered to [-0.5, 0.5]
                int idx = q->index % q->phaseCount + 1;  // 1-based for Halton
                *q->pOutX = halton_seq(idx, 2) - 0.5f;
                *q->pOutY = halton_seq(idx, 3) - 0.5f;
            }
            return FFX_API_RETURN_OK;
        }

        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetGPUMemoryUsage*>(desc);
            if (q->gpuMemoryUsageUpscaler) {
                q->gpuMemoryUsageUpscaler->totalUsageInBytes = 64 * 1024 * 1024;  // ~64 MB estimate
                q->gpuMemoryUsageUpscaler->aliasableUsageInBytes = 0;
            }
            return FFX_API_RETURN_OK;
        }

        default: {
            // Forward non-upscale queries to original DLL
            uint64_t effect = desc->type & FFX_API_EFFECT_MASK;
            if (effect != FFX_API_EFFECT_UPSCALE && effect != 0) {
                load_original_dll();
                if (g_orig_Query) {
                    log_msg("  Non-upscale query type 0x%llx, forwarding to original DLL\n", desc->type);
                    return g_orig_Query(context, desc);
                }
            }
            log_msg("  Unhandled query type 0x%llx\n", desc->type);
            return FFX_API_RETURN_OK;  // Don't error on unknown queries
        }
    }
}

__declspec(dllexport) ffxReturnCode_t ffxDispatch(
    ffxContext* context,
    const ffxDispatchDescHeader* desc)
{
    if (!context || !*context || !desc) {
        log_msg("ffxDispatch: null context or desc\n");
        return FFX_API_RETURN_ERROR_PARAMETER;
    }

    log_msg("ffxDispatch: type=0x%llx\n", desc->type);

    // Forward non-upscale contexts (e.g. frame gen) to original DLL
    if (g_our_contexts.find(*context) == g_our_contexts.end()) {
        if (g_orig_Dispatch) {
            log_msg("  -> Forwarding to original DLL (non-upscale context)\n");
            return g_orig_Dispatch(context, desc);
        }
        return FFX_API_RETURN_OK;
    }

    auto* ctx = static_cast<fsr4::FSR4DLLContext*>(*context);

#ifndef FSR4_RELEASE
    auto& cfg = fsr4::get_debug_config();

    // Check if we should route to original DLL (A/B comparison)
    auto orig_it = g_orig_contexts.find(ctx);
    if (cfg.use_original && orig_it != g_orig_contexts.end() && g_orig_Dispatch) {
        log_msg("  -> Routing to ORIGINAL DLL (use_original=true)\n");
        __try {
            return g_orig_Dispatch(&orig_it->second, desc);
        } __except(EXCEPTION_EXECUTE_HANDLER) {
            log_msg("  ORIGINAL DLL dispatch CRASHED, falling back to FSR4\n");
        }
    }
#endif

    // Only handle upscale dispatch
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE) {
        auto* upscale = reinterpret_cast<const ffxDispatchDescUpscale*>(desc);
        return ctx->dispatch(upscale);
    }

    // Reactive mask generation — no-op for now
    if (desc->type == FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK) {
        log_msg("  GenerateReactiveMask: no-op\n");
        return FFX_API_RETURN_OK;
    }

    log_msg("  Unknown dispatch type 0x%llx\n", desc->type);
    return FFX_API_RETURN_OK;
}

} // extern "C"
