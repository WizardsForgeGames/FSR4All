// FSR 4 All - Binary-compatible FidelityFX API declarations
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Minimal redeclarations of AMD FidelityFX API types for DLL replacement.
// Layout-compatible with FidelityFX SDK (Kits/FidelityFX/api/include/).
// Only the types/enums we actually consume are declared.
// Type/enum names and values mirror AMD's FidelityFX SDK headers
// (Copyright (C) 2025 Advanced Micro Devices, Inc., MIT License -- see LICENSE).
#pragma once

#include <stdint.h>
#include <d3d12.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Return codes
// =========================================================================
typedef uint32_t ffxReturnCode_t;

enum FfxApiReturnCodes
{
    FFX_API_RETURN_OK                     = 0,
    FFX_API_RETURN_ERROR                  = 1,
    FFX_API_RETURN_ERROR_UNKNOWN_DESCTYPE = 2,
    FFX_API_RETURN_ERROR_RUNTIME_ERROR    = 3,
    FFX_API_RETURN_NO_PROVIDER            = 4,
    FFX_API_RETURN_ERROR_MEMORY           = 5,
    FFX_API_RETURN_ERROR_PARAMETER        = 6,
};

// =========================================================================
// Core types
// =========================================================================
typedef void* ffxContext;
typedef uint64_t ffxStructType_t;

typedef struct ffxApiHeader
{
    ffxStructType_t      type;
    struct ffxApiHeader* pNext;
} ffxApiHeader;

typedef ffxApiHeader ffxCreateContextDescHeader;
typedef ffxApiHeader ffxConfigureDescHeader;
typedef ffxApiHeader ffxQueryDescHeader;
typedef ffxApiHeader ffxDispatchDescHeader;

typedef void (*ffxApiMessage)(uint32_t type, const wchar_t* message);

// =========================================================================
// Allocation callbacks
// =========================================================================
typedef void* (*ffxAlloc)(void* pUserData, uint64_t size);
typedef void  (*ffxDealloc)(void* pUserData, void* pMem);

typedef struct ffxAllocationCallbacks
{
    void*     pUserData;
    ffxAlloc  alloc;
    ffxDealloc dealloc;
} ffxAllocationCallbacks;

// =========================================================================
// Resource types
// =========================================================================
struct FfxApiDimensions2D
{
    uint32_t width;
    uint32_t height;
};

struct FfxApiFloatCoords2D
{
    float x;
    float y;
};

enum FfxApiSurfaceFormat
{
    FFX_API_SURFACE_FORMAT_UNKNOWN = 0,
    FFX_API_SURFACE_FORMAT_R32G32B32A32_TYPELESS,
    FFX_API_SURFACE_FORMAT_R32G32B32A32_UINT,
    FFX_API_SURFACE_FORMAT_R32G32B32A32_FLOAT,
    FFX_API_SURFACE_FORMAT_R16G16B16A16_FLOAT,
    FFX_API_SURFACE_FORMAT_R32G32B32_FLOAT,
    FFX_API_SURFACE_FORMAT_R32G32_FLOAT,
    FFX_API_SURFACE_FORMAT_R8_UINT,
    FFX_API_SURFACE_FORMAT_R32_UINT,
    FFX_API_SURFACE_FORMAT_R8G8B8A8_TYPELESS,
    FFX_API_SURFACE_FORMAT_R8G8B8A8_UNORM,
    FFX_API_SURFACE_FORMAT_R8G8B8A8_SNORM,
    FFX_API_SURFACE_FORMAT_R8G8B8A8_SRGB,
    FFX_API_SURFACE_FORMAT_B8G8R8A8_TYPELESS,
    FFX_API_SURFACE_FORMAT_B8G8R8A8_UNORM,
    FFX_API_SURFACE_FORMAT_B8G8R8A8_SRGB,
    FFX_API_SURFACE_FORMAT_R11G11B10_FLOAT,
    FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM,
    FFX_API_SURFACE_FORMAT_R16G16_FLOAT,
    FFX_API_SURFACE_FORMAT_R16G16_UINT,
    FFX_API_SURFACE_FORMAT_R16G16_SINT,
    FFX_API_SURFACE_FORMAT_R16_FLOAT,
    FFX_API_SURFACE_FORMAT_R16_UINT,
    FFX_API_SURFACE_FORMAT_R16_UNORM,
    FFX_API_SURFACE_FORMAT_R16_SNORM,
    FFX_API_SURFACE_FORMAT_R8_UNORM,
    FFX_API_SURFACE_FORMAT_R8G8_UNORM,
    FFX_API_SURFACE_FORMAT_R8G8_UINT,
    FFX_API_SURFACE_FORMAT_R32_FLOAT,
    FFX_API_SURFACE_FORMAT_R9G9B9E5_SHAREDEXP,
    FFX_API_SURFACE_FORMAT_R16G16B16A16_TYPELESS,
    FFX_API_SURFACE_FORMAT_R32G32_TYPELESS,
    FFX_API_SURFACE_FORMAT_R10G10B10A2_TYPELESS,
    FFX_API_SURFACE_FORMAT_R16G16_TYPELESS,
    FFX_API_SURFACE_FORMAT_R16_TYPELESS,
    FFX_API_SURFACE_FORMAT_R8_TYPELESS,
    FFX_API_SURFACE_FORMAT_R8G8_TYPELESS,
    FFX_API_SURFACE_FORMAT_R32_TYPELESS,
};

enum FfxApiResourceState
{
    FFX_API_RESOURCE_STATE_COMMON             = (1 << 0),
    FFX_API_RESOURCE_STATE_UNORDERED_ACCESS   = (1 << 1),
    FFX_API_RESOURCE_STATE_COMPUTE_READ       = (1 << 2),
    FFX_API_RESOURCE_STATE_PIXEL_READ         = (1 << 3),
    FFX_API_RESOURCE_STATE_COPY_SRC           = (1 << 4),
    FFX_API_RESOURCE_STATE_COPY_DEST          = (1 << 5),
    FFX_API_RESOURCE_STATE_RENDER_TARGET      = (1 << 8),
};

enum FfxApiResourceType
{
    FFX_API_RESOURCE_TYPE_BUFFER,
    FFX_API_RESOURCE_TYPE_TEXTURE1D,
    FFX_API_RESOURCE_TYPE_TEXTURE2D,
    FFX_API_RESOURCE_TYPE_TEXTURE_CUBE,
    FFX_API_RESOURCE_TYPE_TEXTURE3D,
};

struct FfxApiResourceDescription
{
    uint32_t type;
    uint32_t format;
    union { uint32_t width;  uint32_t size; };
    union { uint32_t height; uint32_t stride; };
    union { uint32_t depth;  uint32_t alignment; };
    uint32_t mipCount;
    uint32_t flags;
    uint32_t usage;
};

struct FfxApiResource
{
    void*                          resource;
    struct FfxApiResourceDescription description;
    uint32_t                       state;
};

struct FfxApiEffectMemoryUsage
{
    uint64_t totalUsageInBytes;
    uint64_t aliasableUsageInBytes;
};

// =========================================================================
// Descriptor type IDs
// =========================================================================
// Core API descriptor types (from ffx_api.h)
#define FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG1                          0x0000001u
#define FFX_API_QUERY_DESC_TYPE_GET_VERSIONS                              4u
#define FFX_API_DESC_TYPE_OVERRIDE_VERSION                                5u
#define FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION                      6u

#define FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_DX12                     0x0000002u
#define FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE                         0x00010000u
#define FFX_API_DISPATCH_DESC_TYPE_UPSCALE                               0x00010001u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE   0x00010002u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE 0x00010003u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT              0x00010004u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET                  0x00010005u
#define FFX_API_DISPATCH_DESC_TYPE_UPSCALE_GENERATEREACTIVEMASK          0x00010006u
#define FFX_API_CONFIGURE_DESC_TYPE_UPSCALE_KEYVALUE                     0x00010007u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE                 0x00010008u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE_V2              0x00010009u
#define FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS        0x0001000au

// =========================================================================
// DX12 backend descriptor
// =========================================================================
struct ffxCreateBackendDX12Desc
{
    ffxCreateContextDescHeader header;
    ID3D12Device*              device;
};

// =========================================================================
// Upscale quality modes
// =========================================================================
enum FfxApiUpscaleQualityMode
{
    FFX_UPSCALE_QUALITY_MODE_NATIVEAA          = 0,  // 1.0x
    FFX_UPSCALE_QUALITY_MODE_QUALITY           = 1,  // 1.5x
    FFX_UPSCALE_QUALITY_MODE_BALANCED          = 2,  // 1.7x
    FFX_UPSCALE_QUALITY_MODE_PERFORMANCE       = 3,  // 2.0x
    FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE = 4,  // 3.0x
};

// =========================================================================
// Upscale create/dispatch flags
// =========================================================================
enum FfxApiCreateContextUpscaleFlags
{
    FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE                 = (1<<0),
    FFX_UPSCALE_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS  = (1<<1),
    FFX_UPSCALE_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION = (1<<2),
    FFX_UPSCALE_ENABLE_DEPTH_INVERTED                     = (1<<3),
    FFX_UPSCALE_ENABLE_DEPTH_INFINITE                     = (1<<4),
    FFX_UPSCALE_ENABLE_AUTO_EXPOSURE                      = (1<<5),
    FFX_UPSCALE_ENABLE_DYNAMIC_RESOLUTION                 = (1<<6),
    FFX_UPSCALE_ENABLE_DEBUG_CHECKING                     = (1<<7),
    FFX_UPSCALE_ENABLE_NON_LINEAR_COLORSPACE              = (1<<8),
};

enum FfxApiDispatchFsrUpscaleFlags
{
    FFX_UPSCALE_FLAG_DRAW_DEBUG_VIEW       = (1 << 0),
    FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB = (1 << 1),
    FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ   = (1 << 2),
};

// =========================================================================
// Upscale descriptors
// =========================================================================
struct ffxCreateContextDescUpscale
{
    ffxCreateContextDescHeader header;
    uint32_t                   flags;
    struct FfxApiDimensions2D  maxRenderSize;
    struct FfxApiDimensions2D  maxUpscaleSize;
    ffxApiMessage              fpMessage;
};

struct ffxDispatchDescUpscale
{
    ffxDispatchDescHeader      header;
    void*                      commandList;
    struct FfxApiResource      color;
    struct FfxApiResource      depth;
    struct FfxApiResource      motionVectors;
    struct FfxApiResource      exposure;
    struct FfxApiResource      reactive;
    struct FfxApiResource      transparencyAndComposition;
    struct FfxApiResource      output;
    struct FfxApiFloatCoords2D jitterOffset;
    struct FfxApiFloatCoords2D motionVectorScale;
    struct FfxApiDimensions2D  renderSize;
    struct FfxApiDimensions2D  upscaleSize;
    bool                       enableSharpening;
    float                      sharpness;
    float                      frameTimeDelta;
    float                      preExposure;
    bool                       reset;
    float                      cameraNear;
    float                      cameraFar;
    float                      cameraFovAngleVertical;
    float                      viewSpaceToMetersFactor;
    uint32_t                   flags;
};

// =========================================================================
// Query descriptors
// =========================================================================
struct ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode
{
    ffxQueryDescHeader header;
    uint32_t           qualityMode;
    float*             pOutUpscaleRatio;
};

struct ffxQueryDescUpscaleGetRenderResolutionFromQualityMode
{
    ffxQueryDescHeader header;
    uint32_t           displayWidth;
    uint32_t           displayHeight;
    uint32_t           qualityMode;
    uint32_t*          pOutRenderWidth;
    uint32_t*          pOutRenderHeight;
};

struct ffxQueryDescUpscaleGetJitterPhaseCount
{
    ffxQueryDescHeader header;
    uint32_t           renderWidth;
    uint32_t           displayWidth;
    int32_t*           pOutPhaseCount;
};

struct ffxQueryDescUpscaleGetJitterOffset
{
    ffxQueryDescHeader header;
    int32_t            index;
    int32_t            phaseCount;
    float*             pOutX;
    float*             pOutY;
};

struct ffxQueryDescUpscaleGetGPUMemoryUsage
{
    ffxQueryDescHeader header;
    struct FfxApiEffectMemoryUsage* gpuMemoryUsageUpscaler;
};

// =========================================================================
// Core API version query descriptors (from ffx_api.h)
// =========================================================================
struct ffxQueryDescGetVersions
{
    ffxQueryDescHeader header;
    uint64_t           createDescType;  // Effect type to query (e.g. 0x00010000 = upscale)
    void*              device;          // ID3D12Device*
    uint64_t*          outputCount;     // In: capacity. Out: number of versions.
    uint64_t*          versionIds;      // Output array of version IDs (may be null)
    const char**       versionNames;    // Output array of version name strings (may be null)
};

struct ffxQueryGetProviderVersion
{
    ffxQueryDescHeader header;
    uint64_t           versionId;       // Out: ID of provider being used
    const char*        versionName;     // Out: version name for display
};

struct ffxOverrideVersion
{
    ffxApiHeader header;
    uint64_t     versionId;             // Version to use (from ffxQueryDescGetVersions)
};

#ifdef __cplusplus
}
#endif
