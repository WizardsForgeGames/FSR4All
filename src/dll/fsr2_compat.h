// FSR 4 All - FSR2 Legacy API Compatibility Types
// Copyright (c) 2026 TheWizardsForge. MIT License.
//
// Minimal redeclarations of the OLD FidelityFX FSR2 API types.
// Layout-compatible with ffx_fsr2.h from AMD's FidelityFX SDK 2.x.
// Used by fsr2_shim.cpp to accept old FSR2 calls and route them
// through our FSR4 D3D12 compute pipeline.
// Type/enum names and values mirror AMD's FidelityFX SDK headers
// (Copyright (C) 2025 Advanced Micro Devices, Inc., MIT License -- see LICENSE).
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// =========================================================================
// Error codes (FSR2 uses int32_t, not uint32_t like unified API)
// =========================================================================
typedef int32_t FfxErrorCode;

#define FFX_OK                          0
#define FFX_ERROR_INVALID_POINTER       0x80000000
#define FFX_ERROR_INVALID_ALIGNMENT     0x80000001
#define FFX_ERROR_INVALID_SIZE          0x80000002
#define FFX_ERROR_EOF                   0x80000003
#define FFX_ERROR_INVALID_PATH          0x80000004
#define FFX_ERROR_INVALID_ENUM          0x80000009
#define FFX_ERROR_INVALID_ARGUMENT      0x8000000a
#define FFX_ERROR_OUT_OF_RANGE          0x8000000b
#define FFX_ERROR_NULL_DEVICE           0x8000000c
#define FFX_ERROR_BACKEND_API_ERROR     0x8000000d
#define FFX_ERROR_INCOMPLETE_INTERFACE  0x8000000e

// =========================================================================
// Basic types
// =========================================================================
typedef void* FfxDevice;
typedef void* FfxCommandList;
typedef uint32_t FfxVersionNumber;
typedef void (*FfxFsr2Message)(uint32_t type, const wchar_t* message);
typedef void (*ffxMessageCallback)(uint32_t type, const wchar_t* message);

struct FfxApiDimensions2D_FSR2 {
    uint32_t width;
    uint32_t height;
};

struct FfxApiFloatCoords2D_FSR2 {
    float x;
    float y;
};

// =========================================================================
// Resource types (binary-compatible with AMD's FfxApiResource)
// =========================================================================
struct FfxApiResourceDescription_FSR2 {
    uint32_t type;
    uint32_t format;
    union { uint32_t width;  uint32_t size; };
    union { uint32_t height; uint32_t stride; };
    union { uint32_t depth;  uint32_t alignment; };
    uint32_t mipCount;
    uint32_t flags;
    uint32_t usage;
};

struct FfxApiResource_FSR2 {
    void*                              resource;
    struct FfxApiResourceDescription_FSR2 description;
    uint32_t                           state;
};

struct FfxApiEffectMemoryUsage_FSR2 {
    uint64_t totalUsageInBytes;
    uint64_t aliasableUsageInBytes;
};

// =========================================================================
// FfxInterface — we only need the layout to extract the device pointer
// All function pointers are void* for layout compatibility (8 bytes each on x64)
// =========================================================================
struct FfxInterface_Minimal {
    void* fpCallbacks[21];      // 21 function pointer slots (see ffx_interface.h)
    void* scratchBuffer;
    size_t scratchBufferSize;
    FfxDevice device;           // ID3D12Device* on DX12 backend
};

// =========================================================================
// FSR2 quality modes (same values as unified API)
// =========================================================================
enum FfxApiUpscaleQualityMode_FSR2 {
    FFX_FSR2_QUALITY_MODE_NATIVEAA          = 0,
    FFX_FSR2_QUALITY_MODE_QUALITY           = 1,
    FFX_FSR2_QUALITY_MODE_BALANCED          = 2,
    FFX_FSR2_QUALITY_MODE_PERFORMANCE       = 3,
    FFX_FSR2_QUALITY_MODE_ULTRA_PERFORMANCE = 4,
};

// =========================================================================
// FSR2 init flags
// Bits 0-6 match the unified API exactly.
// Bit 7 = TEXTURE1D_USAGE (FSR2-only, we ignore)
// Bit 8 = DEBUG_CHECKING (FSR2-only, we ignore)
// =========================================================================
enum FfxFsr2InitializationFlagBits {
    FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE                  = (1<<0),
    FFX_FSR2_ENABLE_DISPLAY_RESOLUTION_MOTION_VECTORS   = (1<<1),
    FFX_FSR2_ENABLE_MOTION_VECTORS_JITTER_CANCELLATION  = (1<<2),
    FFX_FSR2_ENABLE_DEPTH_INVERTED                      = (1<<3),
    FFX_FSR2_ENABLE_DEPTH_INFINITE                      = (1<<4),
    FFX_FSR2_ENABLE_AUTO_EXPOSURE                       = (1<<5),
    FFX_FSR2_ENABLE_DYNAMIC_RESOLUTION                  = (1<<6),
    FFX_FSR2_ENABLE_TEXTURE1D_USAGE                     = (1<<7),
    FFX_FSR2_ENABLE_DEBUG_CHECKING                      = (1<<8),
};

// =========================================================================
// FSR2 context description (passed to ffxFsr2ContextCreate)
// =========================================================================
struct FfxFsr2ContextDescription {
    uint32_t                      flags;
    struct FfxApiDimensions2D_FSR2   maxRenderSize;
    struct FfxApiDimensions2D_FSR2   displaySize;
    FfxFsr2Message                fpMessage;
    struct FfxInterface_Minimal   backendInterface;
};

// =========================================================================
// FSR2 dispatch description (passed to ffxFsr2ContextDispatch)
// NOTE: FSR2 has NO upscaleSize per-dispatch — display size is fixed from create.
// =========================================================================
struct FfxFsr2DispatchDescription {
    FfxCommandList                    commandList;
    struct FfxApiResource_FSR2        color;
    struct FfxApiResource_FSR2        depth;
    struct FfxApiResource_FSR2        motionVectors;
    struct FfxApiResource_FSR2        exposure;
    struct FfxApiResource_FSR2        reactive;
    struct FfxApiResource_FSR2        transparencyAndComposition;
    struct FfxApiResource_FSR2        output;
    struct FfxApiFloatCoords2D_FSR2   jitterOffset;
    struct FfxApiFloatCoords2D_FSR2   motionVectorScale;
    struct FfxApiDimensions2D_FSR2    renderSize;
    bool                              enableSharpening;
    float                             sharpness;
    float                             frameTimeDelta;
    float                             preExposure;
    bool                              reset;
    float                             cameraNear;
    float                             cameraFar;
    float                             cameraFovAngleVertical;
    float                             viewSpaceToMetersFactor;
    // Auto-reactive fields (FSR2 extension, ignored by shim)
    bool                              enableAutoReactive;
    struct FfxApiResource_FSR2        colorOpaqueOnly;
    float                             autoTcThreshold;
    float                             autoTcScale;
    float                             autoReactiveScale;
    float                             autoReactiveMax;
};

// =========================================================================
// FSR2 generate reactive mask description
// =========================================================================
struct FfxFsr2GenerateReactiveDescription {
    FfxCommandList                    commandList;
    struct FfxApiResource_FSR2        colorOpaqueOnly;
    struct FfxApiResource_FSR2        colorPreUpscale;
    struct FfxApiResource_FSR2        outReactive;
    struct FfxApiDimensions2D_FSR2    renderSize;
    float                             scale;
    float                             cutoffThreshold;
    float                             binaryValue;
    uint32_t                          flags;
};

// =========================================================================
// FSR2 context (opaque data blob — game allocates this)
// We store our internal context pointer at the start.
// =========================================================================
#define FFX_FSR2_CONTEXT_SIZE (1024 * 128)

struct FfxFsr2Context {
    uint32_t data[FFX_FSR2_CONTEXT_SIZE];
};

#ifdef __cplusplus
}
#endif
