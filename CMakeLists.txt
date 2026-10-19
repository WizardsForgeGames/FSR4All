cmake_minimum_required(VERSION 3.18)
project(FSR4All VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static CRT (/MT) for all targets
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)

# Sources are UTF-8; don't let MSVC decode them with the system code page
if(MSVC)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/utf-8> $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>)
endif()

option(FSR4_BUILD_CUDA "Build the CUDA reimplementation of the FSR4 network (needs CUDA Toolkit + Forge)" ON)
option(FSR4_BUILD_DLL  "Build the D3D12 drop-in DLLs (needs headers generated from your FSR4 source)" ON)

# =========================================================================
# Part 1: CUDA reimplementation (standalone inference, tests, benchmark)
# =========================================================================
if(FSR4_BUILD_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)

    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "52")
        set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "CUDA architectures" FORCE)
    endif()

    # Forge GPU compute library (provides int8_gemm / quantize / int8_conv2d primitives)
    set(FORGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Forge" CACHE PATH "Path to the Forge source tree")
    if(NOT EXISTS "${FORGE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Forge not found at '${FORGE_DIR}'. Pass -DFORGE_DIR=<path>, "
            "or configure with -DFSR4_BUILD_CUDA=OFF to build only the D3D12 DLL.")
    endif()
    add_subdirectory(${FORGE_DIR} ${CMAKE_BINARY_DIR}/forge_build)

    add_library(fsr4_model STATIC
        src/model/weights.cu
        src/model/forward.cu
        src/model/forward_nhwc.cu
        src/model/pipeline.cu
    )
    target_include_directories(fsr4_model
        PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_link_libraries(fsr4_model PUBLIC forge)
    target_compile_options(fsr4_model PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            $<$<NOT:$<CONFIG:Debug>>:-O3>
            --use_fast_math
            --expt-relaxed-constexpr
        >
    )
    set_target_properties(fsr4_model PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

    foreach(exe test_weight_loader test_forward test_inference benchmark)
        add_executable(${exe} tests/${exe}.cu)
        target_link_libraries(${exe} PRIVATE fsr4_model)
        set_target_properties(${exe} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
    endforeach()
endif()

# =========================================================================
# Part 2: D3D12 drop-in DLL (runs AMD's FSR4 HLSL shaders; no CUDA)
# =========================================================================
if(FSR4_BUILD_DLL)
    # These headers contain AMD-derived data and are generated locally, never committed:
    #   scripts/compile_shaders.bat + scripts/embed_shaders.py -> fsr4_shader_blobs.h
    #   scripts/embed_initializer.py -> fsr4_initializer_blob.h, fsr4_encoder1_weights.h
    set(_fsr4_generated
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dll/fsr4_shader_blobs.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dll/fsr4_initializer_blob.h
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dll/fsr4_encoder1_weights.h
    )
    foreach(_hdr ${_fsr4_generated})
        if(NOT EXISTS "${_hdr}")
            message(FATAL_ERROR
                "Missing generated header ${_hdr}.\n"
                "Generate it from your copy of the FSR4 source (see README, 'Building the D3D12 DLL'), "
                "or configure with -DFSR4_BUILD_DLL=OFF.")
        endif()
    endforeach()

    set(_fsr4_dll_includes
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/dll
    )

    # Debug-capable build: log file, live-reloaded fsr4_config.json, diagnostics
    add_library(fsr4_dll SHARED
        src/dll/fsr4_dll.cpp
        src/dll/fsr4_context.cpp
        src/dll/fsr4_d3d12_pipeline.cpp
        src/dll/fsr4_dll.def
    )
    target_include_directories(fsr4_dll PRIVATE ${_fsr4_dll_includes})
    target_link_libraries(fsr4_dll PRIVATE d3d12 dxgi)
    set_target_properties(fsr4_dll PROPERTIES
        OUTPUT_NAME "amd_fidelityfx_upscaler_dx12"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dll"
    )

    # Release build: no debug console, logging, config polling, or diagnostic readback
    add_library(fsr4_dll_release SHARED
        src/dll/fsr4_dll.cpp
        src/dll/fsr4_context.cpp
        src/dll/fsr4_d3d12_pipeline.cpp
        src/dll/fsr4_dll.def
    )
    target_include_directories(fsr4_dll_release PRIVATE ${_fsr4_dll_includes})
    target_compile_definitions(fsr4_dll_release PRIVATE FSR4_RELEASE)
    target_link_libraries(fsr4_dll_release PRIVATE d3d12 dxgi)
    set_target_properties(fsr4_dll_release PROPERTIES
        OUTPUT_NAME "amd_fidelityfx_upscaler_dx12"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dll_release"
    )

    # FSR2 shim: drop-in for games using the old FSR2 API (ffx_fsr2_api_x64.dll)
    add_library(fsr2_shim SHARED
        src/dll/fsr2_shim.cpp
        src/dll/fsr4_context.cpp
        src/dll/fsr4_d3d12_pipeline.cpp
        src/dll/fsr2_shim.def
    )
    target_include_directories(fsr2_shim PRIVATE ${_fsr4_dll_includes})
    target_link_libraries(fsr2_shim PRIVATE d3d12 dxgi)
    set_target_properties(fsr2_shim PROPERTIES
        OUTPUT_NAME "ffx_fsr2_api_x64"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/dll"
    )

    # Standalone D3D12 pipeline benchmark
    add_executable(fsr4_d3d12_benchmark
        tests/benchmark_d3d12.cpp
        src/dll/fsr4_d3d12_pipeline.cpp
    )
    target_include_directories(fsr4_d3d12_benchmark PRIVATE ${_fsr4_dll_includes})
    target_link_libraries(fsr4_d3d12_benchmark PRIVATE d3d12 dxgi)

    # Optional D3D12 Agility SDK (NuGet package Microsoft.Direct3D.D3D12, extracted).
    # When set, D3D12Core.dll is copied to D3D12/ next to the benchmark so newer
    # shader models are available on older Windows 10 system runtimes.
    set(D3D12_AGILITY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/d3d12_agility/extracted/build/native/bin/x64"
        CACHE PATH "Directory containing D3D12Core.dll from the D3D12 Agility SDK (optional)")
    if(EXISTS "${D3D12_AGILITY_DIR}/D3D12Core.dll")
        add_custom_command(TARGET fsr4_d3d12_benchmark POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:fsr4_d3d12_benchmark>/D3D12"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${D3D12_AGILITY_DIR}/D3D12Core.dll"
                "$<TARGET_FILE_DIR:fsr4_d3d12_benchmark>/D3D12/D3D12Core.dll"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${D3D12_AGILITY_DIR}/d3d12SDKLayers.dll"
                "$<TARGET_FILE_DIR:fsr4_d3d12_benchmark>/D3D12/d3d12SDKLayers.dll"
            COMMENT "Copying D3D12 Agility SDK DLLs..."
        )
    else()
        message(STATUS "D3D12 Agility SDK not found at D3D12_AGILITY_DIR; fsr4_d3d12_benchmark uses the system D3D12 runtime")
    endif()
endif()
