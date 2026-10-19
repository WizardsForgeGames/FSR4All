# FSR4All

A drop-in D3D12 DLL that runs AMD's FSR4 INT8 upscaler on non-AMD GPUs, plus an earlier (now obsolete) CUDA reimplementation of the FSR4 network.

> **Status: prototype.** This is a project I started, got working, and am sharing with instructions. There is no guarantee it works on your setup. It is not regularly maintained; I update it only when I come back to it.

## How this relates to existing work

Running FSR4 INT8 on NVIDIA GPUs in games is already possible: OptiScaler together with the leaked INT8 build of AMD's FSR4 DLL does this. If you just want FSR4 in a game, that is the more mature option.

This repository has two independent parts:

1. **D3D12 drop-in DLL (`src/dll`)**: the part that is actually used. It exports the FidelityFX upscaler C API (`ffxCreateContext`, `ffxDispatch`, ...) as `amd_fidelityfx_upscaler_dx12.dll` and records AMD's FSR4 HLSL compute shaders, compiled with `dxc`, directly onto the game's command list. The network passes are AMD's shaders; the host side (API, resource and descriptor management, 15 compute pipelines, pre/model/post dispatch, history) is this project's. One shader include is overridden (`shaders/overrides/post_common.hlsli`, see below). Tested in The Outer Worlds 2 and Kingdom Come: Deliverance II. An FSR2-API shim (`ffx_fsr2_api_x64.dll`) is included but was not part of that game testing.
2. **CUDA reimplementation (`src/model`, `include/fsr4`, `tests`)**: obsolete. The network was first rewritten as hand-written CUDA kernels (dp4a INT8 fused blocks, NHWC, per-resolution mega-kernels, CUDA Graphs). Shipping that would have required users to install CUDA, so the DLL replaced it. It remains here as a standalone inference and research tool.

## Performance (RTX 4090)

D3D12 DLL pipeline, measured with `fsr4_d3d12_benchmark.exe --iterations 1000` (GPU timestamps, synthetic inputs):

| Upscale | Full pipeline (median) | Network passes | Post |
|---|---|---|---|
| 1440p → 4K (Quality) | ~2.46 ms | ~1.8 ms | ~0.37 ms |
| 720p → 1440p (Performance) | ~1.12 ms | | ~0.16 ms |

**`post_common.hlsli` override:** replaces the ~18 `exp()` calls per pixel in AMD's Post shader (Gaussian kernel, sigmoid, tanh) with a range-reduced polynomial `fast_exp`. On an RTX 4090 it makes no measurable difference (within run-to-run noise); it has not been re-measured on lower-end GPUs. `fsr4_d3d12_benchmark.exe --amd-post` runs AMD's original Post shader for comparison.

**CUDA reimplementation vs AMD's shaders:** at the same network resolution (3840×2160) on the same GPU, the CUDA forward pass takes ~5.65 ms against ~1.8 ms for AMD's shaders through the DLL. AMD's hand-tuned HLSL is about 3× faster; the CUDA path is kept for reference, not for speed.

CUDA path quality validation: the dp4a INT8 kernels measured 40.33 dB PSNR against an FP32 implementation of the same network, and later tiling/buffer-aliasing changes were bit-identical to the version before them. `scripts/quality_metrics.py` computes PSNR and SSIM between outputs and reference frames.

## Requirements

- Windows 10/11 x64, Visual Studio 2022 (MSVC), CMake ≥ 3.18
- Python 3 for the scripts. Weight extraction uses only the standard library. `quality_metrics.py` also needs NumPy, Pillow and scikit-image.
- **Your own copy of the FSR4 source.** This is AMD's FidelityFX SDK tree containing `Kits/FidelityFX/upscalers/fsr4`, including the `fsr4_model_v07_i8_{quality,balanced,performance,ultraperf}` shader directories. It is not included here. In this README its root is called `FSR4_SOURCE`.
- CUDA part:
  - CUDA Toolkit (developed with 12.4) and an NVIDIA GPU with sm_80 or newer (Forge requires it). Build for your GPU with `-DCMAKE_CUDA_ARCHITECTURES` (only sm_89 tested).
  - The **Forge** GPU compute library. Pass its location with `-DFORGE_DIR=...`; the default is `../Forge`. FSR4All uses Forge's INT8 GEMM, activation-quantization and INT8 conv primitives. Forge is at https://github.com/WizardsForgeGames/Forge. The D3D12 part does not need Forge.
- D3D12 part:
  - Windows SDK with `dxc.exe`. The shaders compile to `cs_6_6`, so the GPU and driver must support Shader Model 6.6 with dp4a.
  - Optionally the D3D12 Agility SDK, for the standalone benchmark only.

## Building the CUDA reimplementation

```bash
# 1. Extract weights from your FSR4 source (writes weights/quality/*.bin + manifest.json)
python scripts/extract_fsr4_weights.py "$FSR4_SOURCE/Kits/FidelityFX/upscalers/fsr4/internal/shaders/fsr4_model_v07_i8_quality"

# 2. Configure and build (CUDA only)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DFSR4_BUILD_DLL=OFF -DFORGE_DIR=/path/to/Forge -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release
```

Set `CMAKE_CUDA_ARCHITECTURES` to match your GPU, e.g. `86` for RTX 30xx.

Run from the repository root. `weights/quality` is the default weight directory, and `--weights <dir>` overrides it.

```bash
build/Release/benchmark.exe --nhwc --1440p            # also --720p, --1080p, --4k, --res WxH, --iters N
build/Release/test_inference.exe --input frame.bmp --scale 2 --output output/
build/Release/test_inference.exe --sequence frames_dir/ --mv-dir mvs/ --output output/seq
build/Release/test_forward.exe                        # forward-pass smoke test
build/Release/test_weight_loader.exe                  # weight loading check
```

`tools/dashboard` is a small Electron viewer for comparing outputs: `npm install`, then `npm start`.

## Building the D3D12 DLL

The DLL embeds compiled shaders and model data generated from your FSR4 source. The generated files are git-ignored and must not be committed.

```bat
set FSR4_SOURCE=C:\path\to\FidelityFX-SDK
set DXC=C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe

:: 1. Compile AMD's HLSL to shaders\compiled\*.cso (4 presets x 3 resolution tiers)
scripts\compile_shaders.bat
:: 2. Embed bytecode -> src\dll\fsr4_shader_blobs.h
python scripts\embed_shaders.py
:: 3. Embed initializers.bin + encoder1 weights -> src\dll\fsr4_initializer_blob.h, fsr4_encoder1_weights.h
python scripts\embed_initializer.py

:: 4. Build (D3D12 only; no CUDA needed)
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DFSR4_BUILD_CUDA=OFF
cmake --build build --config Release
```

Build outputs:
- `build/dll/Release/amd_fidelityfx_upscaler_dx12.dll`: debug-capable. It writes `fsr4_debug.log` and reads `fsr4_config.json` next to itself, re-reading the config every 2 s.
- `build/dll_release/Release/amd_fidelityfx_upscaler_dx12.dll`: no config polling or diagnostics. It writes only a short `fsr4_release.log`.
- `build/dll/Release/ffx_fsr2_api_x64.dll`: the FSR2 shim.
- `build/Release/fsr4_d3d12_benchmark.exe`: a standalone pipeline benchmark. Set `-DD3D12_AGILITY_DIR=<dir containing D3D12Core.dll>` to bundle the Agility SDK runtime with it.

**Deploying:** back up the game's FidelityFX DLL, then replace it with the built one. Some games load a loader DLL that proxies to the upscaler. Kingdom Come: Deliverance II is one: replace `amd_fidelityfx_loader_dx12.dll` there, not the upscaler. The Outer Worlds 2 loads `amd_fidelityfx_upscaler_dx12.dll` directly.

Pre, model and post shaders must be compiled for the same resolution tier (`FFX_MLSR_RESOLUTION`, 0/1/2 = 1080p/2160p/4320p). If they are not, the scratch-buffer layouts disagree and the image corrupts. `compile_shaders.bat` builds every tier, and the DLL picks one at context creation.

## Profiling scripts

`scripts/run_ncu*.bat` run Nsight Compute on `benchmark.exe`. They need administrator rights on Windows. Set `NCU` to the full path of `ncu.exe` if it is not on `PATH`. `scripts/executive_summary.py` converts the exported CSV into an Excel report.

## Legal and attribution

- FSR, FidelityFX and the FSR4 network belong to AMD. **This project is not affiliated with or endorsed by AMD.**
- The network weights, quantization scales and shaders used here derive from AMD's FidelityFX SDK FSR4 source. AMD pushed that source to its public GPUOpen GitHub in August 2025, apparently by accident, under the MIT license. This repository does not redistribute that source, the network weights, or compiled shaders. The only exceptions are a few dozen activation-scale constants and one modified shader include, both listed in `LICENSE`. You supply your own copy, and the scripts generate the derived files locally.
- A few files in this repository contain AMD-derived material. They keep AMD's MIT notice and are listed in [`LICENSE`](LICENSE). Everything else is MIT, © 2026 Darren Sander.
