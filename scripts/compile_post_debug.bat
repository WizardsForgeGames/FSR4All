@echo off
setlocal enabledelayedexpansion
REM Compile ONLY post_debug shaders (FFX_DEBUG_VISUALIZE=1) for all presets x resolutions

REM Required environment:
REM   FSR4_SOURCE  Root of your FidelityFX SDK checkout containing the FSR4 source
REM                (the directory that contains Kits\FidelityFX\upscalers\fsr4).
REM Optional:
REM   DXC          Full path to dxc.exe (default: dxc.exe on PATH). The Windows SDK copy
REM                lives at "C:\Program Files (x86)\Windows Kits\10\bin\<version>\x64\dxc.exe".
if "%FSR4_SOURCE%"=="" (
    echo ERROR: set FSR4_SOURCE to the root of your FSR4 / FidelityFX SDK source checkout.
    exit /b 1
)
if "%DXC%"=="" set "DXC=dxc.exe"
set "REPO=%~dp0.."
set "SRC=%FSR4_SOURCE%\Kits\FidelityFX\upscalers\fsr4"
set "OUT=%REPO%\shaders\compiled"

set COMMON_FLAGS=-no-warnings -O3 -enable-16bit-types -HV 2021 -T cs_6_6
set "FFX_GPU=%FSR4_SOURCE%\Kits\FidelityFX\api\internal\gpu"
set PRE_POST_INCLUDES_ORIG=-I "%SRC%\dx12" -I "%SRC%\include\gpu\fsr4" -I "%SRC%\include\gpu" -I "%SRC%\include\gpu\spd" -I "%SRC%" -I "%FFX_GPU%"

set SHADER_BASE=%SRC%\internal\shaders

for %%Q in (quality balanced performance ultraperf) do (
    for %%R in (1080 2160 4320) do (
        set RES_IDX=0
        if "%%R"=="2160" set RES_IDX=1
        if "%%R"=="4320" set RES_IDX=2

        set PRESET_DIR=%SHADER_BASE%\fsr4_model_v07_i8_%%Q

        echo === Compiling Post Shader [debug] ^(%%Q %%Rp^) ===
        "%DXC%" %COMMON_FLAGS% %PRE_POST_INCLUDES_ORIG% -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_COLORSPACE=0 -DFFX_MLSR_RESOLUTION=!RES_IDX! -DFFX_DEBUG_VISUALIZE=1 -E main "!PRESET_DIR!\post.hlsl" -Fo "%OUT%\post_debug_%%Q_%%R.cso"
        if errorlevel 1 goto :error
    )
)

echo.
echo === All post_debug shaders compiled successfully! ===
goto :end

:error
echo.
echo !!! COMPILATION FAILED !!!
exit /b 1

:end
