@echo off
setlocal enabledelayedexpansion
REM Compile all FSR4 HLSL shaders for DX12 compute backend
REM Compiles FOUR quality presets x THREE resolution tiers
REM Usage: compile_shaders.bat [1080|2160|4320|all] [quality|balanced|performance|ultraperf|all]

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
set MODEL_INCLUDES=-I "%SRC%\dx12" -I "%SRC%\dx12\ml2code_runtime"
set "FFX_GPU=%FSR4_SOURCE%\Kits\FidelityFX\api\internal\gpu"
set "OVERRIDES=%REPO%\shaders\overrides"
set PRE_POST_INCLUDES=-I "%OVERRIDES%" -I "%SRC%\dx12" -I "%SRC%\include\gpu\fsr4" -I "%SRC%\include\gpu" -I "%SRC%\include\gpu\spd" -I "%SRC%" -I "%FFX_GPU%"
set PRE_POST_INCLUDES_ORIG=-I "%SRC%\dx12" -I "%SRC%\include\gpu\fsr4" -I "%SRC%\include\gpu" -I "%SRC%\include\gpu\spd" -I "%SRC%" -I "%FFX_GPU%"

set SHADER_BASE=%SRC%\internal\shaders

if not exist "%OUT%" mkdir "%OUT%"

REM Parse resolution argument
set BUILD_1080=0
set BUILD_2160=0
set BUILD_4320=0
if "%1"=="" (
    set BUILD_1080=1
    set BUILD_2160=1
    set BUILD_4320=1
) else if "%1"=="all" (
    set BUILD_1080=1
    set BUILD_2160=1
    set BUILD_4320=1
) else if "%1"=="1080" (
    set BUILD_1080=1
) else if "%1"=="2160" (
    set BUILD_2160=1
) else if "%1"=="4320" (
    set BUILD_4320=1
) else (
    echo Usage: compile_shaders.bat [1080^|2160^|4320^|all] [quality^|balanced^|performance^|ultraperf^|all]
    exit /b 1
)

REM Parse quality preset argument
set BUILD_QUALITY=0
set BUILD_BALANCED=0
set BUILD_PERFORMANCE=0
set BUILD_ULTRAPERF=0
if "%2"=="" (
    set BUILD_QUALITY=1
    set BUILD_BALANCED=1
    set BUILD_PERFORMANCE=1
    set BUILD_ULTRAPERF=1
) else if "%2"=="all" (
    set BUILD_QUALITY=1
    set BUILD_BALANCED=1
    set BUILD_PERFORMANCE=1
    set BUILD_ULTRAPERF=1
) else if "%2"=="quality" (
    set BUILD_QUALITY=1
) else if "%2"=="balanced" (
    set BUILD_BALANCED=1
) else if "%2"=="performance" (
    set BUILD_PERFORMANCE=1
) else if "%2"=="ultraperf" (
    set BUILD_ULTRAPERF=1
) else (
    echo Usage: compile_shaders.bat [1080^|2160^|4320^|all] [quality^|balanced^|performance^|ultraperf^|all]
    exit /b 1
)

REM Resolution tier definitions:
REM   FFX_MLSR_RESOLUTION=0 -> 1080p (up to 1920x1080 output)
REM   FFX_MLSR_RESOLUTION=1 -> 2160p (up to 3840x2160 output)
REM   FFX_MLSR_RESOLUTION=2 -> 4320p (up to 7680x4320 output)
REM
REM CRITICAL: Pre, Post, and Model passes MUST use matching resolution.
REM           Mismatch causes scratch buffer tensor stride corruption.

REM === RCAS (resolution-independent, quality-independent, compile once) ===
echo === Compiling RCAS Shader ===
"%DXC%" %COMMON_FLAGS% %PRE_POST_INCLUDES% -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_COLORSPACE=0 -DFSR4UPSCALER_BIND_CB_RCAS=2 -E main "%SRC%\internal\shaders\rcas.hlsl" -Fo "%OUT%\rcas.cso"
if errorlevel 1 goto :error

REM === Debug Blend Overlay (resolution-independent, quality-independent, compile once) ===
echo === Compiling Debug Blend Overlay Shader ===
"%DXC%" %COMMON_FLAGS% -E main "%REPO%\shaders\debug_blend_overlay.hlsl" -Fo "%OUT%\debug_blend_overlay.cso"
if errorlevel 1 goto :error

REM === Compile function for one preset x one resolution ===
REM We use CALL :compile_preset to iterate

if %BUILD_QUALITY%==1     call :compile_all_res "quality" "fsr4_model_v07_i8_quality"
if %BUILD_BALANCED%==1    call :compile_all_res "balanced" "fsr4_model_v07_i8_balanced"
if %BUILD_PERFORMANCE%==1 call :compile_all_res "performance" "fsr4_model_v07_i8_performance"
if %BUILD_ULTRAPERF%==1   call :compile_all_res "ultraperf" "fsr4_model_v07_i8_ultraperf"

echo.
echo === All shaders compiled successfully! ===
echo Output: %OUT%
goto :end

REM =====================================================================
REM Subroutine: compile all requested resolutions for a given preset
REM Args: %~1 = preset short name, %~2 = preset directory name
REM =====================================================================
:compile_all_res
set PRESET_NAME=%~1
set PRESET_DIR=%SHADER_BASE%\%~2

if %BUILD_1080%==1 call :compile_tier "%PRESET_NAME%" "%PRESET_DIR%" 1080 0
if %BUILD_2160%==1 call :compile_tier "%PRESET_NAME%" "%PRESET_DIR%" 2160 1
if %BUILD_4320%==1 call :compile_tier "%PRESET_NAME%" "%PRESET_DIR%" 4320 2
goto :eof

REM =====================================================================
REM Subroutine: compile one preset x one resolution tier
REM Args: %~1 = preset name, %~2 = preset dir, %~3 = resolution, %~4 = res index
REM =====================================================================
:compile_tier
set P_NAME=%~1
set P_DIR=%~2
set P_RES=%~3
set P_IDX=%~4

echo.
echo ============================================
echo === %P_NAME% / %P_RES%p tier (FFX_MLSR_RESOLUTION=%P_IDX%) ===
echo ============================================

echo === Compiling Pre Shader (%P_NAME% %P_RES%p) ===
"%DXC%" %COMMON_FLAGS% %PRE_POST_INCLUDES% -DFFX_MLSR_DEPTH_INVERTED=1 -DFFX_MLSR_LOW_RES_MV=1 -DFFX_MLSR_JITTERED_MOTION_VECTORS=1 -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_COLORSPACE=0 -DFFX_MLSR_RESOLUTION=%P_IDX% -E main "%P_DIR%\pre.hlsl" -Fo "%OUT%\pre_%P_NAME%_%P_RES%.cso"
if errorlevel 1 goto :error

echo === Compiling Post Shader [fast_exp] (%P_NAME% %P_RES%p) ===
"%DXC%" %COMMON_FLAGS% %PRE_POST_INCLUDES% -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_COLORSPACE=0 -DFFX_MLSR_RESOLUTION=%P_IDX% -E main "%P_DIR%\post.hlsl" -Fo "%OUT%\post_%P_NAME%_%P_RES%.cso"
if errorlevel 1 goto :error

echo === Compiling Post Shader [normal] (%P_NAME% %P_RES%p) ===
"%DXC%" %COMMON_FLAGS% %PRE_POST_INCLUDES_ORIG% -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_COLORSPACE=0 -DFFX_MLSR_RESOLUTION=%P_IDX% -E main "%P_DIR%\post.hlsl" -Fo "%OUT%\post_normal_%P_NAME%_%P_RES%.cso"
if errorlevel 1 goto :error

echo === Compiling Post Shader [debug] (%P_NAME% %P_RES%p) ===
"%DXC%" %COMMON_FLAGS% %PRE_POST_INCLUDES_ORIG% -DFFX_MLSR_AUTOEXPOSURE_ENABLED=0 -DFFX_MLSR_COLORSPACE=0 -DFFX_MLSR_RESOLUTION=%P_IDX% -DFFX_DEBUG_VISUALIZE=1 -E main "%P_DIR%\post.hlsl" -Fo "%OUT%\post_debug_%P_NAME%_%P_RES%.cso"
if errorlevel 1 goto :error

echo === Compiling Model Passes 1-12 (%P_NAME% %P_RES%p) ===
for /L %%i in (1,1,12) do (
    echo   Pass %%i...
    "%DXC%" %COMMON_FLAGS% %MODEL_INCLUDES% -DMLSR_PASS_%%i -E fsr4_model_v07_i8_pass%%i "%P_DIR%\passes_%P_RES%.hlsl" -Fo "%OUT%\pass_%%i_%P_NAME%_%P_RES%.cso"
    if errorlevel 1 goto :error
)
goto :eof

:error
echo.
echo !!! COMPILATION FAILED !!!
exit /b 1

:end
