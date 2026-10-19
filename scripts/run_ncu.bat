@echo off
REM Self-elevate if not admin
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo Requesting admin privileges...
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

echo === FSR4 Nsight Compute Profiling (NHWC) ===
echo Running as administrator: YES
echo Resolution: 640x360
echo.

if "%NCU%"=="" set "NCU=ncu.exe"
set "REPO=%~dp0.."
cd /d "%REPO%"
if not exist "%REPO%\experiments" mkdir "%REPO%\experiments"

echo Step 1: Profiling...
"%NCU%" --set full --launch-skip 0 --launch-count 50 -o "%REPO%\experiments\ncu_fsr4_nhwc" --force-overwrite "%REPO%\build\Release\benchmark.exe" --nhwc --res 640x360 --warmup 1 --iters 1 --weights "%REPO%\weights\quality"

echo.
echo Step 2: Exporting CSV...
"%NCU%" --import "%REPO%\experiments\ncu_fsr4_nhwc.ncu-rep" --csv --page details > "%REPO%\experiments\ncu_fsr4_nhwc_details.csv"

echo.
echo === Done ===
echo Check: dir "%REPO%\experiments\ncu_fsr4_nhwc*"
dir "%REPO%\experiments\ncu_fsr4_nhwc*"
echo.
pause
