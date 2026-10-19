@echo off
REM Optional environment: NCU = full path to ncu.exe (default: ncu.exe on PATH).
REM Run as administrator: Windows WDDM blocks GPU performance counters otherwise.
if "%NCU%"=="" set "NCU=ncu.exe"
set "REPO=%~dp0.."
cd /d "%REPO%"
if not exist "%REPO%\experiments" mkdir "%REPO%\experiments"
set "APP=%REPO%\build\Release\benchmark.exe"
set "OUT=%REPO%\experiments\ncu_mega"
set "LOG=%REPO%\experiments\ncu_mega_log.txt"

echo === NCU Mega-Kernel Profiling === > "%LOG%" 2>&1
echo Current directory: %CD% >> "%LOG%" 2>&1
echo NCU: "%NCU%" >> "%LOG%" 2>&1
echo APP: "%APP%" >> "%LOG%" 2>&1
echo OUT: "%OUT%" >> "%LOG%" 2>&1
echo. >> "%LOG%" 2>&1

echo === Checking files exist === >> "%LOG%" 2>&1
if exist "%NCU%" (echo NCU exists >> "%LOG%") else (echo NCU NOT FOUND >> "%LOG%")
if exist "%APP%" (echo APP exists >> "%LOG%") else (echo APP NOT FOUND >> "%LOG%")

echo. >> "%LOG%" 2>&1
echo === Running ncu === >> "%LOG%" 2>&1
REM launch-skip 0: Profile from the very first kernel (profiled run bypasses CUDA Graph)
REM launch-count 20: 8 kernels per forward pass, capture ~2.5 passes for reliable data
REM 720p: keeps ncu replay time reasonable
"%NCU%" --set full --launch-skip 0 --launch-count 20 -o "%OUT%" --force-overwrite "%APP%" --nhwc --720p --warmup 0 --iters 1 --weights "%REPO%\weights\quality" >> "%LOG%" 2>&1
echo NCU exit code: %errorlevel% >> "%LOG%" 2>&1

echo. >> "%LOG%" 2>&1
echo === Checking output === >> "%LOG%" 2>&1
if exist "%OUT%.ncu-rep" (
    echo ncu-rep file exists >> "%LOG%" 2>&1
    echo === Exporting CSV === >> "%LOG%" 2>&1
    "%NCU%" --import "%OUT%.ncu-rep" --csv --page details > "%OUT%_details.csv" 2>> "%LOG%"
    echo CSV export exit code: %errorlevel% >> "%LOG%" 2>&1
) else (
    echo ncu-rep file NOT FOUND >> "%LOG%" 2>&1
)

echo === Done === >> "%LOG%" 2>&1
