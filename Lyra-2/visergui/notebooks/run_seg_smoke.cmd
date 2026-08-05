@echo off
rem Segmentation smoke launcher: same MSVC + CUDA env as run_train_smoke.cmd
rem (gsplat renders need the cached JIT kernels + ninja on PATH), forwards args
rem to run_seg_smoke.ps1. Log: _work\seg_smoke.log
set "LOG=%~dp0_work\seg_smoke.log"
if not exist "%~dp0_work" mkdir "%~dp0_work"
echo starting > "%LOG%"
for /d %%D in ("C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.38*") do set "MSVC=%%~D"
set "SDK=C:\Program Files (x86)\Windows Kits\10"
set "SDKVER=10.0.26100.0"
set "CUDA_HOME=%LOCALAPPDATA%\miniconda3\envs\splat\Library"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%MSVC%\bin\Hostx64\x64;%SDK%\bin\%SDKVER%\x64;%CUDA_HOME%\bin;%LOCALAPPDATA%\miniconda3\envs\splat;%LOCALAPPDATA%\miniconda3\envs\splat\Scripts;%PATH%"
set "INCLUDE=%MSVC%\include;%SDK%\Include\%SDKVER%\ucrt;%SDK%\Include\%SDKVER%\shared;%SDK%\Include\%SDKVER%\um;%SDK%\Include\%SDKVER%\winrt"
set "LIB=%MSVC%\lib\x64;%SDK%\Lib\%SDKVER%\ucrt\x64;%SDK%\Lib\%SDKVER%\um\x64;%CUDA_HOME%\lib"
powershell -ExecutionPolicy Bypass -File "%~dp0run_seg_smoke.ps1" %* >> "%LOG%" 2>&1
echo SEG_SMOKE_EXIT=%ERRORLEVEL% >> "%LOG%"
exit /b %ERRORLEVEL%
