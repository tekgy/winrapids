@echo off
cd /d "%~dp0"

set "MSVC_ROOT=C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.44.35207"
set "WINSDK_ROOT=C:\Program Files (x86)\Windows Kits\10"
set "WINSDK_VER=10.0.26100.0"

set "PATH=%MSVC_ROOT%\bin\Hostx64\x64;%PATH%"
set "INCLUDE=%MSVC_ROOT%\include;%WINSDK_ROOT%\Include\%WINSDK_VER%\ucrt;%WINSDK_ROOT%\Include\%WINSDK_VER%\um;%WINSDK_ROOT%\Include\%WINSDK_VER%\shared"
set "LIB=%MSVC_ROOT%\lib\x64;%WINSDK_ROOT%\Lib\%WINSDK_VER%\ucrt\x64;%WINSDK_ROOT%\Lib\%WINSDK_VER%\um\x64"

echo === Building remaining tests ===
nvcc -O2 -arch=sm_120 -DSKIP_CAPACITY_TEST -o gpu_memory_remaining.exe gpu_memory.cu -lcudart 2>&1
if %errorlevel% neq 0 exit /b %errorlevel%
echo === Running remaining tests ===
gpu_memory_remaining.exe
