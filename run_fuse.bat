@echo off
REM ASCII-only on purpose: cmd.exe parses .bat in the OEM codepage.
REM MeshFusion launcher: RS-primary + GW-complement mesh fusion.
REM Usage: run_fuse.bat --rs RS.ply --gw GW.ply --out OUTDIR [options]
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PREFIX=C:\Users\toruh\miniconda3\envs\gaussian_wrapping"
set "PATH=%CUDA_HOME%\bin;%PREFIX%\Scripts;%PREFIX%\Library\bin;%PATH%"
set "DISTUTILS_USE_SDK=1"
set "VSLANG=1033"
set "NVCC_APPEND_FLAGS=-DUSE_CUDA"
set "PYTHONUTF8=1"
if not exist "%PREFIX%\python.exe" (
  echo [ERROR] conda env 'gaussian_wrapping' not found at %PREFIX%
  exit /b 1
)
"%PREFIX%\python.exe" "%~dp0fuse_meshes.py" %*
exit /b %errorlevel%
