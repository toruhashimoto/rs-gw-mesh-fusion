@echo off
REM ASCII-only on purpose: cmd.exe parses .bat in the OEM codepage.
REM MeshFusion desktop app launcher (local Gradio UI, opens in browser).
REM Python resolution order: MESHFUSION_PYTHON env var > .venv > python on PATH.
setlocal
if exist "%~dp0local_env.bat" call "%~dp0local_env.bat"
if defined MESHFUSION_PYTHON (
  set "PY=%MESHFUSION_PYTHON%"
) else if exist "%~dp0.venv\Scripts\python.exe" (
  set "PY=%~dp0.venv\Scripts\python.exe"
) else (
  set "PY=python"
)
if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>nul
set "VSLANG=1033"
set "NVCC_APPEND_FLAGS=-DUSE_CUDA"
set "PYTHONUTF8=1"
"%PY%" "%~dp0app.py"
if errorlevel 1 (
  echo.
  echo [ERROR] app exited with an error. Is the environment installed?
  echo         pip install -r requirements.txt
  pause
)
exit /b %errorlevel%
