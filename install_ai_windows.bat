@echo off
rem Creates .venv-ai: a separate Python with CUDA PyTorch + diffusers for the AI nodes
rem (SDXL Generation, SDXL Lightning, LCM Turbo, SDXL Turbo). The app finds it automatically.
rem It is separate from .venv because PyTorch ships wheels for fewer Python versions than the app supports.
setlocal
cd /d "%~dp0"

set "PY="
for %%V in (3.12 3.13 3.11) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  where python >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.11-3.13 not found. Install Python 3.12 from python.org and run this again.
  pause & exit /b 1
)
echo Using %PY%
%PY% -c "import sys; print(sys.version)"

if not exist .venv-ai (
  %PY% -m venv .venv-ai || (echo Could not create .venv-ai & pause & exit /b 1)
)
call .venv-ai\Scripts\activate.bat
python -m pip install --upgrade pip

rem CUDA 12.8 wheels support RTX 20xx-50xx (Blackwell needs 12.8 or newer).
echo Installing PyTorch with CUDA (about 3 GB)...
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128 || (
  echo.
  echo PyTorch install failed. If your Python version has no CUDA wheel yet, install Python 3.12 and retry.
  pause & exit /b 1
)
echo Installing diffusers...
python -m pip install "diffusers>=0.30" "transformers>=4.44" "accelerate>=0.33" "peft>=0.12" safetensors pillow || (
  pause & exit /b 1
)
python -c "import torch, diffusers; print('torch', torch.__version__, 'CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''); print('diffusers', diffusers.__version__)"
echo.
echo AI environment ready: %CD%\.venv-ai\Scripts\python.exe
echo Download models from Edit ^> Settings ^> AI Models (or AI ^> Models ^& Settings).
pause
