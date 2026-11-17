@echo off
setlocal
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3.11+ not found in PATH. Install it from python.org and tick "Add python.exe to PATH". & pause & exit /b 1)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" || (echo Python 3.10 or newer is required. & pause & exit /b 1)
if not exist .venv (
  python -m venv .venv || (echo Could not create the virtual environment. & pause & exit /b 1)
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
echo Building the C++ core and installing (needs Visual Studio 2022 Build Tools, "Desktop development with C++")...
python -m pip install . || (
  echo.
  echo Build failed. Check that "Visual Studio 2022 Build Tools" with the C++ workload is installed:
  echo   https://visualstudio.microsoft.com/visual-cpp-build-tools/
  pause & exit /b 1
)
python -m pip install pytest pytest-qt scipy >nul
echo.
echo Installed. Start with run_windows.bat
pause
