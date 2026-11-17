@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\vertexcomp.exe (echo Run install_windows.bat first. & pause & exit /b 1)
start "" .venv\Scripts\vertexcomp.exe %*
