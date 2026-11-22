@echo off
setlocal
cd /d "%~dp0.."
set "PYTHON=%CD%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
set "PYTHONPATH=%CD%\src;%CD%;%PYTHONPATH%"
"%PYTHON%" ui\texture_task_app.py
set "EXIT=%ERRORLEVEL%"
if not "%EXIT%"=="0" pause
endlocal & exit /b %EXIT%
