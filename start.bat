@echo off
setlocal
cd /d "%~dp0"
set "PYTHONIOENCODING=utf-8"

REM Prefer the project-local virtual environment.
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
    goto run
)

REM Fall back to python on PATH.
where python >nul 2>&1
if errorlevel 1 goto nopython
set "PY=python"
goto run

:nopython
echo [ERROR] Python not found.
echo.
echo Install Python 3.11+ and add it to PATH, or create a venv first:
echo     python -m venv .venv
echo     .venv\Scripts\python.exe -m pip install -r requirements.txt
echo.
pause
exit /b 1

:run
"%PY%" app.py
pause
