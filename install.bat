@echo off
setlocal enabledelayedexpansion

title KHQuant Native Installer

echo ==========================================
echo   KHQuant Native Installer for Windows
echo ==========================================
echo.

set "PYTHON_EXE="
set "PYTHON_ARGS="
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 --version >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=py"
        set "PYTHON_ARGS=-3.12"
    )
)
if not defined PYTHON_EXE where python >nul 2>nul
if not defined PYTHON_EXE if not errorlevel 1 set "PYTHON_EXE=python"
if not defined PYTHON_EXE (
    echo Python not found. Please install Python 3.12 from https://www.python.org/downloads/
    pause
    exit /b 1
)

%PYTHON_EXE% %PYTHON_ARGS% --version

cd /d "%~dp0"
%PYTHON_EXE% %PYTHON_ARGS% install.py %*

if errorlevel 1 (
    echo.
    echo Installation failed. See errors above.
    pause
    exit /b 1
)

echo.
echo Installation complete.
pause
exit /b 0
