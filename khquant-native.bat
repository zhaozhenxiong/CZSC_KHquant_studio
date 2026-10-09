@echo off
setlocal DisableDelayedExpansion

rem Resolve repo root. Python loads the full .env with no shell evaluation.
cd /d "%~dp0"
set "repo_root=%~dp0"
set "repo_root=%repo_root:~0,-1%"

set "venv_dir="
set "venv_value="
if exist ".env" for /f "usebackq tokens=1,* delims==" %%A in (".env") do if /i "%%A"=="KHQUANT_VENV_DIR" set "venv_value=%%~B"
if defined venv_value for %%I in ("%venv_value%") do set "venv_dir=%%~fI"
if defined venv_dir goto venv_selected
set "venv_dir=%repo_root%\.venv"

:venv_selected

if not defined venv_dir (
    echo Virtual environment not found. Run install.bat first.
    exit /b 1
)

if not exist "%venv_dir%\Scripts\python.exe" (
    echo Python interpreter missing in %venv_dir%
    exit /b 1
)

set "PYTHONPATH=%repo_root%\app"
if /i "%1"=="dashboard" (
    echo Starting KHQuant dashboard using the port configured in .env ...
)
"%venv_dir%\Scripts\python.exe" -m my_strategy.cli %*

exit /b %errorlevel%
