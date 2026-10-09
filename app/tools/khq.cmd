@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

where pwsh >nul 2>nul
if "%ERRORLEVEL%"=="0" (
    pwsh -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%khq.ps1" %*
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%khq.ps1" %*
)

exit /b %ERRORLEVEL%
