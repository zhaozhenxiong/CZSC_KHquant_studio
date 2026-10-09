@echo off
setlocal
cd /d %~dp0
powershell -NoProfile -ExecutionPolicy Bypass -File "%CD%\my_strategy\scripts\run_agent_task.ps1" %*
exit /b %ERRORLEVEL%
