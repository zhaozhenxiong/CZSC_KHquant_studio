@echo off
setlocal
for %%I in ("%~dp0..") do set "repo_root=%%~fI"
call "%repo_root%\khquant-native.bat" update-data --mode custom
set "exit_code=%ERRORLEVEL%"
pause
exit /b %exit_code%
