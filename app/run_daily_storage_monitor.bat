@echo off
setlocal
for %%I in ("%~dp0..") do set "repo_root=%%~fI"
call "%repo_root%\khquant-native.bat" doctor --check-write --strict
exit /b %ERRORLEVEL%
