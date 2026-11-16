@echo off
setlocal
cd /d "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" (
  "%~dp0.venv\Scripts\python.exe" -m shogi.usi %*
) else (
  python -m shogi.usi %*
)
exit /b %errorlevel%
