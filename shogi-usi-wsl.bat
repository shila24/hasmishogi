@echo off
rem EDIT THIS VARIABLE: absolute Linux path of the repository inside WSL.
set "WSL_REPO=/EDIT/THIS/PATH/hasmishogi"
setlocal
wsl.exe --cd "%WSL_REPO%" --exec sh -c "if [ -x .venv/bin/python ]; then exec .venv/bin/python -m shogi.usi; else exec python3 -m shogi.usi; fi"
exit /b %errorlevel%
