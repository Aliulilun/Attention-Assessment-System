@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM ============================================================
REM  Report Viewer launcher - Windows
REM
REM  Read-only viewer for analysis reports. No GPU, no PyTorch,
REM  no model downloads. Works on any Windows PC.
REM
REM  First run creates a local virtual environment and installs
REM  PySide6 (1-3 minutes). Later runs start in a few seconds.
REM
REM  ASCII only, CRLF endings, no ^ continuation, no GOTO.
REM ============================================================

cd /d "%~dp0"

set "PY="
where py >nul 2>nul
if not errorlevel 1 set "PY=py -3"
if not defined PY where python >nul 2>nul
if not defined PY if not errorlevel 1 set "PY=python"

if not defined PY echo [ERROR] Python 3 not found.
if not defined PY echo         Install it from https://www.python.org/downloads/
if not defined PY echo         Remember to tick "Add Python to PATH" during setup.
if not defined PY pause
if not defined PY exit /b 1

if exist ".venv\Scripts\python.exe" echo   Virtual environment ready.

if not exist ".venv\Scripts\python.exe" echo   [1/2] First run: creating virtual environment...
if not exist ".venv\Scripts\python.exe" %PY% -m venv .venv
if not exist ".venv\Scripts\python.exe" echo [ERROR] Could not create the virtual environment.
if not exist ".venv\Scripts\python.exe" pause
if not exist ".venv\Scripts\python.exe" exit /b 1

if not exist ".venv\Lib\site-packages\PySide6" echo   [2/2] Installing PySide6 and pandas - needs internet, 1-3 minutes...
if not exist ".venv\Lib\site-packages\PySide6" ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
if not exist ".venv\Lib\site-packages\PySide6" ".venv\Scripts\python.exe" -m pip install --quiet PySide6 pandas openpyxl

echo.
echo   Starting the viewer...
echo.

start "" ".venv\Scripts\pythonw.exe" "app\ui\viewer.py"

endlocal
