@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  Attention Assessment System - GUI Launcher (DEBUG)
REM
REM  Same as run_ui.bat but uses python.exe instead of pythonw.exe,
REM  so a console stays attached and prints everything the app
REM  writes. Use this when the GUI will not start and you need to
REM  see the error message.
REM
REM  Trade-off: the console window IS the app's console, so
REM  closing it kills the GUI. For normal use run run_ui.bat.
REM
REM  Plain style on purpose: ASCII only, no GOTO or labels, no
REM  multi-line ( ) blocks. The script always reaches the PAUSE
REM  at the bottom, so the window can never vanish unread.
REM ============================================================

cd /d "%~dp0"

set "ENV_NAME=mediapipe_py39"
set "CONDA_BAT="

if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"

echo.
echo   project : %CD%
echo   env     : %ENV_NAME%
echo   conda   : %CONDA_BAT%
echo.

if not defined CONDA_BAT echo [ERROR] conda.bat not found in the usual locations.
if not defined CONDA_BAT echo         Open "Anaconda Prompt" from the Start menu and run:
if not defined CONDA_BAT echo             conda activate %ENV_NAME%
if not defined CONDA_BAT echo             cd /d "%CD%"
if not defined CONDA_BAT echo             python ui\app.py

if defined CONDA_BAT echo   Starting... first launch loads PyTorch, about 10-20 seconds.
if defined CONDA_BAT echo.
if defined CONDA_BAT call "%CONDA_BAT%" activate %ENV_NAME%
if defined CONDA_BAT if errorlevel 1 echo [ERROR] Could not activate env %ENV_NAME%. Run "conda env list" to check.
if defined CONDA_BAT if not errorlevel 1 python ui\app.py

echo.
echo   ---------------------------------------------------------
echo   Press any key to close this window...
pause >nul
endlocal
