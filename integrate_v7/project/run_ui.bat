@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  Attention Assessment System - GUI Launcher
REM  Double-click this file to start the GUI.
REM
REM  Uses pythonw.exe (not python.exe) so the GUI has no console
REM  attached. That matters: with python.exe the terminal window
REM  IS the app's console, so closing it kills the GUI. With
REM  pythonw.exe this launcher window closes right away and the
REM  GUI keeps running on its own.
REM
REM  If the GUI does not appear, run run_ui_debug.bat instead -
REM  that one keeps the console open and shows the error.
REM
REM  Written in a deliberately plain style: ASCII only, no GOTO
REM  or labels, no multi-line ( ) blocks. Those are the parts
REM  that misbehave with LF line endings or a mismatched
REM  codepage, and a broken .bat closes before you can read it.
REM
REM  Chinese documentation: README.md and ui\README_UI.md
REM ============================================================

cd /d "%~dp0"

set "ENV_NAME=mediapipe_py39"
set "CONDA_BAT="

REM --- Find conda.bat -----------------------------------------
REM  Double-clicking opens a plain cmd.exe where conda is NOT on
REM  PATH (only Anaconda Prompt sets that up), so look for the
REM  launcher directly in the usual install locations.
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"

if not defined CONDA_BAT echo.
if not defined CONDA_BAT echo [ERROR] conda.bat not found in the usual locations.
if not defined CONDA_BAT echo         Open "Anaconda Prompt" from the Start menu and run:
if not defined CONDA_BAT echo             conda activate %ENV_NAME%
if not defined CONDA_BAT echo             cd /d "%CD%"
if not defined CONDA_BAT echo             python ui\app.py
if not defined CONDA_BAT echo.
if not defined CONDA_BAT pause

if not defined CONDA_BAT exit /b 1

call "%CONDA_BAT%" activate %ENV_NAME%

if errorlevel 1 echo.
if errorlevel 1 echo [ERROR] Could not activate conda env: %ENV_NAME%
if errorlevel 1 echo         Run "conda env list" to see what is available.
if errorlevel 1 echo.
if errorlevel 1 pause
if errorlevel 1 exit /b 1

REM  start "" ... launches the GUI as an independent process and
REM  lets this script exit immediately, so no window is left behind.
start "" pythonw.exe "ui\app.py"

endlocal
