@echo off
chcp 65001 >nul
setlocal

REM ============================================================
REM  Attention Assessment System - GUI Launcher (ui\ copy)
REM
REM  Identical to C:\project\run_ui.bat except for the location.
REM  This copy keeps the ui\ folder self-contained; the root copy
REM  is there for a one-click start from the project home.
REM  Both cd to the project root first, because the analysis
REM  pipeline uses relative paths such as model\gaze\...
REM
REM  Uses pythonw.exe so the GUI has no console attached; with
REM  python.exe the terminal window IS the app's console and
REM  closing it would kill the GUI.
REM
REM  If the GUI does not appear, run C:\project\run_ui_debug.bat
REM  which keeps the console open and shows the error.
REM ============================================================

REM  %~dp0 is ...\project\ui\ so go one level up to the project root
cd /d "%~dp0.."

set "ENV_NAME=mediapipe_py39"
set "CONDA_BAT="

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

start "" pythonw.exe "ui\app.py"

endlocal
