@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM ============================================================
REM  Attention Assessment System - portable launcher
REM
REM  No installation needed. No Anaconda needed. Just double-click.
REM
REM  First run does three one-time setup steps (5-15 minutes):
REM     1. extract the bundled Python environment
REM     2. conda-unpack - fixes paths inside that environment
REM     3. copy the Whisper / EasyOCR model caches into your
REM        user folder, so nothing has to be downloaded
REM  Every run after that starts in about 15 seconds.
REM
REM  Requirements: Windows 10/11 64-bit, NVIDIA GPU with an
REM  up-to-date driver. See README_FIRST.txt.
REM
REM  ASCII only, no GOTO / labels, no multi-line ( ) blocks.
REM ============================================================

cd /d "%~dp0"
set "ROOT=%CD%"

echo.
echo ============================================================
echo   Attention Assessment System
echo ============================================================
echo   folder : %ROOT%
echo.

REM ---------- sanity check --------------------------------------
if not exist "%ROOT%\app\ui\app.py" echo [ERROR] app\ui\app.py not found.
if not exist "%ROOT%\app\ui\app.py" echo         Make sure you extracted the WHOLE archive, keeping
if not exist "%ROOT%\app\ui\app.py" echo         this file next to the app and env folders.
if not exist "%ROOT%\app\ui\app.py" pause
if not exist "%ROOT%\app\ui\app.py" exit /b 1

REM ---------- step 1: extract the environment --------------------
REM  Shipped as env.tar.gz so the outer archive stays small and
REM  compresses fast. Windows 10 build 17063+ has tar.exe built in.
if exist "%ROOT%\env\python.exe" echo [1/3] Environment already extracted - skipping.

if not exist "%ROOT%\env\python.exe" echo [1/3] Extracting Python environment - this takes 5-15 minutes...
if not exist "%ROOT%\env\python.exe" if not exist "%ROOT%\env.tar.gz" echo [ERROR] env.tar.gz is missing from this folder.
if not exist "%ROOT%\env\python.exe" if not exist "%ROOT%\env.tar.gz" pause
if not exist "%ROOT%\env\python.exe" if not exist "%ROOT%\env.tar.gz" exit /b 1

if not exist "%ROOT%\env\python.exe" mkdir "%ROOT%\env" 2>nul
if not exist "%ROOT%\env\python.exe" tar -xf "%ROOT%\env.tar.gz" -C "%ROOT%\env"

if not exist "%ROOT%\env\python.exe" echo [ERROR] Extraction failed.
if not exist "%ROOT%\env\python.exe" echo         Your Windows may be too old to include tar.exe.
if not exist "%ROOT%\env\python.exe" echo         Install 7-Zip, extract env.tar.gz into a folder
if not exist "%ROOT%\env\python.exe" echo         named "env" next to this file, then run this again.
if not exist "%ROOT%\env\python.exe" pause
if not exist "%ROOT%\env\python.exe" exit /b 1

REM ---------- step 2: conda-unpack -------------------------------
REM  A packed environment still contains absolute paths from the
REM  machine that built it. conda-unpack rewrites them. It must run
REM  exactly once, after extraction, with the env activated.
if exist "%ROOT%\env\.unpacked_ok" echo [2/3] Environment already prepared - skipping.

if not exist "%ROOT%\env\.unpacked_ok" echo [2/3] Preparing environment (conda-unpack)...
if not exist "%ROOT%\env\.unpacked_ok" call "%ROOT%\env\Scripts\activate.bat"
if not exist "%ROOT%\env\.unpacked_ok" "%ROOT%\env\Scripts\conda-unpack.exe"
if not exist "%ROOT%\env\.unpacked_ok" if errorlevel 1 echo [ERROR] conda-unpack failed.
if not exist "%ROOT%\env\.unpacked_ok" if errorlevel 1 pause
if not exist "%ROOT%\env\.unpacked_ok" if errorlevel 1 exit /b 1
if not exist "%ROOT%\env\.unpacked_ok" echo ok > "%ROOT%\env\.unpacked_ok"

REM ---------- step 3: install the model caches -------------------
REM  Whisper and EasyOCR look for their weights in the user's home
REM  folder, not inside the environment, so copy them there once.
REM  Existing files are never overwritten (/XN /XO /XC on robocopy
REM  would be finer-grained; /XC /XN /XO keeps whatever is newer).
echo [3/3] Installing model caches...

if exist "%ROOT%\assets\whisper" mkdir "%USERPROFILE%\.cache\whisper" 2>nul
if exist "%ROOT%\assets\whisper" robocopy "%ROOT%\assets\whisper" "%USERPROFILE%\.cache\whisper" /E /XC /XN /XO /NFL /NDL /NJH /NJS /NP >nul

if exist "%ROOT%\assets\EasyOCR" mkdir "%USERPROFILE%\.EasyOCR" 2>nul
if exist "%ROOT%\assets\EasyOCR" robocopy "%ROOT%\assets\EasyOCR" "%USERPROFILE%\.EasyOCR" /E /XC /XN /XO /NFL /NDL /NJH /NJS /NP >nul

if exist "%ROOT%\assets\Ultralytics" mkdir "%APPDATA%\Ultralytics" 2>nul
if exist "%ROOT%\assets\Ultralytics" robocopy "%ROOT%\assets\Ultralytics" "%APPDATA%\Ultralytics" /E /XC /XN /XO /NFL /NDL /NJH /NJS /NP >nul

REM ---------- launch ---------------------------------------------
REM  Activate so the env's Library\bin is on PATH (CUDA and Qt DLLs
REM  live there), then launch with pythonw.exe so no console window
REM  stays attached - closing a console would kill the GUI.
echo.
echo   Starting the interface - the window appears in 10-20 seconds.
echo.

call "%ROOT%\env\Scripts\activate.bat"
cd /d "%ROOT%\app"
start "" "%ROOT%\env\pythonw.exe" "ui\app.py"

endlocal
