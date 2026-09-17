@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM ============================================================
REM  Build a self-contained release, using conda-pack.
REM
REM  Run this ON THE DEVELOPMENT MACHINE (the one with a working
REM  mediapipe_py39 environment). It produces C:\project\release\
REM  which you then zip and upload.
REM
REM  Expect 15-40 minutes and about 20 GB of free disk space.
REM
REM  Style rules for this file - all learned the hard way:
REM    * ASCII only          - cmd parses .bat with the console
REM                            codepage; UTF-8 Chinese breaks it
REM    * CRLF line endings   - LF-only .bat files get mis-parsed
REM    * no ^ continuation   - ^ followed by LF makes cmd lose its
REM                            place and silently stop mid-script
REM    * no GOTO / labels    - same class of problem
REM    * always reach PAUSE  - the window must never vanish unread
REM ============================================================

set "ENV_NAME=mediapipe_py39"
set "PKG_DIR=%~dp0"
cd /d "%PKG_DIR%.."
set "PROJECT_DIR=%CD%"
set "OUT_DIR=%PROJECT_DIR%\release"

echo.
echo ============================================================
echo   Building release package
echo ============================================================
echo   project : %PROJECT_DIR%
echo   env     : %ENV_NAME%
echo   output  : %OUT_DIR%
echo.

set "CONDA_BAT="
if exist "%USERPROFILE%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%USERPROFILE%\miniconda3\condabin\conda.bat" set "CONDA_BAT=%USERPROFILE%\miniconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%ProgramData%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%ProgramData%\anaconda3\condabin\conda.bat"
if not defined CONDA_BAT if exist "%LOCALAPPDATA%\anaconda3\condabin\conda.bat" set "CONDA_BAT=%LOCALAPPDATA%\anaconda3\condabin\conda.bat"

if not defined CONDA_BAT echo [ERROR] conda.bat not found.
if not defined CONDA_BAT pause
if not defined CONDA_BAT exit /b 1

echo   conda   : %CONDA_BAT%
echo.

echo [1/6] Preparing output folder...
if exist "%OUT_DIR%" rmdir /s /q "%OUT_DIR%"
mkdir "%OUT_DIR%"
mkdir "%OUT_DIR%\app"
mkdir "%OUT_DIR%\assets"

REM  /XD excludes directories by name at ANY depth, so "video" and
REM  "output" also cover hurry\video and hurry\output. Clinical
REM  recordings and existing analysis results never leave this box.
REM  summarize\files and summarize\output are excluded by full path:
REM  they hold ~70 reports containing full speech transcripts.
REM  Kept on ONE line on purpose - see the style rules above.
echo [2/6] Copying source code and models...
robocopy "%PROJECT_DIR%" "%OUT_DIR%\app" /E /NFL /NDL /NJH /NJS /NP /XD video output release packaging .git .vscode __pycache__ .venv "%PROJECT_DIR%\summarize\files" "%PROJECT_DIR%\summarize\output" /XF *.pyc groups.json speech_cache.json

if errorlevel 8 echo [ERROR] robocopy failed.
if errorlevel 8 pause
if errorlevel 8 exit /b 1

mkdir "%OUT_DIR%\app\hurry\video" 2>nul
mkdir "%OUT_DIR%\app\hurry\output" 2>nul
mkdir "%OUT_DIR%\app\video" 2>nul
mkdir "%OUT_DIR%\app\output" 2>nul
mkdir "%OUT_DIR%\app\summarize\files" 2>nul
mkdir "%OUT_DIR%\app\summarize\output" 2>nul
echo Put the videos you want to analyse in THIS folder. > "%OUT_DIR%\app\hurry\video\PUT_VIDEOS_HERE.txt"
echo Analysis results (.mp4 and .txt) will appear here. > "%OUT_DIR%\app\hurry\output\RESULTS_APPEAR_HERE.txt"
echo Put event_record .txt files here, then run summarize.py > "%OUT_DIR%\app\summarize\files\PUT_REPORTS_HERE.txt"

REM  conda-pack only captures the environment itself. Whisper and
REM  EasyOCR download their weights to the user's home folder on
REM  first run, so they must be shipped separately.
REM  Copy ONLY large-v3.pt - the model the pipeline actually loads.
REM  A dev machine often has base / medium / large-v2 lying around
REM  from earlier experiments; shipping those wastes several GB.
echo [3/6] Collecting Whisper / EasyOCR / Ultralytics caches...

set "WHISPER_PT=%USERPROFILE%\.cache\whisper\large-v3.pt"
if exist "%WHISPER_PT%" mkdir "%OUT_DIR%\assets\whisper" 2>nul
if exist "%WHISPER_PT%" copy /y "%WHISPER_PT%" "%OUT_DIR%\assets\whisper\" >nul
if exist "%WHISPER_PT%" echo   Whisper large-v3.pt copied.
if not exist "%WHISPER_PT%" echo   [WARN] large-v3.pt not found - recipient will download it on first run.

if exist "%USERPROFILE%\.EasyOCR" mkdir "%OUT_DIR%\assets\EasyOCR" 2>nul
if exist "%USERPROFILE%\.EasyOCR" robocopy "%USERPROFILE%\.EasyOCR" "%OUT_DIR%\assets\EasyOCR" /E /NFL /NDL /NJH /NJS /NP >nul
if not exist "%USERPROFILE%\.EasyOCR" echo   [WARN] EasyOCR cache not found - recipient will download it on first run.

if exist "%APPDATA%\Ultralytics" mkdir "%OUT_DIR%\assets\Ultralytics" 2>nul
if exist "%APPDATA%\Ultralytics" robocopy "%APPDATA%\Ultralytics" "%OUT_DIR%\assets\Ultralytics" /E /NFL /NDL /NJH /NJS /NP >nul

echo [4/6] Making sure conda-pack is available...
call "%CONDA_BAT%" activate base
if errorlevel 1 echo [ERROR] Could not activate base environment.
if errorlevel 1 pause
if errorlevel 1 exit /b 1

python -m pip install --quiet conda-pack
if errorlevel 1 echo [ERROR] pip install conda-pack failed.
if errorlevel 1 pause
if errorlevel 1 exit /b 1

REM  --ignore-missing-files tolerates packages whose file list no
REM  longer matches disk (common after pip upgrades in a conda env).
echo [5/6] Packing conda environment - this takes 15-30 minutes...
echo.
conda pack -n %ENV_NAME% -o "%OUT_DIR%\env.tar.gz" --ignore-missing-files --force
if errorlevel 1 echo [ERROR] conda pack failed. See the message above.
if errorlevel 1 pause
if errorlevel 1 exit /b 1

echo [6/6] Copying launcher and instructions...
copy /y "%PKG_DIR%release_template\START_UI.bat" "%OUT_DIR%\START_UI.bat"
copy /y "%PKG_DIR%release_template\README_FIRST.txt" "%OUT_DIR%\README_FIRST.txt"

echo.
echo ============================================================
echo   DONE
echo ============================================================
echo   Release folder: %OUT_DIR%
echo.
echo   Check before zipping:
echo     - env.tar.gz is several GB (tiny means conda pack failed)
echo     - assets\whisper\large-v3.pt exists
echo     - START_UI.bat and README_FIRST.txt are present
echo     - app\hurry\video and app\summarize\files hold only the
echo       placeholder .txt files - no clinical recordings
echo.
echo   Then right-click the release folder and compress it.
echo.
pause
endlocal
