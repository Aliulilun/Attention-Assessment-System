@echo off
chcp 65001 >nul
setlocal EnableExtensions

REM ============================================================
REM  Build the cross-platform REPORT VIEWER package.
REM
REM  Much smaller and simpler than build_release.bat: the viewer
REM  only reads report .txt files, so it needs no models, no
REM  PyTorch and no conda environment. It runs on macOS, Windows
REM  and Linux, with or without a GPU.
REM
REM  Output: C:\project\release_viewer\  (a few hundred KB)
REM  Zip that folder and send it.
REM
REM  ASCII only, CRLF endings, no ^ continuation, no GOTO.
REM ============================================================

set "PKG_DIR=%~dp0"
cd /d "%PKG_DIR%.."
set "PROJECT_DIR=%CD%"
set "OUT_DIR=%PROJECT_DIR%\release_viewer"

echo.
echo ============================================================
echo   Building report viewer package
echo ============================================================
echo   project : %PROJECT_DIR%
echo   output  : %OUT_DIR%
echo.

echo [1/4] Preparing output folder...
if exist "%OUT_DIR%" rmdir /s /q "%OUT_DIR%"
mkdir "%OUT_DIR%"
mkdir "%OUT_DIR%\app"
mkdir "%OUT_DIR%\reports"

REM  Only the UI package is needed. modules\stage_scoring is pulled
REM  in by monitor_page, which the viewer never imports - but ship
REM  modules\ anyway so a later version can reuse it without
REM  another round of packaging. It is only a few hundred KB.
echo [2/4] Copying viewer source...
robocopy "%PROJECT_DIR%\ui" "%OUT_DIR%\app\ui" /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ /XF *.pyc groups.json
robocopy "%PROJECT_DIR%\modules" "%OUT_DIR%\app\modules" /E /NFL /NDL /NJH /NJS /NP /XD __pycache__ gaze_estimation /XF *.pyc

if errorlevel 8 echo [ERROR] robocopy failed.
if errorlevel 8 pause
if errorlevel 8 exit /b 1

echo [3/4] Copying launchers and instructions...
copy /y "%PKG_DIR%viewer_template\START_VIEWER.bat" "%OUT_DIR%\START_VIEWER.bat"
copy /y "%PKG_DIR%viewer_template\START_VIEWER.command" "%OUT_DIR%\START_VIEWER.command"
REM  Wildcard copy: the instructions file has a Chinese name and
REM  this .bat must stay pure ASCII, so never type it literally.
copy /y "%PKG_DIR%viewer_template\*.txt" "%OUT_DIR%\"
echo Put analysis report .txt files here. > "%OUT_DIR%\reports\PUT_REPORTS_HERE.txt"

echo [4/4] Done.
echo.
echo ============================================================
echo   DONE
echo ============================================================
echo   Viewer folder: %OUT_DIR%
echo.
echo   Before zipping:
echo     - decide whether to put any .txt reports in reports\
echo       (they contain full speech transcripts - check your
echo        research ethics rules first)
echo.
echo   IMPORTANT for macOS recipients:
echo     Zipping on Windows loses the executable bit on
echo     START_VIEWER.command. The instructions in the txt file
echo     tell them to run it via  bash START_VIEWER.command
echo     which works regardless.
echo.
pause
endlocal
