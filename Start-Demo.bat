@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

:: ── Check virtual environment ───────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ======================================================================
    echo  [ERROR] Virtual environment not found at .venv\
    echo ======================================================================
    echo.
    echo  To set up the environment, open a command prompt in this folder and run:
    echo.
    echo    python -m venv .venv
    echo    .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Check database ──────────────────────────────────────────────────────────
if not exist "data\presento.db" (
    echo [INFO] Initializing database with demo accounts...
    call .venv\Scripts\python.exe -m backend.seed_demo
    echo.
)

:MENU
cls
echo.
echo ======================================================================
echo                        PRESENTO AI VISION DEMO
echo ======================================================================
echo.
echo   1. Laptop Webcam Demo (Recommended for Friends / Offline)
echo   2. Raspberry Pi Camera Demo (Network Stream / Hardware Setup)
echo   Q. Quit
echo.
echo ======================================================================
set /p DEMO_CHOICE="Enter your choice (1, 2, or Q): "

if /i "%DEMO_CHOICE%"=="1" goto WEBCAM_DEMO
if /i "%DEMO_CHOICE%"=="2" goto PI_DEMO
if /i "%DEMO_CHOICE%"=="Q" goto QUIT

echo.
echo [!] Invalid selection. Please enter 1, 2, or Q.
timeout /t 2 >nul
goto MENU

:WEBCAM_DEMO
set CAMERA_SOURCE=webcam
set OPENVINO_DEVICE=AUTO

cls
echo.
echo ======================================================================
echo                    PRESENTO AI VISION - LAPTOP DEMO
echo ======================================================================
echo.
echo  Camera      : Laptop Webcam
echo  AI Device   : Auto-Detect (Intel GPU / CPU Fallback)
echo  Pi Required : No
echo.
echo ----------------------------------------------------------------------
echo  Interactive Controls during Demo:
echo    [1] Mode 1: Face Detection (SCRFD + 5 Landmarks)
echo    [2] Mode 2: Face Recognition (ArcFace + Matcher)
echo    [3] Mode 3: Anti-Spoof (Dual MN3 + MiniFASNet)
echo    [4] Mode 4: Attendance Verification (Gating)
echo    [5] Mode 5: Summary Dashboard and Analytics
echo    [P] Toggle AI Pipeline Architecture Panel
echo    [E] Enroll Unknown Person (In-GUI Interactive Flow)
echo    [R] Retry Recognition / Reset Identity History
echo    [Space] Freeze / Pause Frame
echo    [D] Dump Diagnostic Snapshot to Console
echo    [Q] / [ESC] Stop Demo and Release Camera
echo ----------------------------------------------------------------------
echo.

call .venv\Scripts\python.exe -m edge.demo --source webcam %*

echo.
echo ======================================================================
echo  Demo stopped. Webcam released cleanly.
echo ======================================================================
echo.
pause
exit /b 0

:PI_DEMO
set CAMERA_SOURCE=stream

cls
echo.
echo ======================================================================
echo                 PRESENTO AI VISION - RASPBERRY PI DEMO
echo ======================================================================
echo.
echo  Camera      : Raspberry Pi Network Stream
echo  Stream Port : 5001 (TCP)
echo  Control Port: 5002 (HTTP)
echo  Pi Required : Yes
echo.
echo ----------------------------------------------------------------------
echo  Interactive Controls during Demo:
echo    [1] Mode 1: Face Detection (SCRFD + 5 Landmarks)
echo    [2] Mode 2: Face Recognition (ArcFace + Matcher)
echo    [3] Mode 3: Anti-Spoof (Dual MN3 + MiniFASNet)
echo    [4] Mode 4: Attendance Verification (Gating)
echo    [5] Mode 5: Summary Dashboard and Analytics
echo    [P] Toggle AI Pipeline Architecture Panel
echo    [E] Enroll Unknown Person (In-GUI Interactive Flow)
echo    [R] Retry Recognition / Reset Identity History
echo    [Space] Freeze / Pause Frame
echo    [D] Dump Diagnostic Snapshot to Console
echo    [Q] / [ESC] Stop Demo and Release Camera
echo ----------------------------------------------------------------------
echo.

call .venv\Scripts\python.exe -m edge.demo --source stream %*

echo.
echo ======================================================================
echo  Demo stopped. Camera session released and Pi camera service turned OFF.
echo ======================================================================
echo.
pause
exit /b 0

:QUIT
echo.
echo Exiting launcher.
exit /b 0
