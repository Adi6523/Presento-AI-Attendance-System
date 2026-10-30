@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo.
echo ======================================================================
echo             PRESENTO - STUDENT ENROLLMENT LAUNCHER
echo ======================================================================
echo.

:: ── Check virtual environment ───────────────────────────────────────────────
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .venv\
    echo.
    echo  To fix, open a terminal in this folder and run:
    echo    python -m venv .venv
    echo    .venv\Scripts\pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

:: ── Interactive Student Information Input ──────────────────────────────────
echo ----------------------------------------------------------------------
echo  Please enter the student details for Smart Face Enrollment.
echo  (Press Q or leave blank to cancel)
echo ----------------------------------------------------------------------
echo.

set /p STUDENT_ID="Student ID (e.g. DEMO001): "
if "%STUDENT_ID%"=="" (
    echo [INFO] Enrollment cancelled.
    pause
    exit /b 0
)
if /i "%STUDENT_ID%"=="Q" (
    echo [INFO] Enrollment cancelled.
    pause
    exit /b 0
)

set /p STUDENT_NAME="Student Name (e.g. Demo Student): "
if "%STUDENT_NAME%"=="" (
    echo [INFO] Enrollment cancelled: Student Name is required.
    pause
    exit /b 0
)
if /i "%STUDENT_NAME%"=="Q" (
    echo [INFO] Enrollment cancelled.
    pause
    exit /b 0
)

echo.
echo ----------------------------------------------------------------------
echo  Target Student: %STUDENT_NAME% (%STUDENT_ID%)
echo ----------------------------------------------------------------------
echo.

:: ── Camera Selection ───────────────────────────────────────────────────────
echo  Camera Source:
echo    [1] Laptop Webcam (Default / Offline)
echo    [2] Raspberry Pi Camera (Network Stream)
set /p CAM_CHOICE="Select camera [1 or 2, default=1]: "

if "%CAM_CHOICE%"=="2" (
    set CAMERA_ARG=--source stream
    echo [INFO] Selected Raspberry Pi Stream.
) else (
    set CAMERA_ARG=--source webcam
    echo [INFO] Selected Laptop Webcam.
)

echo.
echo ======================================================================
echo  Starting Face Enrollment Session...
echo ======================================================================
echo  Smart Guided Enrollment Workflow:
echo    - Follow on-screen pose instructions (Front, Left, Right, Up, Down)
echo    - Quality filter automatically rejects blurry or small faces
echo    - Duplicate filter ensures sample diversity across all captures
echo    - Press [C] or [SPACE] to capture each sample
echo    - Press [S] to save early once at least 8 samples are captured
echo    - Press [Q] or [ESC] to cancel without saving
echo ======================================================================
echo.

:: ── Run the Smart Enrollment Engine ────────────────────────────────────────
call .venv\Scripts\python.exe -m edge.enroll_face --id "%STUDENT_ID%" --name "%STUDENT_NAME%" %CAMERA_ARG%
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ======================================================================
if %EXIT_CODE% EQU 0 (
    echo  [SUCCESS] Student %STUDENT_NAME% (%STUDENT_ID%) enrolled successfully!
    echo  Updated embeddings saved in data\face_embeddings.npz
) else (
    echo  [INFO] Enrollment session ended or cancelled.
)
echo  Camera session released.
echo ======================================================================
echo.

:: ── Show updated enrollment store summary ──────────────────────────────────
call .venv\Scripts\python.exe -m edge.verify_enrollment
echo.
pause
