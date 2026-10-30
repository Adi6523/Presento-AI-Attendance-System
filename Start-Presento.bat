@echo off
setlocal
cd /d "%~dp0"

echo.
echo  PRESENTO - Smart Attendance System
echo  ====================================
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

:: ── Check Node / npm ────────────────────────────────────────────────────────
where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js / npm not found.
    echo  Install from https://nodejs.org then try again.
    echo.
    pause
    exit /b 1
)

:: ── Install frontend packages once ──────────────────────────────────────────
if not exist "frontend\node_modules" (
    echo [INFO] Installing frontend packages for the first time...
    call npm --prefix frontend install
    if errorlevel 1 (
        echo [ERROR] npm install failed. Check your internet connection.
        pause
        exit /b 1
    )
    echo.
)

:: ── Initialize and seed demo sessions if missing ─────────────────────────────
if not exist "data\presento.db" (
    echo [INFO] Initializing database with demo accounts and active sessions...
    call .venv\Scripts\python.exe -m backend.seed_demo
    echo.
)

:: ── Pi Camera Stream Discovery Check ───────────────────────────────────────
echo [INFO] Discovering Raspberry Pi Camera Stream...
call .venv\Scripts\python.exe -m edge.discovery --timeout 0.8
echo.

:: ── Start web servers (Camera remains strictly OFF until session starts) ──────
echo [1/2] Starting Presento API on http://127.0.0.1:8000 ...
start "Presento API" /d "%~dp0" cmd /k ".venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [2/2] Starting Presento Dashboard on http://localhost:5173 ...
start "Presento Dashboard" /d "%~dp0frontend" cmd /k "npm run dev"

:: ── Open browser after servers are ready ────────────────────────────────────
echo.
echo Waiting for servers to initialize (6 seconds)...
timeout /t 6 /nobreak >nul
start "" "http://localhost:5173"

echo.
echo ======================================================================
echo  Presento is fully active!
echo  ------------------------------------------------------------------
echo   Dashboard      : http://localhost:5173
echo   API            : http://127.0.0.1:8000
echo   Camera Mode    : On-Demand (OFF during idle, activates for attendance/enrollment)
echo   Camera Status  : Standby / Hardware Released (C270 LED is OFF)
echo ======================================================================
echo.
echo Keep the Presento service windows open while using the system.
echo You can minimize or close THIS launcher window.
echo.
pause
