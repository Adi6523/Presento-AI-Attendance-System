# ==============================================================================
# Presento Smart Attendance - Windows Unified Launcher (PowerShell)
# Starts API, Frontend Dashboard, and Edge Face Recognition Runner with Pi Discovery
# ==============================================================================
[CmdletBinding()]
param (
    [ValidateSet("Full", "RunnerOnly", "WebOnly", "Demo")]
    [string]$Mode = "Full",
    [switch]$SkipBrowser,
    [string]$Source = $null
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "                PRESENTO - SMART ATTENDANCE SYSTEM                   " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Mode: $Mode" -ForegroundColor Yellow
Write-Host ""

# 1. Virtual environment check
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at .venv\" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 2. Database seed check
$DbPath = Join-Path $ScriptDir "data\presento.db"
if (-not (Test-Path $DbPath)) {
    Write-Host "[INFO] Initializing database with demo accounts..." -ForegroundColor Green
    & $PythonExe -m backend.seed_demo
}

# 3. Raspberry Pi Stream Discovery
if ($Mode -in @("Full", "RunnerOnly", "Demo")) {
    Write-Host "[*] Probing Raspberry Pi camera stream..." -ForegroundColor Cyan
    $discArgs = @("-m", "edge.discovery", "--timeout", "0.8")
    if ($Source -and ($Source.ToLower().Trim() -notin @("stream", "pi", "network"))) {
        $discArgs += @("--host", $Source.Split(":")[0])
    }
    & $PythonExe $discArgs
    Write-Host ""
}

# 4. Launch Services based on Mode
if ($Mode -in @("Full", "WebOnly")) {
    Write-Host "[1] Starting Presento API on http://127.0.0.1:8000..." -ForegroundColor Green
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"$PythonExe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`"" -WorkingDirectory $ScriptDir -WindowStyle Normal

    Write-Host "[2] Starting Presento Dashboard on http://localhost:5173..." -ForegroundColor Green
    $FrontendDir = Join-Path $ScriptDir "frontend"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"npm run dev`"" -WorkingDirectory $FrontendDir -WindowStyle Normal
}

if ($Mode -eq "RunnerOnly") {
    Write-Host "[3] Starting Presento Camera Recognition Runner (On-Demand)..." -ForegroundColor Green
    $runnerArgs = "-m edge.runner"
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"$PythonExe $runnerArgs`"" -WorkingDirectory $ScriptDir -WindowStyle Normal
}

if ($Mode -eq "Demo") {
    Write-Host "[3] Starting Presento AI Vision Demo Mode (Interactive HUD)..." -ForegroundColor Green
    $demoArgs = "-m edge.demo"
    if ($Source) {
        $demoArgs += " --source $Source"
    }
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k `"$PythonExe $demoArgs`"" -WorkingDirectory $ScriptDir -WindowStyle Normal
}

# 5. Open Browser
if ($Mode -in @("Full", "WebOnly") -and -not $SkipBrowser) {
    Write-Host ""
    Write-Host "Waiting 5 seconds for web services to initialize..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:5173"
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Presento is running!" -ForegroundColor Green
Write-Host "  - Dashboard      : http://localhost:5173" -ForegroundColor White
Write-Host "  - API Docs       : http://127.0.0.1:8000/docs" -ForegroundColor White
Write-Host "  - Camera Mode    : On-Demand (activates only during attendance or enrollment)" -ForegroundColor White
Write-Host "  - Camera Status  : Standby / Hardware Released (C270 LED is OFF)" -ForegroundColor White
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Keep the service terminal windows open. You may minimize this console." -ForegroundColor Gray
