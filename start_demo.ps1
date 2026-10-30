# ==============================================================================
# Presento AI Vision Demo — Windows PowerShell Launcher
# Starts the live biometric recognition demo with interactive HUD overlay
# ==============================================================================
[CmdletBinding()]
param (
    [string]$Source = $null,
    [ValidateRange(1, 5)]
    [int]$Mode = 1,
    [switch]$ShowPipeline,
    [switch]$Headless
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "             PRESENTO AI VISION - LIVE DEMO LAUNCHER                  " -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Initial Mode : Mode $Mode" -ForegroundColor Yellow
if ($Source) {
    Write-Host " Camera Source: $Source" -ForegroundColor Yellow
}
Write-Host ""

# 1. Virtual environment check
$PythonExe = Join-Path $ScriptDir ".venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    Write-Host "[ERROR] Virtual environment not found at .venv\" -ForegroundColor Red
    Write-Host "Run: python -m venv .venv; .venv\Scripts\pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}

# 2. Raspberry Pi Stream Discovery (Stream mode only)
$effectiveSource = if ($Source) { $Source.ToLower().Trim() } elseif ($env:CAMERA_SOURCE) { $env:CAMERA_SOURCE.ToLower().Trim() } else { "stream" }
$isLocalWebcam = $effectiveSource -in @("webcam", "laptop", "local") -or ($effectiveSource -match "^\d+$")

if (-not $isLocalWebcam) {
    Write-Host "[*] Probing Raspberry Pi camera stream..." -ForegroundColor Cyan
    $discArgs = @("-m", "edge.discovery", "--timeout", "0.8")
    if ($Source -and ($Source.ToLower().Trim() -notin @("stream", "pi", "network"))) {
        $discArgs += @("--host", $Source.Split(":")[0])
    }
    & $PythonExe $discArgs
    Write-Host ""
} else {
    Write-Host "[*] Local webcam selected - skipping Raspberry Pi network discovery." -ForegroundColor Green
    Write-Host ""
}

# 3. Launch Demo
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray
Write-Host " Interactive Keyboard Controls:" -ForegroundColor White
Write-Host "   [1] Mode 1: Face Detection (SCRFD + 5 Landmarks)" -ForegroundColor Cyan
Write-Host "   [2] Mode 2: Face Recognition (ArcFace + Matcher)" -ForegroundColor Cyan
Write-Host "   [3] Mode 3: Anti-Spoof (Dual MN3 + MiniFASNet)" -ForegroundColor Cyan
Write-Host "   [4] Mode 4: Attendance Verification (Gating)" -ForegroundColor Cyan
Write-Host "   [5] Mode 5: Summary Dashboard & Analytics" -ForegroundColor Cyan
Write-Host "   [P] Toggle AI Pipeline Architecture Panel" -ForegroundColor Yellow
Write-Host "   [Space] Freeze / Pause Live Frame" -ForegroundColor Yellow
Write-Host "   [D] Dump Diagnostic Snapshot to Console" -ForegroundColor Yellow
Write-Host "   [Q] / [ESC] Stop Demo & Release Camera" -ForegroundColor Red
Write-Host "----------------------------------------------------------------------" -ForegroundColor Gray
Write-Host ""

$demoArgs = @("-m", "edge.demo", "--mode", "$Mode")
if ($Source) {
    $demoArgs += @("--source", "$Source")
}
if ($ShowPipeline) {
    $demoArgs += @("--debug-pipeline")
}
if ($Headless) {
    $demoArgs += @("--headless")
}

& $PythonExe $demoArgs

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host " Demo stopped. Camera hardware released and Pi service turned OFF." -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
