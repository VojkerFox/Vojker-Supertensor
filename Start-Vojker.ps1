# ====================================================================
# VOJKER SUPERTENSOR: ENVIRONMENT WAKE-UP ROUTINE (Cpk 3.0)
# ====================================================================

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)

if (-not $isAdmin) {
    Write-Host "=========================================================" -ForegroundColor Red
    Write-Host "[!] CRITICAL ERROR: LISÄOIKEUKSIA VAADITAAN" -ForegroundColor Red
    Write-Host "PostgreSQL-palvelimen käynnistäminen vaatii Admin-oikeudet."
    Write-Host "Käynnistä PowerShell valitsemalla 'Run as Administrator'."
    Write-Host "=========================================================" -ForegroundColor Red
    Pause
    Exit
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   VOJKER SYSTEM INITIALIZATION..." -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

$pgService = Get-Service -Name *postgresql* -ErrorAction SilentlyContinue | Select-Object -First 1

if ($pgService) {
    Write-Host "[SYSTEM] Found Database Engine: $($pgService.Name)" -ForegroundColor Gray
    if ($pgService.Status -ne 'Running') {
        Write-Host "[SYSTEM] Database is offline. Igniting PostgreSQL..." -ForegroundColor Yellow
        Start-Service -Name $pgService.Name -ErrorAction Stop
        Write-Host "[OK] PostgreSQL Engine ONLINE." -ForegroundColor Green
    } else {
        Write-Host "[OK] PostgreSQL Engine is already running." -ForegroundColor Green
    }
} else {
    Write-Host "[WARNING] PostgreSQL service auto-detect failed." -ForegroundColor Red
    Write-Host "Varmista, että PostgreSQL on asennettu tälle koneelle." -ForegroundColor Red
}

$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "[SYSTEM] Activating Python Virtual Environment (.venv)..." -ForegroundColor Gray
    . $venvPath
    Write-Host "[OK] .venv Activated." -ForegroundColor Green
} else {
    Write-Host "[WARNING] .venv not found! Running with global Python." -ForegroundColor Yellow
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "   ALL SYSTEMS GO. LAUNCHING MASTER TESTER." -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Start-Sleep -Seconds 1

python master_tester.py

Pause
