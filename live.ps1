# ====================================================================
# VOJKER LIVE TRIAGE ENGINE WAKE-UP
# ====================================================================

# 1. Aktivoidaan virtuaaliympäristö
$venvPath = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    . $venvPath
} else {
    Write-Host "[WARNING] .venv ei löytynyt!" -ForegroundColor Yellow
}

# 2. Tyhjennetään terminaali puhtaalta pöydältä
Clear-Host

# 3. Käynnistetään Live-moottori
python scripts/triage_engine.py
