# scripts/multi_dpi_render_inner.ps1
# Multi-DPI synthetic render of the OpenScore Lieder corpus.
# Runs detached via WMI from multi_dpi_render_launch.ps1. Logs to logs/multi_dpi_render.log.

Set-Location "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$logFile = "logs\multi_dpi_render.log"
$venvPython = "venv-cu132\Scripts\python.exe"

& $venvPython -m src.data.generate_synthetic `
    --mode render `
    --dpis 94 150 300 `
    --output-dir data\processed\synthetic_multi_dpi `
    --workers 8 `
    *>&1 | Tee-Object -FilePath $logFile
