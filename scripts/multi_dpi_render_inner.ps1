# scripts/multi_dpi_render_inner.ps1
# Multi-DPI synthetic render of the OpenScore Lieder corpus.
# Runs detached via WMI from multi_dpi_render_launch.ps1.
#
# Note: WMI-detached PowerShell processes have no console, so PowerShell
# pipelines (Tee-Object, *>&1) silently drop output. We use cmd.exe /c
# redirection instead, which works in detached contexts. Also pass -u to
# python for unbuffered stdout so the log streams in real time.

Set-Location "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$logFile = "logs\multi_dpi_render.log"
$venvPython = "venv-cu132\Scripts\python.exe"

# Activate venv (sets PATH, etc.) — still useful for ImageMagick subprocess discovery
& venv-cu132\Scripts\Activate.ps1

# Build the python command and dispatch via cmd.exe so > redirection works
$pyArgs = @(
    "-u",
    "-m", "src.data.generate_synthetic",
    "--mode", "render",
    "--dpis", "94", "150", "300",
    "--output-dir", "data\processed\synthetic_multi_dpi",
    "--workers", "8"
) -join " "

# cmd.exe /c handles file redirection cleanly in detached contexts
cmd.exe /c "`"$venvPython`" $pyArgs > `"$logFile`" 2>&1"
