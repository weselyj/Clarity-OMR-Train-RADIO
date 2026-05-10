# scripts/multi_dpi_render_inner.ps1
# Multi-DPI synthetic render of the OpenScore Lieder corpus.
# Runs detached via WMI from multi_dpi_render_launch.ps1.
#
# Notes:
# - WMI-detached PowerShell processes have no console; PS pipelines (Tee-Object,
#   *>&1) silently drop output. Use cmd.exe /c redirect for log capture, and
#   pass python -u for unbuffered stdout so the log streams in real time.
# - The default --input-manifest at src\data\manifests\master_manifest.jsonl
#   contains 2400 entries (mixed primus + lieder) but only 600 are usable for
#   Verovio rendering. Pass an intentionally-nonexistent path so the script
#   falls through to scan_default_sources, which picks up all 1462 lieder MXLs
#   via the data\Lieder-main\scores junction (created separately).

Set-Location "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO"

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$logFile = "logs\multi_dpi_render.log"
$venvPython = "venv-cu132\Scripts\python.exe"

& venv-cu132\Scripts\Activate.ps1

$pyArgs = @(
    "-u",
    "-m", "src.data.generate_synthetic",
    "--mode", "render",
    "--dpis", "94", "150", "300",
    "--output-dir", "data\processed\synthetic_multi_dpi",
    "--workers", "8",
    "--input-manifest", "data\__nonexistent_force_full_scan__.jsonl"
) -join " "

cmd.exe /c "`"$venvPython`" $pyArgs > `"$logFile`" 2>&1"
