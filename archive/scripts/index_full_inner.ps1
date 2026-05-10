# Runs detached via WMI; this script's parent (SSH session) has no handle on it.
# Launches the full-corpus dataset indexer with no per-dataset cap.
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv\Scripts\python.exe"
$pyArgs = @(
    "-u", "-m", "src.data.index",
    "--data-root", "data",
    "--split-config", "configs/splits.yaml",
    "--output-manifest", "src/data/manifests/master_manifest_full.jsonl",
    "--output-summary",  "src/data/manifests/master_manifest_full_summary.json"
)
Remove-Item "logs/index_full.log","logs/index_full.err","logs/index_full.pid","logs/index_full_wrapper.log" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "logs/index_full.log" `
    -RedirectStandardError  "logs/index_full.err" `
    -NoNewWindow `
    -PassThru
$proc.Id | Out-File -FilePath "logs/index_full.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/index_full_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/index_full_wrapper.log" -Append
