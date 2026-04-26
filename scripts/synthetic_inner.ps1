# Runs detached via WMI; this script's parent (SSH session) has no handle on it.
# Renders the full synthetic page corpus from lieder MusicXML inputs at seed 1337.
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
# cairocffi (pulled in by cairosvg) needs libcairo-2.dll on PATH; darktable ships one.
$env:PATH = "C:\Program Files\darktable\bin;" + $env:PATH
$py = Join-Path $repo "venv\Scripts\python.exe"
$pyArgs = @(
    "-u", "-m", "src.data.generate_synthetic",
    "--mode", "render",
    "--write-png",
    "--workers", "8",
    "--seed", "1337",
    "--input-manifest", "src/data/manifests/master_manifest_full.jsonl"
)
Remove-Item "logs/synthetic_render.log","logs/synthetic_render.err","logs/synthetic_render.pid","logs/synthetic_render_wrapper.log" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "logs/synthetic_render.log" `
    -RedirectStandardError  "logs/synthetic_render.err" `
    -NoNewWindow `
    -PassThru
$proc.Id | Out-File -FilePath "logs/synthetic_render.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/synthetic_render_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/synthetic_render_wrapper.log" -Append
