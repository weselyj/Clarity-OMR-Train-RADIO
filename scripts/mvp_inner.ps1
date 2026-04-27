# Runs detached via WMI; this script's parent has no handle on it.
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv\Scripts\python.exe"
$pyArgs = @(
    "-u", "src/train/train.py",
    "--stage-configs", "configs/archived_stage2_experiments/train_stage2_radio_mvp.yaml",
    "--mode", "execute",
    "--checkpoint-dir", "checkpoints/mvp_radio_stage2",
    "--token-manifest", "src/data/manifests/token_manifest_full.jsonl",
    "--step-log", "logs/mvp_radio_stage2_steps.jsonl"
)
Remove-Item "logs/mvp_radio_stage2.log","logs/mvp_radio_stage2.err","logs/mvp_radio_stage2.pid" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "logs/mvp_radio_stage2.log" `
    -RedirectStandardError  "logs/mvp_radio_stage2.err" `
    -NoNewWindow `
    -PassThru
$proc.Id | Out-File -FilePath "logs/mvp_radio_stage2.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/mvp_radio_stage2_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/mvp_radio_stage2_wrapper.log" -Append
