# Runs detached via WMI; this script's parent (SSH session) has no handle on it.
# Launches Stage 1 RADIO training (full corpus, ~10-15h on the 5090).
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv\Scripts\python.exe"
$pyArgs = @(
    "-u", "src/train/train.py",
    "--stage-configs", "configs/train_stage1_radio.yaml",
    "--mode", "execute",
    "--checkpoint-dir", "checkpoints/full_radio_stage1",
    "--token-manifest", "src/data/manifests/token_manifest_full.jsonl",
    "--step-log", "logs/full_radio_stage1_steps.jsonl"
)
Remove-Item "logs/full_radio_stage1.log","logs/full_radio_stage1.err","logs/full_radio_stage1.pid","logs/full_radio_stage1_wrapper.log","logs/full_radio_stage1_steps.jsonl" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "logs/full_radio_stage1.log" `
    -RedirectStandardError  "logs/full_radio_stage1.err" `
    -NoNewWindow `
    -PassThru
$proc.Id | Out-File -FilePath "logs/full_radio_stage1.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/full_radio_stage1_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/full_radio_stage1_wrapper.log" -Append
