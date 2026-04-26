# Runs detached via WMI; this script's parent (SSH session) has no handle on it.
# Launches Stage 2 RADIO training (full corpus, ~12h on the 5090).
# Resumes from the final Stage 1 checkpoint (step_0004000.pt).
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv\Scripts\python.exe"
$resume = "checkpoints\full_radio_stage1\stage1-radio-monophonic-foundation_step_0004000.pt"
$pyArgs = @(
    "-u", "src/train/train.py",
    "--stage-configs", "configs/train_stage2_radio.yaml",
    "--mode", "execute",
    "--resume-checkpoint", $resume,
    "--start-stage", "stage2-radio-polyphonic",
    "--checkpoint-dir", "checkpoints/full_radio_stage2",
    "--token-manifest", "src/data/manifests/token_manifest_full.jsonl",
    "--step-log", "logs/full_radio_stage2_steps.jsonl",
    "--validation-batches", "16"
)
Remove-Item "logs/full_radio_stage2.log","logs/full_radio_stage2.err","logs/full_radio_stage2.pid","logs/full_radio_stage2_wrapper.log","logs/full_radio_stage2_steps.jsonl" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "logs/full_radio_stage2.log" `
    -RedirectStandardError  "logs/full_radio_stage2.err" `
    -NoNewWindow `
    -PassThru
$proc.Id | Out-File -FilePath "logs/full_radio_stage2.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/full_radio_stage2_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/full_radio_stage2_wrapper.log" -Append
