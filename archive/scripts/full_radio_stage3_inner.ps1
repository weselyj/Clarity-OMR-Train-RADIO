# Runs detached via WMI; parent SSH session has no handle on it.
# Stage 3 RADIO training. Resumes from Stage 2 final checkpoint.
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv-cu132\Scripts\python.exe"
$resume = "checkpoints\full_radio_stage2\stage2-radio-polyphonic_final.pt"
$pyArgs = @(
    "-u", "src/train/train.py",
    "--stage-configs", "configs/train_stage3_radio.yaml",
    "--mode", "execute",
    "--resume-checkpoint", $resume,
    "--start-stage", "stage3-radio-full-complexity",
    "--checkpoint-dir", "checkpoints/full_radio_stage3",
    "--token-manifest", "src/data/manifests/token_manifest_full.jsonl,data/processed/synthetic/manifests/synthetic_token_manifest.jsonl",
    "--step-log", "logs/full_radio_stage3_steps.jsonl",
    "--validation-batches", "16",
    "--num-workers", "4",
    "--prefetch-factor", "4",
    "--torch-compile",
    "--channels-last"
)
Remove-Item "logs/full_radio_stage3.log","logs/full_radio_stage3.err","logs/full_radio_stage3.pid","logs/full_radio_stage3_wrapper.log","logs/full_radio_stage3_steps.jsonl" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py `
    -ArgumentList $pyArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput "logs/full_radio_stage3.log" `
    -RedirectStandardError  "logs/full_radio_stage3.err" `
    -NoNewWindow `
    -PassThru
$proc.Id | Out-File -FilePath "logs/full_radio_stage3.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/full_radio_stage3_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/full_radio_stage3_wrapper.log" -Append
