$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv-cu132\Scripts\python.exe"
$pyArgs = @(
    "-u", "-m", "eval.run_lieder_eval",
    "--checkpoint", "checkpoints/full_radio_stage3/stage3-radio-full-complexity_best.pt",
    "--config", "configs/train_stage3_radio.yaml",
    "--name", "stage3_best_full"
)
Remove-Item "logs/lieder_eval_stage3_full.log","logs/lieder_eval_stage3_full.err","logs/lieder_eval_stage3_full.pid","logs/lieder_eval_stage3_full_wrapper.log" -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $py -ArgumentList $pyArgs -WorkingDirectory $repo -RedirectStandardOutput "logs/lieder_eval_stage3_full.log" -RedirectStandardError "logs/lieder_eval_stage3_full.err" -NoNewWindow -PassThru
$proc.Id | Out-File -FilePath "logs/lieder_eval_stage3_full.pid" -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath "logs/lieder_eval_stage3_full_wrapper.log" -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath "logs/lieder_eval_stage3_full_wrapper.log" -Append
