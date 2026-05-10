$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
$py = Join-Path $repo "venv\Scripts\python.exe"
$pyArgs = @(
    "-u", "src/train/train.py",
    "--stage-configs", "configs/train_stage2_radio_mvp.yaml",
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
    -WindowStyle Hidden `
    -PassThru
$proc.Id | Out-File -FilePath "logs/mvp_radio_stage2.pid" -Encoding ASCII -NoNewline
Write-Output ("Started PID: " + $proc.Id + " at " + $proc.StartTime)
Start-Sleep -Seconds 35
$running = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
if ($running) {
    Write-Output ("ALIVE @35s: PID=" + $proc.Id + " CPU=" + [math]::Round($running.CPU,1) + "s WS=" + [math]::Round($running.WorkingSet64/1MB,0) + "MB")
} else {
    Write-Output ("DIED <35s: PID=" + $proc.Id + " ExitCode=" + $proc.ExitCode)
}
Write-Output "--- log size (out/err) ---"
$o = Get-Item logs/mvp_radio_stage2.log -ErrorAction SilentlyContinue; if ($o) { Write-Output ("out " + $o.Length) }
$e = Get-Item logs/mvp_radio_stage2.err -ErrorAction SilentlyContinue; if ($e) { Write-Output ("err " + $e.Length) }
Write-Output "--- log tail ---"
if (Test-Path logs/mvp_radio_stage2.log) { Get-Content logs/mvp_radio_stage2.log -Tail 10 }
Write-Output "--- err tail ---"
if (Test-Path logs/mvp_radio_stage2.err) { Get-Content logs/mvp_radio_stage2.err -Tail 10 }
