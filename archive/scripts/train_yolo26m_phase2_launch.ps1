# scripts/train_yolo26m_phase2_launch.ps1
# YOLO26m Phase 2: noise fine-tuning. Run AFTER Phase 1 completes.
#
# Loads Phase 1's best.pt (clean-trained model at converged state) and continues
# training with the scan-noise + page-curvature augmentation pipeline. By this
# point the model has stable cls_loss in normal range, so noise won't push
# gradients into NaN territory as easily.
#
# Config: same as Phase 1 + --noise. Lower epoch cap since we're fine-tuning.
# Results: runs/yolo26m_phase2_noise/

$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\train_yolo_inner.ps1"

$phase1Best = "runs\detect\runs\yolo26m_phase1_clean\weights\best.pt"
$pyArgs  = "--model $phase1Best --data data\processed\mixed_v1\data.yaml --name yolo26m_phase2_noise --project runs --workers 6 --batch 4 --nan-guard --noise --epochs 50 --patience 15"
$logName = "train_yolo26m_phase2"

[Environment]::SetEnvironmentVariable("TRAIN_YOLO_ARGS", $pyArgs,  "User")
[Environment]::SetEnvironmentVariable("TRAIN_YOLO_LOG",  "$logName.log", "User")

$cmd    = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
Write-Output ("ReturnValue=" + $result.ReturnValue + " WrapperPID=" + $result.ProcessId)

Start-Sleep -Seconds 30
$pidFile = Join-Path $repo "logs\$logName.pid"
if (Test-Path $pidFile) {
    $pyPid = [int](Get-Content $pidFile)
    $p = Get-Process -Id $pyPid -ErrorAction SilentlyContinue
    if ($p) { Write-Output ("ALIVE PythonPID=" + $pyPid + " StartTime=" + $p.StartTime) }
    else    { Write-Output ("DIED PythonPID=" + $pyPid) }
} else {
    Write-Output "NO PID FILE YET"
}

$logFile = Join-Path $repo "logs\$logName.log"
$errFile = Join-Path $repo "logs\$logName.err"
if (Test-Path $logFile) { Write-Output "--- log tail ---"; Get-Content $logFile -Tail 12 }
if ((Test-Path $errFile) -and (Get-Item $errFile).Length -gt 0) { Write-Output "--- err tail ---"; Get-Content $errFile -Tail 12 }
Write-Output "--- gpu ---"
& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
