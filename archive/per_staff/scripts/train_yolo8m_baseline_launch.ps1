# scripts/train_yolo8m_baseline_launch.ps1
# Outer launcher: detaches YOLOv8m baseline training from SSH via WMI (Invoke-CimMethod).
#
# Trains on the mixed Stage A dataset produced by Phase 4.
# Hyperparameters: rect=True, cos_lr=True, augmentation disabled (sheet music),
#                  batch=16 (default), no torch.compile (OOMs at imgsz=1920 batch=16).
# Results land in: runs/yolo8m_baseline_v1/

$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\train_yolo_inner.ps1"

$pyArgs  = "--model yolov8m.pt --data data\processed\mixed_v1\data.yaml --name yolo8m_baseline_v1 --project runs"
$logName = "train_yolo8m_baseline"

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
