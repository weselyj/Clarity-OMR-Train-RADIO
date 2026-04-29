# scripts/train_yolo26m_launch.ps1
#
# YOLO26m run config:
#   --workers 6  : caps system RAM at ~80% (vs 96% with default 8 workers).
#   --noise      : scan-noise + page-curvature augmentation pipeline.
#   --nan-guard  : zero out NaN/Inf gradients before optimizer step.
#   AMP enabled (default), batch=8 (default).

$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\train_yolo_inner.ps1"

$pyArgs  = "--model yolo26m.pt --data data\processed\mixed_v1\data.yaml --name yolo26m_v1 --project runs --workers 6 --noise --nan-guard"
$logName = "train_yolo26m"

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
