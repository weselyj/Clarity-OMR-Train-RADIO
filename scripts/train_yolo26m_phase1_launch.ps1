# scripts/train_yolo26m_phase1_launch.ps1
# YOLO26m Phase 1: clean training (no noise), batch=4 to fit in actual VRAM.
#
# Curriculum approach: train on clean synthetic+real data first to get the 1-class
# detection head into a stable regime, then Phase 2 resumes with noise aug for
# scan robustness. Noise pipeline is too aggressive to apply on top of YOLO26m's
# COCO-pretrained head from epoch 1 (cls_loss spikes to 50-80 normal, vs ~1 for
# YOLOv8m); needs a stable model first.
#
# Config:
#   --workers 6   : RAM cap.
#   --batch 4     : YOLO26m at 1920 imgsz with AMP+batch=8 paged 45GB to system RAM
#                   (only 32GB actual VRAM); batch=4 fits cleanly without paging.
#   --nan-guard   : insurance against fp16 overflow at low LR.
#   AMP on (default), noise off (default).
# Results: runs/yolo26m_phase1_clean/

$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\train_yolo_inner.ps1"

$pyArgs  = "--model yolo26m.pt --data data\processed\mixed_v1\data.yaml --name yolo26m_phase1_clean --project runs --workers 6 --batch 4 --nan-guard"
$logName = "train_yolo26m_phase1"

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
