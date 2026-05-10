# Outer launcher: detaches Stage 3 RADIO training inner wrapper from SSH via WMI.
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\full_radio_stage3_inner.ps1"
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
Write-Output ("ReturnValue=" + $result.ReturnValue + " WrapperPID=" + $result.ProcessId)
Start-Sleep -Seconds 30
$pidFile = Join-Path $repo "logs\full_radio_stage3.pid"
if (Test-Path $pidFile) {
    $pyPid = [int](Get-Content $pidFile)
    $p = Get-Process -Id $pyPid -ErrorAction SilentlyContinue
    if ($p) {
        Write-Output ("ALIVE PythonPID=" + $pyPid + " StartTime=" + $p.StartTime)
    } else {
        Write-Output ("DIED PythonPID=" + $pyPid + " (no process found)")
    }
} else {
    Write-Output "NO PID FILE YET"
}
$logFile = Join-Path $repo "logs\full_radio_stage3.log"
$errFile = Join-Path $repo "logs\full_radio_stage3.err"
if (Test-Path $logFile) { Write-Output "--- log tail ---"; Get-Content $logFile -Tail 12 }
if ((Test-Path $errFile) -and (Get-Item $errFile).Length -gt 0) { Write-Output "--- err tail ---"; Get-Content $errFile -Tail 12 }
Write-Output "--- gpu ---"
& nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
