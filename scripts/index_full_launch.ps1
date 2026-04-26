# Outer launcher: detaches the inner wrapper from the SSH session via WMI.
$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\index_full_inner.ps1"
$cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{CommandLine=$cmd}
Write-Output ("ReturnValue=" + $result.ReturnValue + " WrapperPID=" + $result.ProcessId)
Start-Sleep -Seconds 12
$pidFile = Join-Path $repo "logs\index_full.pid"
if (Test-Path $pidFile) {
    $pyPid = [int](Get-Content $pidFile)
    $p = Get-Process -Id $pyPid -ErrorAction SilentlyContinue
    if ($p) {
        Write-Output ("ALIVE PythonPID=" + $pyPid + " StartTime=" + $p.StartTime)
    } else {
        Write-Output ("DIED PythonPID=" + $pyPid + " (no process found)")
    }
} else {
    Write-Output "NO PID FILE YET (wrapper hasn't spawned python)"
}
$logFile = Join-Path $repo "logs\index_full.log"
$errFile = Join-Path $repo "logs\index_full.err"
if (Test-Path $logFile) { Write-Output "--- log tail ---"; Get-Content $logFile -Tail 8 }
if (Test-Path $errFile) {
    $eSize = (Get-Item $errFile).Length
    if ($eSize -gt 0) { Write-Output "--- err tail ---"; Get-Content $errFile -Tail 8 }
}
