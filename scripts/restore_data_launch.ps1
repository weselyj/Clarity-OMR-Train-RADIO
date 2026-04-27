# Outer launcher — uses the WMI detach pattern (matches the repo's
# scripts/full_radio_stage*_launch.ps1 convention) so the inner script
# survives SSH session close.
$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $env:USERPROFILE "restore_data_inner.ps1"
$log   = Join-Path $repo "logs\restore_data.log"
$pidf  = Join-Path $repo "logs\restore_data.pid"

if (-not (Test-Path (Split-Path $log))) {
    New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
}
"=== restore_data launcher start $(Get-Date -Format o) ===" | Out-File -FilePath $log -Encoding utf8

# WMI detach: launches a new powershell process owned by WMIPRVSE, not by
# this SSH session. Survives SSH disconnect.
$cmdLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine }

if ($result.ReturnValue -ne 0) {
    Write-Error "Win32_Process Create failed with ReturnValue=$($result.ReturnValue)"
    exit 1
}

$childPid = $result.ProcessId
$childPid | Out-File -FilePath $pidf -Encoding ascii -NoNewline
Write-Host "DETACHED_PID=$childPid"
Write-Host "LOG=$log"
