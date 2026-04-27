# Detach Phase 6 bench inner via WMI so it survives SSH close.
$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\cu132_phase6_dataloader_inner.ps1"
$cmdLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine }
if ($result.ReturnValue -ne 0) {
    Write-Error "Win32_Process Create failed (ReturnValue=$($result.ReturnValue))"
    exit 1
}
Write-Host "DETACHED_PID=$($result.ProcessId)"
Write-Host "LOG=$repo\logs\cu132_phase6_dataloader.log"
Write-Host "PROFILE=$repo\bench\cu132_dataloader\profile.jsonl"
