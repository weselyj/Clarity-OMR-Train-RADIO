# Launch the Phase 3 bench inner script via WMI Create so it survives SSH close.
$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$inner = Join-Path $repo "scripts\cu132_phase3_bench_inner.ps1"
$cmdLine = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$inner`""
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $cmdLine }
if ($result.ReturnValue -ne 0) {
    Write-Error "Win32_Process Create failed (ReturnValue=$($result.ReturnValue))"
    exit 1
}
Write-Host "DETACHED_PID=$($result.ProcessId)"
Write-Host "LOG=$repo\logs\cu132_phase3_bench.log"
Write-Host "ERR=$repo\logs\cu132_phase3_bench.err"
Write-Host "PROFILE=$repo\bench\cu132_700\profile.jsonl"
