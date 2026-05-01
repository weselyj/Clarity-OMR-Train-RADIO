# scripts/train_yolo_inner.ps1
# Runs detached via WMI (Invoke-CimMethod); this script's parent has no handle on it.
#
# Arg-passing strategy: the outer launcher writes TRAIN_YOLO_ARGS and TRAIN_YOLO_LOG
# to User-scope env vars immediately before spawning this process.  Each launch
# overwrites both vars, so stale-value risk is low for a single-GPU machine where
# only one training job runs at a time.  If concurrent launches ever become needed,
# switch to two dedicated inner scripts (train_yolo8m_baseline_inner.ps1 and
# train_yolo26m_inner.ps1) with args baked in — that eliminates the shared-state
# hazard entirely.
#
# Output: separate .log / .err files + a .pid file, matching the repo convention
# used by full_radio_stage* and synthetic_* launchers.

$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
Set-Location $repo
New-Item -ItemType Directory -Force -Path "logs" | Out-Null

$py      = Join-Path $repo "venv-cu132\Scripts\python.exe"
$logBase = Join-Path $repo "logs\$($env:TRAIN_YOLO_LOG)"
$logFile = $logBase -replace '\.log$', '.log'
$errFile = $logBase -replace '\.log$', '.err'
$pidFile = $logBase -replace '\.log$', '.pid'
$wrapLog = $logBase -replace '\.log$', '_wrapper.log'

# Split the args string back into an array for Start-Process
$pyArgsList = ($env:TRAIN_YOLO_ARGS -split '\s+') + @()
$allArgs    = @("-u", "scripts\train_yolo.py") + $pyArgsList

Remove-Item $logFile, $errFile, $pidFile, $wrapLog -ErrorAction SilentlyContinue

$proc = Start-Process -FilePath $py `
    -ArgumentList $allArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError  $errFile `
    -NoNewWindow `
    -PassThru

$proc.Id | Out-File -FilePath $pidFile -Encoding ASCII -NoNewline
"Wrapper started python PID $($proc.Id) at $(Get-Date -Format o)" | Out-File -FilePath $wrapLog -Append
$proc.WaitForExit()
"Wrapper exit: python PID $($proc.Id) ExitCode=$($proc.ExitCode) at $(Get-Date -Format o)" | Out-File -FilePath $wrapLog -Append
