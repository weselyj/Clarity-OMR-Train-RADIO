<#
Stage-A hardened retrain worker (seder). Codifies the validated schtasks
pattern + the four Sub-plan-B minimal-reliable fixes:
  1. UTF-8 logging (PYTHONIOENCODING + -Encoding utf8)
  2. stderr non-fatal: $ErrorActionPreference='Continue', gate on $LASTEXITCODE
  3. save_period (passed via --save-period; default 5)
  4. resume-from-last: relaunch with --resume if last.pt exists

Deploy to a no-space path (e.g. C:\radio_jobs\run_stagea_hardened_retrain.ps1)
and register:
  schtasks /create /tn radio_stagea_hardened /sc ONCE /st 23:59 /f ^
    /tr "powershell -ExecutionPolicy Bypass -File C:\radio_jobs\run_stagea_hardened_retrain.ps1"
  schtasks /run /tn radio_stagea_hardened
Poll: schtasks /query /tn radio_stagea_hardened  +  the .done/.failed markers.
#>
param(
    [string]$Repo       = "$env:USERPROFILE\Clarity-OMR-Train-RADIO",
    [string]$Venv       = "venv-cu132",
    [string]$Data       = "data/processed/mixed_systems_v1/data.yaml",
    [string]$Model      = "yolo26m.pt",
    [string]$Name       = "yolo26m_systems_hardened",
    [int]   $Epochs     = 100,
    [int]   $Batch      = 4,
    [int]   $SavePeriod = 5,
    [string]$JobTag     = "radio_stagea_hardened"
)

$ErrorActionPreference = "Continue"          # fix #2: child stderr is NOT fatal
$env:PYTHONIOENCODING  = "utf-8"             # fix #1: UTF-8 stdio

Set-Location $Repo
$py        = Join-Path $Repo "$Venv\Scripts\python.exe"
$logDir    = Join-Path $Repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logOut    = Join-Path $logDir "$JobTag.out.log"
$logErr    = Join-Path $logDir "$JobTag.err.log"
$doneMark  = Join-Path $logDir "$JobTag.done"
$failMark  = Join-Path $logDir "$JobTag.failed"
Remove-Item $doneMark,$failMark -ErrorAction SilentlyContinue

# fix #4: resume-from-last if a checkpoint exists for this run name
$lastPt = Join-Path $Repo "runs\detect\runs\$Name\weights\last.pt"
$modelArg = $Model
$resumeArg = @()
if (Test-Path $lastPt) {
    $modelArg  = $lastPt
    $resumeArg = @("--resume")
    "[$JobTag] resuming from $lastPt" | Out-File -Encoding utf8 -Append $logOut
}

$argList = @(
    "scripts/train_yolo.py",
    "--model", $modelArg,
    "--data", $Data,
    "--name", $Name,
    "--project", "runs/detect/runs",
    "--epochs", $Epochs,
    "--batch", $Batch,
    "--device", "0",
    "--amp", "--nan-guard",
    "--noise", "--noise-warmup-steps", "2000",
    "--max-grad-norm", "1.0",
    "--save-period", $SavePeriod
) + $resumeArg

"[$JobTag] launching: $py $($argList -join ' ')" |
    Out-File -Encoding utf8 -Append $logOut

# fix #1+#2: capture both streams to UTF-8 logs; success gates on exit code only
& $py @argList *>> $logOut 2>> $logErr
$code = $LASTEXITCODE

if ($code -eq 0) {
    "[$JobTag] DONE exit=0" | Out-File -Encoding utf8 -Append $logOut
    New-Item -ItemType File -Force -Path $doneMark | Out-Null
} else {
    "[$JobTag] FAILED exit=$code" | Out-File -Encoding utf8 -Append $logErr
    New-Item -ItemType File -Force -Path $failMark | Out-Null
}
exit $code
