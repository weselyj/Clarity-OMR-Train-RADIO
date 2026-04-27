# Mirror data/ to the NAS at \\10.10.1.234\Share\datasets\clarity-omr-data\
#
# Uses robocopy (Windows native) so no rsync install is needed.
# /E   — copy subdirs including empty
# /R:2 — retry twice on transient errors
# /W:5 — 5s between retries
# /MT:8 — 8 worker threads (saturates 1 GbE)
# /NP   — no per-file progress (cleaner log)
# /TEE  — also tee to stdout
# /LOG  — write summary log to logs/backup_data_to_nas.log
#
# Idempotent: re-running only copies new/changed files (robocopy default).
# Safe to run after synthetic_full_page is regenerated (incremental).
#
# Run from any cwd; uses absolute paths.

$ErrorActionPreference = "Stop"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$src   = Join-Path $repo "data"
$dst   = "\\10.10.1.234\Share\datasets\clarity-omr-data"
$log   = Join-Path $repo "logs\backup_data_to_nas.log"

if (-not (Test-Path $src)) {
    Write-Error "Source $src does not exist."
    exit 1
}

New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# Skip the per-dataset .tgz archives — they're huge and easy to re-download
# from the public URLs. The extracted content is what we want backed up.
$xf = @("*.tgz", "CameraPrIMuS.tgz", "grandstaff.tgz", "primusCalvoRizoAppliedSciences2018.tgz")

Write-Host "[backup] $src  ->  $dst"
Write-Host "[backup] excluding: $($xf -join ', ')"

robocopy $src $dst /E /R:2 /W:5 /MT:8 /NP /TEE /LOG:$log /XF $xf

# Robocopy exit codes: 0 = no copy, 1 = files copied, 2 = extra files, 3 = both.
# Anything >= 8 is an error.
$rc = $LASTEXITCODE
if ($rc -ge 8) {
    Write-Error "[backup] robocopy reported errors (exit $rc); see $log"
    exit $rc
}
Write-Host "[backup] complete (robocopy exit $rc, see $log for details)"
