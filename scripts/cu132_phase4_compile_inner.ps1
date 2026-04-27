# Phase 4.2 of the cu132 plan — A/B torch.compile against the cu132 baseline.
#
# Same workload as Phase 3 (700 opt-steps Stage 2) but with --torch-compile
# enabled. Output goes to bench/cu132_compile so direct comparison with
# bench/cu132_700 is a one-line scripts/profile_summary.py invocation.
#
# Important: the first 50-100 opt-steps after compile are slower
# (cache warm-up). --diag-cadence 50 + 200-step skip in profile_summary
# should mask this.

$ErrorActionPreference = "Continue"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$venv  = Join-Path $repo "venv-cu132\Scripts\python.exe"
$resume = Join-Path $repo "checkpoints\full_radio_stage2\stage2-radio-polyphonic_final.pt"
$manifest = Join-Path $repo "src\data\manifests\token_manifest_full.jsonl"
$bench = Join-Path $repo "bench\cu132_compile"
$pidf  = Join-Path $repo "logs\cu132_phase4_compile.pid"
$log   = Join-Path $repo "logs\cu132_phase4_compile.log"
$err   = Join-Path $repo "logs\cu132_phase4_compile.err"

Set-Location $repo
New-Item -ItemType Directory -Force -Path $bench | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

$PID | Out-File -FilePath $pidf -Encoding ascii -NoNewline

$pyArgs = @(
    "-u", "src/train/train.py",
    "--stage-configs", "configs/train_stage2_radio.yaml",
    "--mode", "execute",
    "--resume-checkpoint", $resume,
    "--start-stage", "stage2-radio-polyphonic",
    "--checkpoint-dir", $bench,
    "--token-manifest", $manifest,
    "--step-log", "$bench\steps.jsonl",
    "--max-steps-per-stage", "700",
    "--validation-batches", "1",
    "--diag-cadence", "50",
    "--profile-step-timing",
    "--profile-output", "$bench\profile.jsonl",
    "--torch-compile"
)

& $venv @pyArgs *>$log 2>$err
"=== EXIT_CODE=$LASTEXITCODE ===" | Out-File -FilePath $log -Append -Encoding utf8
