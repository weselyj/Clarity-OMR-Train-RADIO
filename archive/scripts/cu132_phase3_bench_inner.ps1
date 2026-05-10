# Phase 3 of the cu132 plan — re-profile baseline on the new toolchain.
#
# Runs 700 opt-steps of Stage 2 against venv-cu132/ with --profile-step-timing.
# 200-step warm + 500-step measured -> input to scripts/profile_summary.py.
# Resume base: Stage 2's _final.pt (so we're warm-starting from a converged
# adapter, not the Stage 1 monophonic checkpoint, to keep the bench close to
# the cu128 reference workload from issue #2).
#
# Detached via WMI Create (matches the repo's launcher convention) so the
# bench survives SSH session close.
#
# This is the inner launcher. The outer script (cu132_phase3_bench_launch.ps1)
# Win32_Process Create's this with hidden window.

$ErrorActionPreference = "Continue"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$venv  = Join-Path $repo "venv-cu132\Scripts\python.exe"
$resume = Join-Path $repo "checkpoints\full_radio_stage2\stage2-radio-polyphonic_final.pt"
$manifest = Join-Path $repo "src\data\manifests\token_manifest_full.jsonl"
$bench = Join-Path $repo "bench\cu132_700"
$pidf  = Join-Path $repo "logs\cu132_phase3_bench.pid"
$log   = Join-Path $repo "logs\cu132_phase3_bench.log"
$err   = Join-Path $repo "logs\cu132_phase3_bench.err"

Set-Location $repo
New-Item -ItemType Directory -Force -Path $bench | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null

# Record pid for the watcher pattern used elsewhere in this repo
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
    # 2800 stage-steps with grad_accum=8 -> 350 opt-steps (100 warm + 250 measured),
    # matching the cu128 reference profile in issue #2's pinned 2026-04-27 comment.
    "--max-steps-per-stage", "2800",
    "--validation-batches", "1",
    # cadence=1 (every-step) matches the cu128 reference methodology so the
    # comparison is pure-toolchain, not toolchain + diagnostic gating.
    "--diag-cadence", "1",
    "--profile-step-timing",
    "--profile-output", "$bench\profile.jsonl"
)

# Run inline so this WMI-detached process IS the python process effectively.
& $venv @pyArgs *>$log 2>$err
"=== EXIT_CODE=$LASTEXITCODE ===" | Out-File -FilePath $log -Append -Encoding utf8
