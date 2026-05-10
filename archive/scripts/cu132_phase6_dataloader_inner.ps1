# Phase 6 bench — cu132 + DataLoader rewrite, no torch.compile.
# Disambiguates the compile contribution from the DataLoader contribution
# (Phase 4.2's -6.9% wall is the COMPOUND of both).
#
# Same workload + diag-cadence + grad-accum-aware step count as Phase 3.

$ErrorActionPreference = "Continue"
$repo  = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$venv  = Join-Path $repo "venv-cu132\Scripts\python.exe"
$resume = Join-Path $repo "checkpoints\full_radio_stage2\stage2-radio-polyphonic_final.pt"
$manifest = Join-Path $repo "src\data\manifests\token_manifest_full.jsonl"
$bench = Join-Path $repo "bench\cu132_dataloader"
$pidf  = Join-Path $repo "logs\cu132_phase6_dataloader.pid"
$log   = Join-Path $repo "logs\cu132_phase6_dataloader.log"
$err   = Join-Path $repo "logs\cu132_phase6_dataloader.err"

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
    "--max-steps-per-stage", "2800",
    "--validation-batches", "1",
    "--diag-cadence", "1",
    "--profile-step-timing",
    "--profile-output", "$bench\profile.jsonl"
    # NO --torch-compile here. DataLoader is the default code path on this
    # branch since commit 591fc3d, so this run captures cu132 + DataLoader
    # without the compile contribution.
)

& $venv @pyArgs *>$log 2>$err
"=== EXIT_CODE=$LASTEXITCODE ===" | Out-File -FilePath $log -Append -Encoding utf8
