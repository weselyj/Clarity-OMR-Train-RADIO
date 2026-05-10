# Pre-cu132 Baseline (2026-04-27)

Captured before installing the PyTorch nightly built against CUDA 13.2 on the Windows training host (the GPU box).

Plan: [`docs/superpowers/plans/2026-04-27-pytorch-nightly-cu132.md`](../../docs/superpowers/plans/2026-04-27-pytorch-nightly-cu132.md).

## Current PyTorch wheel

```
torch.__version__         = 2.11.0+cu128
torch.version.cuda        = 12.8
torch.cuda.get_arch_list  = ['sm_75', 'sm_80', 'sm_86', 'sm_90', 'sm_100', 'sm_120']
torch.cuda.get_device_name(0)        = NVIDIA GeForce RTX 5090
torch.cuda.get_device_capability(0)  = (12, 0)
```

`sm_120` is already in the arch list and `get_device_capability` returns `(12, 0)`.
**Outcome 1 from Phase 0 Task 0.1 Step 3 applies:** the upgrade is for kernel/perf gains, not Blackwell enablement. Bar for "is this worth it" is therefore higher — any phase that yields < 5% wall improvement is not worth the maintenance burden of running on nightlies.

## Driver

```
NVIDIA-SMI  : 596.21
Driver      : 596.21
CUDA Version: 13.2
GPU         : NVIDIA GeForce RTX 5090 (32 GB, WDDM)
```

Driver already exposes CUDA 13.2. **Phase 1.2 (manual driver install via RDP) is skipped entirely.**

## Current Stage 2 profile (5362 ms / opt-step, 250 measured steps)

Source: issue #2 pinned 2026-04-27 comment. Measured on `configs/train_stage2_radio.yaml` (batch 2, max_seq_len 384, RADIO bf16, grad_accum 16) with `--profile-step-timing` instrumentation at commit `1a768fe`. 100 warm-up steps skipped, 250 measured steps.

| Phase | Mean ms | % of wall |
|---|---|---|
| gpu_backward | 3255 | 60.7 |
| gpu_forward | 1828 | 34.1 |
| gpu_grad_diagnostics | 119 | 2.2 |
| cpu_augment | 76 | 1.4 |
| cpu_encode | 48 | 0.9 |
| gpu_optimizer | 5 | 0.1 |
| cpu_h2d | 5 | 0.1 |
| cpu_log_io | 0.2 | 0.0 |
| cpu_sample | 0.3 | 0.0 |
| **Wall mean** | **5362** | — |

Phase coverage: 99.5%. CPU pipeline (sample + encode + augment + h2d + log_io) sums to **2.4% of wall**. Encoder forward+backward dominates at **94.8%**.

This is the comparison target for Phase 3.

## Rollback procedure

If anything regresses after the upgrade:

1. Halt any running training:
   ```powershell
   $pid = [int](Get-Content "$env:USERPROFILE\Clarity-OMR-Train-RADIO\logs\<stage>.pid" -ErrorAction SilentlyContinue)
   if ($pid -and (Get-Process -Id $pid -ErrorAction SilentlyContinue)) { Stop-Process -Id $pid -Force }
   ```
2. The original venv is preserved at `$env:USERPROFILE\Clarity-OMR-Train-RADIO\venv\` and is never modified by this plan. To switch back, just point launchers at `venv\` instead of `venv-cu132\`.
3. If the issue is in cu132 nightly torch: delete `venv-cu132\` and re-run from scratch.
4. If the issue is in the project deps installed alongside torch: re-run `venv\Scripts\pip.exe install -r requirements.txt` from the original venv.
5. Verify by running `--profile-step-timing --max-steps-per-stage 5` against the original venv and confirming wall ≈ 5362 ms / opt-step.

The rollback freeze file at `docs/perf/freeze/pre-cu132-2026-04-27.txt` (from Phase 0.2) is the authoritative snapshot of the working pip state.
