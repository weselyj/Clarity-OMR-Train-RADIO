# Sub-plan B — Acceptance Runbook (graceful-safe + no-regression)

Run ON SEDER (`venv-cu132`). B PASSES iff: the hardened retrain either
completes clean OR the active guard halts it; AND the selected checkpoint
passes the provenance gate (0 non-finite tensors, total>0); AND it meets
lieder recall >= 0.930. Do NOT overwrite the validated faint-ink best.pt
(runs/detect/runs/yolo26m_systems_faintink/weights/best.pt).

## 0. Pre-flight caveats (READ FIRST — from Task 5/6/7 review)

- **No stale `last.pt` before a fresh run.** Ultralytics `resume=True` (bool)
  resolves the checkpoint via a CWD glob for the newest `last*.pt`, not the
  worker's explicit `$lastPt`. Before the FIRST hardened run, confirm no
  `runs/detect/runs/yolo26m_systems_hardened/weights/last.pt` exists (a stale
  one would make the worker think it must resume). The worker only adds
  `--resume` when that `last.pt` is present.
- **Verify the checkpoint path after the first epoch.** The worker uses
  `--project runs`; Ultralytics writes detect runs to
  `runs/detect/runs/<name>/weights/`. After the first `--save-period`
  checkpoint, `dir`/`ls runs/detect/runs/yolo26m_systems_hardened/weights/`
  to confirm `last.pt` is appearing there (sanity-check the path before
  trusting resume / scoring).
- **Logs are split:** stdout → `logs/radio_stagea_hardened.out.log`,
  stderr → `logs/radio_stagea_hardened.err.log`. Python tracebacks land in
  the `.err.log`; the `[nan-guard]` / `[stagea-hardening]` lines are on
  stdout (`.out.log`).
- **If `venv-cu132\Scripts\python.exe` is missing/mistyped**, the worker may
  exit without writing a `.done`/`.failed` marker; in that case rely on
  `schtasks /query /tn radio_stagea_hardened` last-run result, not just the
  markers.
- **`--noise-warmup-steps 2000`** is intentional for the hardened recipe
  (the faint-ink run used 500); not a mistake.

## 1. Launch the hardened retrain (codified worker)

Deploy + register the worker (no-space path avoids schtasks quoting):

    copy scripts\seder\run_stagea_hardened_retrain.ps1 C:\radio_jobs\
    schtasks /create /tn radio_stagea_hardened /sc ONCE /st 23:59 /f ^
      /tr "powershell -ExecutionPolicy Bypass -File C:\radio_jobs\run_stagea_hardened_retrain.ps1"
    schtasks /run /tn radio_stagea_hardened

Poll: `schtasks /query /tn radio_stagea_hardened` + `logs\radio_stagea_hardened.done|.failed`.
Watch `logs\radio_stagea_hardened.out.log` for `[nan-guard]` / `[stagea-hardening]` lines.

## 2. Select the checkpoint to gate

- Clean completion -> `runs/detect/runs/yolo26m_systems_hardened/weights/best.pt`
- Active-halt fired -> the newest finite checkpoint: prefer `best.pt`; if its
  provenance fails, fall back to the highest-epoch `--save-period` checkpoint
  under `runs/detect/runs/yolo26m_systems_hardened/weights/`.

## 3. Provenance gate (mandatory, blocks scoring)

    venv-cu132\Scripts\python.exe -c "from src.train.stagea_hardening import validate_checkpoint_finite as v; ok,n,t,k=v(r'runs/detect/runs/yolo26m_systems_hardened/weights/best.pt'); print('ok',ok,'nonfinite',n,'total',t,'first',k); import sys; sys.exit(0 if (ok and t>0) else 1)"

A non-zero exit, `ok=False`, OR `total==0` is an automatic B-FAIL regardless
of metrics. (`total==0` means neither a 'model' nor 'ema' tensor was scanned
— wrong/empty checkpoint path; never silently treat that as a pass.)

## 4. Lieder no-regression gate (>= 0.930)

Two-step (same scorer Sub-plan A reads):

    venv-cu132\Scripts\python.exe eval/run_stage_a_only.py ^
      --pdf-dir data/openscore_lieder/pdfs ^
      --yolo-weights runs/detect/runs/yolo26m_systems_hardened/weights/best.pt ^
      --out-dir eval/results/stagea_manifests/hardened

    venv-cu132\Scripts\python.exe eval/score_stage_a_only.py ^
      --manifest-dir eval/results/stagea_manifests/hardened ^
      --scores-dir data/openscore_lieder/scores ^
      --out-csv eval/results/stagea_hardened.csv

Compute aggregate recall with the Sub-plan-A reader and compare to baseline:

    venv-cu132\Scripts\python.exe -c "from eval.robust_stage_a.gate import recall_from_stagea_csv as r; new=r('eval/results/stagea_hardened.csv'); base=r('eval/results/stagea_baseline_pre_faintink.csv'); print('new',new,'base',base,'PASS' if new>=base else 'FAIL')"

## 5. Verdict

B PASS iff: training ended clean-or-guard-halted AND step 3 prints ok=True
with total>0 AND step 4 prints PASS (new >= base, base = 0.930). Record the
outcome + the [nan-guard]/[stagea-hardening] log evidence in the Sub-plan-B
handoff and update memory `project_radio_robust_stagea`.
