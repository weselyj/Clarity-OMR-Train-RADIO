# Stage 3 v3 Encoder-Freeze Fix + Retrain — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the design in [`docs/superpowers/specs/2026-05-11-stage3-v3-retrain-design.md`](../specs/2026-05-11-stage3-v3-retrain-design.md): diagnose with a frankenstein checkpoint, fix the encoder-freeze bug via cache-derived auto-detect, retrain Stage 3 v3 for 9000 steps, and re-evaluate against A2 + demo + lieder. Produce a definitive answer on whether the RADIO architecture works when trained correctly.

**Architecture:** Four sequential phases with explicit gates. Phase 1 is a cheap diagnostic that gates the expensive Phase 3 retrain. Phase 2 is a small focused PR (one fix + tests + assertion). Phase 3 is the long-running training job on seder. Phase 4 is three eval passes plus a written verdict.

**Tech Stack:** Python 3.13, PyTorch (CUDA on seder via `venv-cu132`), PEFT (DoRA adapters), music21 (eval scoring). All training runs on seder; all code edits land on the `feat/stage3-v3-retrain` branch off `main` at `91a1e71`.

**Branch strategy:** All code changes (Phase 1 script, Phase 2 fix, Phase 3 config) land on a single branch `feat/stage3-v3-retrain`. Phase 4 eval scripts already exist on `main`. The audit-results report (Phase 4 §) commits to the same branch as a final commit. One PR at the end covering the fix + tests + config + report.

**Pre-existing context worth noting before starting:**
- The audit (PR #48, merged) identifies the bug at `src/train/train.py:1327-1331` (`_prepare_model_for_dora`) and recommends Option B (auto-detect freeze from cache config). The spec adopts this with the rationale that cache + unfrozen encoder is physically meaningless (cache becomes stale after the first encoder gradient step).
- v2 step log shows the actual best val_loss occurred at step 5500 (0.164), not step 4000 (0.184) as the audit's `_best.pt` metadata read suggested. Task 1 below resolves this discrepancy before launching v3 so we can trust v3's `_best.pt`.
- Seder operational details: SSH alias `10.10.1.29` (configured for user "Jonathan Wesely"), Windows cmd.exe shell — use `&&` not `;`, use `cd /D` to change drive+dir, Python at `venv-cu132\Scripts\python.exe`.
- Stage 3 v2 checkpoint: `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` (on seder only).
- Stage 2 v2 checkpoint: `checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt` (on seder only).
- Encoder cache: `data/cache/encoder/ac8948ae4b5be3e9/` (on seder only).
- Combined training manifest: `src/data/manifests/token_manifest_stage3.jsonl` (in repo).

**Where each phase runs:**
- Tasks 1, 2, 3 (Phase 1 + best-pt investigation): seder
- Task 4 (Phase 2 code fix): local + tests run on seder (CUDA-gated)
- Tasks 5, 6, 7 (Phase 3): config locally, training on seder
- Task 8 (Phase 4 evals): seder
- Task 9 (Phase 4 report): local

---

## File Structure

**New files (`feat/stage3-v3-retrain` branch):**
- `scripts/audit/build_frankenstein_checkpoint.py` — Phase 1 merge script
- `tests/train/test_freeze_encoder.py` — Phase 2 regression tests
- `configs/train_stage3_radio_systems_v3.yaml` — Phase 3 config
- `docs/audits/2026-05-11-stage3-v3-retrain-results.md` — Phase 4 report

**Modified files:**
- `src/train/train.py` — Phase 2 fix in `_prepare_model_for_dora` + pre-flight assertion at the call site

**Generated artifacts (not in git, on seder):**
- `checkpoints/_frankenstein_s2enc_s3dec.pt` — Phase 1 diagnostic checkpoint
- `checkpoints/full_radio_stage3_v3/*.pt` — Phase 3 training output (including `_best.pt` and `_step_NNNNNNN.pt` series)
- `logs/full_radio_stage3_v3_steps.jsonl` — Phase 3 step log
- `eval/results/clarity_demo_stage3_v3_best/*` — Phase 4 demo eval output
- `eval/results/lieder_stage3_v3/*` — Phase 4 lieder eval output
- `audit_results/a2_encoder_stage3_v3.json` — Phase 4 A2 re-run output

---

## Task 1: Best-`_best.pt` accounting investigation (Phase 3 prereq)

**Goal:** Reconcile the discrepancy between the v2 step log (best val_loss at step 5500 = 0.164) and the audit's metadata read (claimed step 4000, val_loss 0.148). Before v3 launches we need to know whether `_best.pt`'s file metadata, the file's *weights*, or the trainer's save logic is wrong. The decision affects whether we trust v3's `_best.pt` later.

**Files:** investigation only — no code changes in this task. Findings recorded inline at the bottom of the v3 results doc (created in Task 9).

- [ ] **Step 1.1: Read what `_best.pt` actually contains**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import torch
ckpt = torch.load(r''checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt'', map_location=''cpu'', weights_only=False)
print(''top-level keys:'', list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__)
for k in (''step'', ''global_step'', ''val_loss'', ''best_val_loss'', ''best_step'', ''epoch''):
    if isinstance(ckpt, dict) and k in ckpt:
        print(f''  {k}: {ckpt[k]}'')
"'
```

Expected output: dict with model state plus metadata fields. Record which fields are present and what they report.

- [ ] **Step 1.2: Compare against the matching `_step_*.pt` files**

If `_best.pt` claims step 4000 in its metadata, compare a model weight (e.g., `decoder.layers.0.self_attn.q_proj.weight`'s mean abs value) against `_step_0004000.pt`'s same weight. If they match, `_best.pt` IS step 4000's weights with possibly-stale val_loss metadata. If they don't match, `_best.pt`'s metadata claim is wrong — the file holds different weights.

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import torch
best = torch.load(r''checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt'', map_location=''cpu'', weights_only=False)
s4 = torch.load(r''checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_step_0004000.pt'', map_location=''cpu'', weights_only=False)
s55 = torch.load(r''checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_step_0005500.pt'', map_location=''cpu'', weights_only=False)
# Find any model parameter key present in all three
def get_sd(c):
    return c.get(''model'') or c.get(''state_dict'') or c
b_sd, s4_sd, s55_sd = get_sd(best), get_sd(s4), get_sd(s55)
keys = list(b_sd.keys())[:1]
for k in keys:
    if k in s4_sd and k in s55_sd:
        b_to_s4 = (b_sd[k].float() - s4_sd[k].float()).abs().max().item()
        b_to_s55 = (b_sd[k].float() - s55_sd[k].float()).abs().max().item()
        print(f''key: {k}'')
        print(f''  best vs step_4000:  max abs diff = {b_to_s4:.8f}'')
        print(f''  best vs step_5500:  max abs diff = {b_to_s55:.8f}'')
"'
```

If `best vs step_4000` is ~0 → `_best.pt` IS step 4000's weights (despite step log saying 5500 was better → trainer's save-best logic has a bug)
If `best vs step_5500` is ~0 → `_best.pt` IS step 5500's weights (matches step log → only the metadata is stale, weights are fine)

- [ ] **Step 1.3: Search the trainer's save-best logic**

```bash
grep -n "best_val\|_best.pt\|best_loss\|save_best\|is_best\|best_so_far" src/train/train.py | head -20
```

Read the surrounding code at the matches. Confirm what condition triggers saving `_best.pt` and what metadata gets written. A common bug pattern: the comparison uses `<=` instead of `<`, OR `best` tracks `train_loss` instead of `val_loss`, OR the comparison happens BEFORE the current step's val pass updates the running best.

- [ ] **Step 1.4: Record the finding**

Write a one-paragraph note titled "v2 `_best.pt` discrepancy investigation" — append at the bottom of `docs/audits/2026-05-11-stage3-v3-retrain-results.md` (the report file you'll create in Task 9). Paragraph should answer:
- What `_best.pt` actually contains (weights from which step?)
- What its metadata claims
- Whether the save-best logic is buggy or just its metadata writeback is stale
- Whether v3's `_best.pt` will be trustworthy (will the bug recur?) and if not, what to do (e.g., always pick best by re-evaluating `_step_*.pt` files post-hoc)

If a real save-best-logic bug is found, **stop the plan here** and discuss with the user — the bug fix becomes its own small PR before v3 launches.

If only the metadata is stale (weights are right), proceed.

- [ ] **Step 1.5: Commit the report stub**

Even if Task 9 isn't done yet, create the report file with the investigation paragraph:

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git checkout -b feat/stage3-v3-retrain
mkdir -p docs/audits
cat > docs/audits/2026-05-11-stage3-v3-retrain-results.md << 'EOF'
# Stage 3 v3 Retrain Results

**Date:** 2026-05-11
**Spec:** [docs/superpowers/specs/2026-05-11-stage3-v3-retrain-design.md](../superpowers/specs/2026-05-11-stage3-v3-retrain-design.md)
**Plan:** [docs/superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md](../superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md)
**Status:** in progress

## v2 `_best.pt` discrepancy investigation (Task 1)

<TASK 1 PARAGRAPH GOES HERE>

## Phase 1 — Diagnostic (frankenstein checkpoint)

<TASK 3 RESULTS GO HERE>

## Phase 2 — Code fix

<TASK 4 SUMMARY GOES HERE>

## Phase 3 — Retrain (v3, 9000 steps)

<TASK 7 SUMMARY GOES HERE>

## Phase 4 — Re-evaluation

<TASK 8 RESULTS GO HERE>

## Verdict

<TASK 9 CONTENT GOES HERE>
EOF
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md
git commit -m "docs(report): start Stage 3 v3 retrain results report

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Replace `<TASK 1 PARAGRAPH GOES HERE>` with the actual finding from Step 1.4.

---

## Task 2: Frankenstein checkpoint build script (Phase 1 setup)

**Goal:** Create `scripts/audit/build_frankenstein_checkpoint.py` that builds a single state_dict containing Stage 2 v2 encoder weights + Stage 3 v2 non-encoder weights, saved to a file the existing eval driver can consume.

**Files:**
- Create: `scripts/audit/build_frankenstein_checkpoint.py`

- [ ] **Step 2.1: Implement the script**

Create `scripts/audit/build_frankenstein_checkpoint.py`:

```python
"""Build a frankenstein inference checkpoint: Stage 2 v2 encoder + Stage 3 v2
non-encoder weights.

Why: the Stage 3 v2 training audit (PR #48) found the encoder DoRA adapters
were silently updated during Stage 3 v2 training. The decoder trained against
encoder features from the Stage 2 v2 cache; at inference the live encoder is
the drifted Stage 3 v2 version. This script reconstructs the train-time
encoder + Stage 3 decoder pair so the demo eval can run against the
"intended" model — a falsifiable test of whether the encoder drift is the
dominant failure mode.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.build_frankenstein_checkpoint \\
        --stage2-ckpt checkpoints\\full_radio_stage2_systems_v2\\stage2-radio-systems-polyphonic_best.pt \\
        --stage3-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --out checkpoints\\_frankenstein_s2enc_s3dec.pt
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def _get_state_dict(checkpoint):
    """Find the model state_dict inside a checkpoint payload.

    Trainer saves often nest: {"model": <state>, "step": N, ...}. Inference
    saves often: {"state_dict": <state>, ...}. Both formats are supported.
    Returns the state_dict and a dict of any non-state metadata for logging.
    """
    if not isinstance(checkpoint, dict):
        return checkpoint, {}
    for key in ("model", "state_dict"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            metadata = {k: v for k, v in checkpoint.items() if k != key and not isinstance(v, dict)}
            return checkpoint[key], metadata
    # Top-level is already a state_dict
    return checkpoint, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-ckpt", type=Path, required=True,
                   help="Stage 2 v2 best.pt — source of the encoder weights that produced the cache")
    p.add_argument("--stage3-ckpt", type=Path, required=True,
                   help="Stage 3 v2 best.pt — source of the trained decoder + bridge weights")
    p.add_argument("--out", type=Path, required=True,
                   help="Output frankenstein checkpoint path")
    p.add_argument("--encoder-key-marker", type=str, default="encoder",
                   help="Substring identifying encoder params (default: 'encoder')")
    args = p.parse_args()

    import torch

    s2_raw = torch.load(args.stage2_ckpt, map_location="cpu", weights_only=False)
    s3_raw = torch.load(args.stage3_ckpt, map_location="cpu", weights_only=False)

    s2_sd, s2_meta = _get_state_dict(s2_raw)
    s3_sd, s3_meta = _get_state_dict(s3_raw)

    print(f"Stage 2 v2 keys: {len(s2_sd)}")
    print(f"Stage 3 v2 keys: {len(s3_sd)}")
    print(f"Stage 2 v2 metadata: {s2_meta}")
    print(f"Stage 3 v2 metadata: {s3_meta}")

    # Build merged state_dict starting from Stage 3 v2 (which has the trained
    # decoder), then overlay Stage 2 v2's encoder params.
    merged = dict(s3_sd)
    n_from_s2 = 0
    n_missing_in_s2 = 0
    n_kept_from_s3 = 0
    only_in_s2 = []
    only_in_s3 = []
    for k in list(s3_sd.keys()):
        if args.encoder_key_marker in k:
            if k in s2_sd:
                merged[k] = s2_sd[k]
                n_from_s2 += 1
            else:
                # Encoder key in S3 but not S2 — keep S3's version, but flag it.
                # Probably means the encoder grew new parameters between S2 and S3
                # (unlikely but possible if architecture changed).
                n_missing_in_s2 += 1
        else:
            n_kept_from_s3 += 1
    # Keys only in S2 that aren't in S3 — for visibility
    for k in s2_sd:
        if args.encoder_key_marker in k and k not in s3_sd:
            only_in_s2.append(k)
    # Keys only in S3 that aren't in S2 (and aren't encoder) — also for visibility
    for k in s3_sd:
        if args.encoder_key_marker not in k and k not in s2_sd:
            only_in_s3.append(k)

    print()
    print(f"Merge result:")
    print(f"  encoder keys taken from S2: {n_from_s2}")
    print(f"  encoder keys missing in S2 (kept S3 version): {n_missing_in_s2}")
    print(f"  non-encoder keys taken from S3: {n_kept_from_s3}")
    print(f"  encoder keys ONLY in S2 (dropped): {len(only_in_s2)}")
    print(f"  non-encoder keys ONLY in S3 (kept): {len(only_in_s3)}")
    if only_in_s2[:5]:
        print(f"  first 5 S2-only encoder keys: {only_in_s2[:5]}")
    if only_in_s3[:5]:
        print(f"  first 5 S3-only non-encoder keys: {only_in_s3[:5]}")

    # Safety check: if more than 1% of encoder keys mismatch, the experiment
    # may be confounded — escalate before drawing a conclusion.
    n_encoder_keys = n_from_s2 + n_missing_in_s2
    if n_encoder_keys > 0:
        mismatch_pct = 100 * (n_missing_in_s2 + len(only_in_s2)) / n_encoder_keys
        print(f"  encoder-key mismatch rate: {mismatch_pct:.2f}%")
        if mismatch_pct > 1.0:
            print(f"  WARNING: encoder-key mismatch rate exceeds 1%. Diagnostic result may be confounded.")

    # Wrap in the inference-friendly format (top-level state_dict) so the
    # eval driver can load it without special handling.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": merged,
        "note": "frankenstein: stage2_v2_encoder + stage3_v2_decoder",
        "source_stage2": str(args.stage2_ckpt),
        "source_stage3": str(args.stage3_ckpt),
        "n_encoder_keys_from_s2": n_from_s2,
        "n_encoder_keys_missing_in_s2": n_missing_in_s2,
    }
    # Preserve any non-model fields from the S3 payload (DoRA config, vocab, etc.).
    if isinstance(s3_raw, dict):
        for k, v in s3_raw.items():
            if k not in payload and k not in ("model", "state_dict"):
                payload[k] = v
    torch.save(payload, args.out)
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Push to seder**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
scp scripts/audit/build_frankenstein_checkpoint.py '10.10.1.29:audit_frankenstein.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_frankenstein.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\build_frankenstein_checkpoint.py"'
```

- [ ] **Step 2.3: Commit**

```bash
git add scripts/audit/build_frankenstein_checkpoint.py
git commit -m "feat(audit): build frankenstein checkpoint script

Merges Stage 2 v2 encoder + Stage 3 v2 non-encoder weights into a single
state_dict for diagnostic eval. Tests the audit's hypothesis that encoder
drift is the dominant failure mode.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Run Phase 1 diagnostic + GATE

**Goal:** Build the frankenstein checkpoint and run the 4-piece demo eval against it. Decide whether to proceed to Phase 2 retrain or escalate.

- [ ] **Step 3.1: Build the frankenstein checkpoint**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.build_frankenstein_checkpoint --stage2-ckpt checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt --stage3-ckpt checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --out checkpoints\_frankenstein_s2enc_s3dec.pt'
```

Expected output: prints S2 / S3 key counts, merge result, and `Wrote checkpoints\_frankenstein_s2enc_s3dec.pt`. If encoder-key mismatch rate > 1%, **stop and report to user** — experiment is confounded.

- [ ] **Step 3.2: Run the 4-piece demo eval against the frankenstein**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m eval.run_clarity_demo_radio_eval --stage-b-ckpt checkpoints\_frankenstein_s2enc_s3dec.pt --yolo-weights runs\detect\runs\yolo26m_systems\weights\best.pt --name frankenstein_s2enc_s3dec'
```

Expected runtime: ~10 min (4 pieces × ~2.5 min inference + scoring). Watch the per-piece `onset_f1` lines. Record the final `mean onset_f1` value.

- [ ] **Step 3.3: GATE DECISION**

Compare the frankenstein's mean onset_f1 against thresholds from the spec:

| Frankenstein mean onset_f1 | Action |
|---|---|
| **≥ 0.15** | Proceed to Task 4 (Phase 2 fix). Record number in the report. |
| **0.10 – 0.15** | **STOP.** Report numbers to user and discuss before proceeding. The retrain may not close the full gap. |
| **< 0.10** | **STOP.** Diagnosis incomplete — there are failure modes beyond encoder drift. Open a follow-up investigation sub-project. Do NOT proceed to Phase 2/3. |

- [ ] **Step 3.4: Record results in the report file**

Pull the summary.json locally:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/clarity_demo_frankenstein_s2enc_s3dec/summary.json' /tmp/frankenstein_summary.json
```

Replace the `<TASK 3 RESULTS GO HERE>` placeholder in the report file with a paragraph containing:
- Mean onset_f1 + per-piece breakdown
- Mean note_f1
- The gate decision (proceed / stop / escalate) and rationale
- Path to the frankenstein checkpoint on seder (it's gitignored)

Commit:

```bash
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md
git commit -m "audit: Phase 1 diagnostic (frankenstein) results

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: Encoder-freeze fix (Phase 2)

**Goal:** Add cache-derived auto-detect of frozen-encoder mode to `_prepare_model_for_dora`. Add regression tests. Add a pre-flight runtime assertion at the call site.

**Files:**
- Modify: `src/train/train.py` (function at lines ~1228-1340; call site at ~2181)
- Create: `tests/train/test_freeze_encoder.py`

- [ ] **Step 4.1: Write the failing tests first (TDD)**

Create `tests/train/test_freeze_encoder.py`:

```python
"""Regression tests for the Stage 3 v2 encoder-freeze bug.

Bug recap (per `docs/audits/2026-05-11-stage3-v2-training-audit.md`): Stage 3 v2
training trained the encoder DoRA adapters despite the config and checkpoint
name claiming a frozen encoder. The fix makes the freeze a structural
consequence of using the encoder cache — when `stage_config.cache_root` and
`stage_config.cache_hash16` are both set, encoder params must end with
requires_grad=False after _prepare_model_for_dora.

These tests are CUDA-gated by tests/conftest.py (path matches CUDA_REQUIRED_DIRS).
They run on seder where venv-cu132 has torch+CUDA available; they are SKIPPED
locally.
"""
from __future__ import annotations


def _build_minimal_stage_b():
    """Construct a tiny RadioStageB model for parameter-grouping tests.

    Uses the same StageBModelConfig the trainer uses but with deliberately
    small dims so the test runs in seconds. CUDA not strictly required for
    the parameter-grouping test, but the import path requires torch+PEFT
    which exist in venv-cu132 only.
    """
    from src.models.radio_stage_b import RadioStageB
    from src.models.radio_stage_b import RadioStageBConfig
    cfg = RadioStageBConfig(
        decoder_dim=64,
        decoder_heads=2,
        decoder_layers=1,
        vocab_size=64,
    )
    return RadioStageB(cfg)


def _minimal_stage_config(*, cache_root=None, cache_hash16=None):
    from src.train.train import StageTrainingConfig
    return StageTrainingConfig(
        stage_name="test_stage",
        stage_b_encoder="radio_h",
        cache_root=cache_root,
        cache_hash16=cache_hash16,
        lr_dora=5e-4,
        lr_new_modules=3e-4,
    )


def test_cache_config_freezes_encoder_lora_params():
    """When the stage config has cache_root + cache_hash16 set,
    _prepare_model_for_dora must leave every encoder-side parameter with
    requires_grad=False — including LoRA adapters added to encoder modules."""
    from src.train.train import _prepare_model_for_dora
    model = _build_minimal_stage_b()
    stage_config = _minimal_stage_config(
        cache_root="data/cache/encoder",
        cache_hash16="ac8948ae4b5be3e9",
    )
    # Apply DoRA. Implementation detail: the second arg used to be
    # dora_config dict; after the fix it accepts the stage_config object
    # so the function can read cache fields.
    model, _ = _prepare_model_for_dora(model, {"r": 8, "alpha": 16}, stage_config=stage_config)
    trainable_encoder = [
        name for name, p in model.named_parameters()
        if "encoder" in name and p.requires_grad
    ]
    assert trainable_encoder == [], (
        f"Encoder params should be frozen when cache is configured, but found "
        f"{len(trainable_encoder)} trainable encoder params. First 5: "
        f"{trainable_encoder[:5]}"
    )


def test_no_cache_unfreezes_encoder_lora_params():
    """Mirror of the above — proves the gate does real work. When NO cache
    is configured, encoder-side LoRA params should get requires_grad=True."""
    from src.train.train import _prepare_model_for_dora
    model = _build_minimal_stage_b()
    stage_config = _minimal_stage_config()  # no cache
    model, _ = _prepare_model_for_dora(model, {"r": 8, "alpha": 16}, stage_config=stage_config)
    encoder_lora_trainable = [
        name for name, p in model.named_parameters()
        if "encoder" in name and "lora_" in name and p.requires_grad
    ]
    assert len(encoder_lora_trainable) > 0, (
        "Expected encoder-side LoRA params to be trainable when no cache is "
        "configured (the cache+freeze pairing should be the only thing that "
        "triggers the freeze)."
    )


def test_decoder_lora_always_trainable():
    """Sanity check: decoder LoRA params get requires_grad=True regardless
    of whether the encoder is frozen."""
    from src.train.train import _prepare_model_for_dora
    for cache_root in (None, "data/cache/encoder"):
        cache_hash16 = "ac8948ae4b5be3e9" if cache_root else None
        model = _build_minimal_stage_b()
        stage_config = _minimal_stage_config(cache_root=cache_root, cache_hash16=cache_hash16)
        model, _ = _prepare_model_for_dora(model, {"r": 8, "alpha": 16}, stage_config=stage_config)
        decoder_lora_trainable = [
            name for name, p in model.named_parameters()
            if "encoder" not in name and "lora_" in name and p.requires_grad
        ]
        assert len(decoder_lora_trainable) > 0, (
            f"Decoder LoRA should be trainable with cache_root={cache_root!r}. "
            f"Got {len(decoder_lora_trainable)} trainable decoder LoRA params."
        )
```

Note the tests reference `_prepare_model_for_dora(model, dora_config, stage_config=stage_config)` — this is the new signature the fix introduces. The test will fail to import or fail at signature mismatch with the old signature.

- [ ] **Step 4.2: Push tests to seder and verify they FAIL**

```bash
scp tests/train/test_freeze_encoder.py '10.10.1.29:test_freeze_encoder.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\test_freeze_encoder.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\tests\train\test_freeze_encoder.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m pytest tests/train/test_freeze_encoder.py -v 2>&1'
```

Expected: tests FAIL with a `TypeError` about `_prepare_model_for_dora` not accepting `stage_config` kwarg, OR with the assertion that encoder lora params are NOT frozen (because the fix isn't in yet). Either failure mode proves the test is wired correctly.

- [ ] **Step 4.3: Read the current function to plan the minimal change**

```bash
sed -n '1228,1340p' src/train/train.py
```

Identify exactly where the `requires_grad` toggling happens (around lines 1327-1331). Identify the new signature:
- Old: `def _prepare_model_for_dora(model, dora_config: Dict[str, object])`
- New: `def _prepare_model_for_dora(model, dora_config: Dict[str, object], *, stage_config: Optional["StageTrainingConfig"] = None)`

Stage config is optional so existing callers don't break, but the call site in Task 4.5 will pass it.

- [ ] **Step 4.4: Implement the fix**

In `src/train/train.py`, modify `_prepare_model_for_dora`:

```python
def _prepare_model_for_dora(
    model,
    dora_config: Dict[str, object],
    *,
    stage_config: "Optional[StageTrainingConfig]" = None,
):
    """Apply DoRA to ``model`` and configure which parameters receive gradients.

    When ``stage_config`` indicates a cache-backed run (both ``cache_root`` and
    ``cache_hash16`` set), encoder-side parameters are kept frozen
    (``requires_grad=False``) — using cached encoder features and updating the
    encoder are physically incompatible (the cache becomes stale after the
    first encoder gradient step).

    Without a stage_config, falls back to legacy behavior (all ``lora_*`` and
    new-module params trainable), preserving compatibility with existing
    callers that haven't migrated.
    """
    # ... existing DoRA application code stays unchanged up to the requires_grad
    # loop at line ~1327 ...

    uses_cache = bool(
        stage_config is not None
        and stage_config.cache_root
        and stage_config.cache_hash16
    )

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if uses_cache and "encoder" in name:
            continue  # cache requires the encoder be frozen
        if "lora_" in name or any(marker in name for marker in new_module_keywords):
            parameter.requires_grad = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model, dora_applied
```

The two-line change inside the loop is the load-bearing fix: add `uses_cache` derivation, add the `continue` guard.

- [ ] **Step 4.5: Update the call site**

In `src/train/train.py` at line ~2181, change:

```python
model, dora_applied = _prepare_model_for_dora(base_model, components["dora_config"])
```

to:

```python
model, dora_applied = _prepare_model_for_dora(
    base_model, components["dora_config"], stage_config=stage
)
```

Confirm the local variable holding the `StageTrainingConfig` is called `stage` — check the surrounding ~20 lines for confirmation.

- [ ] **Step 4.6: Add the pre-flight runtime assertion**

Immediately after the call site (line ~2181), add:

```python
# Pre-flight: when running a cache-backed stage, every encoder-side parameter
# MUST be frozen. This is what the Stage 3 v2 audit (PR #48) recommended; a
# violation here means the freeze fix in _prepare_model_for_dora regressed.
_uses_cache = bool(stage.cache_root and stage.cache_hash16)
if _uses_cache:
    _trainable_encoder = sum(
        1 for n, p in model.named_parameters() if "encoder" in n and p.requires_grad
    )
    _trainable_decoder = sum(
        1 for n, p in model.named_parameters() if "encoder" not in n and p.requires_grad
    )
    print(f"[freeze] trainable encoder params: {_trainable_encoder}")
    print(f"[freeze] trainable decoder params: {_trainable_decoder}")
    if _trainable_encoder != 0:
        raise RuntimeError(
            f"Stage {stage.stage_name!r} uses encoder cache "
            f"(hash16={stage.cache_hash16!r}), so all encoder params must be "
            f"frozen, but {_trainable_encoder} are trainable. "
            f"_prepare_model_for_dora freeze logic regressed; refusing to train."
        )
```

- [ ] **Step 4.7: Push and verify all three tests PASS**

```bash
scp src/train/train.py '10.10.1.29:train.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\train.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\src\train\train.py"'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m pytest tests/train/test_freeze_encoder.py -v 2>&1'
```

Expected: 3 tests PASS. If anything fails, read the assertion message — the new logic should match exactly.

- [ ] **Step 4.8: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git add src/train/train.py tests/train/test_freeze_encoder.py
git commit -m "fix(train): auto-freeze encoder when stage uses encoder cache

Resolves the Stage 3 v2 audit's primary finding (PR #48): the encoder DoRA
adapters were silently updated during Stage 3 v2 training because
_prepare_model_for_dora flipped requires_grad=True for every parameter
matching 'lora_', with no encoder-side guard. The config and checkpoint
filename both claimed a frozen encoder; no code implemented the freeze.

Fix: derive 'uses_cache' from the stage config (cache_root + cache_hash16),
and when set, keep encoder-side parameters frozen even if they would
otherwise have matched 'lora_'. This couples cache-use and encoder-freeze
in code, eliminating the foot-gun (the cache becomes stale after the first
encoder gradient step — cache + unfrozen encoder is physically meaningless).

Add three regression tests in tests/train/test_freeze_encoder.py covering:
  - cache config freezes encoder LoRA params
  - no cache config unfreezes them (proves the gate does real work)
  - decoder LoRA always trainable regardless of cache

Add a runtime pre-flight assertion at the call site: when uses_cache is
True, count trainable encoder params and refuse to train if non-zero.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

Update the report's `<TASK 4 SUMMARY GOES HERE>` placeholder with a one-paragraph summary of what was changed and the test results.

---

## Task 5: Create the v3 config (Phase 3 setup)

**Goal:** Copy `configs/train_stage3_radio_systems.yaml` to `_v3.yaml` with the 9000-step target and v3 output paths.

**Files:**
- Create: `configs/train_stage3_radio_systems_v3.yaml`

- [ ] **Step 5.1: Copy the v2 config and modify**

```bash
cp configs/train_stage3_radio_systems.yaml configs/train_stage3_radio_systems_v3.yaml
```

Open `configs/train_stage3_radio_systems_v3.yaml` and modify:

1. Replace the existing header comment block with:

```yaml
# Stage 3 Phase 1 v3 — encoder-freeze-fix retrain (supersedes v2).
#
# v2's _best.pt was trained with encoder DoRA accidentally unfrozen (audit
# PR #48). This v3 run applies the cache-derived auto-freeze in
# _prepare_model_for_dora and extends the step count to 9000 (v2 was still
# descending at step 5500/6000).
#
# Resumes from Stage 2 v2 best.pt (val_loss 0.148, step 4000), same as v2.
# Encoder freeze is now structural (cache_root + cache_hash16 set => freeze)
# rather than implicit, so the "frozen-encoder" claim is now enforced.
#
# Run command (executed on GPU box at 10.10.1.29):
#   ssh 10.10.1.29 'cd "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
#     venv-cu132\Scripts\python -u src/train/train.py \
#       --stage-configs configs/train_stage3_radio_systems_v3.yaml \
#       --mode execute \
#       --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \
#       --start-stage stage3-radio-systems-frozen-encoder \
#       --checkpoint-dir checkpoints/full_radio_stage3_v3 \
#       --token-manifest src/data/manifests/token_manifest_stage3.jsonl \
#       --step-log logs/full_radio_stage3_v3_steps.jsonl'
```

2. Change the step target:

```yaml
effective_samples_per_epoch: 9000    # OPT-STEP target; v2 capped at 6000 was still descending
```

3. Leave all other fields unchanged (same `cache_hash16: ac8948ae4b5be3e9`, same optimizer, same schedule, same dataset_mix).

- [ ] **Step 5.2: Push to seder**

```bash
scp configs/train_stage3_radio_systems_v3.yaml '10.10.1.29:stage3_v3.yaml'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\stage3_v3.yaml" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\configs\train_stage3_radio_systems_v3.yaml"'
```

- [ ] **Step 5.3: Commit**

```bash
git add configs/train_stage3_radio_systems_v3.yaml
git commit -m "feat(config): Stage 3 v3 training config

Same as v2 but extends target to 9000 opt-steps and routes output to
checkpoints/full_radio_stage3_v3/ + logs/full_radio_stage3_v3_steps.jsonl.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 6: Resume verification smoke test (Phase 3 prep)

**Goal:** Before launching the 9000-step run, confirm the training resume code path works correctly. If broken, the long run is exposed to mid-flight termination losing all progress.

- [ ] **Step 6.1: Start a deliberately-short v3 run**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -u src/train/train.py --stage-configs configs/train_stage3_radio_systems_v3.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/_stage3_v3_smoke --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v3_smoke_steps.jsonl'
```

Watch for the pre-flight assertion output `[freeze] trainable encoder params: 0` in the first few seconds. If it raises, the fix didn't land — investigate before proceeding.

Let it run to step 100 (you'll see step log lines), then stop it (Ctrl-C through the SSH session, or kill from another SSH).

Confirm the `_step_0000500.pt` checkpoint exists (if step 100 is too early for the first save, run to step 500):

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && dir checkpoints\_stage3_v3_smoke /B'
```

- [ ] **Step 6.2: Resume from the smoke-test checkpoint**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -u src/train/train.py --stage-configs configs/train_stage3_radio_systems_v3.yaml --mode execute --resume-checkpoint checkpoints/_stage3_v3_smoke/stage3-radio-systems-frozen-encoder_step_0000500.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/_stage3_v3_smoke_resume --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v3_smoke_resume_steps.jsonl'
```

Watch the step log: the global_step should resume from 501 (or wherever the resume checkpoint ends), not restart from 1.

If `global_step` does restart from 1, **STOP and report**: resume is broken. The 9000-step main run would be at risk. The fix needs to live as a separate small PR before continuing.

Let it run to step 600 or so, then stop. Verify the loss curve at steps 501-600 matches a continuation of the original run's 401-500 (similar magnitude, no spike, no reinit pattern).

- [ ] **Step 6.3: Clean up smoke checkpoints**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && rmdir /S /Q checkpoints\_stage3_v3_smoke checkpoints\_stage3_v3_smoke_resume'
```

(Leave the step logs in `logs/` for the report.)

- [ ] **Step 6.4: Record finding in report**

Add a one-paragraph note under "## Phase 3 — Retrain" in the report file:
- Smoke run reached step N before stop
- Resume from step_NNNNNNN.pt successfully continued from step N+1 (or noted otherwise if broken)
- Loss curve continuation looked clean

No new commit needed if the smoke results just go in the report; the resume verification artifacts are gitignored.

---

## Task 7: Launch and monitor v3 training (Phase 3)

**Goal:** Run the 9000-step Stage 3 v3 training. Watch for the pre-flight assertion. Watch for catastrophic failure modes early (NaN loss, OOM, divergent val_loss). Capture `_best.pt`.

- [ ] **Step 7.1: Launch the main v3 training run**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -u src/train/train.py --stage-configs configs/train_stage3_radio_systems_v3.yaml --mode execute --resume-checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt --start-stage stage3-radio-systems-frozen-encoder --checkpoint-dir checkpoints/full_radio_stage3_v3 --token-manifest src/data/manifests/token_manifest_stage3.jsonl --step-log logs/full_radio_stage3_v3_steps.jsonl' &
```

Launch in background so the SSH session can drop without killing the run. (If using `run_demo_eval_logged.cmd`-style detachment from the prior session's notes, even better — that approach survived SSH drops during the demo eval re-runs.)

Expected wall-clock: ~8h on seder. Step 500 takes a few minutes; the bulk arrives at fixed cadence after that.

- [ ] **Step 7.2: Confirm pre-flight assertion passed**

After 30 seconds, check the early log:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && type logs\full_radio_stage3_v3_steps.jsonl' | head -3
```

(Or use `findstr` to look for "trainable encoder params" in the stdout log if it's captured separately.)

The pre-flight should have printed `[freeze] trainable encoder params: 0` before training started. If you see a non-zero count, training would have aborted with the RuntimeError — confirm the run did NOT proceed.

- [ ] **Step 7.3: Watch the first few validation passes**

Validation runs every 500 steps. Check step log entries with `validation` non-null:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import json
val_points = []
with open(r''logs/full_radio_stage3_v3_steps.jsonl'', encoding=''utf-8'') as f:
    for line in f:
        d = json.loads(line)
        if d.get(''validation'') is not None:
            v = d[''validation'']
            vloss = v.get(''val_loss'') if isinstance(v, dict) else v
            val_points.append((d.get(''global_step''), vloss, d.get(''loss'')))
for s, v, t in val_points:
    print(f''  step {s:>5}: val={v:.4f}  train={t:.4f}'')
print(f''best so far: {min(val_points, key=lambda x: x[1]) if val_points else None}'')
"'
```

Expected progression: val_loss should drop steadily from the starting value (around Stage 2 v2's val 0.148 at step 0 — since we're resuming from there) and continue descending without bouncing wildly (unlike v2's noisy curve, which the audit attributed to encoder drift).

If val_loss is bouncing more than v2's was, or if it spikes after step 1000, **stop and investigate** — something else is wrong.

- [ ] **Step 7.4: Sanity-check encoder grad-norm groups**

The step log records `grad_norm_groups.encoder_adapter`. With the freeze fix in place, this should be `None` or `0.0` for every step (since encoder params don't get gradients).

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import json
nonzero = 0
total = 0
with open(r''logs/full_radio_stage3_v3_steps.jsonl'', encoding=''utf-8'') as f:
    for line in f:
        d = json.loads(line)
        g = d.get(''grad_norm_groups'', {}) or {}
        enc = g.get(''encoder_adapter'')
        total += 1
        if enc is not None and enc > 1e-8:
            nonzero += 1
print(f''steps with nonzero encoder_adapter grad: {nonzero}/{total}'')
"'
```

Expected: 0 nonzero. If any step shows a nonzero encoder gradient, the freeze isn't truly sticking despite the pre-flight check — a deeper bug worth investigating before training completes.

- [ ] **Step 7.5: Wait for training to complete**

Background job should write to `logs/full_radio_stage3_v3_steps.jsonl` until step 9000. When the file stops growing for >5 min, check whether training finished cleanly:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && dir checkpoints\full_radio_stage3_v3 /B' | tail -20
```

Expected: see `_best.pt`, `_final.pt`, and `_step_0000500.pt` through `_step_0009000.pt`. If the run crashed early, the `_step_*` files stop before 9000 — investigate from there.

- [ ] **Step 7.6: Verify `_best.pt` corresponds to a real step**

Reuse the technique from Task 1.2: compare `_best.pt`'s weights against each `_step_NNNNNNN.pt` to find the matching step.

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "
import torch, os
def get_sd(c): return c.get(''model'') or c.get(''state_dict'') or c
best = get_sd(torch.load(r''checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt'', map_location=''cpu'', weights_only=False))
# Pick one reference key
ref_key = next(iter(best))
ref_val = best[ref_key].float()
for step in range(500, 9500, 500):
    path = rf''checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_step_{step:07d}.pt''
    if not os.path.exists(path): continue
    sd = get_sd(torch.load(path, map_location=''cpu'', weights_only=False))
    if ref_key in sd:
        diff = (ref_val - sd[ref_key].float()).abs().max().item()
        if diff < 1e-6:
            print(f''_best.pt matches step {step}'')
            break
else:
    print(''_best.pt does not match any _step file (suspect)'')"'
```

Record the matching step in the report under Phase 3 results.

- [ ] **Step 7.7: Update report**

Replace `<TASK 7 SUMMARY GOES HERE>` in the report file with:
- Training start/end timestamps + wall-clock
- Step where val_loss peaked + value
- Step where `_best.pt` actually corresponds to (Task 7.6 finding)
- Encoder grad-norm sanity result (should be all zero)
- Whether training reached step 9000 or stopped earlier

Commit:

```bash
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md
git commit -m "audit: Phase 3 (v3 training) results

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 8: Phase 4 — Re-evaluation (A2 + demo + lieder)

**Goal:** Run the three re-evaluation passes against `checkpoints/full_radio_stage3_v3/_best.pt`. Apply the decision matrix.

- [ ] **Step 8.1: Re-run A2 encoder parity against v3**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a2_encoder_parity --manifest src/data/manifests/token_manifest_stage3.jsonl --cache-root data/cache/encoder --cache-hash16 ac8948ae4b5be3e9 --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 5 --out audit_results/a2_encoder_stage3_v3.json'
```

Pull and inspect:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a2_encoder_stage3_v3.json' /tmp/a2_v3.json
python3 -c "
import json
d = json.load(open('/tmp/a2_v3.json'))
print('PASS' if d['pass'] else 'FAIL')
print('bilinear max:', d.get('bilinear_vs_cache_overall_max_abs', d.get('overall_max_abs_diff')))
print('bilinear mean:', d.get('bilinear_vs_cache_overall_mean_abs', d.get('overall_mean_abs_diff')))
"
```

Expected: PASS with bilinear max_abs_diff well below 0.01 — the encoder didn't move during training. If FAIL, the freeze didn't actually stick during training despite the pre-flight assertion passing and the grad-norm sanity showing zeros — this is the paranoid case; debug before continuing.

- [ ] **Step 8.2: Run the 4-piece HF demo eval against v3**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m eval.run_clarity_demo_radio_eval --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt --name stage3_v3_best'
```

Expected runtime: ~10 min. Pull summary:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/eval/results/clarity_demo_stage3_v3_best/summary.json' /tmp/demo_v3.json
python3 -c "
import json
d = json.load(open('/tmp/demo_v3.json'))['pieces']
import statistics
of1 = [p['score']['onset_f1'] for p in d.values() if 'onset_f1' in p.get('score',{})]
print(f'mean onset_f1: {statistics.mean(of1):.4f}')
for stem, p in d.items():
    s = p.get('score', {})
    print(f'  {stem[:35]:<35} onset_f1={s.get(\"onset_f1\",0):.4f} note_f1={s.get(\"f1\",0):.4f}')
"
```

Record the mean and per-piece onset_f1.

- [ ] **Step 8.3: Run the 50-piece lieder ship-gate against v3**

Find the canonical lieder eval entry point:

```bash
ls eval/ | grep -i lieder
```

Expected: `eval/run_lieder_eval.py` (or similarly named — check the Subproject 4 references).

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m eval.run_lieder_eval --stage-b-ckpt checkpoints/full_radio_stage3_v3/stage3-radio-systems-frozen-encoder_best.pt --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt --name stage3_v3_best'
```

Expected runtime: 1-3h (50 pieces). Run in background.

Pull summary when done and compute corpus mean onset_f1.

- [ ] **Step 8.4: Apply the decision matrix from the spec**

| A2 | Demo mean onset_f1 | Verdict |
|---|---|---|
| PASS | ≥ 0.241 | **SHIP.** Foundation was the only issue. Close the audit chapter. |
| PASS | 0.10 – 0.241 | Foundation now sound; remaining gap is architecture/data. Next: A3 + Phase B from the audit, or pivot to DaViT baseline. |
| PASS | < 0.10 | Foundation sound but model doesn't generalize. Urgent: architecture investigation. |
| FAIL | — | Freeze didn't stick. Debug Phase 2 implementation. |

Record the verdict.

- [ ] **Step 8.5: Update the report with Phase 4 results**

Replace `<TASK 8 RESULTS GO HERE>` in the report with:
- A2 result (PASS / FAIL, max_abs_diff number)
- Demo eval: per-piece + mean onset_f1, comparison to v2 (mean was 0.0589 post-first-emission)
- Lieder eval: corpus mean onset_f1, comparison to v2 (0.0819)
- Chosen verdict from the matrix

Commit:

```bash
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md
git commit -m "audit: Phase 4 (re-evaluation) results

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 9: Final report and PR

**Goal:** Write the verdict paragraph, fill out the report's remaining placeholders, open the PR.

- [ ] **Step 9.1: Replace the `<TASK 9 CONTENT GOES HERE>` placeholder**

Open `docs/audits/2026-05-11-stage3-v3-retrain-results.md` and replace the `<TASK 9 CONTENT GOES HERE>` placeholder under "## Verdict" with a clear paragraph:

- Which decision-matrix branch the v3 results landed in
- The single number that's most diagnostic (A2 max_abs_diff and/or demo mean onset_f1)
- What this means for the project (e.g., SHIP / pivot to architecture investigation / further audit needed)
- What the next sub-project, if any, should be

Also verify every other `<...>` placeholder in the file has been filled.

- [ ] **Step 9.2: Final commit**

```bash
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md
git commit -m "audit: Stage 3 v3 retrain final report and verdict

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 9.3: Push and open PR**

```bash
git push -u origin feat/stage3-v3-retrain
gh pr create --title "feat(train): fix encoder-freeze bug and retrain Stage 3 v3" --body "$(cat <<'EOF'
## Summary

Resolves the encoder-DoRA-not-frozen bug diagnosed in the Stage 3 v2 training
audit (PR #48). Adds a cache-derived auto-freeze in `_prepare_model_for_dora`,
three regression tests, a runtime pre-flight assertion, and a new training
config for the v3 retrain.

## Files

- `src/train/train.py` — fix in `_prepare_model_for_dora` + pre-flight at the call site
- `tests/train/test_freeze_encoder.py` — three regression tests (CUDA-gated)
- `configs/train_stage3_radio_systems_v3.yaml` — v3 training config (9000 steps)
- `scripts/audit/build_frankenstein_checkpoint.py` — Phase 1 diagnostic helper
- `docs/audits/2026-05-11-stage3-v3-retrain-results.md` — final report

## Phase 1 diagnostic result

<COPY FROM REPORT — frankenstein eval mean onset_f1 and gate decision>

## Phase 4 verdict

<COPY FROM REPORT — A2 result + demo eval result + lieder result + matrix branch>

## Test plan

- [ ] Reviewer: confirm `pytest tests/train/test_freeze_encoder.py -v` shows 3 PASSED on seder
- [ ] Reviewer: spot-check that the pre-flight assertion fires when the freeze regresses (run any cache-backed config with the loop's `continue` line commented out — should `RuntimeError` at startup)
- [ ] Reviewer: verify `audit_results/a2_encoder_stage3_v3.json` reports PASS

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `<COPY FROM REPORT — ...>` placeholders before submitting with the actual numbers from the report file.

---

## Self-review checklist

- [x] **Spec coverage:** Each section of the spec maps to tasks.
  - Phase 1 → Tasks 2 + 3
  - Phase 2 → Task 4
  - Phase 3 → Tasks 5 + 6 + 7
  - Phase 4 → Task 8 + 9
  - Best-checkpoint accounting (spec Phase 3 §) → Task 1
- [x] **Placeholder scan:** Real placeholders only exist where output data will fill them in at execution time (`<TASK N ... GOES HERE>` markers in the report file template, and `<COPY FROM REPORT — ...>` markers in the PR body). All code blocks contain complete code.
- [x] **Type consistency:** `_prepare_model_for_dora` signature matches between Task 4.1 (test), Task 4.4 (implementation), Task 4.5 (call site update). `StageTrainingConfig` field names `cache_root` and `cache_hash16` match the existing code (`src/train/train.py:101-102`).
- [x] **Path resolution:** Manifest path resolved in Task 2 of the audit plan (`src/data/manifests/token_manifest_stage3.jsonl`); cache root resolved (`data/cache/encoder`). Both reused directly here without re-discovery.
- [x] **Gate handling:** Task 3.3 explicitly halts the plan if Phase 1 gate fails. Task 6.2 explicitly halts if resume is broken. Task 1.4 explicitly halts if `_best.pt` save-best logic is buggy (not just metadata stale).
