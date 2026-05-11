# Stage 3 v2 Failure-Mode Investigation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the design in [`docs/superpowers/specs/2026-05-11-stage3-v2-failure-mode-investigation-design.md`](../specs/2026-05-11-stage3-v2-failure-mode-investigation-design.md): two diagnostic streams that identify the dominant failure mode for Stage 3 v2 (now that the audit's encoder-drift hypothesis is disproved). Produce a revised verdict that steers the next sub-project.

**Architecture:** Two scripts on `feat/stage3-v3-retrain` (the existing diagnostic-infrastructure branch). Stream A reuses sample-picker from prior audit; Stream B is new instrumentation. Both write JSON outputs consumed by the report.

**Tech Stack:** Python 3.13, PyTorch on seder via `venv-cu132`, music21, PIL. No new third-party deps.

---

## File Structure

**New files (`feat/stage3-v3-retrain` branch):**
- `scripts/audit/a3_decoder_on_training.py` — Stream A: decoder round-trip on training data
- `scripts/audit/pipeline_note_loss.py` — Stream B: pipeline-stage note-count diagnostic

**Modified files:**
- `docs/audits/2026-05-11-stage3-v3-retrain-results.md` — extend with Stream A + Stream B sections + revised verdict

**Generated artifacts (on seder, not in git):**
- `audit_results/a3_decoder_stage3_v2.json`
- `audit_results/pipeline_note_loss_clair_de_lune.json` (one per piece run)

---

## Task 1: Stream A — A3 decoder-on-training script

**Goal:** Run end-to-end Stage A+B inference on 20 training samples, compare predicted tokens to ground-truth labels, report token accuracy and per-class accuracy.

**Files:**
- Create: `scripts/audit/a3_decoder_on_training.py`

- [ ] **Step 1.1: Implement the script**

Create `scripts/audit/a3_decoder_on_training.py` with the implementation sketched in the original audit plan (Task 4 of `docs/superpowers/plans/2026-05-11-stage3-v2-training-audit-plan.md`, which was skipped at fast-exit):

```python
"""A3: Decoder behavior on training data.

The model has been trained on these exact samples (with whatever
augmentation was active). At inference time, on the same images, the
predicted token sequence should closely match the ground-truth labels.

Pass criteria (triage thresholds, not certification):
  - token accuracy >= 80% averaged across samples
  - exact-match sequence rate >= 30%
  - per-class accuracy for time-sig tokens >= 80%
  - per-class accuracy for key-sig tokens >= 80%

Lower numbers indicate the model didn't memorize its training data
(unusual for a typical autoregressive transformer with enough capacity),
which would suggest either training didn't converge or there's a
preprocessing skew that A1/A2 didn't catch.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a3_decoder_on_training \\
        --manifest src\\data\\manifests\\token_manifest_stage3.jsonl \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --n-per-corpus 5 \\
        --out audit_results\\a3_decoder.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def _token_accuracy(predicted: list, target: list) -> float:
    """Per-position accuracy over min(len(predicted), len(target)).

    Conservative: short predictions get 100% on their prefix; exact-match
    catches the length-mismatch case separately.
    """
    n = min(len(predicted), len(target))
    if n == 0:
        return 0.0
    correct = sum(1 for a, b in zip(predicted[:n], target[:n]) if a == b)
    return correct / n


def _per_class_accuracy(predicted: list, target: list, prefix: str) -> tuple[int, int]:
    """Returns (correct, total) for positions where the target token starts with prefix."""
    correct, total = 0, 0
    n = min(len(predicted), len(target))
    for i in range(n):
        if target[i].startswith(prefix):
            total += 1
            if predicted[i] == target[i]:
                correct += 1
    return correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--max-decode-steps", type=int, default=2048)
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from src.inference.checkpoint_load import load_stage_b_for_inference
    from src.inference.decoder_runtime import _load_stage_b_crop_tensor
    from src.inference.system_pipeline import _encode_staff_image, _decode_stage_b_tokens
    from scripts.audit._sample_picker import pick_audit_samples

    device = torch.device("cuda")
    bundle = load_stage_b_for_inference(args.stage_b_ckpt, device, use_fp16=False)

    samples = pick_audit_samples(args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed)
    print(f"Selected {len(samples)} samples; running decoder on each...")

    per_sample = []
    for sample in samples:
        sample_id = sample["sample_id"]
        dataset = sample["dataset"]
        img_path = _REPO / sample["image_path"]
        target = sample.get("token_sequence", [])

        if not img_path.exists():
            per_sample.append({
                "sample_id": sample_id, "dataset": dataset,
                "status": "missing_image",
            })
            continue

        pixel_values = _load_stage_b_crop_tensor(
            img_path,
            image_height=args.image_height,
            image_max_width=args.image_max_width,
            device=device,
        )
        with torch.no_grad():
            memory = _encode_staff_image(bundle.decode_model, pixel_values)
            predicted = _decode_stage_b_tokens(
                model=bundle.model,
                pixel_values=pixel_values,
                vocabulary=bundle.vocab,
                beam_width=1,
                max_decode_steps=args.max_decode_steps,
                length_penalty_alpha=0.4,
                _precomputed={
                    "decode_model": bundle.decode_model,
                    "memory": memory,
                    "token_to_idx": bundle.token_to_idx,
                    "use_fp16": False,
                },
            )

        token_acc = _token_accuracy(predicted, target)
        exact_match = predicted == target
        ts_c, ts_n = _per_class_accuracy(predicted, target, "timeSignature-")
        ks_c, ks_n = _per_class_accuracy(predicted, target, "keySignature-")
        note_c, note_n = _per_class_accuracy(predicted, target, "note-")
        rest_c, rest_n = _per_class_accuracy(predicted, target, "rest")

        per_sample.append({
            "sample_id": sample_id, "dataset": dataset,
            "status": "compared",
            "predicted_len": len(predicted),
            "target_len": len(target),
            "token_accuracy": token_acc,
            "exact_match": exact_match,
            "timeSig_correct": ts_c, "timeSig_total": ts_n,
            "keySig_correct": ks_c, "keySig_total": ks_n,
            "note_correct": note_c, "note_total": note_n,
            "rest_correct": rest_c, "rest_total": rest_n,
            "predicted_first_50": predicted[:50],
            "target_first_50": target[:50],
        })

    compared = [r for r in per_sample if r["status"] == "compared"]
    n_compared = len(compared) or 1

    mean_token_acc = sum(r["token_accuracy"] for r in compared) / n_compared
    exact_match_rate = sum(1 for r in compared if r["exact_match"]) / n_compared

    def _ratio(c, n):
        return c / n if n else None

    ts_c_total = sum(r["timeSig_correct"] for r in compared)
    ts_n_total = sum(r["timeSig_total"] for r in compared)
    ks_c_total = sum(r["keySig_correct"] for r in compared)
    ks_n_total = sum(r["keySig_total"] for r in compared)
    note_c_total = sum(r["note_correct"] for r in compared)
    note_n_total = sum(r["note_total"] for r in compared)
    rest_c_total = sum(r["rest_correct"] for r in compared)
    rest_n_total = sum(r["rest_total"] for r in compared)

    results = {
        "experiment": "a3_decoder_on_training",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_samples": len(per_sample),
        "mean_token_accuracy": mean_token_acc,
        "exact_match_rate": exact_match_rate,
        "timeSig_accuracy": _ratio(ts_c_total, ts_n_total),
        "keySig_accuracy": _ratio(ks_c_total, ks_n_total),
        "note_accuracy": _ratio(note_c_total, note_n_total),
        "rest_accuracy": _ratio(rest_c_total, rest_n_total),
        "per_sample": per_sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"=== A3: Decoder on training data ===")
    print(f"Samples compared: {n_compared}")
    print(f"Mean token accuracy: {mean_token_acc:.3f}")
    print(f"Exact match rate:    {exact_match_rate:.3f}")
    print(f"timeSig accuracy:    {results['timeSig_accuracy']}")
    print(f"keySig accuracy:     {results['keySig_accuracy']}")
    print(f"note accuracy:       {results['note_accuracy']}")
    print(f"rest accuracy:       {results['rest_accuracy']}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 1.2: Push to seder**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
scp scripts/audit/a3_decoder_on_training.py '10.10.1.29:audit_a3.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_a3.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\a3_decoder_on_training.py"'
```

- [ ] **Step 1.3: Smoke-run A3 on 2 samples per corpus**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a3_decoder_on_training --manifest src/data/manifests/token_manifest_stage3.jsonl --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 2 --out audit_results/a3_smoke.json'
```

Expected: 8 samples compared, prints accuracy numbers. If the script crashes mid-run (OOM, missing image, etc.), record the error and report BLOCKED.

- [ ] **Step 1.4: Run full A3 (5 per corpus)**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a3_decoder_on_training --manifest src/data/manifests/token_manifest_stage3.jsonl --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 5 --out audit_results/a3_decoder_stage3_v2.json'
```

Pull and inspect:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a3_decoder_stage3_v2.json' /tmp/a3.json
python3 -c "
import json
d = json.load(open('/tmp/a3.json'))
print(f'tok_acc:     {d[\"mean_token_accuracy\"]:.3f}')
print(f'exact_match: {d[\"exact_match_rate\"]:.3f}')
print(f'timeSig:     {d.get(\"timeSig_accuracy\")}')
print(f'keySig:      {d.get(\"keySig_accuracy\")}')
print(f'note:        {d.get(\"note_accuracy\")}')
print(f'rest:        {d.get(\"rest_accuracy\")}')
from collections import Counter
status = Counter(r['status'] for r in d['per_sample'])
print(f'status:      {dict(status)}')
print()
print('per-corpus token accuracy:')
by_corpus = {}
for r in d['per_sample']:
    if r['status'] == 'compared':
        by_corpus.setdefault(r['dataset'], []).append(r['token_accuracy'])
for c, vals in sorted(by_corpus.items()):
    print(f'  {c:<26} n={len(vals)} mean={sum(vals)/len(vals):.3f}')
"
```

Record the numbers in the report (Step 1.5).

- [ ] **Step 1.5: Update report with Stream A results**

Open `docs/audits/2026-05-11-stage3-v3-retrain-results.md`. Replace the existing `## Phase 2 — Code fix` heading (now obsolete since we're not retraining) and the placeholder under it. Insert a new section after the Phase 1 frankenstein results:

```markdown
## Stream A — Decoder round-trip on training data

**Goal:** Test whether the Stage 3 v2 decoder reproduces its own training labels at inference. Confounding caveat: decoder was trained against cached features; this run uses the live (drifted) encoder, so a low result could mean either decoder is broken OR decoder is fine but mismatched with the live encoder.

**Numbers (full run, 20 samples across 4 corpora, `audit_results/a3_decoder_stage3_v2.json` on seder):**

| Metric | Value |
|---|---|
| Mean token accuracy | <FILL FROM RESULT> |
| Exact match rate | <FILL FROM RESULT> |
| timeSig accuracy | <FILL FROM RESULT> |
| keySig accuracy | <FILL FROM RESULT> |
| note accuracy | <FILL FROM RESULT> |
| rest accuracy | <FILL FROM RESULT> |

Per-corpus token accuracy:
- synthetic_systems: <FILL FROM RESULT>
- grandstaff_systems: <FILL FROM RESULT>
- primus_systems: <FILL FROM RESULT>
- cameraprimus_systems: <FILL FROM RESULT>

**Interpretation.** <1-2 sentences describing what the numbers mean given the triage thresholds (>= 80% / 50-80% / < 50%). Note the confound and how it shapes interpretation. Compare to the frankenstein result for cross-checking.>
```

Fill the `<FILL FROM RESULT>` markers with the exact numbers from the JSON. Commit:

```bash
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md scripts/audit/a3_decoder_on_training.py
git commit -m "feat(audit): Stream A — A3 decoder round-trip on training data

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: Stream B — Pipeline-stage note-loss script

**Goal:** Instrument the inference pipeline to count notes at six stages (raw decoder output → MusicXML file) plus the reference, on a chosen demo piece. Identify where notes are lost.

**Files:**
- Create: `scripts/audit/pipeline_note_loss.py`

- [ ] **Step 2.1: Implement the script**

Create `scripts/audit/pipeline_note_loss.py`:

```python
"""Stream B: per-stage note-count diagnostic for the inference pipeline.

Runs Stage A + Stage B + Stage D end-to-end on one piece, but hooks the
intermediate boundaries to count tokens matching `^note-` at each stage.
Final count is from re-parsing the written MusicXML via music21.
Compares against the reference .mxl note count.

The most informative comparison is the drop BETWEEN stages: if the raw
decoder produces N notes and the final MusicXML has 0.2*N, the assembly
pipeline is dropping notes. If the raw decoder produces 0.2*N already,
the decoder is the bottleneck.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.pipeline_note_loss \\
        --pdf data\\clarity_demo\\pdf\\clair-de-lune-debussy.pdf \\
        --ref data\\clarity_demo\\mxl\\clair-de-lune-debussy.mxl \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --yolo-weights runs\\detect\\runs\\yolo26m_systems\\weights\\best.pt \\
        --out audit_results\\pipeline_note_loss_clair_de_lune.json
"""
from __future__ import annotations
import argparse
import json
import sys
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

_NOTE_RE = re.compile(r"^note-")


def _count_note_tokens(tokens) -> int:
    """Count tokens matching `^note-` in a sequence."""
    return sum(1 for t in tokens if isinstance(t, str) and _NOTE_RE.match(t))


def _count_music21_notes(score) -> int:
    """Count Note objects (including those inside Chords) in a music21 score, after stripTies."""
    import music21
    s = score.stripTies(retainContainers=True)
    flat = s.flatten()
    n_notes = len(flat.getElementsByClass(music21.note.Note))
    # Each Chord contains multiple notes — count them too.
    for ch in flat.getElementsByClass(music21.chord.Chord):
        n_notes += len(ch.notes)
    return n_notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True,
                   help="Reference .mxl file for ground-truth note count")
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--yolo-weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--page-dpi", type=int, default=300)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--max-decode-steps", type=int, default=2048)
    args = p.parse_args()

    import music21
    import torch
    from PIL import Image
    import fitz

    from src.inference.system_pipeline import (
        SystemInferencePipeline,
        _encode_staff_image,
    )
    from src.inference.decoder_runtime import (
        _load_stage_b_crop_tensor,
        _decode_stage_b_tokens,
    )
    from src.data.convert_tokens import _split_staff_sequences_for_validation
    from src.pipeline.assemble_score import (
        assemble_score,
        _enforce_global_key_time,
        StaffRecognitionResult,
        StaffLocation,
    )
    from src.pipeline.export_musicxml import (
        StageDExportDiagnostics,
        assembled_score_to_music21_with_diagnostics,
    )

    # --- Reference count ---
    ref_score = music21.converter.parse(str(args.ref))
    ref_notes = _count_music21_notes(ref_score)
    print(f"Reference notes (after stripTies): {ref_notes}")

    # --- Build pipeline (loads model once) ---
    pipeline = SystemInferencePipeline(
        yolo_weights=args.yolo_weights,
        stage_b_ckpt=args.stage_b_ckpt,
        page_dpi=args.page_dpi,
        image_height=args.image_height,
        image_max_width=args.image_max_width,
        max_decode_steps=args.max_decode_steps,
    )

    # --- Stage 1: Raw decoder output (per system) ---
    # Re-run the page+system loop manually so we can capture intermediates.
    all_token_lists = []
    all_locations = []
    with fitz.open(str(args.pdf)) as doc:
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=args.page_dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            systems = pipeline._stage_a.detect_systems(img)
            for sys_d in systems:
                x1, y1, x2, y2 = sys_d["bbox_extended"]
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                tokens = pipeline._decode_one_crop(crop)
                all_token_lists.append(tokens)
                all_locations.append({
                    "system_index": sys_d["system_index"],
                    "bbox": sys_d["bbox_extended"],
                    "page_index": page_index,
                    "conf": sys_d["conf"],
                })

    stage1_count = sum(_count_note_tokens(tokens) for tokens in all_token_lists)
    print(f"Stage 1 (raw decoder output): {stage1_count} note tokens across {len(all_token_lists)} systems")

    # --- Stage 2: After staff split ---
    stage2_count = 0
    stage2_skipped_systems = 0
    staves = []  # for stage 3+
    for sys_tokens, sys_loc in zip(all_token_lists, all_locations):
        try:
            per_staff = _split_staff_sequences_for_validation(sys_tokens)
        except ValueError:
            stage2_skipped_systems += 1
            continue
        if not per_staff:
            continue
        n = len(per_staff)
        sys_idx = int(sys_loc["system_index"])
        page_idx = int(sys_loc.get("page_index", 0))
        x1, y1, x2, y2 = sys_loc["bbox"]
        sys_h = float(y2) - float(y1)
        for i, staff_tokens in enumerate(per_staff):
            stage2_count += _count_note_tokens(staff_tokens)
            y_top = float(y1) + i * sys_h / n
            y_bottom = float(y1) + (i + 1) * sys_h / n
            location = StaffLocation(
                page_index=page_idx,
                y_top=y_top, y_bottom=y_bottom,
                x_left=float(x1), x_right=float(x2),
            )
            staves.append(StaffRecognitionResult(
                sample_id=f"page{page_idx:04d}_sys{sys_idx:02d}_staff{i:02d}",
                tokens=list(staff_tokens),
                location=location,
                system_index_hint=sys_idx,
            ))
    print(f"Stage 2 (after staff split): {stage2_count} note tokens across {len(staves)} staves (skipped {stage2_skipped_systems} systems)")

    # --- Stage 3: After _enforce_global_key_time + post_process_tokens (run via assemble_score) ---
    # assemble_score runs _enforce_global_key_time + per-system _normalize_measure_count + post_process_tokens.
    score = assemble_score(staves)
    stage3_count = 0
    for system in score.systems:
        for staff in system.staves:
            stage3_count += _count_note_tokens(staff.tokens)
    print(f"Stage 3 (after _enforce_global_key_time + post_process): {stage3_count} note tokens")

    # --- Stage 4: AssembledScore tokens (identical to stage 3 in current code, but capture separately for symmetry) ---
    stage4_count = stage3_count

    # --- Stage 5: music21 Note count (in-memory) ---
    diagnostics = StageDExportDiagnostics()
    music_score = assembled_score_to_music21_with_diagnostics(score, diagnostics, strict=False)
    stage5_count = _count_music21_notes(music_score)
    print(f"Stage 5 (music21 in-memory Notes): {stage5_count}")

    # --- Stage 6: Re-parse the written MusicXML ---
    out_musicxml = args.out.with_suffix(".musicxml")
    music_score.write("musicxml", str(out_musicxml))
    reparsed = music21.converter.parse(str(out_musicxml))
    stage6_count = _count_music21_notes(reparsed)
    print(f"Stage 6 (re-parsed MusicXML): {stage6_count}")

    # --- Diagnostics ---
    print()
    print("Stage D diagnostics (informational):")
    print(f"  padded_measures: {diagnostics.padded_measures}")
    print(f"  skipped_systems: {diagnostics.skipped_systems}")
    print(f"  skipped_notes: {diagnostics.skipped_notes}")
    print(f"  skipped_chords: {diagnostics.skipped_chords}")
    print(f"  missing_durations: {diagnostics.missing_durations}")
    print(f"  unknown_tokens: {diagnostics.unknown_tokens}")
    print(f"  fallback_rests: {diagnostics.fallback_rests}")

    results = {
        "experiment": "pipeline_note_loss",
        "pdf": str(args.pdf),
        "ref": str(args.ref),
        "checkpoint": str(args.stage_b_ckpt),
        "reference_notes": ref_notes,
        "stages": {
            "1_raw_decoder_output": stage1_count,
            "2_after_staff_split": stage2_count,
            "3_after_enforce_and_post_process": stage3_count,
            "4_assembled_score_tokens": stage4_count,
            "5_music21_in_memory": stage5_count,
            "6_reparsed_musicxml": stage6_count,
        },
        "deltas": {
            "ref_to_stage1": stage1_count - ref_notes,
            "stage1_to_stage2": stage2_count - stage1_count,
            "stage2_to_stage3": stage3_count - stage2_count,
            "stage3_to_stage5": stage5_count - stage3_count,
            "stage5_to_stage6": stage6_count - stage5_count,
        },
        "n_systems": len(all_token_lists),
        "n_staves": len(staves),
        "stage2_skipped_systems": stage2_skipped_systems,
        "stage_d_diagnostics": {
            "padded_measures": diagnostics.padded_measures,
            "skipped_systems": diagnostics.skipped_systems,
            "skipped_notes": diagnostics.skipped_notes,
            "skipped_chords": diagnostics.skipped_chords,
            "missing_durations": diagnostics.missing_durations,
            "unknown_tokens": diagnostics.unknown_tokens,
            "fallback_rests": diagnostics.fallback_rests,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Push to seder**

```bash
scp scripts/audit/pipeline_note_loss.py '10.10.1.29:audit_note_loss.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_note_loss.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\pipeline_note_loss.py"'
```

- [ ] **Step 2.3: Run on Clair de Lune (largest under-generation case)**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.pipeline_note_loss --pdf data/clarity_demo/pdf/clair-de-lune-debussy.pdf --ref data/clarity_demo/mxl/clair-de-lune-debussy.mxl --stage-b-ckpt checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt --yolo-weights runs/detect/runs/yolo26m_systems/weights/best.pt --out audit_results/pipeline_note_loss_clair_de_lune.json'
```

Expected runtime: ~2.5 min (single piece inference + scoring). Prints per-stage counts and Stage D diagnostics.

- [ ] **Step 2.4: Run on the other 3 demo pieces**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && (for %s in (fugue-no-2-bwv-847-in-c-minor gnossienne-no-1 prelude-in-d-flat-major-op31-no1-scriabin) do @venv-cu132\Scripts\python.exe -m scripts.audit.pipeline_note_loss --pdf data\clarity_demo\pdf\%s.pdf --ref data\clarity_demo\mxl\%s.mxl --stage-b-ckpt checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --yolo-weights runs\detect\runs\yolo26m_systems\weights\best.pt --out audit_results\pipeline_note_loss_%s.json)'
```

Expected runtime: 4 pieces × ~2.5 min = ~10 min total. (Note: each invocation reloads the model — could batch in a single script but not worth the complexity for 4 pieces.)

Pull all 4 results locally:

```bash
for stem in clair-de-lune-debussy fugue-no-2-bwv-847-in-c-minor gnossienne-no-1 prelude-in-d-flat-major-op31-no1-scriabin; do
    scp "10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/pipeline_note_loss_${stem}.json" "/tmp/nl_${stem}.json"
done
python3 -c "
import json
import os
for stem in ['clair-de-lune-debussy', 'fugue-no-2-bwv-847-in-c-minor', 'gnossienne-no-1', 'prelude-in-d-flat-major-op31-no1-scriabin']:
    path = f'/tmp/nl_{stem}.json'
    if not os.path.exists(path): continue
    d = json.load(open(path))
    print(f'=== {stem} (ref={d[\"reference_notes\"]}) ===')
    s = d['stages']
    print(f\"  1 raw decoder:     {s['1_raw_decoder_output']:>5}\")
    print(f\"  2 staff-split:     {s['2_after_staff_split']:>5}\")
    print(f\"  3 post-process:    {s['3_after_enforce_and_post_process']:>5}\")
    print(f\"  5 music21 mem:     {s['5_music21_in_memory']:>5}\")
    print(f\"  6 reparsed MXL:    {s['6_reparsed_musicxml']:>5}\")
    print(f\"  ref:               {d['reference_notes']:>5}\")
"
```

Record the per-piece stage counts.

- [ ] **Step 2.5: Update report with Stream B results**

Replace the (now-irrelevant) `## Phase 3 — Retrain` and `## Phase 4 — Re-evaluation` sections in `docs/audits/2026-05-11-stage3-v3-retrain-results.md` with a new Stream B section:

```markdown
## Stream B — Pipeline-stage note loss

**Goal:** Find which pipeline stage(s) drop notes between the raw decoder output and the final MusicXML file.

**Per-piece stage counts (`audit_results/pipeline_note_loss_*.json` on seder):**

| Piece | ref | 1 raw decoder | 2 staff-split | 3 post-process | 5 music21 mem | 6 reparsed MXL |
|---|---:|---:|---:|---:|---:|---:|
| clair-de-lune | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| fugue-no-2 | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| gnossienne-no-1 | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |
| prelude-op31 | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> | <FILL> |

**Interpretation.** <Identify where the largest drop occurs across pieces. Is it consistent across pieces? Is the decoder undergenerating from the start (stage 1 << reference), or does the pipeline drop notes downstream (stage 5 << stage 1)? Cite the most-affected piece and stage.>
```

Fill the `<FILL>` markers with exact numbers. Commit:

```bash
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md scripts/audit/pipeline_note_loss.py
git commit -m "feat(audit): Stream B — pipeline-stage note-loss diagnostic

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: Revised verdict + final commit + PR

**Goal:** Synthesize Stream A and Stream B into a revised verdict that identifies the dominant failure mode and recommends a follow-up sub-project. Open the PR for the diagnostic branch.

- [ ] **Step 3.1: Write the revised verdict**

Open `docs/audits/2026-05-11-stage3-v3-retrain-results.md`. Replace the placeholder under `## Verdict` (and remove any obsolete `<TASK N>` markers from earlier in the document — Tasks 4-9 of the v3 retrain plan were not executed). The verdict should be ~3-5 paragraphs covering:

1. **Recap:** what we set out to test (Stage 3 v2 audit's encoder-drift hypothesis); what the frankenstein experiment showed (hypothesis disproved).
2. **What Stream A revealed:** decoder's ability to reproduce its own training labels (cite token accuracy, exact-match rate, per-class). Note the live-encoder confound but interpret in context.
3. **What Stream B revealed:** where in the pipeline notes are lost (cite the biggest drop stage from the per-piece table).
4. **Dominant failure mode:** based on A and B together, one or two sentences identifying what's primarily broken.
5. **Recommendation for next sub-project:** which of the design's "What follows" branches applies (decoder retrain / assembly fix / hybrid / methodology). Be specific about the next concrete action.

- [ ] **Step 3.2: Final commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO
git add docs/audits/2026-05-11-stage3-v3-retrain-results.md
git commit -m "audit: Stage 3 v2 failure-mode investigation verdict

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

- [ ] **Step 3.3: Push and open PR**

```bash
git push -u origin feat/stage3-v3-retrain
gh pr create --title "audit: Stage 3 v2 failure-mode investigation (encoder drift disproved + revised diagnosis)" --body "$(cat <<'EOF'
## Summary

The Stage 3 v2 training audit (PR #48) diagnosed that encoder DoRA was unfrozen during training and concluded this train/eval skew was the primary cause of the eval gap. This branch tested that conclusion and **disproved it**: the frankenstein checkpoint (Stage 2 v2 encoder + Stage 3 v2 decoder) gives essentially the same demo mean onset_f1 (0.0599) as the broken Stage 3 v2 (0.0589).

Two follow-up diagnostic streams then identified the actual dominant failure mode.

## Files

- `scripts/audit/build_frankenstein_checkpoint.py` — Phase 1 diagnostic checkpoint builder
- `scripts/audit/a3_decoder_on_training.py` — Stream A: decoder round-trip on training data
- `scripts/audit/pipeline_note_loss.py` — Stream B: pipeline-stage note-count diagnostic
- `src/inference/checkpoint_io.py` — loader fix (accept both `model` and `model_state_dict` keys)
- `docs/superpowers/specs/2026-05-11-stage3-v3-retrain-design.md` — original (now-obsolete) v3 retrain spec
- `docs/superpowers/specs/2026-05-11-stage3-v2-failure-mode-investigation-design.md` — follow-up investigation spec
- `docs/superpowers/plans/2026-05-11-stage3-v3-retrain-plan.md` — original v3 plan (Tasks 4-9 NOT executed)
- `docs/superpowers/plans/2026-05-11-stage3-v2-failure-mode-investigation-plan.md` — follow-up plan
- `docs/audits/2026-05-11-stage3-v3-retrain-results.md` — full report with Phase 1 + Stream A + Stream B + revised verdict

## Phase 1 result

Frankenstein mean onset_f1 = **0.0599** (vs Stage 3 v2 = 0.0589). Encoder drift is **not** the dominant failure mode. The v3 retrain (Tasks 4-9 of the original plan) was halted.

## Stream A result

<COPY FROM REPORT — mean token accuracy + interpretation>

## Stream B result

<COPY FROM REPORT — biggest stage-to-stage note drop + interpretation>

## Revised verdict

<COPY FROM REPORT — dominant failure mode + recommended next sub-project>

## Test plan

- [ ] Reviewer: confirm `audit_results/*.json` files on seder match the numbers in the report
- [ ] Reviewer: confirm the loader fix doesn't regress existing inference (run `eval/run_clarity_demo_radio_eval.py --name regression_smoke` against Stage 3 v2 best.pt and verify the demo mean onset_f1 is still ~0.06)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace `<COPY FROM REPORT — ...>` markers with the actual numbers/text from the report.

---

## Self-review checklist

- [x] **Spec coverage:**
  - Stream A → Task 1
  - Stream B → Task 2
  - Revised verdict + PR → Task 3
  - All spec sections mapped.
- [x] **Placeholder scan:** Real placeholders only exist where output data fills them at execution time (`<FILL FROM RESULT>` and `<COPY FROM REPORT — ...>` markers). All code blocks contain complete code.
- [x] **Type consistency:** `pick_audit_samples`, `_load_stage_b_crop_tensor`, `_encode_staff_image`, `_decode_stage_b_tokens`, `_split_staff_sequences_for_validation`, `assemble_score`, `assembled_score_to_music21_with_diagnostics` — all names match the codebase. `StaffRecognitionResult`, `StaffLocation`, `StageDExportDiagnostics` likewise. Verified against the audit plan's earlier task definitions where these same APIs were referenced.
- [x] **Path consistency:** Manifest path `src/data/manifests/token_manifest_stage3.jsonl` and Stage 3 v2 checkpoint path match what was used in the audit plan and verified by Task 1 of the prior plan. YOLO weights path `runs/detect/runs/yolo26m_systems/weights/best.pt` matches the eval driver's expectations.
