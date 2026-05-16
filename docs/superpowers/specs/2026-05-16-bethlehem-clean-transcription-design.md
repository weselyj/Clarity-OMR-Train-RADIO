# Bethlehem Clean-Transcription Design (2026-05-16)

## Motivation

"O Little Town of Bethlehem" (`Scanned_20251208-0833.jpg`) is the canonical
real-world beginner-piano scan used to gauge end-to-end transcription quality.
v4 (the scan-realistic Stage 3 retrain) did not make it transcribe cleanly, and
its failure mode is identical to v3's. Diagnosis this session isolated **two
independent defects**, each with hard evidence:

1. **Stage A miss.** YOLO detects only 3 of the 4 printed staff-systems. The
   2nd system ("bove thy deep and dreamless sleep", measures 5–8) sits in a
   736 px vertical blind gap (between detected boxes ending y=1311 and starting
   y=2047) with **zero** detections at the default conf. A conf sweep shows it
   only reappears at conf ≤ 0.03, with an intrinsic detection confidence of
   **0.046** — YOLO effectively cannot see that leading system. Result: 4
   measures never reach the decoder (pred 13 vs GT 17 measures).

2. **Stage B clef bias.** On a system it *does* see (system_02, a clean crop
   with a plainly legible bass clef), the decoder emits `clef-G2` for the
   bottom staff, immediately followed by `rest _whole`. The bass staff is
   almost entirely rests; with no pitch anchors the decoder defaults to the
   corpus-dominant treble clef. This is a learned bias, not scan degradation —
   which is why the v4 scan-realistic data did not fix it.

Ground truth now exists:
`/home/ari/musicxml/Scanned_20251208-0833_20260516.musicxml` — 2 parts, 17
measures, Part 0 clef G2 (39 notes), Part 1 clef F4 (18 notes). This makes the
target measurable.

## Goal

Bethlehem transcribes to match its ground truth: all 4 systems / 17 measures
present, treble staff = G2, bass staff = F4. Iterated against a real metric,
Defect 1 (Stage A) first, Defect 2 (Stage B) second.

## Non-goals / scope boundaries

- Not a general Stage A or Stage B rework. The two fixes are scoped to the
  failure modes above.
- Defect 1 fix is **augmentation-only** (no new labeled real scans). If
  augmentation cannot clear the gate, the escalation is a user-provided custom
  corpus — that corpus and its labeling are **out of scope for this spec** and
  would get their own spec.
- Pitch/rhythm fidelity beyond clef correctness is not a gate. The note
  onset-F1 metric is tracked for regression visibility, not as a pass bar.

## Component 1 — Bethlehem⇄GT scoring harness

The iteration instrument. Without a number, "iterate" is eyeballing crops.

- **Input:** a predicted MusicXML + the GT MusicXML.
- **Output:** three metrics, printed and JSON-dumped:
  - **measure_recall** — fraction of GT measures present in the prediction
    (Defect 1 signal: 13/17 today → must reach 17/17).
  - **clef_accuracy** — per-staff clef correctness, treble staff expects G2,
    bass staff expects F4 (Defect 2 signal: bass currently flips to G2 on the
    rest-heavy system).
  - **note_onset_f1** — overall fidelity, tracked for regression visibility
    only, not a pass bar.
- **Properties:** pure CPU, deterministic, seconds to run, no pipeline
  dependency (consumes already-produced MusicXML). Committed once, reused every
  iteration. Lives at `scripts/audit/score_against_gt.py` with unit tests on
  synthetic MXML fixtures.

## Component 2 — Stage A: faint-ink augmentation + YOLO retrain (Defect 1)

**Root cause:** the Stage A scan-noise augmentation (`src/train/scan_noise.py`,
`BASE_NOISE_PROBABILITIES`) covers JPEG compression, sensor noise, blur,
brightness/contrast (±0.15, mild), rotation, grid/elastic distortion. It has
**no faint-ink / low-coverage arm**. Bethlehem's leading system is barely
inked; the training distribution never shows YOLO a system that faint, so it
never learns to detect one.

**Change:** add a `faint_ink` transform group to `BASE_NOISE_PROBABILITIES`
with its own probability, scaled by the existing intensity ramp via
`scaled_probabilities` (no change to warmup mechanics). The transform
simulates a barely-inked system:

- Asymmetric brightness shift toward white (ink washes out — distinct from the
  symmetric ±0.15 `brightness_contrast`),
- Contrast reduction beyond the current limit,
- Light morphological erosion (stroke thinning), kernel small enough to keep
  staff lines connected.

Retrain Stage A with `scripts/train_yolo.py --noise` (YOLO26m, existing
`data.yaml` — **no new labeled scans**), strengthened pipeline active.

**Gate (both required):**
- Bethlehem system 2 detected at the **default conf 0.25** (confident
  detection, not the current 0.046), verified via
  `scripts/audit/dump_system_crops.py` → 4 systems, clean geometry, no junk.
- **No regression** on Stage A system detection. The plan must first locate
  the eval/metric used by the PR #38 Stage A rebuild; if no reusable harness
  exists, build a minimal fixed regression set (a held-out sample of scored
  scans with system-box labels) and baseline the current checkpoint on it
  before retraining. "No regression" = aggregate system-detection
  recall/precision ≤ 1 pt below that baseline.

**Escalation (out of scope here):** if the gate fails, fall back to a
user-provided custom corpus of Bethlehem-like real scans with labeled system
boxes. That is a separate spec.

## Component 3 — Stage B: rest-heavy-bass clef bias (Defect 2, second)

Only meaningfully measurable once Component 2 delivers all 4 crops (until then
the rest-heavy system may not even be reaching the decoder consistently).

- Re-score Bethlehem with the Component-1 harness after Component 2 lands. If
  `clef_accuracy` shows the bass staff still flips to G2 on rest-heavy systems:
- Iterate via **overfit-prove → targeted live-tier fine-tune**: first prove the
  decoder *can* learn "bass-clef glyph + all rests = clef-F4" by overfitting a
  tiny rest-heavy-bass set (minutes, no encoder-cache rebuild); then a focused
  live-tier fine-tune on a targeted slice. Avoids the ~8 h cache rebuild that
  makes full Stage 3 retrains non-iterable.
- **Gate:** Bethlehem bass clef = F4 on the rest-heavy system **AND** a lieder
  no-regression check. Baseline = v3's documented lieder mean onset_f1
  **0.2398** (the established ship-gate reference; the v4 lieder run is
  incomplete — parked at 129/146 — so it is not used as the baseline). The
  fine-tuned model's lieder mean must be ≥ 0.2398 so the fix is not Bethlehem
  memorization.

## Sequencing

1. Component 1 (harness) — prerequisite for measuring anything.
2. Component 2 (Stage A faint-ink retrain) — recover all 4 systems.
3. Component 3 (Stage B clef) — only after 4 crops are reaching the decoder.

Components 2 and 3 are gated independently; 3 does not start until 2 passes.

## Risks

- **Faint-ink augmentation over-degrades training systems** → YOLO detection
  regresses corpus-wide. Mitigation: own probability key + intensity ramp;
  the no-regression gate catches it; tune probability/erosion-kernel down if
  it triggers.
- **Augmentation insufficient** — synthetic faint systems don't span the real
  Bethlehem appearance. Mitigation: the gate fails cleanly and we escalate to
  the custom-corpus path (user already opted into this fallback).
- **Component 3 overfit doesn't generalize** — the lieder no-regression gate is
  mandatory specifically to catch Bethlehem memorization.
- **Stage A retrain wall-clock** — YOLO retrain is the long pole; it is a
  single retrain per augmentation change, not an inner-loop cost.

## Evidence appendix

Conf sweep on Bethlehem (`dump_system_crops.py`, default weights):

| conf | systems | system 2 recovered | junk |
|---|---:|---|---|
| 0.25 / 0.15 / 0.10 / 0.05 | 3 | no | — |
| 0.03 / 0.02 | 4 | yes — bbox (0,1319,2621,2035), intrinsic conf 0.046 | none |

GT vs v4 prediction:

| | GT | v4 pred | defect |
|---|---|---|---|
| measures | 17 | 13 | Stage A miss (~1 system) |
| Part 0 (treble) clef | G2 | G2 ✓ | — |
| Part 1 (bass) clef | F4 | F4,F4,**G2** | Stage B bias |
| notes (T/B) | 39/18 | 30/13 | downstream of both |
