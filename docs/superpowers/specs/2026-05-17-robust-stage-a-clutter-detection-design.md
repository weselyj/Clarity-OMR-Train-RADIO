# Robust Stage-A — clutter/noise-resilient system detection (design)

**Date:** 2026-05-17
**Status:** design, pending user review → implementation plan

## Context & problem

Stage-A is a single-class ("system") YOLO detector that localizes staff systems on a
page; its boxes are cropped and fed to Stage-B. It is trained on a mix of
engraved/synthetic corpora (`build_grandstaff_systems`, `build_synthetic_systems_v1`,
`derive_audiolabs_systems`, `derive_sparse_augment_systems` → `mixed_systems_v1`).

Spot-checking the post-faint-ink model on heterogeneous real scans
(`beginner_scans`) exposed a competence gap on out-of-distribution input: on
"Receipt - Primary Chord Progression" the detector **simultaneously** boxed a
title block as a system (false positive) **and** missed a real system at the
bottom of the page (false negative), with uniformly low confidence (0.69–0.75
vs 0.91–0.96 on clean scans). It also emits confident phantom systems on
non-music documents (a Warranty Deed → 4 systems @ 0.95). Stage-A has no concept
of "music vs. not"; it fires its learned horizontal-band prior anywhere.

Because every Stage-A error poisons every downstream RADIO stage, a downstream
filter only rejects bad data — it cannot recover a system the detector never
proposed. The fix must be at the model + training-data level.

## Goal & success definition

A clutter/noise-resilient Stage-A that, balanced (neither side dominating):
1. **never** boxes text/garbage/non-music content as a system, and
2. **reliably finds true staff systems amid clutter** (surrounding text, titles,
   annotations, scanner noise, skew).

**Ship gate (strictest — "any bbox error is a RADIO failure"):** on a held-out
real archetype set, **every scenario** must have **zero bbox error**:
- no false system (a predicted box matching no ground-truth system),
- no missed system (a ground-truth system with no matching prediction),
- every matched box tight enough for Stage-B and **not clipping the lyric band**.

Operationally, GT↔pred are matched by IoU; below the Stage-B-usable IoU
threshold (or clipping lyrics) counts as a geometry failure. The gate is binary
**per scenario** — one failing archetype blocks ship; aggregate metrics may not
hide a bad mode.

Additionally, to ship: **no regression vs the existing 0.930 lieder Stage-A
baseline** (`eval/results/stagea_baseline_pre_faintink.csv`), **and** a dedicated
**lyric-system recall sub-metric** at or above baseline so a vocal-music
regression cannot hide inside the aggregate.

## Non-goals

- Downstream filtering / post-hoc rejection band-aids.
- Note-level transcription accuracy (a Stage-B concern).
- The bass-clef-flip / Stage-B clef bias (Phase 3 — its own spec, sequenced after this).
- Creating the held-out real archetype set: that is a **user-provided input/dependency**.

## Approach

**A — spine (chosen).** Keep Stage-A single-class (`nc=1`); reuse the entire
proven `mixed_systems_v1` training/eval harness so the 0.930 lieder no-regression
baseline stays directly comparable. Robustness is achieved through the data
engine (below) plus an iterative train→eval→mine loop. Rationale: reuses proven
infrastructure, synthetic composition yields exact ground-truth boxes for free,
and the iterative hard-example loop is the only realistic route to an extremely
strict bar.

**C — calibrated abstention (folded in).** Tune the detector's confidence
operating point so an unsure model emits **nothing** rather than a likely-wrong
box — a precision safety net (a wrong box is fatal). This addresses the
precision half only; a suppressed real staff is still a gate failure, so it is a
net, not a standalone answer.

**B — explicit `text/non-music` class (hot-swappable fallback only).** Adopted
**only if** A plateaus on precision in the loop (cannot drive false-system rate
to zero on the held-out set within the escalation bound). If adopted, two
clauses are **mandatory**: (1) class-1 (text) may **never** suppress a class-0
(system) via cross-class NMS — class-1 is a discarded auxiliary signal only;
(2) any text inside a system's vertical band is ineligible to be class-1. B
carries an annotation-completeness tax and the lyrics trap (below); it is not
adopted up front.

**Bundled — retrain-hardening (prerequisite).** Gradient clipping and an LR/AMP
loss-scale review to eliminate the epoch-~34 NaN instability observed in the
faint-ink retrain; seder worker robustness (UTF-8 logging, stderr handling,
scheduled-task hygiene); and best.pt-provenance validation per the Bethlehem
plan Step 4.4 learnings. A perfect bar cannot be chased on a training process
that randomly NaNs.

## Data engine

Three sources with distinct, non-interchangeable roles:

1. **Synthetic clutter composition (training volume).** Extend the existing
   scan-noise / synthetic tooling to composite real/synthetic systems onto
   text/garbage/noisy/skewed backgrounds and around titles, headers, and
   annotations — exact GT boxes for free. **Deliberately floods lyric-bearing
   positives**: staff + lyrics/dynamics/tempo/expression text where the GT box
   *includes* that text. This is the bulk of training data.
2. **Mined hard negatives (cheap precision signal).** Staff-free
   non-music / text / front-matter pages from existing corpora plus generated
   receipts/forms/garbage. **Label-free.** Every candidate passes a
   staff-presence audit so no negative accidentally contains a staff.
3. **Held-out real archetype set (the ship gate; user-provided).** ~15–24 real
   scans spanning ~6–8 deliberately-distinct failure modes (title/header over a
   system, dense text wrapping a staff, handwritten annotations, heavy scanner
   noise/skew, multi-column/mixed layout, a pure non-music page, and explicit
   lyrics/vocal archetypes). **Never trained on** — eval and synthetic-realism
   calibration only. Diversity over count: one+ scenario per failure mode beats
   many near-duplicates.

**Negative-definition rule (written annotation guideline, non-optional):** a
region is "garbage" **only by staff-absence**, never by text-presence. Every
"text near staff" instance must be paired overwhelmingly with positive
supervision so the discriminative cue the model learns is the 5-line staff
signature, not "text is present."

## The lyrics-as-garbage regression risk & mitigations

This is the central domain risk and gets explicit, non-optional treatment.
Sheet music legitimately contains text (lyrics, tempo/dynamics, fingering,
rehearsal marks). The 0.930 no-regression baseline is **openscore_lieder — art
songs, i.e. staves with lyrics by definition**. A careless "text ⇒ reject"
signal would regress on exactly the forbidden corpus, by **missing or clipping
vocal systems that carry lyrics**.

Mitigations, in priority order:
1. **Staff-absence negative definition** (above) — the root mitigation.
2. **Flood lyric-bearing positives** where the GT box includes the lyrics, so
   staff+lyrics = one system is overwhelmingly the learned association.
3. **Negatives staff-free by construction**; mined real negatives audited for
   accidental staves.
4. **Dedicated lyrics regression slice in the strict eval** (dense-lyrics
   system, lyrics-system with a title above it, heavy-expression-text system);
   a clipped-lyrics box is a gate failure; the lyric-system recall sub-metric
   prevents a vocal regression hiding in the aggregate.
5. **A-as-spine is itself the primary structural mitigation** — no `text` class
   exists that *can* learn to suppress lyrics; mitigations 1–4 are good practice
   for A and mandatory for B.
6. **If B is triggered:** the two mandatory decoupling clauses above.
7. **Per-iteration lyrics-slice watch:** a new lyric-system miss is answered
   with more lyric *positives* (hard-positive mining), never more negatives.

## Iterative loop & error handling

Loop: train → eval (held-out per-scenario gate + lieder no-regression + lyric
slice) → on any failure, **mine and re-synthesize**: false boxes → analogous
hard negatives; misses → analogous positives-in-clutter (incl. lyric variants);
geometry failures → tighten synthesis. Retrain, repeat.

Bounded with escalation:
- If precision plateaus (false-system rate cannot reach zero on the held-out set
  within the loop bound) → trigger the **B fallback**.
- If B also plateaus → **escalate to the user** (likely real-data labeling — the
  source-3 corpus expanded for training, not just eval).
- Each iteration explicitly monitors the lyrics slice; the precision/recall tilt
  on vocal music is caught the iteration it appears.

Retrain-hardening is a prerequisite to entering the loop, not an optimization.

## Testing strategy

CPU-unit-testable (must be, and runnable off the CUDA gate):
- Synthetic composition: boxes are pixel-correct; lyric-bearing positives' GT
  boxes include the lyrics; mined/generated negatives are provably staff-free.
- Abstention / confidence-operating-point logic.
- Eval harness: the per-scenario binary gate, IoU matching, and the
  lyric-system recall sub-metric.

GPU (seder, via the hardened worker): training and full eval. The held-out real
archetype set is the integration gate — nothing ships without it green plus the
lieder no-regression and lyric-slice checks.

## Dependencies & risks

- **User dependency:** the held-out real archetype set (size/diversity per §
  Data engine). Implementation cannot reach the ship gate without it.
- **Central risk:** synthetic→real transfer gap. Mitigated by the held-out real
  set, synthetic-realism calibration against it, and the iterative loop.
- **Seder training fragility:** the epoch-~34 NaN and worker issues — addressed
  by the bundled retrain-hardening prerequisite.
- **Lieder/lyrics trap:** addressed by the dedicated mitigations section.

## Sequencing vs. Phase 3

This effort **precedes Phase 3 (bass-clef / Stage-B clef bias)**: Stage-A errors
poison every downstream stage including any clef work, so robust detection is
the foundation. Phase 3 remains its own spec, after this.

## Decisions deferred to the implementation plan (with recommended defaults)

These are scoping decisions for the plan, not unresolved design gaps:
- **B trigger:** recommended default — B is a fallback triggered by a precision
  plateau, not adopted up front.
- **Loop/escalation bound:** the exact max-iteration count and the precise
  "precision plateau" / "B plateau" thresholds that trigger escalation.
- **Geometry gate:** the exact Stage-B-usable IoU threshold and the operational
  definition of "clipping the lyric band."
- **Held-out set finalization:** exact size and the concrete archetype list,
  finalized with the user's actual scenarios.
- **Abstention calibration:** the method for choosing the confidence operating
  point.
