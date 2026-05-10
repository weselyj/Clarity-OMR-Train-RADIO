# Subproject 4 — Plain-English Overview (for review)

A plain-English companion to the technical spec at [`2026-05-10-radio-subproject4-design.md`](2026-05-10-radio-subproject4-design.md). The technical doc is what the implementer follows; this one exists so a reviewer can sanity-check the plan without reading the codebase.

## What is this project, in one paragraph?

**Clarity-OMR-Train-RADIO** is an Optical Music Recognition (OMR) project — software that takes scanned sheet music (PDFs, photos) and converts it into a machine-readable music format called **MusicXML**, which can then be played, transposed, edited in notation software, etc.

The recognition is done by a deep learning pipeline:

- **Stage A** is a **YOLO** object detector (the same family of models used for "find the dog in this photo"). Its job is to look at a page and locate the music.
- **Stage B** is a **transformer** model (like the ones used in language models). It takes a small image of music and emits a sequence of tokens describing what notes, rests, accidentals, etc. it sees.
- **Assembly** stitches the Stage B outputs back into a single coherent MusicXML score.

The "RADIO" branch is a research effort to improve recognition by training Stage B on cleaner inputs.

## What's a "staff" vs. a "system"?

A **staff** is one set of five horizontal lines that holds notes. A piano part has **two staves stacked together** (one for each hand) — they form a **system**. A typical hymn is one system per line; a piano piece is also one system per line (just with two staves in the system).

This distinction matters because we recently changed how Stage A talks to Stage B.

## What's the recent history?

Until last week, Stage A produced one image per **staff**, and Stage B was trained on those staff-shaped images. But for piano music, the two staves of a system are inherently linked (they share rhythm, key signatures, etc.) and treating them independently throws away that linkage.

So the team retrained Stage B (Stage 3 v2) on **system-shaped images** — multi-staff stacks where appropriate. That training finished and the new model is good (validation loss went from 0.564 down to 0.148 — a ~3× improvement).

But when the team tried to **evaluate** the new model end-to-end on real sheet music, the inference pipeline was still chopping pages into per-staff images and feeding those to a system-trained model. That format mismatch produced bad results that looked like model weakness but were actually a wiring bug.

PR #43 (just merged) **cleaned up** by archiving all the per-staff inference code into `archive/per_staff/`, leaving the codebase coherent but with no working end-to-end inference path. That's what Subproject 4 builds.

## What does Subproject 4 do?

Build the replacement inference pipeline that takes a PDF and produces a MusicXML file, this time using the system-shaped image format that the model actually expects. Then run it on 50 lieder (German art songs from the OpenScore Lieder corpus) and produce a real evaluation number.

The work breaks down into:

1. **A new Stage A wrapper** that runs the system-detection YOLO model and produces one image per system (instead of per staff).
2. **A library class** (`SystemInferencePipeline`) that loads YOLO + Stage B once and exposes a simple "give me a PDF, get a parsed score back" method.
3. **A thin command-line wrapper** so a single PDF can be processed from the terminal.
4. **A small extension to the assembler** to handle multi-staff token sequences from one system.
5. **The evaluation driver** — taken back out of the archive and rewired to use the new library — that runs the pipeline on 50 lieder pieces and writes a CSV report.

## What does "ship-gate" mean here?

A **ship-gate** is the criterion the team uses to call this work "done." For Subproject 4 it is **not** a specific accuracy number. It is:

- The pipeline runs end-to-end on at least 40 of 50 pieces without crashing.
- It produces a CSV with one row per piece, including an `onset_f1` score (which measures how well the predicted notes line up with the ground-truth notes).
- The smoke-test piece (`lc6623145`) produces a parseable MusicXML and a numeric `onset_f1`. **No specific F1 threshold gates this** — but if the smoke piece scores at or near the broken-pipeline baseline of `0.067`, that's a strong signal something is still wired wrong, and the corpus run shouldn't proceed until that's understood.

Whether the average `onset_f1` across the 50 pieces is good or bad is **a finding for the next phase**, not a Subproject 4 blocker. The point of this subproject is to **stop measuring a wiring bug and start measuring the model**.

## Key design choices the spec commits to

| Choice | Why |
|---|---|
| **Library + thin CLI** rather than a single monolithic script | The 50-piece eval can load the (large) models once instead of 50 times — saves several hours of wall-clock time |
| **Two-pass eval** — inference in-process, scoring in subprocess-per-piece | A previous all-in-one-process attempt OOMed at 43 GB by piece 6/20 because music21 / zss accumulate memory across pieces. The split is a documented safety property, not a stylistic choice |
| **Per-staff y-coordinates by even-splitting the system bbox** | The model emits the staves in top-to-bottom order; we don't need a second detector to localize them within the system |
| **TEDN metric is opt-in** (`--tedn` flag, default off) | TEDN is slow (~300 sec per piece worst case) and the architectural ship-gate doesn't need it |
| **Explicit shared Stage B loader** (`load_stage_b_for_inference`) | Loading a Stage B checkpoint takes 8 ordered steps; the spec factors them into one helper so the pipeline class doesn't get them wrong |
| **Diagnostics sidecar (`.musicxml.diagnostics.json`) emitted with every prediction** | The downstream scorer expects it; without it Stage-D skip counters silently disappear |
| **Lift the archived eval driver back** | It's 700 lines of high-quality orchestration (resume from JSONL, ETA logging, OOM-aware throttling); rewriting from scratch would be wasted work |

## What was the first review round, and what did it change?

A first reviewer caught five real issues, all of which have been addressed in the spec:

1. **In-process scoring would have OOMed** — the archived driver explicitly documents a 43 GB OOM at piece 6/20 from accumulated music21/zss state. Fixed: scoring stays in subprocess-per-piece (Phase 2), inference stays in-process (Phase 1).
2. **Stage B loader was underspecified** — the spec pointed at a helper that only unwraps an already-built model, not loads one. Fixed: explicit `load_stage_b_for_inference` helper factored from the existing 8-step sequence.
3. **Required `--vocab` flag was a documentation trap** — the codebase already builds the vocab in-code via `build_default_vocabulary()`. Fixed: dropped the flag.
4. **MusicXML export would have lost the diagnostics sidecar** — downstream scoring expects `.musicxml.diagnostics.json` alongside each prediction. Fixed: pipeline uses the `_with_diagnostics` export path.
5. **Internal contradiction** — non-goals said "no specific F1 target" but ship-gate set `onset_f1 ≥ 0.5` for smoke. Fixed: smoke gate is now "produces a parseable MusicXML + numeric F1"; the 0.067 floor is a stop-and-investigate signal, not a hard threshold.

## What would a second reviewer check now?

1. **Is the two-pass eval architecture clear and correct?** Phase 1 (inference, in-process, library-driven, writes XML + sidecar + status JSONL) → Phase 2 (scoring, subprocess-per-piece, separate CLI). Does that division of labor make sense?
2. **Are we missing a risk?** The spec lists 6 risks (brace-margin double-counting, inference OOM, zero-detection pages, mismatched staff counts, multi-page assembly, TEDN flakiness). Anything else?
3. **Is even-splitting y coordinates within a system reasonable?** Alternatives: (a) re-running a per-staff detector inside each system (slow, redundant), (b) treating the whole system as a single "staff" location (loses within-system ordering), (c) what we're doing.
4. **Are the test boundaries right?** Seven test files cover the new YOLO wrapper, the assembly extension, the new Stage B loader, the pipeline integration (with diagnostics sidecar assertion), the CLI argparse smoke (no `--vocab`), and the inference/scoring isolation in the eval driver. Anything obvious missing?
5. **Deferred follow-ups (TorchAO experimentation, refactoring `evaluate_stage_b_checkpoint.py` onto the new loader, an inference-side subprocess fallback)** — are any of these actually blockers we shouldn't defer?

## Glossary

- **OMR** — Optical Music Recognition. Software that "reads" sheet music images.
- **MusicXML** — XML-based interchange format for sheet music. Compatible with most notation software (MuseScore, Finale, Sibelius, etc.). `.mxl` is its compressed-archive form.
- **Stage A** — In this pipeline, the object-detection step that finds music on a page.
- **Stage B** — The transformer model that converts a music image into a token sequence.
- **YOLO** — A family of object-detection neural networks. We use a fine-tuned variant called YOLOv8 (here labeled `yolo26m`).
- **Token** — A discrete symbol the model emits. The vocabulary includes things like `<note>`, `<rest>`, `<staff_end>`, `<bos>` (beginning-of-sequence), etc.
- **`onset_f1`** — A score (0 to 1) for how well predicted note start-times match the ground truth. 1.0 is perfect.
- **`linearized_ser`** — String edit rate over the serialized MusicXML. Lower is better.
- **TEDN** — Tree-Edit Distance on a kern-format conversion. Captures deeper structural similarity but expensive to compute.
- **lc6623145** — The single piece used as the smoke test. (lc = lieder corpus identifier.)
- **lieder** — German art song. The 1,462-piece OpenScore Lieder corpus is the training and eval source.
- **PR #43** — The pull request that just landed, archiving the broken per-staff inference path.
