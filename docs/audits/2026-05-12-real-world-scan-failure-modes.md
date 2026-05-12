# Real-World Scan Failure Modes — Bethlehem + TimeMachine

**Date:** 2026-05-12
**Scope:** Two real-world piano-score scans run through the post-PR #54 inference pipeline.
**Branch this lands on:** `feat/predict-pdf-post-decode` → PR (TBD)
**Related docs:**
- [Stage 3 v3 data-rebalance results](2026-05-12-stage3-v3-data-rebalance-results.md) — overall v3 verdict
- [train-vs-inference gap diagnostic (revised)](2026-05-12-stage3-v3-train-vs-inference-gap.md) — earlier failure-mode framing
- Data-pipeline audit (the Phase 1 report): [2026-05-12-data-pipeline-and-capacity-audit.md](2026-05-12-data-pipeline-and-capacity-audit.md)

## TL;DR

User ran two real-world scanned beginner-piano scores through the Stage 3 v3 best.pt:
- `Scanned_20251208-0833.jpg` ("O Little Town of Bethlehem", 4 systems)
- `Receipt - Restoration Hardware.pdf` ("The Time Machine", 4 systems, the file is misnamed)

Both produced MusicXML with alternating-clef bugs on the bass part. Token-level
diagnostics revealed **three distinct failure modes**, not one:

1. **Stage A YOLO missed-system / merged-system** (Bethlehem). YOLO returned
   only 3 of 4 visible systems at default conf=0.25, with one detection
   spanning two visual systems and emitting 3 staves of mixed-clef content.
2. **Stage B phantom-staff hallucination** (TimeMachine, 2 of 4 systems).
   Decoder emitted 3 `<staff_start>` blocks on a 2-staff grand-staff system —
   typically a duplicate of the bass staff under a wrong clef, OR an all-rest
   "draft" block followed by the real bass staff.
3. **Stage B bass-clef-as-treble misread** (Bethlehem sys 2, TimeMachine sys 3).
   Decoder emitted `clef-G2` for a visually-bass staff. The pitches were ALSO
   decoded under the wrong-clef assumption, so they're off by ~20 semitones —
   a clef-only repair makes the clef tag correct but leaves pitches wrong.

This session built diagnostic tooling, fixed failures (1) and (2) with cheap
post-processing, and documented (3) as the residual underlying-generalization
gap to address with retraining.

## Tools added this session (now on main)

### `scripts/predict_pdf.py` — single-PDF/image inference CLI

Wraps `SystemInferencePipeline.run_pdf` + `run_image` + `export_musicxml` in one command.

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf <input> <output.musicxml>'
```

- Defaults to Stage 3 v3 best.pt + standard YOLO weights
- `.pdf` → `run_pdf` (multi-page); image → `run_image` (single page)
- `--diagnostics-out PATH` — writes Stage D diagnostics sidecar
- `--dump-tokens PATH` — JSONL of raw decoder output per system (key debug tool)
- `--yolo-conf FLOAT` — Stage A confidence threshold (default 0.25)
- `--no-postprocess` — disable default post-decode cleanup
- `--repair-bass-clef` — experimental: opt-in clef repair (see caveat below)

Merged in PRs #51 (CLI scaffold), #52 (image input), #53 (dump-tokens), #54 (yolo-conf).

### `src/pipeline/post_decode.py` — post-decode cleanup heuristics (this branch)

Two pure-logic functions operating on per-system token lists, hooked into
`SystemInferencePipeline.run_pdf` and `run_image` after `_decode_one_crop`
returns tokens, before assembly:

1. **`drop_phantom_staves` (default ON)** — when a system has >2 staff chunks,
   identifies and drops duplicates (same note-event signature under different
   clefs) and all-rest "draft" chunks. Never collapses systems to fewer than
   2 staves except when a stave is provably phantom.
2. **`repair_bass_clef` (default OFF)** — when the bottom staff of a system
   has `clef-G2` but its notes' median octave is < 4, swap to `clef-F4`. OFF
   by default because **it doesn't transpose pitches**: the decoder produces
   pitches under the wrong-clef assumption and they're off by ~20 semitones
   even after the clef tag is corrected. Enable only when visual clef-correctness
   matters more than pitch correctness.

19/19 unit tests pass at `tests/test_post_decode.py`. Tests are pure-Python
and run locally (not CUDA-gated).

## The diagnostic process

### Bethlehem token dump (default YOLO conf 0.25)

```
sys 0  y=584-1311   conf=0.557  2 staves  [clef-G2, clef-F4]      ✓ correct
sys 1  y=2047-2767  conf=0.343  3 staves  [clef-G2, clef-G2, clef-F4]  ← merger
sys 2  y=2785-3554  conf=0.966  2 staves  [clef-G2, clef-G2]      ← bass→treble
```

- YOLO returned 3 detections for 4 visible systems (gap y=1311–2047 spans visual system 2 — uncovered).
- sys 1 spans visual systems 2+3 with 3 staves of mixed clefs — assembly stripes these across parts producing the alternating-clef MXL pattern.
- sys 2 was a clean detection (conf 0.966) but the decoder still misread the bass clef.

At `--yolo-conf 0.01` Bethlehem produces 5 detections: the missing visual system 2 is recovered at conf=0.046, but a junk detection on the title area appears at conf=0.013. A more selective threshold (~0.05) would likely keep the recovery without the junk.

### TimeMachine token dump (default YOLO conf 0.25)

```
sys 0  y=821-1632   conf=0.964  2 staves  [clef-G2, clef-F4]            ✓ correct
sys 1  y=1591-2366  conf=0.973  3 staves  [clef-G2, clef-G2, clef-F4]   ← phantom
sys 2  y=2321-3167  conf=0.964  2 staves  [clef-G2, clef-G2]            ← bass→treble
sys 3  y=3151-4090  conf=0.972  3 staves  [clef-G2, clef-G2, clef-F4]   ← phantom
```

All 4 systems detected at conf ≥ 0.96 — Stage A worked perfectly.

The Stage B phantom is structured: for sys 1, chunks 2 and 3 contain
**identical bass content** under different clefs:
```
chunk 1: <staff_idx_0> clef-G2  ... [treble melody]
chunk 2: <staff_idx_1> clef-G2  ... [rest_whole, <chord_start> note-C3 note-G3 ...]
chunk 3: <staff_idx_2> clef-F4  ... [rest_whole, <chord_start> note-C3 note-G3 ...]  ← identical
```

For sys 3, chunk 2 is all-rest and chunk 3 is the real bass:
```
chunk 1: <staff_idx_0> clef-G2  ... [busy melodic line]
chunk 2: <staff_idx_1> clef-G2  ... [rest, rest, rest, rest_half, fermata, rest]
chunk 3: <staff_idx_2> clef-F4  ... [rest, rest, rest_half, note-G3, note-G3, ...]  ← real
```

Two distinct patterns of the same underlying "decoder hesitation → emit phantom" failure.

## After the post-decode fix

### TimeMachine — phantom-drop ON, bass-clef repair OFF (default)

Part 1 (treble): `G2, G2, G2, G2` ✓
Part 2 (bass):   `F4, F4, G2, F4`

The 3-staff phantoms on sys 1 and sys 3 collapsed to 2 staves with the correct
`F4` clef preserved (the heuristic picks the bass-clef chunk when the note
content is in bass register). The remaining `G2` on sys 2 is the underlying
Stage B bass-clef misread — phantom-drop can't fix it (only one chunk emitted,
no phantom to drop).

Before fix: Part 2 had 6 clef tags alternating `F, G, F, G, G, F`.
After fix:  Part 2 has 4 clef tags, only one wrong.

### Bethlehem — phantom-drop ON (default)

Part 1 (treble): `G2, G2, G2` (3 systems detected from 4 visual)
Part 2 (bass):   `F4, F4, G2`

The 3-staff merger on sys 1 collapsed correctly. The `G2` on sys 3 is the
underlying bass-clef misread on the bottom system.

To recover the missing visual system 2: `--yolo-conf 0.05` (with the caveat
that conf < 0.05 admits junk detections in the title region).

## What we DIDN'T fix

The bass-clef-as-treble misread is the residual failure mode that the
post-decoder can't safely fix. The decoder's pitches are entangled with
its clef assumption: if it misreads bass as treble, the pitches it emits
are positioned as if the staff lines mapped to treble pitches. Correcting
just the clef tag makes a note-with-pitch-tag-`E4` appear as E4 in a bass
staff, which is in the wrong visual position. The correct fix would
either:

1. **Transpose pitches alongside the clef swap.** The treble↔bass staff-line
   mapping difference is roughly 20-21 semitones (treble bottom line E4 vs
   bass bottom line G2 = 21 semitones; treble top line F5 vs bass top line
   A3 = 20 semitones), so pitches would need a music-theory-aware down-shift
   by an octave + a major 6th. Implementable but non-trivial.
2. **Re-decode the staff with the clef forced to bass.** Expensive but
   correct. Would require a beam-search variant that conditions on a clef
   prior.
3. **Retrain with more bass-clef-on-real-scans coverage.** The cleanest fix
   but the most expensive. Likely warranted given this failure recurs on
   both scans — see [the v3 results doc](2026-05-12-stage3-v3-data-rebalance-results.md)'s
   bottom-quartile-failure-mode-analysis recommendation.

## Counts & metrics

Token diagnostic timing on RTX 5090, post-postprocess:
- Bethlehem (single image, 3 systems): 12.4s end-to-end
- TimeMachine (PDF, 4 systems): 15.6s end-to-end

Both well within the wall-clock budget for interactive use.

## What a fresh session needs to know

### Branch state
- `main` is at PR #54 (`778b001`) — has predict_pdf CLI with --dump-tokens + --yolo-conf
- This work is on `feat/predict-pdf-post-decode` (PR TBD) — adds post_decode module + --no-postprocess + --repair-bass-clef

### Files touched this session (all merged to main except this branch)
- `src/inference/system_pipeline.py` — `run_image` method, `token_log` capture, `yolo_conf` kwarg, `postprocess` + `postprocess_repair_bass_clef` kwargs
- `scripts/predict_pdf.py` — the CLI tool
- `src/pipeline/post_decode.py` (new, this branch) — heuristics module
- `tests/test_post_decode.py` (new, this branch) — 19 pure-logic tests

### Quick repro commands

```bash
# Default (phantom-drop on, bass-clef repair off):
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf <input> <output.musicxml>'

# With raw decoder dump for debugging:
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf <input> <output.musicxml> \
    --dump-tokens <output.tokens.jsonl>'

# Lower YOLO conf to recover missed systems:
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf <input> <output.musicxml> \
    --yolo-conf 0.05'

# Experimental: also repair bass-clef misread (cosmetic only; pitches stay wrong):
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf <input> <output.musicxml> \
    --repair-bass-clef'

# Bypass post-decode entirely (see the model's raw output in the MXL):
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && \
  venv-cu132\Scripts\python.exe -m scripts.predict_pdf <input> <output.musicxml> \
    --no-postprocess'
```

### Recommended next sub-projects

1. **Stratified failure-mode analysis on bottom-quartile lieder pieces** (33 pieces from the v3 lieder eval with onset_f1 < 0.10). Cluster on observable features (clef type, staff count, page layout, instrument set) to find consistent drivers. Bethlehem + TimeMachine are 2 informal data points for the bass-clef-misread cluster; lieder has 31+ more. Lightweight, half-day of analysis with Phase-1-style audit scripts.
2. **Pitch-aware bass-clef repair** (defer until 1 confirms it's the dominant failure mode). Implement the treble↔bass staff-position transposition alongside the clef tag swap. Would require a small music-theory utility for staff-position → pitch lookup.
3. **YOLO Stage A retrain or threshold-aware system gap detector** (defer; lower priority than 1 and 2 because Bethlehem's Stage A failure was the less common pattern). If Bethlehem-style missing-system failures cluster on a recognizable scan type, augment training data; otherwise add a geometric gap-detection pass after YOLO.
4. **Add a `--yolo-conf-retry` mode** that runs Stage A twice — once at the user's conf, once at a lower threshold over the gap regions only — and merges results, suppressing junk detections by bbox-area or y-position heuristics. Closes the "Bethlehem at conf=0.05 admits junk" issue.

### Open questions for the new session

- Is pitch-correctness or visual-correctness more important for the immediate use case? (Determines whether to default `--repair-bass-clef` on/off.)
- Are these two scans representative, or unusually-formatted? (Both are beginner-piano arrangements with sparse bass parts; could be a corpus-specific failure mode.)
- Is the v3 checkpoint a fixed dependency, or open to a retrain? (Determines whether to invest in heuristics vs training-data fixes.)
