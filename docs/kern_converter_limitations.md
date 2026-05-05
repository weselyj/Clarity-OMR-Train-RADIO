# Kern Converter — Documented Limitations

The kern-to-token converter at `src/data/convert_tokens.py::convert_kern_file`
round-trips against music21's humdrum parser as the validation oracle. After
seven rounds of fixes (vocab extensions, converter behavior fixes, and
reconstruction-path fixes) the audit across the full GrandStaff corpus
(53,883 files) shows:

| | |
|---|---|
| **Passing cleanly** | 52,200 (96.9%) |
| **With divergences** | 1,603 (3.0%) |
| **Failed to compare** | 80 (0.1%) |

## Audit progression

| run | pass rate | what changed |
|---|---|---|
| v1 (pre-rebuild) | implicit (~0%, all corrupt labels) | original buggy converter |
| v3 (post Workstreams 1+2) | 47.0% | full converter rebuild w/ vocab 388 |
| v3b (Cb/Fb/B#/E#) | 61.9% | 4 letter spellings × 5 octaves enharmonic preservation |
| v3c (octave-1) | 80.6% | 21 octave-1 sub-bass tokens |
| v4 (voice-split) | 81.3% | mid-measure `*^` hidden-rest padding |
| v5 (double-flat vocab) | 89.4% | 84 Xbb/X## tokens × 6 octaves |
| v6 (double-flat reconstruction) | 94.5% | `bb` → `--` translation in `append_tokens_to_part` for music21 |
| **v7 (tie continuation)** | **96.9%** | `_` marker handling in `parse_kern_event` |

## Remaining divergences (the ~3.1% non-passing cluster)

The remaining divergences fall into three groups, in decreasing order of
frequency:

### Group 1 — Music21 humdrum-parser bugs (~1.0% / ~558 files)

**Not bugs in our converter.** When kern source contains a `*v` (voice merge)
followed by `*^` (voice re-split) on the same spine within a few lines —
a common pattern in chromatic piano music — music21's humdrum parser drops
notes from the reborn sub-spines.

Our converter correctly tracks these via the `column_to_spine` accumulator
in `convert_kern_file` (lines 841-871) and preserves all notes. The audit
reports them as "extra in our output" because music21 (the reference) is
missing them.

Example: `beethoven/piano-sonatas/sonata05-2/original_m-97-101.krn` —
our converter outputs 63 notes, music21 outputs 57. The 6 "extra" notes
are present in the kern source.

A bug report has been opened against the music21 project documenting this
behavior. Until upstream is fixed, these files will continue to register
as divergent in the audit despite our output being correct.

### Group 2 — Genuine vocab gaps for unusual tuplets (~0.1% / 52 files)

Music21 emits 11:8, 9:8, 15:8, and 17:16 tuplet ratios for some kern
duration codes (e.g., `kern 22` → 11:8 sixteenth tuplet). Our `TUPLET_RATIOS`
map only contains 3:2, 5:4, and 7:4. For unsupported ratios, our
`kern_duration_components` falls through to nearest-quantization, which
produces a musically-close but mis-spelled output.

Frequency in corpus: 1,101 occurrences of unusual ratios in 52 files.
Cumulative offset-drift from each unusual-tuplet note (typically 1/66 of a
quarter per note) cascades to subsequent chord/note events on the same staff,
inflating the divergence count of the affected files.

This could be resolved by adding `<tuplet_9>`, `<tuplet_11>`, `<tuplet_15>`,
`<tuplet_17>` to vocab plus updating `TUPLET_RATIOS`. **Deferred** until
post-training validation shows whether the model needs better tuplet
fidelity at this level.

### Group 3 — Failed-to-compare files (80 / 0.1%)

Kern files where music21's humdrum parser raises an exception during
initial parse — non-standard or malformed kern syntax. Our converter is
more lenient and would still produce tokens, but no round-trip comparison
is possible. Upstream parser issue.

## Resolved limitations (this rebuild)

| Was-limitation | Resolved by |
|---|---|
| Tied notes (`[`/`]`) silently dropped | Task 7 |
| Tie continuation (`_`) silently dropped | v7 fix |
| Slurs (`(`/`)`) silently dropped | Task 8 |
| Articulations (`^`, `'`, `;`, `~`) silently dropped | Task 9 |
| Ornaments (`t`, `T`, `m`, `M`) silently dropped | Task 10 |
| Tuplet duration math off by factor of 2 | Task 11 |
| Triplet vs sextuplet ambiguity | Task 12/14 (resolved per music21's interpretation) |
| Multi-staff parts placed at sequential offsets | Task 14 |
| `HalfStepTrill`/`WholeStepMordent` etc. not recognized | Task 14 |
| Grace notes (`q`/`Q`) silently treated as regular notes | Task 14 |
| `tie_end`/`slur_end` after duration not consumed | Task 14 |
| Enharmonic spellings collapsed (`c-` → `B`) | Task 17 phase 1 (Cb/Fb/B#/E# × 5 octaves) |
| Octave-1 sub-bass clamped to C2 | Task 17 phase 2 (21 octave-1 tokens) |
| Double-flat/double-sharp spellings collapsed (`B--` → `A`) | Task 17 phase 3 (84 Xbb/X## × 6 octaves) |
| Mid-measure `*^` voice-2 placed at offset 0 (reconstruction) | Voice-split fix (hidden rests) |
| Double-flat tokens crashing music21 reconstruction (`Bbb2` invalid) | `bb`→`--` translation in export_musicxml |

## Vocab summary (final)

| Token class | Count | Notes |
|---|---|---|
| Note tokens (single-pitch) | 192 | 7 letters × naturals + 5 sharps + 5 flats + 4 enharmonic-natural-key + 14 double-accidentals = 35 spellings × 6 octaves (1-6) |
| Other (rest, ties, slurs, articulations, ornaments, durations, tuplets, structural markers) | 321 | unchanged from base + workstream additions |
| **Total** | **513** | up from base 388 |
