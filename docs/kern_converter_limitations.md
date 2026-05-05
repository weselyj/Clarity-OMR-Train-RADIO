# Kern Converter — Documented Limitations

The kern-to-token converter at `src/data/convert_tokens.py::convert_kern_file`
round-trips against music21's humdrum parser. After the v3 vocab extension
(Cb/Fb/B#/E# enharmonic preservation + octave-1 sub-bass tokens), the audit
across the full GrandStaff corpus (53,883 files) shows:

| | |
|---|---|
| **Passing cleanly** | 43,450 (80.6%) |
| **With divergences** | 10,353 (19.2%) |
| **Failed to compare** | 80 (0.1%) |

The remaining divergences are concentrated in five categories. **All five trace
to a single root cause: `append_tokens_to_part`'s reconstruction of voice
splits.** The kern → token conversion is correct; only the token → music21
reconstruction path is buggy, which means **training data quality is not
affected** by the remaining divergences (the model trains on the tokens,
which are correct). The reconstruction bug only matters for visual fidelity
verification (Task 22) and human-readable score export.

| Category | Files | % | Cause |
|---|---|---|---|
| `note` | 7,399 | 13.73% | Mid-measure `*^` voice splits — voice 2 notes appear at measure offset 0 instead of correct elapsed offset |
| `chord` | 5,155 | 9.57% | Same root cause — chords in mid-measure-split voice 2 |
| `tie_open` | 1,493 | 2.77% | Tie position cascades from voice-split offset error |
| `tie_close` | 1,462 | 2.71% | Tie position cascades from voice-split offset error |
| `rest` | 589 | 1.09% | Rest position cascades from voice-split offset error |

## Voice-split timing limitation (planned fix before Task 22)

When kern's `*^` voice-split fires in the **middle** of a measure (after some
notes have already been emitted), the converter correctly tracks the split in
its `column_to_spine` mapping and emits the correct tokens for both voice 1
and voice 2. However, when `src/pipeline/export_musicxml.py::append_tokens_to_part`
reconstructs music21 voices from those tokens, it places every `<voice_2>`
block at offset 0 of the measure — it has no information about how much time
had already elapsed in voice 1 before the split.

**Concrete example** (`beethoven/piano-sonatas/sonata01-2/maj2_down_m-0-5.krn`,
staff=0, offset=7.0): A `*^` fires mid-measure; voice-2's `4f` (F4 quarter)
appears at measure offset 0 (score offset 7.0) instead of the correct measure
offset 1.0 (score offset 8.0). Subsequent voice-2 events cascade one beat
early, generating chains of "missing in our output" / "extra in our output"
divergences.

**Fix scope:** approximately 1-2 days. Requires either:

1. Tracking "quarter-lengths consumed in current measure per voice" before
   the split, then emitting padding rests in voice 2 to fill the gap.
2. Encoding the voice start offset as a token (e.g., `<voice_offset_X>`) so
   the reconstructor knows where to place voice 2.

This will be addressed before Task 22's visual fidelity verification, since
the visual review depends on `append_tokens_to_part` producing correct
music21 output. Until then, the voice-split divergences are a known
limitation of the round-trip path only.

## Failed-to-compare files (80 / 0.1%)

These are kern files where music21's humdrum parser raises an exception
during initial parse — typically due to non-standard or malformed kern
syntax. Our converter would still produce tokens for these files (it is
more lenient), but no round-trip comparison is possible. This is upstream
behavior and out of scope.

## Resolved limitations (now passing)

| Was-limitation | Resolved by |
|---|---|
| Tied notes (`[`/`]`) silently dropped | Task 7 |
| Slurs (`(`/`)`) silently dropped | Task 8 |
| Articulations (`^`, `'`, `;`, `~`) silently dropped | Task 9 |
| Ornaments (`t`, `T`, `m`, `M`) silently dropped | Task 10 |
| Tuplet duration math off by factor of 2 | Task 11 |
| Triplet vs sextuplet ambiguity | Task 12 (resolved per music21's interpretation) |
| Multi-staff parts placed at sequential offsets | Task 14 |
| `HalfStepTrill`/`WholeStepMordent` etc. not recognized | Task 14 |
| Grace notes silently treated as regular notes | Task 14 |
| `tie_end`/`slur_end` after duration not consumed | Task 14 |
| Enharmonic spellings collapsed (`c-` → `B`) | Task 17 (Cb/Fb/B#/E# vocab extension) |
| Octave-1 sub-bass clamped to C2 | Task 17 (octave-1 vocab extension) |
