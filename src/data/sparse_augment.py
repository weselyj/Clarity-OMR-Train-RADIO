"""MusicXML transformations to produce sparse-content training pages.

These transformations target underrepresented visual patterns in the synthetic
training set:
- strip_silent_intro: remove parts that contain only rests across the entire
  piece. (v1 limitation: music21 doesn't easily support per-measure-range
  part removal; this is a strict superset of the intended behavior.)
- add_multibar_rest: insert whole rests into measures 1..N of a part to
  produce the bold-bar-count visual.
- thin_vocal_part: replace all notes EXCEPT the first N in a part with rests
  of the same duration (preserves measure timing) — produces the sparse
  vocal-with-pickup pattern from lc6623145 Sys 0.
"""
from __future__ import annotations

from pathlib import Path

import music21


def strip_silent_intro(source: Path, out_path: Path, intro_measures: int = 4) -> None:
    """Remove parts that contain only rests across the entire piece.

    NOTE: v1 limitation. The intended behavior is to remove a part for just
    the first `intro_measures`; music21 doesn't expose a clean way to do that.
    This implementation strips parts that are silent ENTIRELY — a strict
    superset, but useful when intro-only-silent parts exist alongside
    fully-silent parts in a real corpus.

    Args:
        source: input MusicXML/MXL path
        out_path: where to write the transformed score
        intro_measures: kept for API compatibility; currently unused
    """
    score = music21.converter.parse(str(source))
    silent_parts = []
    for part in score.parts:
        notes = list(part.recurse().notes)
        if not notes:
            silent_parts.append(part)
    for part in silent_parts:
        score.remove(part)
    score.write("musicxml", fp=str(out_path))


def add_multibar_rest(
    out_path: Path,
    source: Path,
    part_name: str,
    n_measures: int = 8,
) -> None:
    """Replace the contents of measures 1..n_measures of `part_name` with whole rests."""
    score = music21.converter.parse(str(source))
    part = next((p for p in score.parts if p.partName == part_name), None)
    if part is None:
        raise ValueError(f"Part {part_name!r} not in score")
    for m_idx in range(1, n_measures + 1):
        m = part.measure(m_idx)
        if m is None:
            break
        # Remove existing notes/rests
        for el in list(m.notesAndRests):
            m.remove(el)
        # Append a whole-measure rest. quarterLength=4 covers 4/4 only;
        # for v1 we accept this limitation (lieder is overwhelmingly 4/4 or 3/4).
        ts = m.timeSignature or part.recurse().getElementsByClass("TimeSignature").first()
        if ts is not None:
            rest_ql = ts.barDuration.quarterLength
        else:
            rest_ql = 4.0
        m.append(music21.note.Rest(quarterLength=rest_ql))
    score.write("musicxml", fp=str(out_path))


def thin_vocal_part(
    source: Path,
    out_path: Path,
    part_name: str,
    keep_n_notes: int = 1,
) -> None:
    """Keep first `keep_n_notes` notes in `part_name`; replace the rest with rests."""
    score = music21.converter.parse(str(source))
    part = next((p for p in score.parts if p.partName == part_name), None)
    if part is None:
        raise ValueError(f"Part {part_name!r} not in score")
    kept = 0
    for el in list(part.recurse().notes):
        if kept < keep_n_notes:
            kept += 1
            continue
        measure = el.activeSite
        ql = el.quarterLength
        offset = el.offset
        rest = music21.note.Rest(quarterLength=ql)
        measure.remove(el)
        measure.insert(offset, rest)
    score.write("musicxml", fp=str(out_path))
