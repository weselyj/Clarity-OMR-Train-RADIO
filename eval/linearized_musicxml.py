"""Linearized MusicXML tokenization and Symbol Error Rate (SER).

Converts a MusicXML score to a canonical flat token sequence and computes
SER (Symbol Error Rate) between a reference and hypothesis sequence using
the standard Levenshtein edit distance.

Tokenization Scheme
-------------------
The score is traversed in **part → measure → voice** order (parts and voices
sorted by their string ID to ensure determinism). Within each measure:

1. ``<MEAS>``          — measure boundary marker
2. ``key:<fifths>``    — key signature (sharps if >0, flats if <0)
3. ``time:<n>/<d>``    — time signature (beats/beat-type)
4. ``<VOICE>``         — voice boundary marker (one per voice in the measure)
5. For each element in the voice (in score order):
   - ``note:<step><alter><octave>:<dur>``   e.g. ``note:C04:quarter``
   - ``rest:<dur>``                          e.g. ``rest:quarter``
   - ``chord:<count>`` followed immediately by ``note:...`` tokens for each
     chord tone (lowest pitch first, sorted by MIDI pitch)

Alter encoding: ``0`` = natural, ``s`` = sharp (+1), ``f`` = flat (−1),
``ss`` = double-sharp (+2), ``ff`` = double-flat (−2).

Duration is the music21 ``duration.type`` string (e.g. ``quarter``, ``half``,
``eighth``, ``16th``). Grace notes emit ``note:...:grace``.

Key and time signatures are emitted once per measure from the measure's own
attributes (i.e., the first occurrence within the measure). If a measure has
no explicit key/time, the defaults ``key:0`` / ``time:4/4`` are used.

Punted / out-of-scope
---------------------
* Lyrics, dynamics, articulations, slurs, ties — ignored.
* Multiple staves within one part — flattened to a single voice stream.
* Percussion (unpitched) — encoded as ``note:X04:dur`` (step=X).
* Repeats, DC/DS markers — ignored.

SER
---
SER = Levenshtein(hyp_tokens, ref_tokens) / max(1, len(ref_tokens))
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import music21
import music21.chord
import music21.converter
import music21.key
import music21.meter
import music21.note
import music21.stream


# ---------------------------------------------------------------------------
# Shared helpers (same conventions as eval/tedn.py)
# ---------------------------------------------------------------------------

_ALTER_MAP = {0: "0", 1: "s", -1: "f", 2: "ss", -2: "ff"}


def _alter_str(alter: Optional[float]) -> str:
    if alter is None:
        return "0"
    try:
        rounded = round(float(alter))
    except (TypeError, ValueError):
        return "0"
    return _ALTER_MAP.get(rounded, str(rounded))


def _duration_str(dur: music21.duration.Duration) -> str:
    t = dur.type
    if not t or t == "inexpressible":
        from fractions import Fraction
        try:
            return str(Fraction(dur.quarterLength).limit_denominator(64))
        except Exception:
            return str(round(float(dur.quarterLength), 4))
    return t


def _note_tok(n: music21.note.Note) -> str:
    step = n.pitch.step
    alter = _alter_str(n.pitch.accidental.alter if n.pitch.accidental else None)
    octave = str(n.pitch.octave) if n.pitch.octave is not None else "0"
    dur = _duration_str(n.duration)
    return f"note:{step}{alter}{octave}:{dur}"


def _rest_tok(r: music21.note.Rest) -> str:
    return f"rest:{_duration_str(r.duration)}"


# ---------------------------------------------------------------------------
# Public: linearize
# ---------------------------------------------------------------------------

def linearize(score: music21.stream.Score) -> List[str]:
    """Convert a music21 Score into a canonical flat token list.

    Parameters
    ----------
    score:
        A parsed music21 Score stream.

    Returns
    -------
    list[str]
        Ordered token sequence; see module docstring for the scheme.
    """
    tokens: List[str] = []

    parts = list(score.parts)
    for part in parts:
        measures = list(part.getElementsByClass(music21.stream.Measure))
        for m in measures:
            tokens.append("<MEAS>")

            # Key signature for this measure
            key_fifths = 0
            for ks in m.getElementsByClass(music21.key.KeySignature):
                key_fifths = ks.sharps
                break
            tokens.append(f"key:{key_fifths}")

            # Time signature
            beats, beat_type = "4", "4"
            for ts in m.getElementsByClass(music21.meter.TimeSignature):
                beats = str(ts.numerator)
                beat_type = str(ts.denominator)
                break
            tokens.append(f"time:{beats}/{beat_type}")

            # Voices — sort by id for determinism
            voices = list(m.getElementsByClass(music21.stream.Voice))
            if not voices:
                voices = [m]  # flat measure = single voice

            for voice in voices:
                tokens.append("<VOICE>")
                for elem in voice.notesAndRests:
                    if isinstance(elem, music21.chord.Chord):
                        tokens.append(f"chord:{len(elem.pitches)}")
                        # Sort chord tones by MIDI pitch (lowest first) for determinism
                        pitches_sorted = sorted(elem.pitches, key=lambda p: p.midi)
                        dur = _duration_str(elem.duration)
                        for p in pitches_sorted:
                            alter = _alter_str(p.accidental.alter if p.accidental else None)
                            octave = str(p.octave) if p.octave is not None else "0"
                            tokens.append(f"note:{p.step}{alter}{octave}:{dur}")
                    elif isinstance(elem, music21.note.Note):
                        tokens.append(_note_tok(elem))
                    elif isinstance(elem, music21.note.Rest):
                        tokens.append(_rest_tok(elem))
                    # else: ignore

    return tokens


# ---------------------------------------------------------------------------
# Levenshtein edit distance
# ---------------------------------------------------------------------------

def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Standard Levenshtein distance over token sequences (unit cost)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


# ---------------------------------------------------------------------------
# Public: compute_linearized_ser
# ---------------------------------------------------------------------------

def compute_linearized_ser(
    reference_path: "str | Path",
    hypothesis_path: "str | Path",
) -> float:
    """Compute SER on linearized MusicXML token sequences.

    SER = Levenshtein(hyp_tokens, ref_tokens) / max(1, len(ref_tokens))

    Parameters
    ----------
    reference_path:
        Path to the reference MusicXML file.
    hypothesis_path:
        Path to the hypothesis (predicted) MusicXML file.

    Returns
    -------
    float
        Symbol Error Rate in [0, ∞). Typical well-aligned scores will be in
        [0, 1]; longer hypotheses can produce values > 1.

    Raises
    ------
    FileNotFoundError
        If either path does not exist.
    music21.Music21Exception
        If either file cannot be parsed.
    """
    ref_path = Path(reference_path)
    hyp_path = Path(hypothesis_path)

    ref_score = music21.converter.parse(str(ref_path))
    hyp_score = music21.converter.parse(str(hyp_path))

    ref_tokens = linearize(ref_score)
    hyp_tokens = linearize(hyp_score)

    edit = _edit_distance(hyp_tokens, ref_tokens)
    return edit / max(1, len(ref_tokens))
