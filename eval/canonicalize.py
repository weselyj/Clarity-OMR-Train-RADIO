"""Normalize MusicXML so engine-output differences that don't reflect
transcription quality (part names, divisions, tied-note encoding, enharmonic
spelling, layout hints) don't contaminate scoring."""

import copy
from pathlib import Path
from typing import Union
from music21 import converter, stream

CANONICAL_DIVISIONS = 480


def canonicalize(source: Union[Path, str, stream.Score]) -> stream.Score:
    """Return a canonicalized COPY of the score. Does not mutate the input.

    Invariants after canonicalize():
    - Tied notes merged via stripTies() -- a half-note encoded as quarter+quarter-tied
      collapses to a single sounding event. Required for playback-equivalent scoring
      and matches the upstream Clarity-OMR reference scorer (gist 9992cca340).
    - Parts renamed to P1, P2, ... in score order
    - Note durations rounded to the nearest 1/120 quarter (matches divisions=480)
    - Layout/engraver attributes stripped
    """
    if isinstance(source, (str, Path)):
        score = converter.parse(str(source))
    else:
        score = copy.deepcopy(source)

    # 1. Merge tied notes -- same sounding note = single event regardless of notation
    try:
        score = score.stripTies()
    except Exception:
        # stripTies can fail on malformed scores; fall through with the original
        pass

    # 2. Rename parts
    for i, part in enumerate(score.parts, start=1):
        part.partName = f"P{i}"
        part.partAbbreviation = None

    # 3. Normalize duration quantization to divisions=480 (tick = 4/CANONICAL_DIVISIONS quarter)
    tick = 4.0 / CANONICAL_DIVISIONS
    for n in score.recurse().notesAndRests:
        q = float(n.quarterLength)
        n.quarterLength = round(q / tick) * tick

    # 4. Strip layout / engraver attributes.
    LAYOUT_CLASS_NAMES = ("PageLayout", "SystemLayout", "StaffLayout", "LayoutBase")
    for element in list(score.recurse()):
        if any(c in element.classes for c in LAYOUT_CLASS_NAMES):
            site = element.activeSite
            if site is not None:
                site.remove(element, recurse=False)

    return score


def extract_midi_sequence(source: Union[Path, str, stream.Score]) -> list[tuple[int, float, float]]:
    """Return [(midi_pitch, onset_quarters, duration_quarters), ...] in time order.

    Erases enharmonic spelling by using pitch.midi. Called by both symbol_f1 and
    playback scorers downstream.
    """
    if isinstance(source, (str, Path)):
        score = converter.parse(str(source))
    else:
        score = source

    events = []
    for note in score.flatten().notes:
        if note.isChord:
            for p in note.pitches:
                events.append((p.midi, float(note.offset), float(note.quarterLength)))
        else:
            events.append((note.pitch.midi, float(note.offset), float(note.quarterLength)))
    events.sort(key=lambda e: (e[1], e[0]))
    return events
