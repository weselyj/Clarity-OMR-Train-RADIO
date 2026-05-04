"""Round-trip validation framework: compare music21's humdrum parser output
against our convert_kern_file output, both canonicalized to the level our
OMR token vocabulary can express.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class CanonicalEvent:
    """One musical event in score order, normalized to vocab-level detail."""
    offset_ql: float                    # quarterLength offset within the score
    kind: str                           # 'note' | 'rest' | 'chord' | 'tie_open' | 'tie_close'
                                        # | 'slur_open' | 'slur_close' | 'articulation'
                                        # | 'ornament' | 'measure_boundary' | 'clef'
                                        # | 'key' | 'time'
    payload: Any                        # kind-specific (pitch+dur tuple, articulation type, etc.)
    staff_idx: int                      # 0-based, top-down


@dataclass
class Divergence:
    """One mismatch between reference and our canonical output."""
    kind: str
    offset_ql: float
    staff_idx: int
    ref_value: Any                      # what music21 emitted
    our_value: Any                      # what we emitted (None = missing in our output)
    note: str = ""                      # human-readable description


@dataclass
class CompareResult:
    """Outcome of comparing one .krn file via both paths."""
    kern_path: Path
    ref_canonical: List[CanonicalEvent]
    our_canonical: List[CanonicalEvent]
    divergences: List[Divergence] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.divergences) == 0


def canonicalize_score(score) -> List[CanonicalEvent]:
    """Walk a music21 score and emit a flat ordered list of canonical events.

    Strips music21 richness our vocab can't express (lyrics, fingerings, beaming,
    exact stem direction, etc.). The output is sorted by (staff_idx, offset_ql).
    """
    import music21

    events: List[CanonicalEvent] = []

    parts = list(getattr(score, "parts", []) or [])
    for staff_idx, part in enumerate(parts):
        for elem in part.recurse().notesAndRests:
            try:
                offset_ql = float(elem.getOffsetInHierarchy(score))
            except Exception:
                offset_ql = float(getattr(elem, "offset", 0.0))
            duration_ql = float(elem.duration.quarterLength)

            if isinstance(elem, music21.chord.Chord) and not isinstance(elem, music21.note.Note):
                pitches = tuple(p.nameWithOctave for p in elem.pitches)
                events.append(
                    CanonicalEvent(
                        offset_ql=offset_ql,
                        kind="chord",
                        payload=(pitches, duration_ql),
                        staff_idx=staff_idx,
                    )
                )
            elif isinstance(elem, music21.note.Rest):
                events.append(
                    CanonicalEvent(
                        offset_ql=offset_ql,
                        kind="rest",
                        payload=duration_ql,
                        staff_idx=staff_idx,
                    )
                )
            elif isinstance(elem, music21.note.Note):
                events.append(
                    CanonicalEvent(
                        offset_ql=offset_ql,
                        kind="note",
                        payload=(elem.pitch.nameWithOctave, duration_ql),
                        staff_idx=staff_idx,
                    )
                )

            # Tie open/close events (emit alongside the underlying note/chord)
            if hasattr(elem, "tie") and elem.tie is not None:
                tie_type = getattr(elem.tie, "type", None)
                if tie_type in ("start", "continue"):
                    events.append(
                        CanonicalEvent(
                            offset_ql=offset_ql,
                            kind="tie_open",
                            payload=None,
                            staff_idx=staff_idx,
                        )
                    )
                if tie_type in ("stop", "continue"):
                    events.append(
                        CanonicalEvent(
                            offset_ql=offset_ql,
                            kind="tie_close",
                            payload=None,
                            staff_idx=staff_idx,
                        )
                    )

            # Articulations (each maps to a name like 'accent', 'staccato')
            articulation_name_map = {
                "Accent": "accent",
                "Staccato": "staccato",
                "Staccatissimo": "staccatissimo",
                "Tenuto": "tenuto",
                "StrongAccent": "marcato",
            }
            for art in getattr(elem, "articulations", []) or []:
                name = articulation_name_map.get(type(art).__name__)
                if name is not None:
                    events.append(
                        CanonicalEvent(
                            offset_ql=offset_ql,
                            kind="articulation",
                            payload=name,
                            staff_idx=staff_idx,
                        )
                    )

            # Ornaments + fermata (live in .expressions)
            ornament_name_map = {
                "Trill": "trill",
                "Mordent": "mordent",
                "InvertedMordent": "inverted_mordent",
                "Turn": "turn",
                "Fermata": "fermata",
            }
            for expr in getattr(elem, "expressions", []) or []:
                name = ornament_name_map.get(type(expr).__name__)
                if name is not None:
                    events.append(
                        CanonicalEvent(
                            offset_ql=offset_ql,
                            kind="ornament",
                            payload=name,
                            staff_idx=staff_idx,
                        )
                    )

    events.sort(key=lambda e: (e.staff_idx, e.offset_ql, e.kind))
    return events
