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
