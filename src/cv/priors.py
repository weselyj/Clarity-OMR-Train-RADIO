"""Data structures for CV-detected musical elements."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DIATONIC_NOTES = ["C", "D", "E", "F", "G", "A", "B"]

# Bottom line pitch for each clef (note_name, octave)
CLEF_BOTTOM_LINE: Dict[str, Tuple[str, int]] = {
    "clef-G2": ("E", 4),
    "clef-F4": ("G", 2),
    "clef-C3": ("F", 3),
    "clef-C4": ("D", 3),
    "clef-C1": ("C", 4),
    "clef-G2_8vb": ("E", 3),
    "clef-G2_8va": ("E", 5),
}


def staff_position_to_pitch(position: float, clef: str) -> Optional[str]:
    """Convert staff position to pitch name.

    Position 0 = bottom line, 1 = first space, 2 = second line, etc.
    Each position is one diatonic step. Supports ledger lines (negative
    positions below staff, positions > 8 above staff).
    """
    if clef not in CLEF_BOTTOM_LINE:
        return None
    base_note, base_octave = CLEF_BOTTOM_LINE[clef]
    base_index = DIATONIC_NOTES.index(base_note)
    rounded = round(position)
    total_index = base_index + rounded
    octave = base_octave + total_index // 7
    note = DIATONIC_NOTES[total_index % 7]
    return f"note-{note}{octave}"


@dataclass
class StaffLineInfo:
    y_positions: List[int]
    spacing: float
    top: int
    bottom: int


@dataclass
class NoteheadDetection:
    x: float
    y: float
    w: int
    h: int
    area: int
    staff_position: float = 0.0
    estimated_pitch: Optional[str] = None
    confidence: float = 0.0
    is_filled: bool = True


@dataclass
class BarlineDetection:
    x: float
    x_normalized: float = 0.0
    confidence: float = 0.0


@dataclass
class OnsetCluster:
    x_center: float
    noteheads: List[NoteheadDetection] = field(default_factory=list)
    is_chord: bool = False
    note_count: int = 0
    confidence: float = 0.0


@dataclass
class MeasureSkeleton:
    index: int
    start_x: float
    end_x: float
    onsets: List[OnsetCluster] = field(default_factory=list)
    note_count: int = 0


@dataclass
class StaffSkeleton:
    """Complete CV analysis result for one staff crop."""

    staff_lines: Optional[StaffLineInfo] = None
    barlines: List[BarlineDetection] = field(default_factory=list)
    noteheads: List[NoteheadDetection] = field(default_factory=list)
    onset_clusters: List[OnsetCluster] = field(default_factory=list)
    measures: List[MeasureSkeleton] = field(default_factory=list)

    estimated_clef: Optional[str] = None
    clef_confidence: float = 0.0

    total_note_count: int = 0
    note_count_confidence: float = 0.0
    estimated_measure_count: int = 0
    measure_count_confidence: float = 0.0

    image_width: int = 0
    image_height: int = 0

    def summary(self) -> str:
        lines = [
            f"StaffSkeleton: {self.image_width}x{self.image_height}",
            f"  Staff lines: {len(self.staff_lines.y_positions) if self.staff_lines else 0} "
            f"(spacing={self.staff_lines.spacing:.1f}px)" if self.staff_lines else "  Staff lines: none",
            f"  Barlines: {len(self.barlines)}",
            f"  Noteheads: {len(self.noteheads)}",
            f"  Onset clusters: {len(self.onset_clusters)} "
            f"({sum(1 for c in self.onset_clusters if c.is_chord)} chords)",
            f"  Measures: {self.estimated_measure_count} (conf={self.measure_count_confidence:.2f})",
            f"  Clef: {self.estimated_clef} (conf={self.clef_confidence:.2f})",
        ]
        if self.onset_clusters:
            chord_sizes = [c.note_count for c in self.onset_clusters if c.is_chord]
            if chord_sizes:
                lines.append(f"  Chord sizes: {chord_sizes}")
        return "\n".join(lines)
