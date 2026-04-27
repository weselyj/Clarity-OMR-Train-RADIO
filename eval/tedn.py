"""TEDn — Tree Edit Distance (normalized) over MusicXML scores.

Converts a MusicXML score to an ordered labeled tree and computes the
Zhang-Shasha tree edit distance between two such trees using the ``zss``
package, then normalizes by the size of the reference tree.

Tree Structure
--------------
score
└── part:<part-id>
    └── measure:<key-fifths>:<beats>/<beat-type>
        └── voice:<voice-id>
            ├── note:<step><alter><octave>:<duration-type>
            ├── rest:<duration-type>
            └── chord:<count>          ← has note children

Node label conventions
----------------------
* ``note:C04:quarter``   — step=C, alter=0 (natural), octave=4, duration-type=quarter
* ``note:Cs4:quarter``   — C-sharp (alter=+1 → 's'), ``note:Cf4:quarter`` = C-flat (alter=-1 → 'f')
* ``rest:quarter``
* ``chord:3``            — a chord of 3 simultaneous notes (children are note nodes)
* ``measure:0:4/4``      — key signature sharps=0, 4/4 time
* ``part:P1``
* ``score``

Normalization
-------------
TEDn = TED(hyp, ref) / |T_ref|

where |T_ref| is the total number of nodes in the reference tree.
If the reference tree is empty, TEDn is defined as 0.0 (both empty)
or 1.0 (reference empty, hypothesis non-empty).

Punted / out-of-scope
---------------------
* Percussion parts (unpitched notes) — labeled as ``note:X0:duration`` (step=X, octave=0)
* Lyrics, dynamics, articulation, slurs, ties — all ignored (only pitch+duration retained)
* Multiple staves within a single part — flattened into a single voice stream
* Grace notes — labeled with duration-type ``grace``
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import music21
import music21.stream
import music21.note
import music21.chord
import music21.key
import music21.meter
import zss


# ---------------------------------------------------------------------------
# Node — lightweight tree node wrapping zss.Node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """Ordered labeled tree node for Zhang-Shasha TED via ``zss``."""

    label: str
    children: List["Node"] = field(default_factory=list)

    def add_child(self, child: "Node") -> "Node":
        self.children.append(child)
        return self

    def to_zss(self) -> zss.Node:
        """Convert to a ``zss.Node`` tree (recursive)."""
        z = zss.Node(self.label)
        for child in self.children:
            z.addkid(child.to_zss())
        return z


# ---------------------------------------------------------------------------
# Label helpers
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
    """Return a canonical duration type string."""
    t = dur.type
    if not t or t == "inexpressible":
        # Fall back to rounded quarter-length as a fraction string
        ql = dur.quarterLength
        try:
            from fractions import Fraction
            return str(Fraction(ql).limit_denominator(64))
        except Exception:
            return str(round(ql, 4))
    return t


def _note_label(n: music21.note.Note) -> str:
    step = n.pitch.step
    alter = _alter_str(n.pitch.accidental.alter if n.pitch.accidental else None)
    octave = str(n.pitch.octave) if n.pitch.octave is not None else "0"
    dur = _duration_str(n.duration)
    return f"note:{step}{alter}{octave}:{dur}"


def _rest_label(r: music21.note.Rest) -> str:
    dur = _duration_str(r.duration)
    return f"rest:{dur}"


def _chord_label(c: music21.chord.Chord) -> str:
    return f"chord:{len(c.pitches)}"


def _measure_label(m: music21.stream.Measure) -> str:
    # Extract key signature from the measure (or default C major = 0)
    key_fifths = 0
    for ks in m.getElementsByClass(music21.key.KeySignature):
        key_fifths = ks.sharps
        break

    # Time signature
    beats = "4"
    beat_type = "4"
    for ts in m.getElementsByClass(music21.meter.TimeSignature):
        beats = str(ts.numerator)
        beat_type = str(ts.denominator)
        break

    return f"measure:{key_fifths}:{beats}/{beat_type}"


# ---------------------------------------------------------------------------
# score_to_tree
# ---------------------------------------------------------------------------

def score_to_tree(score: music21.stream.Score) -> Node:
    """Convert a music21 Score to an ordered labeled Node tree.

    The returned tree has the structure::

        score
        └── part:<id>
            └── measure:<key>:<time>
                └── voice:<id>
                    ├── note:...
                    ├── rest:...
                    └── chord:<n>
                        ├── note:...
                        └── ...

    Parameters
    ----------
    score:
        A parsed music21 Score stream.

    Returns
    -------
    Node
        Root node with label ``"score"``.
    """
    root = Node("score")

    parts = list(score.parts)
    for part in parts:
        part_id = str(part.id) if part.id else "P"
        part_node = Node(f"part:{part_id}")
        root.add_child(part_node)

        # Use chordify + voice flattening to handle multi-staff gracefully.
        # We iterate measures directly to preserve voice structure.
        measures = list(part.getElementsByClass(music21.stream.Measure))
        for m in measures:
            m_node = Node(_measure_label(m))
            part_node.add_child(m_node)

            # Collect voices; fall back to measure-level flat iteration
            voices = list(m.getElementsByClass(music21.stream.Voice))
            if not voices:
                # Wrap flat measure content into a synthetic voice
                voices = [m]  # treat measure as a single voice

            for voice in voices:
                # Determine voice id
                if isinstance(voice, music21.stream.Voice):
                    vid = str(voice.id) if voice.id else "1"
                else:
                    vid = "1"
                voice_node = Node(f"voice:{vid}")
                m_node.add_child(voice_node)

                # Walk elements; detect chords (consecutive notes with <chord/> tag
                # are presented by music21 as Chord objects directly).
                for elem in voice.notesAndRests:
                    if isinstance(elem, music21.chord.Chord):
                        ch_node = Node(_chord_label(elem))
                        for p in elem.pitches:
                            # Build a synthetic note label for each chord tone
                            alter = _alter_str(p.accidental.alter if p.accidental else None)
                            octave = str(p.octave) if p.octave is not None else "0"
                            dur = _duration_str(elem.duration)
                            ch_node.add_child(Node(f"note:{p.step}{alter}{octave}:{dur}"))
                        voice_node.add_child(ch_node)
                    elif isinstance(elem, music21.note.Note):
                        voice_node.add_child(Node(_note_label(elem)))
                    elif isinstance(elem, music21.note.Rest):
                        voice_node.add_child(Node(_rest_label(elem)))
                    # else: ignore other element types (barlines, etc.)

    return root


# ---------------------------------------------------------------------------
# Tree size helper
# ---------------------------------------------------------------------------

def _tree_size(node: Node) -> int:
    return 1 + sum(_tree_size(c) for c in node.children)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_tedn(
    reference_path: "str | Path",
    hypothesis_path: "str | Path",
) -> float:
    """Compute Tree Edit Distance normalized (TEDn) between two MusicXML files.

    TEDn = TED(hyp, ref) / |T_ref|

    where TED is the Zhang-Shasha tree edit distance (unit cost: insert/delete/rename)
    and |T_ref| is the number of nodes in the reference tree.

    Parameters
    ----------
    reference_path:
        Path to the reference MusicXML file.
    hypothesis_path:
        Path to the hypothesis (predicted) MusicXML file.

    Returns
    -------
    float
        Normalized tree edit distance in [0, ~1] for well-formed scores.
        Returns 0.0 if both trees are empty.
        Returns 1.0 if the reference tree is empty but hypothesis is non-empty
        (undefined normalization falls back to 1.0).

    Raises
    ------
    FileNotFoundError
        If either path does not exist.
    music21.Music21Exception
        If either file cannot be parsed by music21.
    """
    ref_path = Path(reference_path)
    hyp_path = Path(hypothesis_path)

    ref_score = music21.converter.parse(str(ref_path))
    hyp_score = music21.converter.parse(str(hyp_path))

    ref_tree = score_to_tree(ref_score)
    hyp_tree = score_to_tree(hyp_score)

    ref_size = _tree_size(ref_tree)
    if ref_size == 0:
        hyp_size = _tree_size(hyp_tree)
        return 0.0 if hyp_size == 0 else 1.0

    ted = zss.simple_distance(
        hyp_tree.to_zss(),
        ref_tree.to_zss(),
        get_children=lambda n: n.children,
        get_label=lambda n: n.label,
        label_dist=lambda a, b: 0 if a == b else 1,
    )

    return ted / ref_size
