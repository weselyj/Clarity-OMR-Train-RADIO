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


def _tuplet_ratio(elem) -> tuple | None:
    """Return (numNotesActual, numNotesNormal) of the outermost tuplet on this element, or None."""
    tuplets = list(getattr(elem.duration, "tuplets", []) or [])
    if not tuplets:
        return None
    t = tuplets[0]
    actual = int(getattr(t, "numberNotesActual", 0) or 0)
    normal = int(getattr(t, "numberNotesNormal", 0) or 0)
    if actual == 0 or normal == 0:
        return None
    return (actual, normal)


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
                        payload=(pitches, duration_ql, _tuplet_ratio(elem)),
                        staff_idx=staff_idx,
                    )
                )
            elif isinstance(elem, music21.note.Rest):
                events.append(
                    CanonicalEvent(
                        offset_ql=offset_ql,
                        kind="rest",
                        payload=(duration_ql, _tuplet_ratio(elem)),
                        staff_idx=staff_idx,
                    )
                )
            elif isinstance(elem, music21.note.Note):
                events.append(
                    CanonicalEvent(
                        offset_ql=offset_ql,
                        kind="note",
                        payload=(elem.pitch.nameWithOctave, duration_ql, _tuplet_ratio(elem)),
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


def compare_via_music21(kern_path: Path) -> CompareResult:
    """Round-trip validation:
      1. Parse kern_path via music21's humdrum subconverter -> reference score.
      2. Parse kern_path via our convert_kern_file + append_tokens_to_part -> our score.
      3. Canonicalize both, diff, return CompareResult.
    """
    import music21
    from src.data.convert_tokens import convert_kern_file
    from src.pipeline.export_musicxml import append_tokens_to_part

    # Reference: music21 humdrum parser.
    ref_score = music21.converter.parse(str(kern_path), format="humdrum")
    ref_canonical = canonicalize_score(ref_score)

    # Our path: tokens -> music21 score (one Part per top-level <staff_start>...<staff_end> block).
    tokens = convert_kern_file(kern_path)
    our_score = music21.stream.Score()
    cur_part: List[str] | None = None
    for tok in tokens:
        if tok == "<staff_start>":
            cur_part = []
        elif tok == "<staff_end>":
            if cur_part is not None:
                part = music21.stream.Part()
                # Skip the optional <staff_idx_N> marker token (it's metadata, not music).
                staff_tokens = [t for t in cur_part if not t.startswith("<staff_idx_")]
                try:
                    append_tokens_to_part(part, staff_tokens)
                except Exception:
                    # Conversion failure: still append the empty part so staff_idx alignment holds.
                    pass
                our_score.append(part)
                cur_part = None
        elif tok in {"<bos>", "<eos>"}:
            continue
        elif cur_part is not None:
            cur_part.append(tok)

    our_canonical = canonicalize_score(our_score)

    # Diff: simple ordered alignment by (staff_idx, offset_ql, kind).
    divergences: List[Divergence] = []
    ref_keyed = {(e.staff_idx, round(e.offset_ql, 4), e.kind): e for e in ref_canonical}
    our_keyed = {(e.staff_idx, round(e.offset_ql, 4), e.kind): e for e in our_canonical}

    all_keys = set(ref_keyed) | set(our_keyed)
    for key in sorted(all_keys):
        staff_idx, offset_ql, kind = key
        ref = ref_keyed.get(key)
        our = our_keyed.get(key)
        if ref is not None and our is not None:
            if ref.payload != our.payload:
                divergences.append(
                    Divergence(
                        kind=kind, offset_ql=offset_ql, staff_idx=staff_idx,
                        ref_value=ref.payload, our_value=our.payload,
                        note="payload mismatch",
                    )
                )
        elif ref is not None:
            divergences.append(
                Divergence(
                    kind=kind, offset_ql=offset_ql, staff_idx=staff_idx,
                    ref_value=ref.payload, our_value=None, note="missing in our output",
                )
            )
        else:
            divergences.append(
                Divergence(
                    kind=kind, offset_ql=offset_ql, staff_idx=staff_idx,
                    ref_value=None, our_value=our.payload, note="extra in our output",
                )
            )

    return CompareResult(
        kern_path=kern_path,
        ref_canonical=ref_canonical,
        our_canonical=our_canonical,
        divergences=divergences,
    )


def summarize_divergences(results: List[CompareResult]) -> dict:
    """Frequency-sorted divergence-category report."""
    from collections import defaultdict

    per_kind_count: dict = defaultdict(int)
    per_kind_files: dict = defaultdict(set)
    per_kind_samples: dict = defaultdict(list)

    for r in results:
        kinds_in_file: set = set()
        for d in r.divergences:
            per_kind_count[d.kind] += 1
            kinds_in_file.add(d.kind)
            if len(per_kind_samples[d.kind]) < 5:
                per_kind_samples[d.kind].append(
                    {
                        "kern_path": str(r.kern_path),
                        "offset_ql": d.offset_ql,
                        "staff_idx": d.staff_idx,
                        "ref_value": str(d.ref_value),
                        "our_value": str(d.our_value),
                        "note": d.note,
                    }
                )
        for kind in kinds_in_file:
            per_kind_files[kind].add(str(r.kern_path))

    total_files = len(results)
    summary: dict = {}
    for kind in per_kind_count:
        summary[kind] = {
            "occurrence_count": per_kind_count[kind],
            "files_with_kind": len(per_kind_files[kind]),
            "files_with_kind_pct": (len(per_kind_files[kind]) / total_files * 100.0) if total_files else 0.0,
            "sample_divergences": per_kind_samples[kind],
        }
    # Sort by files_with_kind_pct descending.
    return dict(sorted(summary.items(), key=lambda item: -item[1]["files_with_kind_pct"]))
